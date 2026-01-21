import os
import time
import traceback
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

from shared import (
    supabase_client_env,
    load_sweetspot,
    fetch_hl_universe,
    build_macro_series,
    detect_call_for_coin,
    db_upsert_calls,
    db_read_calls,
    update_call_status,
    db_update_call,
    fetch_candles,
    to_ms,
)

WORKER_VERSION = "v2026-01-21-ui-scan-snapshot"

HEARTBEAT_NAME = "scanner"
SCAN_SLEEP_SECONDS = int(os.getenv("WORKER_SLEEP_SECONDS", "60"))
STATUS_UPDATE_TOP_N = int(os.getenv("STATUS_UPDATE_TOP_N", "120"))
UPDATE_OPEN_LOOKBACK_HOURS = int(os.getenv("UPDATE_OPEN_LOOKBACK_HOURS", "24"))
HEARTBEAT_PROGRESS_EVERY = int(os.getenv("HEARTBEAT_PROGRESS_EVERY", "10"))

SCAN_ONLY_ON_NEW_15M_BAR = os.getenv("SCAN_ONLY_ON_NEW_15M_BAR", "1") == "1"
STATUS_UPDATE_EVERY_N_SCANS = int(os.getenv("STATUS_UPDATE_EVERY_N_SCANS", "3"))

def hb(sb, note: str):
    sb.table("worker_heartbeat").upsert(
        {
            "name": HEARTBEAT_NAME,
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "note": f"{WORKER_VERSION} | {note}",
        },
        on_conflict="name",
    ).execute()
    print(f"[hb] {WORKER_VERSION} | {note}", flush=True)

def upsert_scan_snapshots(sb, snapshots: List[Dict[str, Any]]) -> int:
    if not snapshots:
        return 0
    sb.table("scan_snapshots").upsert(snapshots, on_conflict="coin,bar_time").execute()
    return len(snapshots)

def get_last_bar_snapshot(coin: str, now_utc: datetime) -> Optional[Dict[str, Any]]:
    """
    Fetch last 2 candles to compute 15m change%.
    Extra HL call per coin, but we only run on new 15m bar.
    """
    start = now_utc - timedelta(hours=2)
    df = fetch_candles(coin, to_ms(start), to_ms(now_utc))
    if df is None or df.empty or len(df) < 2:
        return None

    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    last = df.iloc[-1]
    prev = df.iloc[-2]

    last_close = float(last["close"])
    prev_close = float(prev["close"]) if float(prev["close"]) != 0 else last_close
    chg = (last_close / prev_close - 1.0) * 100.0

    bar_time = pd.to_datetime(last["timestamp"], utc=True).to_pydatetime()

    return {
        "coin": str(coin),
        "bar_time": bar_time.isoformat(),
        "updated_time": now_utc.isoformat(),
        "close_price": last_close,
        "change_15m_pct": float(chg),
    }

def main():
    sb = supabase_client_env()
    hb(sb, "BOOT: started")

    sweet = load_sweetspot()
    sweet["coin"] = sweet["coin"].astype(str)
    coins = sweet["coin"].tolist()
    hb(sb, f"Loaded sweetspot coins: {len(coins)}")

    try:
        uni = set(fetch_hl_universe())
        coins = [c for c in coins if c in uni]
        hb(sb, f"HL universe ok. sweetspot_on_hl={len(coins)}")
    except Exception as e:
        hb(sb, f"WARN HL universe check failed: {type(e).__name__}: {str(e)[:120]}")

    last_bar_ts = None
    scan_count = 0

    while True:
        loop_start = datetime.now(timezone.utc)
        scanned = 0
        found = 0
        upserted = 0
        snap_upserted = 0

        try:
            hb(sb, "LOOP: start")

            macro = build_macro_series(loop_start)
            if macro is None or macro.empty:
                hb(sb, "macro empty -> sleep")
                time.sleep(SCAN_SLEEP_SECONDS)
                continue

            if SCAN_ONLY_ON_NEW_15M_BAR:
                try:
                    current_bar_ts = pd.to_datetime(macro["timestamp"].iloc[-1], utc=True)
                except Exception:
                    current_bar_ts = None

                if current_bar_ts is not None and last_bar_ts is not None and current_bar_ts <= last_bar_ts:
                    hb(sb, f"no new 15m bar -> skip ({current_bar_ts.isoformat()})")
                    time.sleep(SCAN_SLEEP_SECONDS)
                    continue

                last_bar_ts = current_bar_ts
                hb(sb, f"new 15m bar -> scanning ({last_bar_ts.isoformat() if last_bar_ts is not None else 'na'})")

            new_rows = []
            snapshots = []

            for i, coin in enumerate(coins, start=1):
                scanned += 1
                if i % HEARTBEAT_PROGRESS_EVERY == 0:
                    hb(sb, f"progress {i}/{len(coins)}")

                base_wr = 0.55
                try:
                    wr = sweet.loc[sweet["coin"] == coin, "winrate"]
                    if len(wr) > 0:
                        base_wr = float(wr.iloc[0])
                except Exception:
                    pass

                # Snapshot for UI (price + 15m change)
                try:
                    s = get_last_bar_snapshot(coin, loop_start)
                    if s:
                        snapshots.append(s)
                except Exception as e:
                    print(f"[snap][WARN] {coin} snapshot failed: {e}", flush=True)

                # Signal detection
                try:
                    call = detect_call_for_coin(
                        coin=coin,
                        macro=macro,
                        base_wr=base_wr,
                        end=loop_start,
                        detected_time=loop_start,
                    )
                except Exception as e:
                    print(f"[scan][WARN] {coin} detect failed: {e}", flush=True)
                    continue

                if call is not None:
                    new_rows.append(call)
                    found += 1

            scan_count += 1

            if new_rows:
                try:
                    upserted = db_upsert_calls(sb, new_rows)
                except Exception as e:
                    hb(sb, f"ERROR upsert calls: {type(e).__name__}: {str(e)[:120]}")
                    print("[ERROR] upsert calls failed:", e, flush=True)

            if snapshots:
                try:
                    snap_upserted = upsert_scan_snapshots(sb, snapshots)
                except Exception as e:
                    hb(sb, f"ERROR upsert snapshots: {type(e).__name__}: {str(e)[:120]}")
                    print("[ERROR] upsert snapshots failed:", e, flush=True)

            # Status update less often
            if scan_count % STATUS_UPDATE_EVERY_N_SCANS == 0:
                try:
                    calls_df = db_read_calls(sb, limit=4000)
                    if not calls_df.empty:
                        cutoff = loop_start - timedelta(hours=UPDATE_OPEN_LOOKBACK_HOURS)
                        open_df = calls_df[(calls_df["call_time"] >= cutoff) & (calls_df["status"] == "OPEN")].copy()
                        if not open_df.empty:
                            open_df = open_df.sort_values("call_time", ascending=False).head(STATUS_UPDATE_TOP_N)

                        for _, row in open_df.iterrows():
                            try:
                                upd = update_call_status(row, end=loop_start)
                                if (
                                    str(upd.get("status")) != str(row.get("status"))
                                    or float(upd.get("pnl_pct", 0.0)) != float(row.get("pnl_pct", 0.0))
                                    or float(upd.get("last_price", 0.0)) != float(row.get("last_price", 0.0))
                                ):
                                    db_update_call(
                                        sb=sb,
                                        coin=str(row["coin"]),
                                        call_time=pd.to_datetime(row["call_time"], utc=True),
                                        status=str(upd.get("status", row.get("status"))),
                                        last_price=float(upd.get("last_price", row.get("last_price") or 0.0)),
                                        pnl_pct=float(upd.get("pnl_pct", row.get("pnl_pct") or 0.0)),
                                    )
                            except Exception as e:
                                print("[status][WARN]", e, flush=True)
                except Exception as e:
                    hb(sb, f"ERROR status: {type(e).__name__}: {str(e)[:120]}")
                    print("[ERROR] status update failed:", e, flush=True)

            dur = (datetime.now(timezone.utc) - loop_start).total_seconds()
            hb(sb, f"done {dur:.1f}s | scanned={scanned} found={found} calls_upserted={upserted} snaps_upserted={snap_upserted}")
            print(f"[loop] done {dur:.1f}s | scanned={scanned} found={found} calls={upserted} snaps={snap_upserted} -> sleep {SCAN_SLEEP_SECONDS}s", flush=True)
            time.sleep(SCAN_SLEEP_SECONDS)

        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:140]}"
            try:
                hb(sb, f"FATAL loop: {msg}")
            except Exception:
                pass
            print("[FATAL] loop crashed:", msg, flush=True)
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    main()
