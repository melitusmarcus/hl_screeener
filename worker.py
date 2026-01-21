import os
import time
import traceback
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

from shared import (
    supabase_client_env,
    load_sweetspot,
    fetch_hl_universe,
    build_macro_series,
    analyze_coin_for_scan,
    db_upsert_calls,
    db_read_calls,
    update_call_status,
    db_update_call,
)

WORKER_VERSION = "v2026-01-21-diagnostics"

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

def upsert_scan_snapshots(sb, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    sb.table("scan_snapshots").upsert(rows, on_conflict="coin,bar_time").execute()
    return len(rows)

def main():
    sb = supabase_client_env()
    hb(sb, "BOOT: started")

    sweet = load_sweetspot()
    sweet["coin"] = sweet["coin"].astype(str)

    coins_all = sweet["coin"].tolist()
    hb(sb, f"Loaded sweetspot coins: {len(coins_all)}")

    # keep only coins that exist on HL
    try:
        uni = set(fetch_hl_universe())
        coins = [c for c in coins_all if c in uni]
        hb(sb, f"HL universe ok. sweetspot_on_hl={len(coins)} / sweetspot_total={len(coins_all)}")
    except Exception as e:
        coins = coins_all
        hb(sb, f"WARN HL universe check failed: {type(e).__name__}: {str(e)[:120]}")

    last_bar_ts = None
    scan_count = 0

    while True:
        loop_start = datetime.now(timezone.utc)
        scanned = 0
        found = 0
        calls_upserted = 0
        snaps_upserted = 0

        try:
            hb(sb, "LOOP: start")

            macro = build_macro_series(loop_start)
            if macro is None or macro.empty:
                hb(sb, "macro empty -> sleep")
                time.sleep(SCAN_SLEEP_SECONDS)
                continue

            # only scan when macro has a new 15m bar
            if SCAN_ONLY_ON_NEW_15M_BAR:
                try:
                    current_bar_ts = pd.to_datetime(macro["timestamp"].iloc[-1], utc=True)
                except Exception:
                    current_bar_ts = None

                if current_bar_ts is not None and last_bar_ts is not None and current_bar_ts <= last_bar_ts:
                    hb(sb, "no new 15m bar -> skip")
                    time.sleep(SCAN_SLEEP_SECONDS)
                    continue

                last_bar_ts = current_bar_ts
                hb(sb, f"new 15m bar -> scan ({last_bar_ts.isoformat() if last_bar_ts is not None else 'na'})")

            scan_count += 1

            snapshots = []
            new_calls = []

            for i, coin in enumerate(coins, start=1):
                scanned += 1
                if i % HEARTBEAT_PROGRESS_EVERY == 0:
                    hb(sb, f"progress {i}/{len(coins)}")

                base_wr = 0.55
                try:
                    w = sweet.loc[sweet["coin"] == coin, "winrate"]
                    if len(w) > 0:
                        base_wr = float(w.iloc[0])
                except Exception:
                    pass

                try:
                    snap, call = analyze_coin_for_scan(
                        coin=coin,
                        macro=macro,
                        base_wr=base_wr,
                        end=loop_start,
                        detected_time=loop_start,
                    )
                    if snap:
                        snapshots.append(snap)
                    if call:
                        new_calls.append(call)
                        found += 1
                except Exception as e:
                    # keep going, don’t crash whole scan
                    print(f"[scan][WARN] {coin} failed: {e}", flush=True)

            # write snapshots & calls
            try:
                snaps_upserted = upsert_scan_snapshots(sb, snapshots)
            except Exception as e:
                hb(sb, f"ERROR upsert snapshots: {type(e).__name__}: {str(e)[:120]}")
                print("[ERROR] upsert snapshots failed:", e, flush=True)

            if new_calls:
                try:
                    calls_upserted = db_upsert_calls(sb, new_calls)
                except Exception as e:
                    hb(sb, f"ERROR upsert calls: {type(e).__name__}: {str(e)[:120]}")
                    print("[ERROR] upsert calls failed:", e, flush=True)

            # status update less often
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
                                # null-safe
                                new_status = str(upd.get("status", row.get("status")))
                                new_last = float(upd.get("last_price", row.get("last_price") or row.get("call_price") or 0.0))
                                new_pnl = float(upd.get("pnl_pct", row.get("pnl_pct") or 0.0))

                                if (
                                    new_status != str(row.get("status"))
                                    or abs(new_last - float(row.get("last_price") or 0.0)) > 1e-12
                                    or abs(new_pnl - float(row.get("pnl_pct") or 0.0)) > 1e-12
                                ):
                                    db_update_call(
                                        sb=sb,
                                        coin=str(row["coin"]),
                                        call_time=pd.to_datetime(row["call_time"], utc=True),
                                        status=new_status,
                                        last_price=new_last,
                                        pnl_pct=new_pnl,
                                    )
                            except Exception as e:
                                print("[status][WARN]", e, flush=True)
                except Exception as e:
                    hb(sb, f"ERROR status: {type(e).__name__}: {str(e)[:120]}")
                    print("[ERROR] status update failed:", e, flush=True)

            dur = (datetime.now(timezone.utc) - loop_start).total_seconds()
            hb(sb, f"done {dur:.1f}s | scanned={scanned} found={found} calls_upserted={calls_upserted} snaps_upserted={snaps_upserted}")
            print(f"[loop] done {dur:.1f}s | scanned={scanned} found={found} calls={calls_upserted} snaps={snaps_upserted} -> sleep {SCAN_SLEEP_SECONDS}s", flush=True)

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
