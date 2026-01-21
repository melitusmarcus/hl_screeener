import os
import time
import traceback
import pandas as pd
from datetime import datetime, timezone, timedelta

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
)

WORKER_VERSION = "v2026-01-21-crashsafe"

HEARTBEAT_NAME = "scanner"
SCAN_SLEEP_SECONDS = int(os.getenv("WORKER_SLEEP_SECONDS", "60"))
STATUS_UPDATE_TOP_N = int(os.getenv("STATUS_UPDATE_TOP_N", "120"))
UPDATE_OPEN_LOOKBACK_HOURS = int(os.getenv("UPDATE_OPEN_LOOKBACK_HOURS", "24"))
HEARTBEAT_PROGRESS_EVERY = int(os.getenv("HEARTBEAT_PROGRESS_EVERY", "10"))

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

def main():
    sb = supabase_client_env()
    hb(sb, "BOOT: started")

    # Load sweetspot
    sweet = load_sweetspot()
    sweet["coin"] = sweet["coin"].astype(str)
    coins = sweet["coin"].tolist()
    hb(sb, f"Loaded sweetspot coins: {len(coins)}")

    # Validate HL universe once
    try:
        uni = set(fetch_hl_universe())
        coins = [c for c in coins if c in uni]
        hb(sb, f"HL universe ok. sweetspot_on_hl={len(coins)}")
    except Exception as e:
        hb(sb, f"WARN HL universe check failed: {type(e).__name__}: {str(e)[:120]}")

    while True:
        loop_start = datetime.now(timezone.utc)
        scanned = 0
        found = 0
        upserted = 0

        try:
            hb(sb, "LOOP: start")

            macro = build_macro_series(loop_start)
            if macro is None or macro.empty:
                hb(sb, "macro empty -> sleep")
                time.sleep(SCAN_SLEEP_SECONDS)
                continue

            new_rows = []
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

            if new_rows:
                try:
                    upserted = db_upsert_calls(sb, new_rows)
                except Exception as e:
                    hb(sb, f"ERROR upsert: {type(e).__name__}: {str(e)[:120]}")
                    print("[ERROR] upsert failed:", e, flush=True)
                    upserted = 0

            # Update statuses for recent OPEN calls
            try:
                calls_df = db_read_calls(sb, limit=4000)
                if not calls_df.empty:
                    cutoff = loop_start - timedelta(hours=UPDATE_OPEN_LOOKBACK_HOURS)
                    open_df = calls_df[(calls_df["call_time"] >= cutoff) & (calls_df["status"] == "OPEN")].copy()
                    if not open_df.empty:
                        open_df = open_df.sort_values("call_time", ascending=False).head(STATUS_UPDATE_TOP_N)

                    updated = 0
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
                                updated += 1
                        except Exception as e:
                            print("[status][WARN]", e, flush=True)

                    if updated:
                        print(f"[status] updated_open={updated}", flush=True)

            except Exception as e:
                hb(sb, f"ERROR status: {type(e).__name__}: {str(e)[:120]}")
                print("[ERROR] status update failed:", e, flush=True)

            dur = (datetime.now(timezone.utc) - loop_start).total_seconds()
            hb(sb, f"done {dur:.1f}s | scanned={scanned} found={found} upserted={upserted}")
            print(f"[loop] done {dur:.1f}s | scanned={scanned} found={found} upserted={upserted} -> sleep {SCAN_SLEEP_SECONDS}s", flush=True)
            time.sleep(SCAN_SLEEP_SECONDS)

        except Exception as e:
            # NEVER die; report and sleep
            msg = f"{type(e).__name__}: {str(e)[:140]}"
            try:
                hb(sb, f"FATAL loop: {msg}")
            except Exception:
                pass
            print("[FATAL] loop crashed:", msg, flush=True)
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Report fatal startup error to heartbeat if possible
        try:
            sb = supabase_client_env()
            sb.table("worker_heartbeat").upsert(
                {
                    "name": HEARTBEAT_NAME,
                    "last_seen": datetime.now(timezone.utc).isoformat(),
                    "note": f"{WORKER_VERSION} | FATAL STARTUP: {type(e).__name__}: {str(e)[:180]}",
                },
                on_conflict="name",
            ).execute()
        except Exception:
            pass
        raise
