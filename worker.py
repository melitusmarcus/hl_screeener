# -*- coding: utf-8 -*-
import os
import time
import traceback
import pandas as pd
from datetime import datetime, timezone

from shared import (
    supabase_client_env,
    load_sweetspot,
    fetch_hl_universe,
    build_macro_series,
    detect_call_for_coin,
    db_upsert_calls,
    db_read_calls,
    db_update_call,
    update_call_status,
)

WORKER_SLEEP_SECONDS = int(os.getenv("WORKER_SLEEP_SECONDS", "60"))
STATUS_UPDATE_TOP_N = int(os.getenv("STATUS_UPDATE_TOP_N", "120"))

def log(msg: str):
    print(msg, flush=True)

def main():
    log(f"✅ Worker booting at {datetime.now(timezone.utc).isoformat()}")
    log(f"ENV SUPABASE_URL set: {bool(os.getenv('SUPABASE_URL'))}")
    log(f"ENV SUPABASE_SERVICE_KEY set: {bool(os.getenv('SUPABASE_SERVICE_KEY'))}")
    log(f"WORKER_SLEEP_SECONDS={WORKER_SLEEP_SECONDS}, STATUS_UPDATE_TOP_N={STATUS_UPDATE_TOP_N}")

    sb = supabase_client_env()
    log("✅ Supabase client created")

    while True:
        loop_start = datetime.now(timezone.utc)
        log(f"💓 heartbeat {loop_start.isoformat()}")

        scanned = 0
        inserted = 0
        updated_n = 0
        new_calls_count = 0

        try:
            now = loop_start
            detected_time = now

            sweet = load_sweetspot().reset_index(drop=True)
            log(f"Loaded sweetspot coins: {len(sweet)}")

            hl_uni = fetch_hl_universe()
            if hl_uni:
                hl_set = set(hl_uni)
                universe = sweet[sweet["coin"].astype(str).isin(hl_set)].copy().reset_index(drop=True)
                log(f"HL universe ok. sweetspot_on_hl={len(universe)} / sweetspot_total={len(sweet)}")
            else:
                universe = sweet.copy()
                log("HL universe fetch failed. Scanning sweetspot as-is.")

            macro = build_macro_series(now)
            if macro.empty:
                log("⚠️ Macro empty (BTC candles missing / rate-limit). Sleeping...")
                time.sleep(WORKER_SLEEP_SECONDS)
                continue

            new_calls = []
            for _, row in universe.iterrows():
                coin = str(row["coin"])
                base_wr = float(row.get("winrate", 0.55))
                scanned += 1
                try:
                    c = detect_call_for_coin(
                        sb=sb,
                        coin=coin,
                        macro=macro,
                        base_wr=base_wr,
                        end=now,
                        detected_time=detected_time,
                    )
                    if c:
                        new_calls.append(c)
                except Exception:
                    continue

            new_calls_count = len(new_calls)
            inserted = db_upsert_calls(sb, new_calls)

            calls_recent = db_read_calls(sb, limit=2000)
            if not calls_recent.empty:
                recent = calls_recent.sort_values("call_time", ascending=False).head(STATUS_UPDATE_TOP_N).copy()
                for _, r in recent.iterrows():
                    try:
                        upd = update_call_status(r, now)
                        if (upd["status"] != r["status"]) or (abs(float(upd["pnl_pct"]) - float(r["pnl_pct"])) > 0.01):
                            db_update_call(
                                sb,
                                str(r["coin"]),
                                pd.to_datetime(r["call_time"], utc=True),
                                upd["status"],
                                float(upd["last_price"]),
                                float(upd["pnl_pct"]),
                            )
                            updated_n += 1
                    except Exception:
                        continue

            loop_end = datetime.now(timezone.utc)
            dt = (loop_end - loop_start).total_seconds()
            log(
                f"✅ loop done in {dt:.1f}s | scanned={scanned} new_calls={new_calls_count} "
                f"upserted={inserted} status_updates={updated_n}"
            )

        except Exception:
            log("❌ Worker loop exception (full traceback below):")
            traceback.print_exc()

        time.sleep(WORKER_SLEEP_SECONDS)

if __name__ == "__main__":
    main()
