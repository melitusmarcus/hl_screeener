import os
import time
import pandas as pd
from datetime import datetime, timezone

from shared import (
    supabase_client_env,
    load_sweetspot, fetch_hl_universe, build_macro_series, detect_call_for_coin,
    db_upsert_calls, db_read_calls, db_update_call, update_call_status,
)

WORKER_SLEEP_SECONDS = int(os.getenv("WORKER_SLEEP_SECONDS", "60"))
STATUS_UPDATE_TOP_N = int(os.getenv("STATUS_UPDATE_TOP_N", "120"))

def main():
    sb = supabase_client_env()
    print("Worker started.")

    while True:
        try:
            now = datetime.now(timezone.utc)
            detected_time = now

            sweet = load_sweetspot().reset_index(drop=True)

            hl_uni = fetch_hl_universe()
            if hl_uni:
                hl_set = set(hl_uni)
                universe = sweet[sweet["coin"].astype(str).isin(hl_set)].copy().reset_index(drop=True)
            else:
                universe = sweet.copy()

            macro = build_macro_series(now)
            if macro.empty:
                print("Macro empty; sleeping...")
                time.sleep(WORKER_SLEEP_SECONDS)
                continue

            new_calls = []
            scanned = 0
            for _, row in universe.iterrows():
                coin = str(row["coin"])
                base_wr = float(row.get("winrate", 0.55))
                scanned += 1
                try:
                    c = detect_call_for_coin(
                        coin=coin,
                        macro=macro,
                        base_wr=base_wr,
                        end=now,
                        detected_time=detected_time
                    )
                    if c:
                        new_calls.append(c)
                except Exception:
                    pass

            inserted = db_upsert_calls(sb, new_calls)
            print(f"[{now.isoformat()}] scanned={scanned} new_calls={len(new_calls)} upserted={inserted}")

            # status updates
            calls_recent = db_read_calls(sb, limit=2000)
            updated_n = 0
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
                                float(upd["pnl_pct"])
                            )
                            updated_n += 1
                    except Exception:
                        pass
            print(f"status_updates={updated_n}")

        except Exception as e:
            print("Worker loop error:", e)

        time.sleep(WORKER_SLEEP_SECONDS)

if __name__ == "__main__":
    main()
