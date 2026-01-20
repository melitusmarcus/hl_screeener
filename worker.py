import time
import traceback
from datetime import datetime, timezone, timedelta

import pandas as pd

from shared import (
    supabase_client_env,
    load_sweetspot,
    build_macro_series,
    detect_call_for_coin,
    db_upsert_calls,
    db_read_calls,
    update_call_status,
    db_update_call,
)

# =========================
# Worker config
# =========================
HEARTBEAT_NAME = "scanner"
SCAN_SLEEP_SECONDS = 60
UPDATE_OPEN_LOOKBACK_HOURS = 12
HEARTBEAT_PROGRESS_EVERY = 10


def hb(sb, note: str):
    """Failsafe heartbeat – får aldrig krascha workern."""
    try:
        sb.table("worker_heartbeat").upsert(
            {
                "name": HEARTBEAT_NAME,
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "note": note,
            },
            on_conflict="name",
        ).execute()
        print(f"[hb] {note}")
    except Exception as e:
        print(f"[hb][WARN] {e}")


def main():
    print("[boot] worker starting...")
    sb = supabase_client_env()
    hb(sb, "BOOT: worker started")

    # --- Load sweetspot coins ---
    sweet = load_sweetspot()
    if sweet.empty or "coin" not in sweet.columns:
        hb(sb, "FATAL: sweetspot empty or missing coin")
        raise RuntimeError("sweetspot_coins.csv invalid")

    sweet["coin"] = sweet["coin"].astype(str).str.upper().str.strip()
    if "winrate" not in sweet.columns:
        sweet["winrate"] = 0.55

    coins = sweet["coin"].dropna().unique().tolist()
    print(f"[universe] loaded coins={len(coins)} sample={coins[:10]}")
    hb(sb, f"BOOT: coins={len(coins)}")

    # --- Main loop ---
    while True:
        loop_start = datetime.now(timezone.utc)
        hb(sb, "LOOP: start")

        try:
            macro = build_macro_series(loop_start)
            if macro is None or macro.empty:
                hb(sb, "macro empty -> sleep")
                time.sleep(SCAN_SLEEP_SECONDS)
                continue

            checked = 0
            found = 0
            new_rows = []

            for i, coin in enumerate(coins, start=1):
                checked += 1

                if i % HEARTBEAT_PROGRESS_EVERY == 0:
                    hb(sb, f"progress {i}/{len(coins)}")

                try:
                    base_wr = float(
                        sweet.loc[sweet["coin"] == coin, "winrate"].iloc[0]
                        if (sweet["coin"] == coin).any()
                        else 0.55
                    )
                except Exception:
                    base_wr = 0.55

                try:
                    call = detect_call_for_coin(
                        coin=coin,
                        macro=macro,
                        base_wr=base_wr,
                        end=loop_start,
                        detected_time=loop_start,
                    )
                except Exception as e:
                    print(f"[scan][WARN] {coin} failed: {e}")
                    continue

                if call is not None:
                    new_rows.append(call)
                    found += 1
                    print(
                        f"[CALL] {coin} dump={call['dump_pct']:.2f}% "
                        f"vol_ratio={call['vol_ratio']:.2f} "
                        f"vol_z={call['vol_z']:.2f} "
                        f"chance={call['chance_pct']:.0f}%"
                    )

            hb(sb, f"LOOP: scanned={checked} found={found}")
            print(f"[scan] scanned={checked} found={found}")

            if new_rows:
                try:
                    n = db_upsert_calls(sb, new_rows)
                    print(f"[db] upserted={n}")
                except Exception as e:
                    print(f"[db][ERROR] {e}")
                    traceback.print_exc()
            else:
                print("[db] upserted=0")

            # --- Update OPEN calls ---
            try:
                calls_df = db_read_calls(sb, limit=5000)
                cutoff = loop_start - timedelta(hours=UPDATE_OPEN_LOOKBACK_HOURS)

                open_df = calls_df[
                    (calls_df["status"].astype(str) == "OPEN")
                    & (calls_df["call_time"] >= cutoff)
                ]

                updated = 0
                for _, row in open_df.iterrows():
                    upd = update_call_status(row, end=loop_start)
                    if upd["status"] != row["status"]:
                        db_update_call(
                            sb=sb,
                            coin=row["coin"],
                            call_time=row["call_time"],
                            status=upd["status"],
                            last_price=upd["last_price"],
                            pnl_pct=upd["pnl_pct"],
                        )
                        updated += 1

                if updated:
                    print(f"[status] updated_open={updated}")

            except Exception as e:
                print(f"[status][WARN] {e}")

            dur = (datetime.now(timezone.utc) - loop_start).total_seconds()
            hb(sb, f"LOOP: done {dur:.1f}s")
            print(f"[loop] done in {dur:.1f}s -> sleep {SCAN_SLEEP_SECONDS}s")
            time.sleep(SCAN_SLEEP_SECONDS)

        except Exception as e:
            print("[loop][ERROR]", e)
            traceback.print_exc()
            hb(sb, f"ERROR: {type(e).__name__}")
            time.sleep(30)


if __name__ == "__main__":
    main()
