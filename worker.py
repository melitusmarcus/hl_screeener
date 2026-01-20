import time
import traceback
from datetime import datetime, timezone, timedelta
from collections import Counter

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

SCAN_SLEEP_SECONDS = 60          # tid mellan scan-loopar
UPDATE_OPEN_LOOKBACK_HOURS = 12  # uppdatera OPEN calls inom senaste N timmar
HEARTBEAT_PROGRESS_EVERY = 10    # heartbeat/log var N:e coin


def hb(sb, note: str):
    """
    Failsafe heartbeat: får ALDRIG stoppa workern.
    Skriver till public.worker_heartbeat om tabellen finns.
    """
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


def safe_upsert_calls(sb, rows):
    try:
        n = db_upsert_calls(sb, rows)
        print(f"[db] upserted={n}")
        return n
    except Exception as e:
        print(f"[db][ERROR] upsert failed: {e}")
        traceback.print_exc()
        hb(sb, f"db ERROR: {type(e).__name__}")
        return 0


def safe_update_open_calls(sb, now_utc: datetime):
    """
    Uppdaterar status/last_price/pnl_pct för OPEN calls i ett tidsfönster.
    """
    try:
        calls_df = db_read_calls(sb, limit=5000)
        if calls_df.empty:
            return 0

        cutoff = now_utc - timedelta(hours=UPDATE_OPEN_LOOKBACK_HOURS)
        calls_df = calls_df[pd.to_datetime(calls_df["call_time"], utc=True, errors="coerce") >= cutoff].copy()

        if calls_df.empty or "status" not in calls_df.columns:
            return 0

        open_df = calls_df[calls_df["status"].astype(str).str.upper() == "OPEN"].copy()
        if open_df.empty:
            return 0

        updated = 0
        for _, row in open_df.iterrows():
            try:
                upd = update_call_status(row, end=now_utc)
                if not isinstance(upd, dict):
                    continue

                # skriv bara om det verkligen ändrats / uppdaterats
                old_status = str(row.get("status"))
                new_status = str(upd.get("status", old_status))
                old_last = row.get("last_price")
                new_last = upd.get("last_price", old_last)
                old_pnl = row.get("pnl_pct")
                new_pnl = upd.get("pnl_pct", old_pnl)

                changed = (new_status != old_status) or (new_last != old_last) or (new_pnl != old_pnl)
                if not changed:
                    continue

                db_update_call(
                    sb=sb,
                    coin=str(row["coin"]),
                    call_time=pd.to_datetime(row["call_time"], utc=True),
                    status=new_status,
                    last_price=float(new_last) if new_last is not None else float(old_last or 0.0),
                    pnl_pct=float(new_pnl) if new_pnl is not None else float(old_pnl or 0.0),
                )
                updated += 1

            except Exception as e:
                print(f"[status][WARN] update failed for {row.get('coin')}: {e}")

        if updated:
            print(f"[status] updated_open={updated}")
        return updated

    except Exception as e:
        print(f"[status][WARN] failed updating open calls: {e}")
        traceback.print_exc()
        hb(sb, f"status WARN: {type(e).__name__}")
        return 0


def main():
    print("[boot] worker starting...")
    sb = supabase_client_env()
    hb(sb, "BOOT: worker started")

    # --- Load sweetspot list (must be in repo) ---
    try:
        sweet = load_sweetspot()
    except Exception as e:
        print("[universe][FATAL] Failed to load sweetspot_coins.csv")
        print(e)
        traceback.print_exc()
        hb(sb, "FATAL: sweetspot load failed")
        raise

    if sweet.empty or "coin" not in sweet.columns:
        hb(sb, "FATAL: sweetspot empty/missing coin")
        raise RuntimeError("sweetspot_coins.csv empty or missing 'coin' column")

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
            # Macro series (BTC vol_z)
            macro = build_macro_series(loop_start)
            if macro is None or macro.empty:
                print("[macro] empty macro series (BTC candles missing) -> sleep")
                hb(sb, "macro empty -> sleep")
                time.sleep(SCAN_SLEEP_SECONDS)
                continue

            checked = 0
            found = 0
            new_rows = []
            reasons = Counter()

            # Detect calls
            for i, coin in enumerate(coins, start=1):
                checked += 1

                # progress heartbeat
                if i % HEARTBEAT_PROGRESS_EVERY == 0:
                    hb(sb, f"progress {i}/{len(coins)}")

                # winrate per coin
                try:
                    base_wr = float(
                        sweet.loc[sweet["coin"] == coin, "winrate"].iloc[0]
                        if (sweet["coin"] == coin).any()
                        else 0.55
                    )
                except Exception:
                    base_wr = 0.55

                try:
                    # debug=True ger oss _reason vid fail (om du har patchat shared.py enligt tidigare)
                    call = detect_call_for_coin(
                        coin=coin,
                        macro=macro,
                        base_wr=base_wr,
                        end=loop_start,
                        detected_time=loop_start,
                        debug=True,  # safe även om din funktion ignorerar param (men då måste den acceptera debug)
                    )
                except TypeError:
                    # om din detect_call_for_coin inte har debug-parameter än
                    call = detect_call_for_coin(
                        coin=coin,
                        macro=macro,
                        base_wr=base_wr,
                        end=loop_start,
                        detected_time=loop_start,
                    )
                except Exception as e:
                    print(f"[scan][WARN] {coin} detect failed: {e}")
                    continue

                if call is None:
                    continue

                if isinstance(call, dict) and "_reason" in call:
                    reasons[str(call["_reason"])] += 1
                    continue

                # New signal
                if isinstance(call, dict):
                    new_rows.append(call)
                    found += 1
                    print(
                        f"[CALL] {coin} time={call.get('call_time')} dump={call.get('dump_pct'):.2f}% "
                        f"vol_ratio={call.get('vol_ratio'):.2f} liq_ratio={call.get('liq_ratio'):.2f} "
                        f"vol_z={call.get('vol_z'):.2f} chance={call.get('chance_pct'):.0f}%"
                    )

            # Summary
            top_reasons = ", ".join([f"{k}={v}" for k, v in reasons.most_common(5)])
            print(f"[scan] scanned={checked} found={found}")
            if top_reasons:
                print(f"[reasons] {top_reasons}")

            hb(sb, f"LOOP: scanned={checked} found={found}")

            # Write new calls
            upserted = 0
            if new_rows:
                upserted = safe_upsert_calls(sb, new_rows)
            else:
                print("[db] upserted=0 (no new calls)")

            # Update open calls
            updated_open = safe_update_open_calls(sb, now_utc=loop_start)

            # Loop done
            dur = (datetime.now(timezone.utc) - loop_start).total_seconds()
            hb(sb, f"LOOP: done {dur:.1f}s upserted={upserted} upd_open={updated_open}")
            print(f"[loop] done in {dur:.1f}s | upserted={upserted} updated_open={updated_open} -> sleep {SCAN_SLEEP_SECONDS}s")
            time.sleep(SCAN_SLEEP_SECONDS)

        except Exception as e:
            print("[loop][ERROR] Unhandled exception:", e)
            traceback.print_exc()
            hb(sb, f"ERROR: {type(e).__name__}")
            time.sleep(30)


if __name__ == "__main__":
    main()
