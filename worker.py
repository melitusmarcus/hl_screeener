import os
import time
import traceback
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from shared import (
    supabase_client_env,
    load_sweetspot,
    fetch_hl_universe,
    fetch_candles,
    build_btc_vol_z,
    detect_call_for_coin,
    update_call_status,
    db_upsert_calls,
    db_read_calls,
    db_update_call,
    db_upsert_latest_scan,
    db_upsert_heartbeat,
    ensure_utc_ts,
    utc_now,
    to_ms,
    VOL_WIN,
    DROP_MIN,
    DROP_MAX,
    VOL_Z_MAX,
    VOL_SPIKE_MULT,
    VOL_FLOOR_MULT,
)

VERSION_TAG = "v2026-01-21-diagnostics-fix"

WORKER_SLEEP_SECONDS = int(os.getenv("WORKER_SLEEP_SECONDS", "60"))
STATUS_UPDATE_TOP_N = int(os.getenv("STATUS_UPDATE_TOP_N", "120"))

def _log(msg: str):
    print(msg, flush=True)

def _warn(msg: str):
    print(msg, flush=True)

def compute_latest_scan_row(
    coin: str,
    base_wr: float,
    btc_z: Optional[float],
    macro_df: Optional[pd.DataFrame],
    end: datetime,
) -> Dict[str, Any]:
    """
    For UI table: compute latest bar metrics + gate-lights.
    Uses short candle range (~10h).
    """
    end = end.astimezone(timezone.utc)
    start = end - timedelta(hours=10)

    row: Dict[str, Any] = {
        "coin": coin,
        "bar_close_utc": ensure_utc_ts(end),  # will be overwritten with candle timestamp
        "updated_time_utc": ensure_utc_ts(end),
        "price": None,
        "chg_15m_pct": None,
        "dump_pct": None,
        "vol_ratio": None,
        "btc_vol_z": None if btc_z is None else float(btc_z),
        "gate_dump": False,
        "gate_macro": False,
        "gate_spike": False,
        "gate_floor": False,
        "signal": False,
        "err": None,
    }

    try:
        df = fetch_candles(coin, to_ms(start), to_ms(end + timedelta(seconds=1)))
        if df.empty or len(df) < (VOL_WIN + 2):
            row["err"] = "too_few_candles"
            return row

        df = df.sort_values("timestamp").reset_index(drop=True)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        bar_close = ensure_utc_ts(last["timestamp"])
        row["bar_close_utc"] = bar_close

        price = float(last["close"])
        row["price"] = price

        # 15m change: close vs prev close (mer stabilt i UI)
        prev_close = float(prev["close"])
        chg_15m = (price / prev_close - 1.0) * 100.0
        row["chg_15m_pct"] = float(chg_15m)

        # dump wick open->low (samma som signal)
        dump = float((float(last["open"]) - float(last["low"])) / float(last["open"])) * 100.0
        row["dump_pct"] = dump

        medv = df["volume"].rolling(VOL_WIN).median().iloc[-1]
        if medv and float(medv) > 0:
            vol_ratio = float(last["volume"]) / float(medv)
            row["vol_ratio"] = float(vol_ratio)
        else:
            row["vol_ratio"] = None

        # gates
        dump_ok = (dump/100.0 >= DROP_MIN) and (dump/100.0 <= DROP_MAX)
        macro_ok = (btc_z is not None) and (abs(float(btc_z)) <= VOL_Z_MAX)
        spike_ok = (row["vol_ratio"] is not None) and (float(row["vol_ratio"]) >= VOL_SPIKE_MULT)
        floor_ok = (row["vol_ratio"] is not None) and (float(row["vol_ratio"]) >= VOL_FLOOR_MULT)

        row["gate_dump"] = bool(dump_ok)
        row["gate_macro"] = bool(macro_ok)
        row["gate_spike"] = bool(spike_ok)
        row["gate_floor"] = bool(floor_ok)
        row["signal"] = bool(dump_ok and macro_ok and spike_ok and floor_ok)

        return row

    except Exception as e:
        row["err"] = str(e)[:240]
        return row

def main():
    sb = supabase_client_env()

    _log(f"[hb] {VERSION_TAG} | BOOT: started")

    # sweetspot
    sweet = load_sweetspot()
    sweet["coin"] = sweet["coin"].astype(str)
    sweet_map = {str(r["coin"]): float(r.get("winrate", 0.55)) for _, r in sweet.iterrows()}
    sweet_coins = list(sweet_map.keys())

    _log(f"[hb] {VERSION_TAG} | Loaded sweetspot coins: {len(sweet_coins)}")

    # intersect with HL universe (avoid scanning coins not on HL)
    hl_uni = fetch_hl_universe()
    hl_set = set(hl_uni)
    coins = [c for c in sweet_coins if c in hl_set]

    _log(f"[hb] {VERSION_TAG} | HL universe ok. sweetspot_on_hl={len(coins)}")

    while True:
        loop_start = utc_now()
        found = 0
        upserted = 0

        try:
            _log(f"[loop] start {loop_start.isoformat()}")

            # BTC macro
            btc_z, macro_df = build_btc_vol_z(loop_start)
            if macro_df is None:
                _warn("[macro][WARN] BTC macro missing (HL may be rate-limiting).")
                macro_df = pd.DataFrame(columns=["timestamp","vol_z"])

            scan_rows: List[Dict[str, Any]] = []
            new_calls: List[Dict[str, Any]] = []

            total = len(coins)
            for i, coin in enumerate(coins, start=1):
                base_wr = float(sweet_map.get(coin, 0.55))

                # UI row
                scan_row = compute_latest_scan_row(
                    coin=coin,
                    base_wr=base_wr,
                    btc_z=btc_z,
                    macro_df=macro_df,
                    end=loop_start,
                )
                scan_rows.append(scan_row)

                # signal -> insert call using full detect (still optimized)
                try:
                    if macro_df is not None and not macro_df.empty:
                        call = detect_call_for_coin(
                            coin=coin,
                            macro=macro_df,
                            base_wr=base_wr,
                            end=loop_start,
                            detected_time=loop_start,
                        )
                        if call:
                            new_calls.append(call)
                            found += 1
                except Exception as e:
                    _warn(f"[scan][WARN] {coin} detect failed: {e}")

                if i % 10 == 0 or i == total:
                    _log(f"[hb] {VERSION_TAG} | progress {i}/{total}")

            # write latest_scan snapshot
            try:
                db_upsert_latest_scan(sb, scan_rows)
            except Exception as e:
                _warn(f"[latest_scan][WARN] upsert failed: {e}")

            # upsert new calls
            if new_calls:
                try:
                    upserted = db_upsert_calls(sb, new_calls)
                except Exception as e:
                    _warn(f"[calls][WARN] upsert failed: {e}")

            # status update (OPEN only)
            try:
                calls = db_read_calls(sb, limit=5000)
                if not calls.empty:
                    calls = calls.sort_values("detected_time", ascending=False)
                    open_calls = calls[calls["status"].astype(str) == "OPEN"].head(STATUS_UPDATE_TOP_N)

                    for _, r in open_calls.iterrows():
                        try:
                            upd = update_call_status(r, loop_start)
                            if (upd["status"] != r["status"]) or (abs(float(upd["pnl_pct"]) - float(r["pnl_pct"])) > 0.01):
                                db_update_call(
                                    sb,
                                    coin=str(r["coin"]),
                                    call_time=ensure_utc_ts(r["call_time"]),
                                    status=str(upd["status"]),
                                    last_price=float(upd["last_price"]),
                                    pnl_pct=float(upd["pnl_pct"]),
                                )
                        except Exception as e:
                            _warn(f"[status][WARN] {r.get('coin')} update failed: {e}")
            except Exception as e:
                _warn(f"[status][WARN] batch failed: {e}")

            dt = (utc_now() - loop_start).total_seconds()
            note = f"LOOP: done {dt:.1f}s | scanned={len(coins)} found={found} upserted={upserted}"
            db_upsert_heartbeat(sb, name="scanner", note=note)

            _log(f"[loop] {note} -> sleep {WORKER_SLEEP_SECONDS}s")
            time.sleep(WORKER_SLEEP_SECONDS)

        except Exception as e:
            err = f"CRASH: {e}"
            _warn(f"[fatal] {err}")
            _warn(traceback.format_exc())
            try:
                db_upsert_heartbeat(sb, name="scanner", note=err)
            except Exception:
                pass
            time.sleep(max(10, WORKER_SLEEP_SECONDS))

if __name__ == "__main__":
    main()