import os
import time
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from shared import (
    supabase_client_env,
    fetch_hl_universe,
    load_sweetspot,
    build_macro_series,
    fetch_candles,
    detect_call_for_coin,
    update_call_status,
    db_upsert_calls,
    db_read_calls,
    db_update_call,
    to_ms,
    attach_macro_asof,
)

# ---------- ENV ----------
WORKER_SLEEP_SECONDS = int(os.getenv("WORKER_SLEEP_SECONDS", "60"))
STATUS_UPDATE_TOP_N = int(os.getenv("STATUS_UPDATE_TOP_N", "120"))
SCAN_CHUNK = int(os.getenv("SCAN_CHUNK", "18"))

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def log(msg: str):
    print(msg, flush=True)

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def compute_latest_scan_row(coin: str, macro: pd.DataFrame, now: datetime) -> Optional[Dict[str, Any]]:
    """
    Latest-scan row for UI. Must fetch enough candles to compute median(20).
    """
    end = now
    # 50 candles (~12.5h) -> enough for rolling median(20) + stable last/prev
    start = end - timedelta(minutes=15 * 50)

    df = fetch_candles(coin, to_ms(start), to_ms(end))
    if df.empty or len(df) < 2:
        return None

    df = df.sort_values("timestamp").reset_index(drop=True)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    price = safe_float(last["close"])
    if price is None:
        return None

    chg_15m_pct = (price / float(prev["close"]) - 1.0) * 100.0

    dump_pct = (float(last["open"]) - float(last["low"])) / float(last["open"])
    dump_pct = dump_pct * 100.0

    vol_ratio = None
    if len(df) >= 22:
        medv = df["volume"].rolling(20).median().iloc[-1]
        if medv and medv > 0:
            vol_ratio = float(last["volume"]) / float(medv)

    # Attach BTC vol_z using ASOF join (robust)
    btc_vol_z = None
    if macro is not None and not macro.empty:
        tmp = attach_macro_asof(df[["timestamp"]].copy(), macro, tolerance_minutes=20)
        if "vol_z" in tmp.columns and pd.notna(tmp["vol_z"].iloc[-1]):
            btc_vol_z = float(tmp["vol_z"].iloc[-1])

    gate_dump = (dump_pct/100.0) >= 0.005 and (dump_pct/100.0) <= 0.05
    gate_macro = (btc_vol_z is not None) and (abs(btc_vol_z) <= 0.60)
    gate_spike = (vol_ratio is not None) and (vol_ratio >= 2.0)
    gate_floor = (vol_ratio is not None) and (vol_ratio >= 1.2)

    signal = bool(gate_dump and gate_macro and gate_spike and gate_floor)

    return {
        "coin": coin,
        "price": float(price),
        "chg_15m_pct": float(chg_15m_pct),
        "dump_pct": float(dump_pct),
        "vol_ratio": float(vol_ratio) if vol_ratio is not None else None,
        "btc_vol_z": float(btc_vol_z) if btc_vol_z is not None else None,
        "signal": signal,
        "gate_dump": bool(gate_dump),
        "gate_macro": bool(gate_macro),
        "gate_spike": bool(gate_spike),
        "gate_floor": bool(gate_floor),
        "bar_close_utc": pd.to_datetime(last["timestamp"], utc=True).isoformat(),
        "updated_time_utc": now.isoformat(),
    }

def upsert_latest_scan(sb, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    sb.table("latest_scan").upsert(rows, on_conflict="coin").execute()

def main():
    sb = supabase_client_env()

    sweet = load_sweetspot()
    sweet_coins = sweet["coin"].astype(str).tolist()
    hl_uni = fetch_hl_universe()
    hl_set = set(hl_uni)
    coins = [c for c in sweet_coins if c in hl_set]
    coins = sorted(list(set(coins)))

    version = "v2026-01-22-asof-macro-fix"
    log(f"[BOOT] {version} started. coins={len(coins)}")

    idx = 0

    while True:
        loop_start = utcnow()
        found = 0
        upserted = 0

        try:
            macro = build_macro_series(loop_start)
        except Exception as e:
            macro = pd.DataFrame()
            log(f"[macro][WARN] failed: {e}")

        chunk = coins[idx:idx + SCAN_CHUNK]
        if len(chunk) < SCAN_CHUNK:
            chunk += coins[0:max(0, SCAN_CHUNK - len(chunk))]
        idx = (idx + SCAN_CHUNK) % max(1, len(coins))

        latest_rows = []
        new_calls = []

        for coin in chunk:
            try:
                row = compute_latest_scan_row(coin, macro, loop_start)
                if row:
                    latest_rows.append(row)

                # Only attempt full detect if row says signal True
                if row and row.get("signal"):
                    base_wr = float(
                        sweet.loc[sweet["coin"].astype(str) == coin, "winrate"].iloc[0]
                    ) if "winrate" in sweet.columns else 0.55

                    call = detect_call_for_coin(
                        coin=coin,
                        macro=macro,
                        base_wr=base_wr,
                        end=loop_start,
                        detected_time=loop_start,
                    )
                    if call:
                        new_calls.append(call)
                        found += 1

            except Exception as e:
                log(f"[scan][WARN] {coin} failed: {e}")

            time.sleep(0.05)

        try:
            upsert_latest_scan(sb, latest_rows)
        except Exception as e:
            log(f"[latest_scan][WARN] upsert failed: {e}")

        try:
            upserted = db_upsert_calls(sb, new_calls)
        except Exception as e:
            log(f"[calls][WARN] upsert failed: {e}")

        # Update OPEN only
        try:
            calls = db_read_calls(sb, limit=5000)
            if not calls.empty:
                open_calls = calls[calls["status"].astype(str) == "OPEN"].copy()
                open_calls = open_calls.sort_values("detected_time", ascending=False).head(STATUS_UPDATE_TOP_N)

                for _, r in open_calls.iterrows():
                    try:
                        upd = update_call_status(r, loop_start)
                        if (upd["status"] != r["status"]) or (abs(float(upd["pnl_pct"]) - float(r["pnl_pct"])) > 0.01):
                            db_update_call(
                                sb,
                                coin=str(r["coin"]),
                                call_time=pd.to_datetime(r["call_time"], utc=True),
                                status=str(upd["status"]),
                                last_price=float(upd["last_price"]),
                                pnl_pct=float(upd["pnl_pct"]),
                            )
                    except Exception as e:
                        log(f"[status][WARN] {r.get('coin')} failed: {e}")

        except Exception as e:
            log(f"[status][WARN] batch failed: {e}")

        dur = (utcnow() - loop_start).total_seconds()
        log(f"[loop] done {dur:.1f}s scanned={len(chunk)} found={found} upserted={upserted} -> sleep {WORKER_SLEEP_SECONDS}s")
        time.sleep(WORKER_SLEEP_SECONDS)

if __name__ == "__main__":
    main()
