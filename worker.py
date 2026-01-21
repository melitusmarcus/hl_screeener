import os
import time
import random
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
    HL_INFO_URL,
)

# ---------- ENV ----------
WORKER_SLEEP_SECONDS = int(os.getenv("WORKER_SLEEP_SECONDS", "60"))
STATUS_UPDATE_TOP_N = int(os.getenv("STATUS_UPDATE_TOP_N", "120"))

# hur många coins per loop (för att inte dö av 429)
SCAN_CHUNK = int(os.getenv("SCAN_CHUNK", "18"))

# ---------- helpers ----------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def log(msg: str):
    print(msg, flush=True)

def hb(sb, note: str):
    # table: public.worker_heartbeat(name text primary key, last_seen timestamptz, note text)
    sb.table("worker_heartbeat").upsert({
        "name": "scanner",
        "last_seen": utcnow().isoformat(),
        "note": note
    }, on_conflict="name").execute()

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def compute_latest_scan_row(coin: str, macro: pd.DataFrame, now: datetime) -> Optional[Dict[str, Any]]:
    """
    Hämtar senaste 2 candles => kan räkna 15m% och dump% och volspik.
    Skriver alltid updated_time_utc när vi lyckas hämta data.
    """
    end = now
    start = end - timedelta(minutes=15 * 6)  # lite marginal för stabilitet

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

    # dump% på sista candle: (open-low)/open i %
    dump_pct = (float(last["open"]) - float(last["low"])) / float(last["open"])
    dump_pct = dump_pct * 100.0

    # vol spike approx: last volume / median(20)
    vol_ratio = None
    if len(df) >= 22:
        medv = df["volume"].rolling(20).median().iloc[-1]
        if medv and medv > 0:
            vol_ratio = float(last["volume"]) / float(medv)

    # BTC macro z vid samma timestamp (merge på timestamp)
    btc_vol_z = None
    if not macro.empty:
        m = pd.merge(
            df[["timestamp"]],
            macro[["timestamp", "vol_z"]],
            on="timestamp",
            how="left"
        )
        if "vol_z" in m.columns and pd.notna(m["vol_z"].iloc[-1]):
            btc_vol_z = float(m["vol_z"].iloc[-1])

    # gates (samma som detect)
    gate_dump = (dump_pct/100.0) >= 0.005 and (dump_pct/100.0) <= 0.05
    gate_macro = (btc_vol_z is not None) and abs(btc_vol_z) <= 0.60
    gate_spike = (vol_ratio is not None) and vol_ratio >= 2.0
    gate_floor = (vol_ratio is not None) and vol_ratio >= 1.2

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

# ---------- main loop ----------
def main():
    sb = supabase_client_env()

    # sweetspot coins (kan senare bytas mot mcap-filter)
    sweet = load_sweetspot()
    sweet_coins = sweet["coin"].astype(str).tolist()
    hl_uni = fetch_hl_universe()
    hl_set = set(hl_uni)
    coins = [c for c in sweet_coins if c in hl_set]
    coins = sorted(list(set(coins)))

    version = "v2026-01-21-archive+latestscan"
    log(f"[BOOT] {version} started. coins={len(coins)}")
    hb(sb, f"BOOT {version}")

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

        # ----- scan chunk -----
        chunk = coins[idx:idx + SCAN_CHUNK]
        if len(chunk) < SCAN_CHUNK:
            chunk += coins[0:max(0, SCAN_CHUNK - len(chunk))]
        idx = (idx + SCAN_CHUNK) % max(1, len(coins))

        latest_rows = []
        new_calls = []

        for i, coin in enumerate(chunk, start=1):
            try:
                # latest scan row (uppdateras även om ingen signal)
                row = compute_latest_scan_row(coin, macro, loop_start)
                if row:
                    latest_rows.append(row)

                # signal detect (bara om row signal, annars skip => spar HL-calls)
                if row and row.get("signal"):
                    base_wr = float(sweet.loc[sweet["coin"].astype(str) == coin, "winrate"].iloc[0]) if "winrate" in sweet.columns else 0.55
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

            time.sleep(0.05)  # extra mild pacing

        # upsert latest scan
        try:
            upsert_latest_scan(sb, latest_rows)
        except Exception as e:
            log(f"[latest_scan][WARN] upsert failed: {e}")

        # upsert new calls
        try:
            upserted = db_upsert_calls(sb, new_calls)
        except Exception as e:
            log(f"[calls][WARN] upsert failed: {e}")

        # ----- status update (OPEN ONLY => archive lock) -----
        try:
            calls = db_read_calls(sb, limit=5000)
            if not calls.empty:
                open_calls = calls[calls["status"].astype(str) == "OPEN"].copy()
                open_calls = open_calls.sort_values("detected_time", ascending=False).head(STATUS_UPDATE_TOP_N)

                for _, r in open_calls.iterrows():
                    try:
                        upd = update_call_status(r, loop_start)
                        # om TP/SL/EXPIRED => detta "låser" last_price, och vi rör inte igen senare
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
        hb(sb, f"LOOP: done {dur:.1f}s scanned={len(chunk)} found={found} upserted={upserted}")
        log(f"[loop] done {dur:.1f}s scanned={len(chunk)} found={found} upserted={upserted} -> sleep {WORKER_SLEEP_SECONDS}s")
        time.sleep(WORKER_SLEEP_SECONDS)

if __name__ == "__main__":
    main()
