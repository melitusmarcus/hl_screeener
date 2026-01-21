import os
import time
import random
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from supabase import create_client, Client

HL_INFO_URL = "https://api.hyperliquid.xyz/info"

# =========================
# CONFIG (shared)
# =========================
INTERVAL = "15m"  # HL candle interval

DROP_MIN = 0.005
DROP_MAX = 0.05

VOL_WIN = 20
VOL_Z_MAX = 0.60
VOL_SPIKE_MULT = 2.00
VOL_FLOOR_MULT = 1.20

TP = 0.015
SL = 0.020
HOLD_BARS = 24  # 24 * 15m = 6h

FEE_PER_SIDE = 0.00045
SLIP_PER_SIDE = 0.00030

SWEETSPOT_PATH = "sweetspot_coins.csv"

# =========================
# Supabase (ENV only)
# =========================
def supabase_client_env() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_KEY env vars.")
    return create_client(url, key)

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, str) and x.strip() == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def iso(dt: Any) -> str:
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    return str(dt)

def db_upsert_calls(sb: Client, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    payload = []
    for r in rows:
        payload.append({
            "call_time": iso(r["call_time"]),
            "detected_time": iso(r["detected_time"]),
            "coin": str(r["coin"]),
            "call_price": safe_float(r["call_price"]),
            "tp_price": safe_float(r["tp_price"]),
            "sl_price": safe_float(r["sl_price"]),
            "expiry_time": iso(r["expiry_time"]),
            "dump_pct": safe_float(r["dump_pct"]),
            "vol_z": safe_float(r["vol_z"]),
            "vol_ratio": safe_float(r["vol_ratio"]),
            "liq_ratio": safe_float(r["liq_ratio"]),
            "chance_pct": safe_float(r["chance_pct"]),
            "status": str(r.get("status", "OPEN")),
            "last_price": safe_float(r.get("last_price", r["call_price"])),
            "pnl_pct": safe_float(r.get("pnl_pct", 0.0)),
        })

    sb.table("calls").upsert(payload, on_conflict="coin,call_time").execute()
    return len(payload)

def db_read_calls(sb: Client, limit: int = 5000) -> pd.DataFrame:
    res = sb.table("calls").select("*").order("detected_time", desc=True).limit(int(limit)).execute()
    data = getattr(res, "data", None) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["call_time"] = pd.to_datetime(df["call_time"], utc=True, errors="coerce")
    df["expiry_time"] = pd.to_datetime(df["expiry_time"], utc=True, errors="coerce")
    if "detected_time" in df.columns:
        df["detected_time"] = pd.to_datetime(df["detected_time"], utc=True, errors="coerce")
    else:
        df["detected_time"] = df["call_time"]
    return df

def db_update_call(sb: Client, coin: str, call_time: pd.Timestamp, status: str, last_price: float, pnl_pct: float):
    sb.table("calls").update({
        "status": str(status),
        "last_price": safe_float(last_price),
        "pnl_pct": safe_float(pnl_pct),
    }).eq("coin", str(coin)).eq("call_time", call_time.isoformat()).execute()

# =========================
# HL client (rate-limited)
# =========================
SESSION = requests.Session()
_last_call_ts = 0.0
REQUEST_MIN_DELAY = 0.30
MAX_RETRIES = 6

def hl_post(payload: Dict[str, Any]) -> Any:
    """
    HL POST with:
    - rate limiting
    - retry/backoff
    - DEBUG info on failures (status, snippet)
    """
    global _last_call_ts
    now = time.time()
    wait = REQUEST_MIN_DELAY - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)

    backoff = 0.7
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = SESSION.post(HL_INFO_URL, json=payload, timeout=20)

            # store info for debug
            status = r.status_code
            text_snip = ""
            try:
                text_snip = (r.text or "")[:180]
            except Exception:
                pass

            if status == 429:
                last_err = f"HTTP 429 rate-limited. snip={text_snip}"
                time.sleep(backoff + random.uniform(0, 0.4))
                backoff = min(backoff * 1.7, 10.0)
                continue

            if status in (500, 502, 503, 504):
                last_err = f"HTTP {status} server error. snip={text_snip}"
                time.sleep(backoff + random.uniform(0, 0.4))
                backoff = min(backoff * 1.7, 10.0)
                continue

            r.raise_for_status()
            _last_call_ts = time.time()
            return r.json()

        except requests.Timeout:
            last_err = "Timeout"
            time.sleep(backoff + random.uniform(0, 0.4))
            backoff = min(backoff * 1.7, 10.0)

        except requests.RequestException as e:
            last_err = f"RequestException: {type(e).__name__}: {str(e)[:140]}"
            time.sleep(backoff + random.uniform(0, 0.4))
            backoff = min(backoff * 1.7, 10.0)

    raise RuntimeError(f"HL request failed after retries. last_err={last_err}")


def fetch_hl_universe() -> List[str]:
    meta = hl_post({"type": "meta"})
    coins: List[str] = []
    if isinstance(meta, dict):
        uni = meta.get("universe")
        if isinstance(uni, list):
            for item in uni:
                if isinstance(item, dict):
                    name = item.get("name")
                    if isinstance(name, str) and name:
                        coins.append(name)
    return sorted(list(set(coins)))

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def fetch_candles(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    data = hl_post({
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": INTERVAL, "startTime": start_ms, "endTime": end_ms}
    })
    if not data:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(data)
    if "t" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    for src, dst in [("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")]:
        df[dst] = pd.to_numeric(df.get(src), errors="coerce")

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df

def load_sweetspot() -> pd.DataFrame:
    df = pd.read_csv(SWEETSPOT_PATH)
    df["coin"] = df["coin"].astype(str)
    if "winrate" not in df.columns:
        df["winrate"] = 0.55
    return df

def compute_chance_pct(base_wr: float, dump_pct: float, vol_z: float, vol_ratio: float, liq_ratio: float) -> float:
    wr = float(np.clip(base_wr, 0.35, 0.80))
    score = (wr - 0.50)

    dump_norm = np.clip((dump_pct - DROP_MIN) / (DROP_MAX - DROP_MIN), 0, 1)
    score += 0.12 * dump_norm

    vol_norm = np.clip((vol_ratio - VOL_SPIKE_MULT) / max(0.1, (4.0 - VOL_SPIKE_MULT)), 0, 1)
    score += 0.10 * vol_norm

    z_norm = np.clip(abs(vol_z) / max(0.01, VOL_Z_MAX), 0, 1)
    score -= 0.10 * z_norm

    liq_norm = np.clip((liq_ratio - VOL_FLOOR_MULT) / max(0.1, (2.0 - VOL_FLOOR_MULT)), 0, 1)
    score += 0.04 * liq_norm

    chance = 50 + score * 100
    return float(np.clip(chance, 5, 95))

def build_macro_series(end: datetime) -> pd.DataFrame:
    start = end - timedelta(days=3)
    btc = fetch_candles("BTC", to_ms(start), to_ms(end))
    if btc.empty or len(btc) < 120:
        return pd.DataFrame(columns=["timestamp", "vol_z"])

    ret = np.log(btc["close"]).diff()
    rv = ret.rolling(40).std(ddof=0)
    mu = rv.rolling(40).mean()
    sd = rv.rolling(40).std(ddof=0)
    vol_z = (rv - mu) / sd
    out = pd.DataFrame({"timestamp": btc["timestamp"], "vol_z": vol_z}).dropna()
    return out

def detect_call_for_coin(
    coin: str,
    macro: pd.DataFrame,
    base_wr: float,
    end: datetime,
    detected_time: datetime
) -> Optional[Dict[str, Any]]:
    start = end - timedelta(days=3)
    df = fetch_candles(coin, to_ms(start), to_ms(end))
    if df.empty or len(df) < (VOL_WIN + 5):
        return None

    m = pd.merge(df, macro, on="timestamp", how="inner").dropna()
    if m.empty or len(m) < (VOL_WIN + 5):
        return None

    m = m.reset_index(drop=True)
    last = m.iloc[-1]

    dump_pct = float((last["open"] - last["low"]) / last["open"])
    vol_z = float(last["vol_z"])

    medv = m["volume"].rolling(VOL_WIN).median().iloc[-1]
    if not np.isfinite(medv) or medv <= 0:
        return None

    vol_ratio = float(last["volume"] / medv)
    liq_ratio = float(last["volume"] / medv)

    dump_ok = (dump_pct >= DROP_MIN) and (dump_pct <= DROP_MAX)
    macro_ok = (abs(vol_z) <= VOL_Z_MAX)
    spike_ok = (vol_ratio >= VOL_SPIKE_MULT)
    floor_ok = (liq_ratio >= VOL_FLOOR_MULT)

    if not (dump_ok and macro_ok and spike_ok and floor_ok):
        return None

    call_price = float(last["close"])
    call_time = pd.to_datetime(last["timestamp"], utc=True).to_pydatetime()
    tp_price = call_price * (1 + TP)
    sl_price = call_price * (1 - SL)
    expiry_time = call_time + timedelta(minutes=15 * HOLD_BARS)

    chance_pct = compute_chance_pct(
        base_wr=float(base_wr),
        dump_pct=float(dump_pct),
        vol_z=float(vol_z),
        vol_ratio=float(vol_ratio),
        liq_ratio=float(liq_ratio),
    )

    return {
        "call_time": pd.Timestamp(call_time, tz=timezone.utc),
        "detected_time": pd.Timestamp(detected_time, tz=timezone.utc),
        "coin": coin,
        "call_price": call_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "expiry_time": pd.Timestamp(expiry_time, tz=timezone.utc),
        "dump_pct": dump_pct * 100.0,
        "vol_z": vol_z,
        "vol_ratio": vol_ratio,
        "liq_ratio": liq_ratio,
        "chance_pct": chance_pct,
        "status": "OPEN",
        "last_price": call_price,
        "pnl_pct": 0.0,
    }

def update_call_status(call_row: pd.Series, end: datetime) -> Dict[str, Any]:
    coin = str(call_row["coin"])
    call_time = pd.to_datetime(call_row["call_time"], utc=True)
    start = call_time - timedelta(minutes=15)

    df = fetch_candles(coin, to_ms(start.to_pydatetime()), to_ms(end))
    if df.empty:
        return dict(call_row)

    df = df[df["timestamp"] >= call_time].copy()
    if df.empty:
        return dict(call_row)

    tp_price = safe_float(call_row.get("tp_price"))
    sl_price = safe_float(call_row.get("sl_price"))
    call_price = safe_float(call_row.get("call_price"))
    expiry_time = pd.to_datetime(call_row.get("expiry_time"), utc=True, errors="coerce")

    hit_tp = (df["high"] >= tp_price).any() if tp_price > 0 else False
    hit_sl = (df["low"] <= sl_price).any() if sl_price > 0 else False

    status = str(call_row.get("status", "OPEN"))
    if hit_sl:
        status = "SL"
    elif hit_tp:
        status = "TP"
    elif isinstance(expiry_time, pd.Timestamp) and pd.notna(expiry_time):
        if datetime.now(timezone.utc) >= expiry_time.to_pydatetime():
            status = "EXPIRED"

    last_price = safe_float(df["close"].iloc[-1], default=call_price)
    pnl_pct = ((last_price / call_price) - 1.0) * 100.0 if call_price > 0 else 0.0

    updated = dict(call_row)
    updated["status"] = status
    updated["last_price"] = last_price
    updated["pnl_pct"] = pnl_pct
    return updated

def simulate_pnl(calls: pd.DataFrame, start_equity: float, notional_per_trade: float, apply_friction: bool) -> Dict[str, Any]:
    if calls.empty:
        return {"equity": start_equity, "pnl": 0.0, "pnl_pct": 0.0, "open_count": 0, "closed_count": 0}

    df = calls.copy()
    df["call_time"] = pd.to_datetime(df["call_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["call_time"]).sort_values("call_time")

    friction_rt = 0.0
    if apply_friction:
        friction_rt = 2.0 * (FEE_PER_SIDE + SLIP_PER_SIDE)

    equity = float(start_equity)
    closed = 0
    open_ = 0

    for _, r in df.iterrows():
        entry = safe_float(r.get("call_price"), default=0.0)
        if entry <= 0:
            continue

        lastp = safe_float(r.get("last_price"), default=entry)
        status = str(r.get("status", "OPEN"))

        if status == "TP":
            exitp = safe_float(r.get("tp_price"), default=lastp)
            closed += 1
        elif status == "SL":
            exitp = safe_float(r.get("sl_price"), default=lastp)
            closed += 1
        elif status == "EXPIRED":
            exitp = lastp
            closed += 1
        else:
            exitp = lastp
            open_ += 1

        ret = (exitp / entry) - 1.0
        ret -= friction_rt
        equity += float(notional_per_trade) * ret

    pnl = equity - float(start_equity)
    return {
        "equity": equity,
        "pnl": pnl,
        "pnl_pct": (pnl / float(start_equity)) * 100.0 if start_equity else 0.0,
        "open_count": open_,
        "closed_count": closed
    }
