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

# =========================
# Helpers
# =========================
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def as_utc_ts(x) -> pd.Timestamp:
    # robust UTC parse
    ts = pd.to_datetime(x, utc=True, errors="coerce")
    if isinstance(ts, pd.Timestamp) and ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts

def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

# =========================
# HL client (rate-limited)
# =========================
SESSION = requests.Session()
_last_call_ts = 0.0
REQUEST_MIN_DELAY = float(os.getenv("HL_REQUEST_MIN_DELAY", "0.35"))  # slower to reduce 429
MAX_RETRIES = int(os.getenv("HL_MAX_RETRIES", "8"))

def hl_post(payload: Dict[str, Any]) -> Any:
    global _last_call_ts
    now = time.time()
    wait = REQUEST_MIN_DELAY - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)

    backoff = 0.8
    last_err = None

    for _ in range(MAX_RETRIES):
        try:
            r = SESSION.post(HL_INFO_URL, json=payload, timeout=25)

            if r.status_code == 429:
                last_err = "HTTP 429 rate-limited"
                time.sleep(backoff + random.uniform(0, 0.6))
                backoff = min(backoff * 1.8, 12.0)
                continue

            if r.status_code in (500, 502, 503, 504):
                last_err = f"HTTP {r.status_code} server error"
                time.sleep(backoff + random.uniform(0, 0.6))
                backoff = min(backoff * 1.8, 12.0)
                continue

            r.raise_for_status()
            _last_call_ts = time.time()
            return r.json()

        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(backoff + random.uniform(0, 0.6))
            backoff = min(backoff * 1.8, 12.0)

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

def fetch_candles(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    data = hl_post({
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": INTERVAL, "startTime": int(start_ms), "endTime": int(end_ms)}
    })
    if not data:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    df = pd.DataFrame(data)
    if "t" not in df.columns:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    for src, dst in [("o","open"),("h","high"),("l","low"),("c","close"),("v","volume")]:
        df[dst] = pd.to_numeric(df.get(src), errors="coerce")

    df = df[["timestamp","open","high","low","close","volume"]].dropna()
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df

def load_sweetspot() -> pd.DataFrame:
    df = pd.read_csv(SWEETSPOT_PATH)
    df["coin"] = df["coin"].astype(str)
    if "winrate" not in df.columns:
        df["winrate"] = 0.55
    return df

# =========================
# Macro series (BTC vol_z)
# =========================
def build_macro_series(end: datetime) -> pd.DataFrame:
    start = end - timedelta(days=3)
    btc = fetch_candles("BTC", to_ms(start), to_ms(end))
    if btc.empty or len(btc) < 120:
        return pd.DataFrame(columns=["timestamp","vol_z"])

    ret = np.log(btc["close"]).diff()
    rv = ret.rolling(40).std(ddof=0)
    mu = rv.rolling(40).mean()
    sd = rv.rolling(40).std(ddof=0)
    vol_z = (rv - mu) / sd
    out = pd.DataFrame({"timestamp": btc["timestamp"], "vol_z": vol_z}).dropna()
    return out

# =========================
# Chance heuristic
# =========================
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

# =========================
# Detect call
# =========================
def detect_call_for_coin(
    coin: str,
    macro: pd.DataFrame,
    base_wr: float,
    end: datetime,
    detected_time: datetime,
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

# =========================
# Position status update (OPEN -> TP/SL/EXPIRED) + live last_price/pnl
# =========================
def update_call_status(call_row: pd.Series, end: datetime) -> Dict[str, Any]:
    coin = str(call_row.get("coin"))
    call_time = as_utc_ts(call_row.get("call_time"))
    if pd.isna(call_time):
        return dict(call_row)

    # fetch from just before call_time to now
    start = call_time - pd.Timedelta(minutes=15)
    df = fetch_candles(coin, to_ms(start.to_pydatetime()), to_ms(end))
    if df.empty:
        return dict(call_row)

    df = df[df["timestamp"] >= call_time].copy()
    if df.empty:
        return dict(call_row)

    tp_price = safe_float(call_row.get("tp_price"))
    sl_price = safe_float(call_row.get("sl_price"))
    call_price = safe_float(call_row.get("call_price"))
    expiry_time = as_utc_ts(call_row.get("expiry_time"))

    if not np.isfinite(call_price) or call_price <= 0:
        return dict(call_row)

    hit_tp = np.isfinite(tp_price) and (df["high"] >= tp_price).any()
    hit_sl = np.isfinite(sl_price) and (df["low"] <= sl_price).any()

    status = str(call_row.get("status", "OPEN"))
    if status != "OPEN":
        # already closed -> keep static (archived)
        return dict(call_row)

    # conservative: SL first
    new_status = "OPEN"
    if hit_sl:
        new_status = "SL"
        last_price = float(sl_price) if np.isfinite(sl_price) else float(df["close"].iloc[-1])
    elif hit_tp:
        new_status = "TP"
        last_price = float(tp_price) if np.isfinite(tp_price) else float(df["close"].iloc[-1])
    elif (not pd.isna(expiry_time)) and (end >= expiry_time.to_pydatetime()):
        new_status = "EXPIRED"
        last_price = float(df["close"].iloc[-1])
    else:
        # still open: mark-to-market on last close
        last_price = float(df["close"].iloc[-1])

    pnl_pct = (last_price / call_price - 1.0) * 100.0

    updated = dict(call_row)
    updated["status"] = new_status
    updated["last_price"] = last_price
    updated["pnl_pct"] = pnl_pct
    return updated

# =========================
# DB ops
# =========================
def db_upsert_calls(sb: Client, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    payload = []
    for r in rows:
        payload.append({
            "call_time": as_utc_ts(r["call_time"]).isoformat(),
            "detected_time": as_utc_ts(r["detected_time"]).isoformat(),
            "coin": str(r["coin"]),
            "call_price": float(r["call_price"]),
            "tp_price": float(r["tp_price"]),
            "sl_price": float(r["sl_price"]),
            "expiry_time": as_utc_ts(r["expiry_time"]).isoformat(),
            "dump_pct": float(r["dump_pct"]),
            "vol_z": float(r["vol_z"]),
            "vol_ratio": float(r["vol_ratio"]),
            "liq_ratio": float(r["liq_ratio"]),
            "chance_pct": float(r["chance_pct"]),
            "status": str(r["status"]),
            "last_price": float(r["last_price"]),
            "pnl_pct": float(r["pnl_pct"]),
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
    df["detected_time"] = pd.to_datetime(df.get("detected_time", df["call_time"]), utc=True, errors="coerce")
    return df

def db_update_call(sb: Client, coin: str, call_time: pd.Timestamp, status: str, last_price: float, pnl_pct: float):
    # update exact position row
    sb.table("calls").update({
        "status": str(status),
        "last_price": float(last_price),
        "pnl_pct": float(pnl_pct),
    }).eq("coin", str(coin)).eq("call_time", as_utc_ts(call_time).isoformat()).execute()

# =========================
# PnL simulation
# =========================
def simulate_pnl(calls: pd.DataFrame, start_equity: float, notional_per_trade: float, apply_friction: bool) -> Dict[str, Any]:
    if calls.empty:
        return {"equity": start_equity, "pnl": 0.0, "pnl_pct": 0.0, "open_count": 0, "closed_count": 0}

    df = calls.copy()
    df["call_time"] = pd.to_datetime(df["call_time"], utc=True)
    df = df.sort_values("call_time")

    friction_rt = 0.0
    if apply_friction:
        friction_rt = 2.0 * (FEE_PER_SIDE + SLIP_PER_SIDE)

    equity = start_equity
    closed = 0
    open_ = 0

    for _, r in df.iterrows():
        entry = safe_float(r.get("call_price"))
        lastp = safe_float(r.get("last_price"))
        status = str(r.get("status", "OPEN"))

        if not np.isfinite(entry) or entry <= 0:
            continue

        if status == "TP":
            exitp = safe_float(r.get("tp_price"), lastp)
            closed += 1
        elif status == "SL":
            exitp = safe_float(r.get("sl_price"), lastp)
            closed += 1
        elif status == "EXPIRED":
            exitp = lastp
            closed += 1
        else:
            exitp = lastp
            open_ += 1

        if not np.isfinite(exitp) or exitp <= 0:
            continue

        ret = (exitp / entry) - 1.0
        ret -= friction_rt
        equity += notional_per_trade * ret

    pnl = equity - start_equity
    return {
        "equity": equity,
        "pnl": pnl,
        "pnl_pct": (pnl / start_equity) * 100.0,
        "open_count": open_,
        "closed_count": closed
    }
