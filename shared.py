import os
import time
import random
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

from supabase import create_client, Client

HL_INFO_URL = "https://api.hyperliquid.xyz/info"

# =========================
# CONFIG
# =========================
INTERVAL = "15m"

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
# Time helpers (UTC-safe)
# =========================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def to_ms(dt: datetime) -> int:
    # dt MUST be tz-aware UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def ensure_utc_ts(x) -> pd.Timestamp:
    """
    Returns tz-aware UTC pandas Timestamp.
    - If x is naive -> localize UTC
    - If x is aware -> convert to UTC
    """
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return ts
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

def iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

# =========================
# Supabase
# =========================
def supabase_client_env() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_KEY env vars.")
    return create_client(url, key)

def db_upsert_calls(sb: Client, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    payload = []
    for r in rows:
        payload.append({
            "call_time": ensure_utc_ts(r["call_time"]).isoformat(),
            "detected_time": ensure_utc_ts(r["detected_time"]).isoformat(),
            "coin": str(r["coin"]),
            "call_price": float(r["call_price"]),
            "tp_price": float(r["tp_price"]),
            "sl_price": float(r["sl_price"]),
            "expiry_time": ensure_utc_ts(r["expiry_time"]).isoformat(),
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
    ct = ensure_utc_ts(call_time).isoformat()
    sb.table("calls").update({
        "status": str(status),
        "last_price": float(last_price),
        "pnl_pct": float(pnl_pct),
    }).eq("coin", str(coin)).eq("call_time", ct).execute()

def db_upsert_latest_scan(sb: Client, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    payload = []
    for r in rows:
        payload.append({
            "coin": str(r["coin"]),
            "bar_close_utc": ensure_utc_ts(r["bar_close_utc"]).isoformat(),
            "updated_time_utc": ensure_utc_ts(r["updated_time_utc"]).isoformat(),
            "price": None if r.get("price") is None else float(r["price"]),
            "chg_15m_pct": None if r.get("chg_15m_pct") is None else float(r["chg_15m_pct"]),
            "dump_pct": None if r.get("dump_pct") is None else float(r["dump_pct"]),
            "vol_ratio": None if r.get("vol_ratio") is None else float(r["vol_ratio"]),
            "btc_vol_z": None if r.get("btc_vol_z") is None else float(r["btc_vol_z"]),
            "gate_dump": bool(r.get("gate_dump", False)),
            "gate_macro": bool(r.get("gate_macro", False)),
            "gate_spike": bool(r.get("gate_spike", False)),
            "gate_floor": bool(r.get("gate_floor", False)),
            "signal": bool(r.get("signal", False)),
            "err": None if not r.get("err") else str(r["err"])[:240],
        })
    sb.table("latest_scan").upsert(payload, on_conflict="coin").execute()
    return len(payload)

def db_upsert_heartbeat(sb: Client, name: str, note: str):
    sb.table("worker_heartbeat").upsert(
        {
            "name": str(name),
            "last_seen": iso_utc(utc_now()),
            "note": str(note)[:240],
        },
        on_conflict="name"
    ).execute()

# =========================
# HL client (adaptive throttling)
# =========================
SESSION = requests.Session()
_last_call_ts = 0.0
REQUEST_MIN_DELAY = float(os.getenv("HL_MIN_DELAY_SECONDS", "0.45"))  # start slower than before
MAX_RETRIES = 8

# adaptive delay if 429 happens a lot
_dynamic_delay = REQUEST_MIN_DELAY

def hl_post(payload: Dict[str, Any]) -> Any:
    global _last_call_ts, _dynamic_delay

    now = time.time()
    wait = _dynamic_delay - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)

    backoff = 0.8
    last_err = None

    for _ in range(MAX_RETRIES):
        try:
            r = SESSION.post(HL_INFO_URL, json=payload, timeout=20)

            if r.status_code == 429:
                last_err = "HTTP 429 rate-limited"
                # increase delay a bit when rate-limited
                _dynamic_delay = min(_dynamic_delay * 1.15 + 0.05, 2.0)
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff = min(backoff * 1.6, 10.0)
                continue

            if r.status_code in (500, 502, 503, 504):
                last_err = f"HTTP {r.status_code} server error"
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff = min(backoff * 1.6, 10.0)
                continue

            r.raise_for_status()
            _last_call_ts = time.time()

            # gently relax delay when things work
            _dynamic_delay = max(REQUEST_MIN_DELAY, _dynamic_delay * 0.985)

            return r.json()

        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff = min(backoff * 1.6, 10.0)

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

    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True, errors="coerce")
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
# Features / signal logic
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

def build_btc_vol_z(end: datetime) -> Tuple[Optional[float], Optional[pd.DataFrame]]:
    """
    Return latest btc vol_z and full series.
    Uses only ~2 days, enough for 120+ candles.
    """
    end = end.astimezone(timezone.utc)
    start = end - timedelta(days=2)
    btc = fetch_candles("BTC", to_ms(start), to_ms(end + timedelta(seconds=1)))
    if btc.empty or len(btc) < 120:
        return None, None

    ret = np.log(btc["close"]).diff()
    rv = ret.rolling(40).std(ddof=0)
    mu = rv.rolling(40).mean()
    sd = rv.rolling(40).std(ddof=0)
    vol_z = (rv - mu) / sd
    out = pd.DataFrame({"timestamp": btc["timestamp"], "vol_z": vol_z}).dropna()
    if out.empty:
        return None, None
    latest_z = float(out["vol_z"].iloc[-1])
    return latest_z, out

def detect_call_for_coin(
    coin: str,
    macro: pd.DataFrame,
    base_wr: float,
    end: datetime,
    detected_time: datetime,
) -> Optional[Dict[str, Any]]:
    """
    Optimized: only fetch last ~ (VOL_WIN+10) candles instead of 3 days.
    """
    end = end.astimezone(timezone.utc)
    detected_time = detected_time.astimezone(timezone.utc)

    # need ~30 candles => 30*15m = 7.5h
    start = end - timedelta(hours=10)

    df = fetch_candles(coin, to_ms(start), to_ms(end + timedelta(seconds=1)))
    if df.empty or len(df) < (VOL_WIN + 5):
        return None

    m = pd.merge(df, macro, on="timestamp", how="inner").dropna()
    if m.empty or len(m) < (VOL_WIN + 5):
        return None

    m = m.reset_index(drop=True)
    last = m.iloc[-1]
    prev = m.iloc[-2]

    # dump wick from open to low (same as innan)
    dump_pct = float((last["open"] - last["low"]) / last["open"])
    vol_z = float(last["vol_z"])

    # vol spike vs rolling median
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
    call_time = ensure_utc_ts(last["timestamp"])
    tp_price = call_price * (1 + TP)
    sl_price = call_price * (1 - SL)
    expiry_time = call_time + pd.Timedelta(minutes=15 * HOLD_BARS)

    chance_pct = compute_chance_pct(
        base_wr=float(base_wr),
        dump_pct=float(dump_pct),
        vol_z=float(vol_z),
        vol_ratio=float(vol_ratio),
        liq_ratio=float(liq_ratio),
    )

    return {
        "call_time": call_time,
        "detected_time": ensure_utc_ts(detected_time),
        "coin": coin,
        "call_price": call_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "expiry_time": expiry_time,
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
    end = end.astimezone(timezone.utc)

    coin = str(call_row["coin"])
    call_time = ensure_utc_ts(call_row["call_time"])
    expiry_time = ensure_utc_ts(call_row["expiry_time"])

    # only need window from call_time-15m to now (cap to 8h)
    start = (call_time - pd.Timedelta(minutes=15)).to_pydatetime()
    start_dt = start if start.tzinfo else start.replace(tzinfo=timezone.utc)

    # cap range
    if end - start_dt > timedelta(hours=10):
        start_dt = end - timedelta(hours=10)

    df = fetch_candles(coin, to_ms(start_dt), to_ms(end + timedelta(seconds=1)))
    if df.empty:
        return dict(call_row)

    df = df[df["timestamp"] >= call_time].copy()
    if df.empty:
        return dict(call_row)

    tp_price = float(call_row["tp_price"])
    sl_price = float(call_row["sl_price"])
    call_price = float(call_row["call_price"])

    hit_tp = (df["high"] >= tp_price).any()
    hit_sl = (df["low"] <= sl_price).any()

    status = str(call_row.get("status", "OPEN"))
    if status in ("TP", "SL", "EXPIRED"):
        # already final; just refresh last_price/pnl
        pass
    else:
        # conservative: SL first
        if hit_sl:
            status = "SL"
        elif hit_tp:
            status = "TP"
        elif utc_now() >= expiry_time.to_pydatetime():
            status = "EXPIRED"
        else:
            status = "OPEN"

    last_price = float(df["close"].iloc[-1])
    pnl_pct = (last_price / call_price - 1.0) * 100.0

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
    df = df.sort_values("call_time")

    friction_rt = 0.0
    if apply_friction:
        friction_rt = 2.0 * (FEE_PER_SIDE + SLIP_PER_SIDE)

    equity = float(start_equity)
    closed = 0
    open_ = 0

    for _, r in df.iterrows():
        entry = float(r["call_price"])
        lastp = r.get("last_price", None)
        status = str(r.get("status", "OPEN"))

        if lastp is None or (isinstance(lastp, float) and np.isnan(lastp)):
            lastp = entry

        lastp = float(lastp)

        if status == "TP":
            exitp = float(r["tp_price"])
            closed += 1
        elif status == "SL":
            exitp = float(r["sl_price"])
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
        "pnl_pct": (pnl / float(start_equity)) * 100.0,
        "open_count": open_,
        "closed_count": closed
    }
