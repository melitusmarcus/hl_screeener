import time
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from supabase import create_client, Client

HL_INFO_URL = "https://api.hyperliquid.xyz/info"

# =========================
# CONFIG
# =========================
INTERVAL = "15m"  # keep 15m; UI will use detected_time so it shows 0s on detection

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

REFRESH_SECONDS = 60
MAX_CALLS_DISPLAY = 30

SWEETSPOT_PATH = "sweetspot_coins.csv"

# =========================
# STYLE
# =========================
st.set_page_config(page_title="HL Whale-Dump Bounce Scanner", layout="wide")

CUSTOM_CSS = """
<style>
:root { color-scheme: dark; }
html, body, [data-testid="stApp"] { background: #05070c !important; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3, h4, p, div, span, label { color: #F1F5F9 !important; }
.small { font-size: 12px; opacity: 0.92; }

.card {
  background: linear-gradient(180deg, #0B1220 0%, #070B12 100%);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 14px 30px rgba(0,0,0,0.50);
}

.pill {
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  margin-right: 6px;
  font-size: 12px;
}

.good { color: #7CFF9B !important; }
.bad { color: #FF6B6B !important; }
.neutral { color: #9DB2FF !important; }

div[data-testid="stDataFrame"] {
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  overflow: hidden;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# RERUN helper
# =========================
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.stop()

# =========================
# Supabase
# =========================
@st.cache_resource
def supabase_client() -> Client:
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_KEY in Streamlit Secrets.")
    return create_client(url, key)

def db_upsert_calls(rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    sb = supabase_client()

    payload = []
    for r in rows:
        payload.append({
            "call_time": r["call_time"].isoformat(),
            "detected_time": r["detected_time"].isoformat(),
            "coin": r["coin"],
            "call_price": float(r["call_price"]),
            "tp_price": float(r["tp_price"]),
            "sl_price": float(r["sl_price"]),
            "expiry_time": r["expiry_time"].isoformat(),
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

def db_read_calls(limit: int = 5000) -> pd.DataFrame:
    sb = supabase_client()
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

def db_update_call(coin: str, call_time: pd.Timestamp, status: str, last_price: float, pnl_pct: float):
    sb = supabase_client()
    sb.table("calls").update({
        "status": str(status),
        "last_price": float(last_price),
        "pnl_pct": float(pnl_pct),
    }).eq("coin", str(coin)).eq("call_time", call_time.isoformat()).execute()

# =========================
# HL Client (rate limited)
# =========================
SESSION = requests.Session()
_last_call_ts = 0.0
REQUEST_MIN_DELAY = 0.20
MAX_RETRIES = 8

def hl_post(payload: Dict[str, Any]) -> Any:
    global _last_call_ts
    now = time.time()
    wait = REQUEST_MIN_DELAY - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)

    backoff = 0.6
    for _ in range(MAX_RETRIES):
        try:
            r = SESSION.post(HL_INFO_URL, json=payload, timeout=20)

            if r.status_code == 429:
                time.sleep(backoff + random.uniform(0, 0.4))
                backoff = min(backoff * 1.7, 10.0)
                continue

            if r.status_code in (500, 502, 503, 504):
                time.sleep(backoff + random.uniform(0, 0.4))
                backoff = min(backoff * 1.7, 10.0)
                continue

            r.raise_for_status()
            _last_call_ts = time.time()
            return r.json()

        except requests.RequestException:
            time.sleep(backoff + random.uniform(0, 0.4))
            backoff = min(backoff * 1.7, 10.0)

    raise RuntimeError("HL request failed after retries")

@st.cache_data(show_spinner=False, ttl=300)
def fetch_hl_universe() -> List[str]:
    """
    Pull all coins available on HL.
    Hyperliquid 'meta' typically returns a universe list with symbols/names.
    If schema changes, we fail gracefully (return empty).
    """
    try:
        meta = hl_post({"type": "meta"})
        coins: List[str] = []
        if isinstance(meta, dict):
            uni = meta.get("universe")
            if isinstance(uni, list):
                for item in uni:
                    if isinstance(item, dict):
                        # common key is "name" for coin symbol
                        name = item.get("name")
                        if isinstance(name, str) and name:
                            coins.append(name)
        return sorted(list(set(coins)))
    except Exception:
        return []

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def fetch_candles(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    data = hl_post({
        "type": "candleSnapshot",
        "req": {"coin": coin, "interval": INTERVAL, "startTime": start_ms, "endTime": end_ms}
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

@st.cache_data(show_spinner=False)
def load_sweetspot() -> pd.DataFrame:
    df = pd.read_csv(SWEETSPOT_PATH)
    df["coin"] = df["coin"].astype(str)
    if "winrate" not in df.columns:
        df["winrate"] = 0.55
    return df

def human_age(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "-"
    delta = datetime.now(timezone.utc) - dt.to_pydatetime()
    sec = int(delta.total_seconds())
    if sec < 0:
        sec = 0
    if sec < 60: return f"{sec}s"
    mins = sec // 60
    if mins < 60: return f"{mins}m"
    hrs = mins // 60
    if hrs < 48: return f"{hrs}h"
    days = hrs // 24
    return f"{days}d"

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
        return pd.DataFrame(columns=["timestamp","vol_z"])

    ret = np.log(btc["close"]).diff()
    rv = ret.rolling(40).std(ddof=0)
    mu = rv.rolling(40).mean()
    sd = rv.rolling(40).std(ddof=0)
    vol_z = (rv - mu) / sd
    out = pd.DataFrame({"timestamp": btc["timestamp"], "vol_z": vol_z}).dropna()
    return out

def detect_call_for_coin(coin: str, macro: pd.DataFrame, base_wr: float, end: datetime, detected_time: datetime) -> Optional[Dict[str, Any]]:
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

    tp_price = float(call_row["tp_price"])
    sl_price = float(call_row["sl_price"])
    call_price = float(call_row["call_price"])
    expiry_time = pd.to_datetime(call_row["expiry_time"], utc=True)

    hit_tp = (df["high"] >= tp_price).any()
    hit_sl = (df["low"] <= sl_price).any()

    status = "OPEN"
    if hit_sl:
        status = "SL"
    elif hit_tp:
        status = "TP"
    elif datetime.now(timezone.utc) >= expiry_time.to_pydatetime():
        status = "EXPIRED"

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
    df["call_time"] = pd.to_datetime(df["call_time"], utc=True)
    df = df.sort_values("call_time")

    friction_rt = 0.0
    if apply_friction:
        friction_rt = 2.0 * (FEE_PER_SIDE + SLIP_PER_SIDE)

    equity = start_equity
    closed = 0
    open_ = 0

    for _, r in df.iterrows():
        entry = float(r["call_price"])
        lastp = float(r["last_price"])
        status = str(r["status"])

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
        equity += notional_per_trade * ret

    pnl = equity - start_equity
    return {
        "equity": equity,
        "pnl": pnl,
        "pnl_pct": (pnl / start_equity) * 100.0,
        "open_count": open_,
        "closed_count": closed
    }

# =========================
# UI
# =========================
st.markdown(
    f"""
    <div class="card">
      <h1 style="margin:0;">HL Whale-Dump Bounce Scanner</h1>
      <div class="small">
        Persisted to Supabase • Since=detected_time (0s when your scanner sees it) • Candle source=HL {INTERVAL}
      </div>
      <div style="margin-top:10px;">
        <span class="pill">TP +{TP*100:.1f}%</span>
        <span class="pill">SL -{SL*100:.1f}%</span>
        <span class="pill">Hold {int(15*HOLD_BARS/60)}h</span>
        <span class="pill">VOL_Z_MAX {VOL_Z_MAX}</span>
        <span class="pill">SPIKE {VOL_SPIKE_MULT}x</span>
        <span class="pill">AUTO {REFRESH_SECONDS}s</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Export
colx, coly = st.columns([1, 2])
with colx:
    if st.button("Export calls (CSV)"):
        calls_df = db_read_calls(limit=200000)
        if calls_df.empty:
            st.warning("No calls in DB yet.")
        else:
            csv = calls_df.sort_values("detected_time", ascending=True).to_csv(index=False).encode("utf-8")
            st.download_button("Download calls_export.csv", data=csv, file_name="calls_export.csv", mime="text/csv")
with coly:
    st.markdown("<div class='small neutral'>Universe = sweetspot_coins.csv ∩ HL universe</div>", unsafe_allow_html=True)

# Load sweetspot + HL universe and intersect
try:
    sweet = load_sweetspot().reset_index(drop=True)
except Exception as e:
    st.error(f"Could not load {SWEETSPOT_PATH}. Put it next to app.py. Error: {e}")
    st.stop()

hl_uni = fetch_hl_universe()
sweet_coins = sweet["coin"].astype(str).tolist()

if hl_uni:
    hl_set = set(hl_uni)
    universe = sweet[sweet["coin"].astype(str).isin(hl_set)].copy().reset_index(drop=True)
    missing_on_hl = sorted(list(set(sweet_coins) - set(universe["coin"].astype(str).tolist())))
else:
    # If HL universe fetch fails, fall back to scanning the CSV as-is
    universe = sweet.copy()
    missing_on_hl = []

now = datetime.now(timezone.utc)
detected_time = now

colA, colB = st.columns([1.15, 1])

with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live scan")

    inserted = 0
    scanned = 0
    candles_ok = 0

    with st.spinner("Building BTC macro + scanning…"):
        macro = build_macro_series(now)

        if macro.empty:
            st.error("BTC macro series is empty. HL may be rate-limiting or returned no BTC candles.")
        else:
            new_calls = []
            for _, row in universe.iterrows():
                coin = str(row["coin"])
                base_wr = float(row.get("winrate", 0.55))
                scanned += 1
                try:
                    c = detect_call_for_coin(coin=coin, macro=macro, base_wr=base_wr, end=now, detected_time=detected_time)
                    if c:
                        new_calls.append(c)
                except Exception:
                    pass

            inserted = db_upsert_calls(new_calls)

    st.markdown(
        f"<div class='small neutral'>Sweetspot CSV coins: <b>{len(sweet)}</b> | On HL: <b>{len(universe)}</b> | Scanned: <b>{scanned}</b> | New calls upserted: <b>{inserted}</b></div>",
        unsafe_allow_html=True
    )
    if missing_on_hl:
        st.markdown(f"<div class='small'>Not on HL (filtered out): {len(missing_on_hl)} (first 10): {', '.join(missing_on_hl[:10])}</div>", unsafe_allow_html=True)

    # Update statuses
    calls_recent = db_read_calls(limit=2000)
    updated_n = 0
    if not calls_recent.empty:
        recent = calls_recent.sort_values("call_time", ascending=False).head(80).copy()
        for _, r in recent.iterrows():
            try:
                upd = update_call_status(r, now)
                if (upd["status"] != r["status"]) or (abs(float(upd["pnl_pct"]) - float(r["pnl_pct"])) > 0.01):
                    db_update_call(
                        str(r["coin"]),
                        pd.to_datetime(r["call_time"], utc=True),
                        upd["status"],
                        float(upd["last_price"]),
                        float(upd["pnl_pct"])
                    )
                    updated_n += 1
            except Exception:
                pass
        st.markdown(f"<div class='small'>Status updates this refresh: <b>{updated_n}</b></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("PnL since start (sim)")

    calls_all = db_read_calls(limit=200000)
    sim = simulate_pnl(
        calls=calls_all,
        start_equity=100_000.0,
        notional_per_trade=2_000.0,
        apply_friction=True
    )

    pnl = sim["pnl"]
    pnl_pct = sim["pnl_pct"]
    eq = sim["equity"]
    pnl_color = "good" if pnl >= 0 else "bad"

    st.markdown(
        f"""
        <div style="display:flex; gap:18px; flex-wrap:wrap;">
          <div>
            <div class="small">Equity</div>
            <div style="font-size:28px;" class="{pnl_color}">${eq:,.0f}</div>
          </div>
          <div>
            <div class="small">PnL</div>
            <div style="font-size:28px;" class="{pnl_color}">${pnl:,.0f}</div>
          </div>
          <div>
            <div class="small">PnL %</div>
            <div style="font-size:28px;" class="{pnl_color}">{pnl_pct:+.2f}%</div>
          </div>
        </div>
        <div class="small" style="margin-top:10px;">
          Trades simulated: closed <b>{sim["closed_count"]}</b>, open <b>{sim["open_count"]}</b>.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Recent calls
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Recent calls (Since = detected_time)")

calls = db_read_calls(limit=5000)
if calls.empty:
    st.info("No calls yet. (Filters may be strict or not enough live candles yet.)")
else:
    view = calls.sort_values("detected_time", ascending=False).head(MAX_CALLS_DISPLAY).copy()
    view["Since"] = view["detected_time"].apply(human_age)

    def status_tag(s: str) -> str:
        s = str(s)
        if s == "TP": return "TP ✅"
        if s == "SL": return "SL ❌"
        if s == "EXPIRED": return "EXPIRED ⏳"
        return "OPEN •"

    view["Status"] = view["status"].apply(status_tag)
    view["Chance"] = view["chance_pct"].map(lambda x: f"{float(x):.0f}%")
    view["Call px"] = view["call_price"].map(lambda x: f"{float(x):.6g}")
    view["Now px"] = view["last_price"].map(lambda x: f"{float(x):.6g}")
    view["Change %"] = view["pnl_pct"].map(lambda x: f"{float(x):+.2f}%")
    view["Dump %"] = view["dump_pct"].map(lambda x: f"{float(x):.2f}%")
    view["BTC vol_z"] = view["vol_z"].map(lambda x: f"{float(x):+.2f}")
    view["Vol spike x"] = view["vol_ratio"].map(lambda x: f"{float(x):.2f}x")

    out = view[[
        "detected_time","Since","call_time","coin","Status","Chance","Call px","Now px",
        "Change %","Dump %","BTC vol_z","Vol spike x"
    ]].copy()
    out.rename(columns={
        "detected_time": "Detected (UTC)",
        "call_time": "Candle close (UTC)",
        "coin": "Coin"
    }, inplace=True)

    st.dataframe(out, use_container_width=True, height=540)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='small'>"
    "<b>Note:</b> 15m signals are based on the last closed 15m candle. Your UI now shows 0s when detected_time is written."
    "</div>",
    unsafe_allow_html=True
)

# Auto-refresh LAST
st.markdown(
    f"<div class='small neutral'>Auto-refresh enabled. Next refresh in ~{REFRESH_SECONDS}s.</div>",
    unsafe_allow_html=True
)
time.sleep(REFRESH_SECONDS)
safe_rerun()
