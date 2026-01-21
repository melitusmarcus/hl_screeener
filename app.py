import os
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from supabase import create_client, Client

from shared import simulate_pnl, TP, SL, HOLD_BARS, INTERVAL

REFRESH_SECONDS = 30
MAX_CALLS_DISPLAY = 30
MAX_SCAN_DISPLAY = 250

st.set_page_config(page_title="HL Whale-Dump Bounce Scanner", layout="wide")

CUSTOM_CSS = """
<style>
:root { color-scheme: dark; }
html, body, [data-testid="stApp"] { background: #05070c !important; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1450px; }
h1,h2,h3,h4,p,div,span,label { color: #F1F5F9 !important; }

.small { font-size: 12px; opacity: 0.92; }
.muted { opacity: 0.78; }

.card {
  background: linear-gradient(180deg, #0B1220 0%, #070B12 100%);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 14px 30px rgba(0,0,0,0.50);
}

.good { color: #7CFF9B !important; }
.bad { color: #FF6B6B !important; }
.neutral { color: #9DB2FF !important; }

.header-row {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:14px;
}

.live-wrap { display:flex; align-items:center; gap:10px; }
.live-dot {
  width:10px; height:10px; border-radius:999px;
  background: #7CFF9B;
  box-shadow: 0 0 0 rgba(124,255,155, 0.7);
  animation: pulse 1.35s infinite;
}
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 rgba(124,255,155,0.55); }
  70%  { box-shadow: 0 0 0 10px rgba(124,255,155,0.0); }
  100% { box-shadow: 0 0 0 0 rgba(124,255,155,0.0); }
}

div[data-testid="stDataFrame"] {
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  overflow: hidden;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource
def supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", None)
    key = os.getenv("SUPABASE_SERVICE_KEY") or st.secrets.get("SUPABASE_SERVICE_KEY", None)
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_KEY.")
    return create_client(url, key)

def human_age(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "-"
    delta = datetime.now(timezone.utc) - dt.to_pydatetime()
    sec = int(delta.total_seconds())
    if sec < 0: sec = 0
    if sec < 60: return f"{sec}s"
    mins = sec // 60
    if mins < 60: return f"{mins}m"
    hrs = mins // 60
    if hrs < 48: return f"{hrs}h"
    days = hrs // 24
    return f"{days}d"

def db_read_calls(limit: int = 5000) -> pd.DataFrame:
    sb = supabase_client()
    res = sb.table("calls").select("*").order("detected_time", desc=True).limit(int(limit)).execute()
    data = getattr(res, "data", None) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["call_time"] = pd.to_datetime(df["call_time"], utc=True, errors="coerce")
    df["expiry_time"] = pd.to_datetime(df["expiry_time"], utc=True, errors="coerce")
    df["detected_time"] = pd.to_datetime(df.get("detected_time", df["call_time"]), utc=True, errors="coerce")
    return df

def db_read_heartbeat(name: str = "scanner") -> pd.DataFrame:
    sb = supabase_client()
    res = sb.table("worker_heartbeat").select("*").eq("name", name).limit(1).execute()
    data = getattr(res, "data", None) or []
    return pd.DataFrame(data)

def db_read_scan_latest(limit: int = 250) -> pd.DataFrame:
    sb = supabase_client()
    r1 = sb.table("scan_snapshots").select("bar_time").order("bar_time", desc=True).limit(1).execute()
    d1 = getattr(r1, "data", None) or []
    if not d1:
        return pd.DataFrame()
    latest_bar_time = d1[0].get("bar_time")
    if not latest_bar_time:
        return pd.DataFrame()

    r2 = (
        sb.table("scan_snapshots")
        .select("*")
        .eq("bar_time", latest_bar_time)
        .order("change_15m_pct", desc=True)
        .limit(int(limit))
        .execute()
    )
    data = getattr(r2, "data", None) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["bar_time"] = pd.to_datetime(df["bar_time"], utc=True, errors="coerce")
    df["updated_time"] = pd.to_datetime(df["updated_time"], utc=True, errors="coerce")
    return df

# ---- Auto refresh (component if available, else meta refresh)
def ui_autorefresh(seconds: int):
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=seconds * 1000, key="ui_refresh")
    except Exception:
        # fallback (full page refresh)
        st.markdown(f"<meta http-equiv='refresh' content='{int(seconds)}'>", unsafe_allow_html=True)

ui_autorefresh(REFRESH_SECONDS)

# ---------- Header ----------
hb = db_read_heartbeat("scanner")
hb_since = "-"
if not hb.empty:
    hb_dt = pd.to_datetime(hb.loc[0, "last_seen"], utc=True, errors="coerce")
    hb_since = human_age(hb_dt)

st.markdown(
    f"""
    <div class="card">
      <div class="header-row">
        <div>
          <h1 style="margin:0;">HL Whale-Dump Bounce Scanner</h1>
          <div class="small muted">HL {INTERVAL} • TP +{TP*100:.1f}% • SL -{SL*100:.1f}% • Hold {int(15*HOLD_BARS/60)}h</div>
        </div>
        <div class="live-wrap">
          <div class="live-dot"></div>
          <div class="small">Live • {hb_since}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- PnL ----------
calls_all = db_read_calls(limit=200000)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("PnL since start")
if calls_all.empty:
    st.info("No calls.")
else:
    sim = simulate_pnl(calls=calls_all, start_equity=100_000.0, notional_per_trade=2_000.0, apply_friction=True)
    pnl = sim["pnl"]
    pnl_pct = sim["pnl_pct"]
    eq = sim["equity"]
    pnl_color = "good" if pnl >= 0 else "bad"

    st.markdown(
        f"""
        <div style="display:flex; gap:18px; flex-wrap:wrap;">
          <div>
            <div class="small muted">Equity</div>
            <div style="font-size:28px;" class="{pnl_color}">${eq:,.0f}</div>
          </div>
          <div>
            <div class="small muted">PnL</div>
            <div style="font-size:28px;" class="{pnl_color}">${pnl:,.0f}</div>
          </div>
          <div>
            <div class="small muted">PnL %</div>
            <div style="font-size:28px;" class="{pnl_color}">{pnl_pct:+.2f}%</div>
          </div>
        </div>
        <div class="small muted" style="margin-top:10px;">
          Closed {sim["closed_count"]} • Open {sim["open_count"]}
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Latest Scan ----------
scan_df = db_read_scan_latest(limit=MAX_SCAN_DISPLAY)

latest_bar_age = "-"
latest_bar_time_str = "-"
if not scan_df.empty:
    latest_bar_time = scan_df["bar_time"].max()
    latest_bar_age = human_age(latest_bar_time)
    latest_bar_time_str = latest_bar_time.strftime("%Y-%m-%d %H:%M:%S")

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader(f"Latest scan • {latest_bar_age}")

if scan_df.empty:
    st.info("No scan data yet.")
else:
    view = scan_df.copy()

    # IMPORTANT: Since = bar close age (not updated_time)
    view["Since"] = view["bar_time"].apply(human_age)

    def dot(v: bool) -> str:
        return "🟢" if bool(v) else "🔴"

    view["Signal"] = view["signal_ok"].apply(dot)
    view["Dump"] = view["gate_dump_ok"].apply(dot)
    view["Macro"] = view["gate_macro_ok"].apply(dot)
    view["Spike"] = view["gate_spike_ok"].apply(dot)
    view["Floor"] = view["gate_floor_ok"].apply(dot)

    view["Price"] = view["close_price"].map(lambda x: f"{float(x):.6g}" if pd.notna(x) else "-")
    view["15m %"] = view["change_15m_pct"].map(lambda x: f"{float(x):+.2f}%" if pd.notna(x) else "-")
    view["Dump %"] = view["dump_pct"].map(lambda x: f"{float(x):.2f}%" if pd.notna(x) else "-")
    view["Vol x"] = view["vol_ratio"].map(lambda x: f"{float(x):.2f}x" if pd.notna(x) else "-")
    view["BTC z"] = view["btc_vol_z"].map(lambda x: f"{float(x):+.2f}" if pd.notna(x) else "-")

    out = view[[
        "coin", "Price", "15m %", "Dump %", "Vol x", "BTC z",
        "Signal", "Dump", "Macro", "Spike", "Floor",
        "Since", "bar_time"
    ]].copy()

    out.rename(columns={
        "coin": "Coin",
        "bar_time": "Bar close (UTC)"
    }, inplace=True)

    st.dataframe(out, use_container_width=True, height=560)

st.markdown("</div>", unsafe_allow_html=True)

# ---------- Recent Calls ----------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Recent calls")

if calls_all.empty:
    st.info("No calls.")
else:
    view = calls_all.sort_values("detected_time", ascending=False).head(MAX_CALLS_DISPLAY).copy()
    view["Since"] = view["detected_time"].apply(human_age)

    def status_tag(s: str) -> str:
        s = str(s)
        if s == "TP": return "TP ✅"
        if s == "SL": return "SL ❌"
        if s == "EXPIRED": return "EXPIRED ⏳"
        return "OPEN •"

    view["Status"] = view["status"].apply(status_tag)
    view["Chance"] = view["chance_pct"].map(lambda x: f"{float(x):.0f}%")
    view["Call"] = view["call_price"].map(lambda x: f"{float(x):.6g}")
    view["Now"] = view["last_price"].map(lambda x: f"{float(x):.6g}" if pd.notna(x) else "-")
    view["Pnl %"] = view["pnl_pct"].map(lambda x: f"{float(x):+.2f}%" if pd.notna(x) else "-")
    view["Dump %"] = view["dump_pct"].map(lambda x: f"{float(x):.2f}%")

    out = view[["detected_time", "Since", "call_time", "coin", "Status", "Chance", "Call", "Now", "Pnl %", "Dump %"]].copy()
    out.rename(columns={
        "detected_time": "Detected (UTC)",
        "call_time": "Candle close (UTC)",
        "coin": "Coin"
    }, inplace=True)

    st.dataframe(out, use_container_width=True, height=540)

st.markdown("</div>", unsafe_allow_html=True)
