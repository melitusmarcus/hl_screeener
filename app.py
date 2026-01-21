import os
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from supabase import create_client, Client

from shared import simulate_pnl, TP, SL, HOLD_BARS, INTERVAL

REFRESH_SECONDS = 60
MAX_CALLS_DISPLAY = 30
MAX_SCAN_DISPLAY = 200  # show up to N coins in latest scan list

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

def db_read_scan_latest(limit: int = 200) -> pd.DataFrame:
    """
    Reads latest scan snapshots, one row per coin per bar.
    We show the latest bar_time only.
    """
    sb = supabase_client()
    # Get latest bar_time first
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

# ---------- Header ----------
hb = db_read_heartbeat("scanner")
hb_since = "-"
hb_note = ""
if not hb.empty:
    hb_dt = pd.to_datetime(hb.loc[0, "last_seen"], utc=True, errors="coerce")
    hb_since = human_age(hb_dt)
    hb_note = str(hb.loc[0, "note"] or "")

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

# ---------- Top actions ----------
left, right = st.columns([1, 2])
with left:
    if st.button("Export calls (CSV)"):
        calls_df = db_read_calls(limit=200000)
        if calls_df.empty:
            st.warning("No calls.")
        else:
            csv = calls_df.sort_values("detected_time", ascending=True).to_csv(index=False).encode("utf-8")
            st.download_button("Download", data=csv, file_name="calls_export.csv", mime="text/csv")

# ---------- PnL ----------
calls_all = db_read_calls(limit=200000)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("PnL since start")
if calls_all.empty:
    st.info("No calls.")
else:
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
latest_scan_age = "-"
if not scan_df.empty:
    # use updated_time max as "latest"
    latest_upd = scan_df["updated_time"].max()
    latest_scan_age = human_age(latest_upd)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader(f"Latest scan • {latest_scan_age}")

if scan_df.empty:
    st.info("No scan data yet.")
else:
    view = scan_df.copy()
    view["Since"] = view["updated_time"].apply(human_age)
    view["Price"] = view["close_price"].map(lambda x: f"{float(x):.6g}")
    view["15m %"] = view["change_15m_pct"].map(lambda x: f"{float(x):+.2f}%")

    out = view[["coin", "Price", "15m %", "updated_time", "Since"]].copy()
    out.rename(columns={"coin": "Coin", "updated_time": "Updated (UTC)"}, inplace=True)
    st.dataframe(out, use_container_width=True, height=520)
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
    view["Now"] = view["last_price"].map(lambda x: f"{float(x):.6g}")
    view["Pnl %"] = view["pnl_pct"].map(lambda x: f"{float(x):+.2f}%")
    view["Dump %"] = view["dump_pct"].map(lambda x: f"{float(x):.2f}%")

    out = view[["detected_time", "Since", "call_time", "coin", "Status", "Chance", "Call", "Now", "Pnl %", "Dump %"]].copy()
    out.rename(columns={
        "detected_time": "Detected (UTC)",
        "call_time": "Candle close (UTC)",
        "coin": "Coin"
    }, inplace=True)

    st.dataframe(out, use_container_width=True, height=540)

st.markdown("</div>", unsafe_allow_html=True)

# Auto refresh (no extra component)
st.markdown(f"<div class='small muted'>Auto-refresh: {REFRESH_SECONDS}s</div>", unsafe_allow_html=True)
st.experimental_rerun if False else None
