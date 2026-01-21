import os
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from supabase import create_client, Client

from shared import simulate_pnl, TP, SL, HOLD_BARS, INTERVAL

# -------------------------
# UI CONFIG
# -------------------------
MAX_CALLS_DISPLAY = 30
LATEST_SCAN_LIMIT = 200

# fragment update intervals (seconds)
RUN_EVERY_PNL = 30
RUN_EVERY_CALLS = 30
RUN_EVERY_LATEST = 20

st.set_page_config(page_title="HL Whale-Dump Bounce Scanner", layout="wide")

CUSTOM_CSS = """
<style>
:root { color-scheme: dark; }
html, body, [data-testid="stApp"] { background: #05070c !important; }
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3, h4, p, div, span, label { color: #F1F5F9 !important; }
.small { font-size: 12px; opacity: 0.88; }
.muted { opacity: 0.75; }
.card {
  background: linear-gradient(180deg, #0B1220 0%, #070B12 100%);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 14px 30px rgba(0,0,0,0.45);
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

.dot {
  display:inline-block;
  width:10px; height:10px; border-radius:999px;
  margin-right:8px;
  border:1px solid rgba(255,255,255,0.25);
  vertical-align:middle;
}
.dot-ok { background: #2BD576; }
.dot-no { background: #FF4D4D; }
.dot-na { background: #666; }

div[data-testid="stDataFrame"] {
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  overflow: hidden;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------
# SUPABASE
# -------------------------
@st.cache_resource
def supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", None)
    key = os.getenv("SUPABASE_SERVICE_KEY") or st.secrets.get("SUPABASE_SERVICE_KEY", None)
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_KEY (env vars or Streamlit secrets).")
    return create_client(url, key)

def db_read_calls(limit: int = 5000) -> pd.DataFrame:
    sb = supabase_client()
    res = sb.table("calls").select("*").order("detected_time", desc=True).limit(int(limit)).execute()
    data = getattr(res, "data", None) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df

    # timestamps
    df["call_time"] = pd.to_datetime(df["call_time"], utc=True, errors="coerce")
    df["expiry_time"] = pd.to_datetime(df["expiry_time"], utc=True, errors="coerce")
    if "detected_time" in df.columns:
        df["detected_time"] = pd.to_datetime(df["detected_time"], utc=True, errors="coerce")
    else:
        df["detected_time"] = df["call_time"]

    return df

def db_read_latest_scan(limit: int = 200) -> pd.DataFrame:
    sb = supabase_client()
    res = sb.table("latest_scan").select("*").order("updated_time_utc", desc=True).limit(int(limit)).execute()
    data = getattr(res, "data", None) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df

    if "bar_close_utc" in df.columns:
        df["bar_close_utc"] = pd.to_datetime(df["bar_close_utc"], utc=True, errors="coerce")
    if "updated_time_utc" in df.columns:
        df["updated_time_utc"] = pd.to_datetime(df["updated_time_utc"], utc=True, errors="coerce")
    return df

# -------------------------
# HELPERS
# -------------------------
def human_age(dt: pd.Timestamp) -> str:
    if dt is None or pd.isna(dt):
        return "-"
    now = datetime.now(timezone.utc)
    delta = now - dt.to_pydatetime()
    sec = int(delta.total_seconds())
    if sec < 0:
        sec = 0
    if sec < 60:
        return f"{sec}s"
    mins = sec // 60
    if mins < 60:
        return f"{mins}m"
    hrs = mins // 60
    if hrs < 48:
        return f"{hrs}h"
    days = hrs // 24
    return f"{days}d"

def status_tag(s: str) -> str:
    s = str(s)
    if s == "TP":
        return "TP ✅"
    if s == "SL":
        return "SL ❌"
    if s == "EXPIRED":
        return "EXPIRED ⏳"
    return "OPEN •"

def fmt_num(x, fmt="{:.4g}", default="-"):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return fmt.format(float(x))
    except Exception:
        return default

def dot_html(ok: bool | None) -> str:
    if ok is True:
        return "<span class='dot dot-ok'></span>"
    if ok is False:
        return "<span class='dot dot-no'></span>"
    return "<span class='dot dot-na'></span>"

# "Since" ska vara statisk (fryser per call_key)
def get_static_since(call_key: str, detected_time: pd.Timestamp) -> str:
    if "since_cache" not in st.session_state:
        st.session_state["since_cache"] = {}

    cache = st.session_state["since_cache"]
    if call_key not in cache:
        cache[call_key] = human_age(detected_time)
    return cache[call_key]

# -------------------------
# HEADER
# -------------------------
st.markdown(
    f"""
    <div class="card">
      <h1 style="margin:0;">HL Whale-Dump Bounce Scanner</h1>
      <div class="small muted">
        HL {INTERVAL} • TP +{TP*100:.1f}% • SL -{SL*100:.1f}% • Hold {int(15*HOLD_BARS/60)}h
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

top_left, top_right = st.columns([1, 2])
with top_left:
    if st.button("Export calls (CSV)"):
        calls_df = db_read_calls(limit=200000)
        if calls_df.empty:
            st.warning("No calls in DB yet.")
        else:
            csv = calls_df.sort_values("detected_time", ascending=True).to_csv(index=False).encode("utf-8")
            st.download_button("Download calls_export.csv", data=csv, file_name="calls_export.csv", mime="text/csv")

with top_right:
    st.markdown("<div class='small neutral'>Worker writes continuously.</div>", unsafe_allow_html=True)

# -------------------------
# PnL CARD (fragment)
# -------------------------
@st.fragment(run_every=RUN_EVERY_PNL)
def section_pnl():
    calls_all = db_read_calls(limit=200000)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("PnL since start")
    if calls_all.empty:
        st.info("No calls in DB yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

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
        <div class="small muted" style="margin-top:10px;">
          Trades: closed <b>{sim["closed_count"]}</b>, open <b>{sim["open_count"]}</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

section_pnl()

# -------------------------
# LATEST SCAN (fragment)
# -------------------------
@st.fragment(run_every=RUN_EVERY_LATEST)
def section_latest_scan():
    scan = db_read_latest_scan(limit=LATEST_SCAN_LIMIT)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if scan.empty:
        st.subheader("Latest scan")
        st.info("No latest_scan rows yet (worker needs to upsert latest_scan).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # display age by updated_time_utc (NOT bar close)
    latest_upd = scan["updated_time_utc"].max() if "updated_time_utc" in scan.columns else None
    age = human_age(latest_upd)
    st.subheader(f"Latest scan • {age}")

    # Prep columns
    view = scan.copy()

    # sort: strongest dumps first (optional)
    if "chg_15m_pct" in view.columns:
        view = view.sort_values("chg_15m_pct", ascending=True)

    view["Coin"] = view["coin"].astype(str)
    view["Price"] = view["price"].map(lambda x: fmt_num(x, "{:.6g}"))
    view["15m %"] = view["chg_15m_pct"].map(lambda x: fmt_num(x, "{:+.2f}%"))
    view["Dump %"] = view["dump_pct"].map(lambda x: fmt_num(x, "{:.2f}%"))
    view["Vol x"] = view["vol_ratio"].map(lambda x: fmt_num(x, "{:.2f}x"))
    view["BTC z"] = view["btc_vol_z"].map(lambda x: fmt_num(x, "{:+.2f}"))

    # gates dots
    def gate_col(colname: str, label: str):
        if colname not in view.columns:
            view[label] = " "
            return
        view[label] = view[colname].map(lambda v: dot_html(bool(v)) if pd.notna(v) else dot_html(None))

    gate_col("signal", "Signal")
    gate_col("gate_dump", "Dump")
    gate_col("gate_macro", "Macro")
    gate_col("gate_spike", "Spike")
    gate_col("gate_floor", "Floor")

    view["Since"] = view["updated_time_utc"].map(human_age) if "updated_time_utc" in view.columns else "-"
    view["Bar close (UTC)"] = view["bar_close_utc"].map(lambda x: x.isoformat() if pd.notna(x) else "-")

    out = view[[
        "Coin","Price","15m %","Dump %","Vol x","BTC z",
        "Signal","Dump","Macro","Spike","Floor",
        "Since","Bar close (UTC)"
    ]].copy()

    # Render dots as HTML: use st.dataframe won't render HTML. We use st.markdown table alternative.
    # But to keep it simple + nice, we render as dataframe without HTML, using emojis instead.
    # Replace dot html with emojis:
    def to_emoji(v):
        if isinstance(v, str) and "dot-ok" in v:
            return "🟢"
        if isinstance(v, str) and "dot-no" in v:
            return "🔴"
        return "⚪"

    for c in ["Signal", "Dump", "Macro", "Spike", "Floor"]:
        out[c] = out[c].map(to_emoji)

    st.dataframe(out, use_container_width=True, height=520)
    st.markdown("</div>", unsafe_allow_html=True)

section_latest_scan()

# -------------------------
# RECENT CALLS (fragment)
# -------------------------
@st.fragment(run_every=RUN_EVERY_CALLS)
def section_calls():
    calls_all = db_read_calls(limit=200000)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Recent calls")

    if calls_all.empty:
        st.info("No calls yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    view = calls_all.sort_values("detected_time", ascending=False).head(MAX_CALLS_DISPLAY).copy()

    # static "Since" per call
    def make_key(r):
        # unique key for the call
        ct = pd.to_datetime(r["call_time"], utc=True)
        return f"{r.get('coin','?')}|{ct.isoformat()}"

    view["Since"] = view.apply(lambda r: get_static_since(make_key(r), pd.to_datetime(r["detected_time"], utc=True)), axis=1)

    view["Status"] = view["status"].apply(status_tag)
    view["Chance"] = view["chance_pct"].map(lambda x: fmt_num(x, "{:.0f}%", default="-"))
    view["Call"] = view["call_price"].map(lambda x: fmt_num(x, "{:.6g}", default="-"))
    view["Now"] = view["last_price"].map(lambda x: fmt_num(x, "{:.6g}", default="-"))
    view["PnL %"] = view["pnl_pct"].map(lambda x: fmt_num(x, "{:+.2f}%", default="-"))
    view["Dump %"] = view["dump_pct"].map(lambda x: fmt_num(x, "{:.2f}%", default="-"))

    out = view[[
        "detected_time","Since","call_time","coin","Status","Chance","Call","Now","PnL %","Dump %"
    ]].copy()

    out.rename(columns={
        "detected_time": "Detected (UTC)",
        "call_time": "Candle close (UTC)",
        "coin": "Coin"
    }, inplace=True)

    st.dataframe(out, use_container_width=True, height=520)
    st.markdown("</div>", unsafe_allow_html=True)

section_calls()
