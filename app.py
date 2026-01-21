import os
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

from supabase import create_client, Client

from shared import simulate_pnl, TP, SL, HOLD_BARS, INTERVAL, db_mark_ui_first_seen

REFRESH_SECONDS = 60
MAX_CALLS_DISPLAY = 30
MAX_SCAN_DISPLAY = 120

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

.good { color: #7CFF9B !important; }
.bad { color: #FF6B6B !important; }
.neutral { color: #9DB2FF !important; }

.dot {
  display:inline-block; width:10px; height:10px; border-radius:50%;
  background:#2dff7a; box-shadow:0 0 10px rgba(45,255,122,0.6);
  margin-right:10px;
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
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_KEY (env vars or Streamlit secrets).")
    return create_client(url, key)

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
    if "ui_first_seen_time" in df.columns:
        df["ui_first_seen_time"] = pd.to_datetime(df["ui_first_seen_time"], utc=True, errors="coerce")
    else:
        df["ui_first_seen_time"] = pd.NaT
    return df

def db_read_latest_scan(limit: int = 200) -> pd.DataFrame:
    sb = supabase_client()
    res = sb.table("latest_scan").select("*").order("coin").limit(int(limit)).execute()
    data = getattr(res, "data", None) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df
    if "bar_close_utc" in df.columns:
        df["bar_close_utc"] = pd.to_datetime(df["bar_close_utc"], utc=True, errors="coerce")
    if "updated_time_utc" in df.columns:
        df["updated_time_utc"] = pd.to_datetime(df["updated_time_utc"], utc=True, errors="coerce")
    return df

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

def status_tag(s: str) -> str:
    s = str(s)
    if s == "TP": return "TP ✅"
    if s == "SL": return "SL ❌"
    if s == "EXPIRED": return "EXPIRED ⏳"
    return "OPEN •"

def _format_static_since(detected: pd.Timestamp, first_seen: pd.Timestamp) -> str:
    if pd.isna(detected) or pd.isna(first_seen):
        return "-"
    sec = int((first_seen - detected).total_seconds())
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

# Header
st.markdown(
    f"""
    <div class="card">
      <div style="display:flex; align-items:center; gap:10px;">
        <span class="dot"></span>
        <h1 style="margin:0;">HL Whale-Dump Bounce Scanner</h1>
      </div>
      <div class="small">
        HL {INTERVAL} • TP +{TP*100:.1f}% • SL -{SL*100:.1f}% • Hold {int(15*HOLD_BARS/60)}h
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
    st.markdown("<div class='small neutral'>UI updates without full reload.</div>", unsafe_allow_html=True)

# ---- PnL ----
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("PnL since start")
calls_all = db_read_calls(limit=200000)

if calls_all.empty:
    st.info("No calls in DB yet.")
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
          Trades: closed <b>{sim["closed_count"]}</b>, open <b>{sim["open_count"]}</b>.
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)

# ---- Latest scan fragment ----
@st.fragment(run_every=f"{REFRESH_SECONDS}s")
def section_latest_scan():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    scan = db_read_latest_scan(limit=MAX_SCAN_DISPLAY)
    if scan.empty:
        st.subheader("Latest scan")
        st.info("No latest_scan rows yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if "bar_close_utc" in scan.columns:
        scan["Since"] = scan["bar_close_utc"].apply(human_age)
        latest_bar = scan["bar_close_utc"].max()
        age = human_age(latest_bar)
    else:
        scan["Since"] = "-"
        age = "-"

    def dot(ok: bool) -> str:
        return "🟢" if bool(ok) else "🔴"

    def fmt_price(x):
        try: return f"{float(x):.6g}"
        except: return "-"

    def fmt_pct(x):
        try: return f"{float(x):+.2f}%"
        except: return "-"

    def fmt_dump(x):
        try: return f"{float(x):.2f}%"
        except: return "-"

    def fmt_v(x, suf="x"):
        try: return f"{float(x):.2f}{suf}"
        except: return "-"

    def fmt_z(x):
        try: return f"{float(x):+.2f}"
        except: return "-"

    out = pd.DataFrame()
    out["Coin"] = scan.get("coin", "-")
    out["Price"] = scan.get("price", None).map(fmt_price)
    out["15m %"] = scan.get("chg_15m_pct", None).map(fmt_pct)
    out["Dump %"] = scan.get("dump_pct", None).map(fmt_dump)
    out["Vol x"] = scan.get("vol_ratio", None).map(lambda x: fmt_v(x, "x"))
    out["BTC z"] = scan.get("btc_vol_z", None).map(fmt_z)
    out["Signal"] = scan.get("signal", False).map(dot)
    out["Dump"] = scan.get("gate_dump", False).map(dot)
    out["Macro"] = scan.get("gate_macro", False).map(dot)
    out["Spike"] = scan.get("gate_spike", False).map(dot)
    out["Floor"] = scan.get("gate_floor", False).map(dot)
    out["Since"] = scan["Since"]
    out["Bar close (UTC)"] = scan["bar_close_utc"].astype(str) if "bar_close_utc" in scan.columns else "-"

    st.subheader(f"Latest scan • {age}")
    st.dataframe(out, use_container_width=True, height=520)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Recent calls (STATIC Since) fragment ----
@st.fragment(run_every=f"{REFRESH_SECONDS}s")
def section_recent_calls_static_since():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Recent calls")

    sb = supabase_client()
    calls = db_read_calls(limit=200000)
    if calls.empty:
        st.info("No calls yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    view = calls.sort_values("detected_time", ascending=False).head(MAX_CALLS_DISPLAY).copy()

    # Set ui_first_seen_time ONCE for visible rows missing it
    now_utc = datetime.now(timezone.utc)
    missing = view[view["ui_first_seen_time"].isna()][["coin", "call_time"]]
    if not missing.empty:
        for _, r in missing.iterrows():
            try:
                db_mark_ui_first_seen(sb, coin=str(r["coin"]), call_time=pd.to_datetime(r["call_time"], utc=True), now_utc=now_utc)
            except Exception:
                pass

        # re-read to display fixed timestamps
        calls = db_read_calls(limit=200000)
        view = calls.sort_values("detected_time", ascending=False).head(MAX_CALLS_DISPLAY).copy()

    view["Since"] = view.apply(lambda r: _format_static_since(r["detected_time"], r["ui_first_seen_time"]), axis=1)
    view["Status"] = view["status"].apply(status_tag)

    def safe_float(x):
        try:
            return float(x)
        except:
            return None

    view["_call"] = view["call_price"].map(safe_float)
    view["_now"] = view["last_price"].map(safe_float)
    view["_dump"] = view.get("dump_pct", None).map(safe_float)

    def fmt_change(row):
        c = row["_call"]; n = row["_now"]
        if c is None or n is None or c == 0:
            return "-"
        return f"{(n/c - 1.0)*100.0:+.2f}%"

    view["Chance"] = view["chance_pct"].map(lambda x: f"{float(x):.0f}%" if pd.notna(x) else "-")
    view["Call"] = view["_call"].map(lambda x: f"{x:.6g}" if x is not None else "-")
    view["Now"] = view["_now"].map(lambda x: f"{x:.6g}" if x is not None else "-")
    view["Pnl %"] = view.apply(fmt_change, axis=1)
    view["Dump %"] = view["_dump"].map(lambda x: f"{x:.2f}%" if x is not None else "-")

    out = view[[
        "detected_time","Since","call_time","coin","Status","Chance","Call","Now","Pnl %","Dump %"
    ]].copy()
    out.rename(columns={
        "detected_time":"Detected (UTC)",
        "call_time":"Candle close (UTC)",
        "coin":"Coin",
    }, inplace=True)

    st.dataframe(out, use_container_width=True, height=360)
    st.markdown("</div>", unsafe_allow_html=True)

# Render
section_latest_scan()
section_recent_calls_static_since()
