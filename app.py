import os
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

from supabase import create_client, Client
from shared import simulate_pnl, TP, SL, HOLD_BARS, INTERVAL

MAX_CALLS_DISPLAY = 50

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

.dot {
  display:inline-block;
  width:10px;height:10px;
  border-radius:999px;
  background:#7CFF9B;
  box-shadow: 0 0 10px rgba(124,255,155,0.55);
  margin-right:8px;
}

div[data-testid="stDataFrame"] {
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.10);
  overflow: hidden;
}

/* ===== Loader (no text) ===== */
.vix-loader {
  width: 16px; height: 16px;
  border: 2px solid rgba(255,255,255,0.18);
  border-top-color: rgba(255,255,255,0.85);
  border-radius: 50%;
  display: inline-block;
  animation: vixspin 0.75s linear infinite;
  vertical-align: middle;
}
@keyframes vixspin { to { transform: rotate(360deg); } }

/* ===== Macro pill (ALLOW / BLOCK) ===== */
.macro-pill {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-radius: 999px;
  font-weight: 900;
  font-size: 13px;
  border: 1px solid rgba(255,255,255,0.14);
  box-shadow: 0 6px 18px rgba(0,0,0,0.35);
  user-select: none;
}
.macro-pill.red   { background: rgba(255, 59, 48, 0.18); color: rgba(255, 210, 210, 0.95); }
.macro-pill.green { background: rgba(52, 199, 89, 0.18); color: rgba(210, 255, 225, 0.95); }

.macro-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  background: currentColor;
  opacity: 0.95;
}
.macro-main { letter-spacing: 0.2px; }
.macro-sub  { opacity: 0.9; font-weight: 800; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def ui_loader():
    # Visual spinner only (no text)
    st.markdown('<span class="vix-loader"></span>', unsafe_allow_html=True)


def macro_pill(allow_trades: bool, subtext: str):
    cls = "green" if allow_trades else "red"
    main = "ALLOW TRADES" if allow_trades else "BLOCK TRADES"
    html = f"""
    <div class="macro-pill {cls}">
      <span class="macro-dot"></span>
      <span class="macro-main">{main}</span>
      <span class="macro-sub">{subtext}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


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
    return df


def db_read_latest_scan(limit: int = 200) -> pd.DataFrame:
    sb = supabase_client()
    res = sb.table("latest_scan").select("*").order("updated_time_utc", desc=True).limit(int(limit)).execute()
    data = getattr(res, "data", None) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["updated_time_utc"] = pd.to_datetime(df["updated_time_utc"], utc=True, errors="coerce")
    df["bar_close_utc"] = pd.to_datetime(df["bar_close_utc"], utc=True, errors="coerce")
    return df


def human_age(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "-"
    delta = datetime.now(timezone.utc) - dt.to_pydatetime()
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


st.markdown(
    f"""
    <div class="card">
      <h1 style="margin:0;">HL Whale-Dump Bounce Scanner</h1>
      <div class="small">
        HL {INTERVAL} • TP +{TP*100:.1f}% • SL -{SL*100:.1f}% • Hold {int(15*HOLD_BARS/60)}h
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

colx, coly = st.columns([1, 2])
with colx:
    if st.button("Export calls (CSV)"):
        calls_df = db_read_calls(limit=200000)
        if calls_df.empty:
            st.warning("No calls in DB yet.")
        else:
            csv = calls_df.sort_values("detected_time", ascending=True).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download calls_export.csv",
                data=csv,
                file_name="calls_export.csv",
                mime="text/csv",
            )
with coly:
    st.markdown("<div class='small neutral'>UI reads DB • Worker updates positions continuously</div>", unsafe_allow_html=True)


@st.fragment(run_every=10)
def pnl_section():
    calls_all = db_read_calls(limit=200000)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("PnL")
    if calls_all.empty:
        st.info("No calls in DB yet.")
    else:
        sim = simulate_pnl(
            calls=calls_all,
            start_equity=100_000.0,
            notional_per_trade=2_000.0,
            apply_friction=True,
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
              Closed <b>{sim["closed_count"]}</b> • Open <b>{sim["open_count"]}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


@st.fragment(run_every=10)
def positions_section():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Positions")

    # spinner under header while reading
    pos_status = st.empty()
    with pos_status.container():
        ui_loader()

    calls_all = db_read_calls(limit=5000)
    pos_status.empty()

    if calls_all.empty:
        st.info("No calls yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Show OPEN first, then recently closed
    calls_all["is_open"] = calls_all["status"].astype(str).eq("OPEN")
    calls_all = calls_all.sort_values(["is_open", "detected_time"], ascending=[False, False]).head(MAX_CALLS_DISPLAY).copy()

    calls_all["Since"] = calls_all["detected_time"].apply(human_age)

    def status_tag(s: str) -> str:
        s = str(s)
        if s == "TP":
            return "TP ✅"
        if s == "SL":
            return "SL ❌"
        if s == "EXPIRED":
            return "EXPIRED ⏳"
        return "OPEN •"

    calls_all["Status"] = calls_all["status"].apply(status_tag)
    calls_all["Chance"] = calls_all["chance_pct"].map(lambda x: f"{float(x):.0f}%" if pd.notna(x) else "-")
    calls_all["Call"] = calls_all["call_price"].map(lambda x: f"{float(x):.6g}" if pd.notna(x) else "-")
    calls_all["Now"] = calls_all["last_price"].map(lambda x: f"{float(x):.6g}" if pd.notna(x) else "-")
    calls_all["PnL %"] = calls_all["pnl_pct"].map(lambda x: f"{float(x):+.2f}%" if pd.notna(x) else "-")
    calls_all["Dump %"] = calls_all["dump_pct"].map(lambda x: f"{float(x):.2f}%" if pd.notna(x) else "-")

    out = calls_all[
        ["detected_time", "Since", "call_time", "coin", "Status", "Chance", "Call", "Now", "PnL %", "Dump %"]
    ].copy()
    out.rename(
        columns={
            "detected_time": "Detected (UTC)",
            "call_time": "Candle close (UTC)",
            "coin": "Coin",
        },
        inplace=True,
    )

    st.dataframe(out, use_container_width=True, height=540)
    st.markdown("</div>", unsafe_allow_html=True)


@st.fragment(run_every=10)
def latest_scan_section():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # spinner under header while reading
    scan_status = st.empty()
    with scan_status.container():
        ui_loader()

    df = db_read_latest_scan(limit=150)
    scan_status.empty()

    if df.empty:
        st.subheader("Latest scan")
        st.caption("No scan rows yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    last_upd = df["updated_time_utc"].max()
    st.subheader(f"Latest scan • {human_age(last_upd)}")

    # ===== Global MACRO condition: abs(z) <= 0.6 =====
    macro_threshold = 0.6
    if "btc_vol_z" in df.columns and df["btc_vol_z"].notna().any():
        btc_z = float(df["btc_vol_z"].median())
        allow = abs(btc_z) <= macro_threshold
        macro_sub = f"|BTC vol_z| {abs(btc_z):.2f} ≤ {macro_threshold:.1f}"
    else:
        allow = False
        macro_sub = "BTC vol_z missing"

    macro_pill(allow, macro_sub)
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    # =================================================

    def dot(ok: bool) -> str:
        return "🟢" if bool(ok) else "🔴"

    view = df.sort_values("coin").copy()
    view["15m %"] = view["chg_15m_pct"].map(lambda x: f"{float(x):+.2f}%" if pd.notna(x) else "-")
    view["Price"] = view["price"].map(lambda x: f"{float(x):.6g}" if pd.notna(x) else "-")
    view["Dump %"] = view["dump_pct"].map(lambda x: f"{float(x):.2f}%" if pd.notna(x) else "-")
    view["Vol x"] = view["vol_ratio"].map(lambda x: f"{float(x):.2f}x" if pd.notna(x) else "-")
    view["BTC z"] = view["btc_vol_z"].map(lambda x: f"{float(x):+.2f}" if pd.notna(x) else "-")

    view["Signal"] = view["signal"].map(dot)
    view["Dump"] = view["gate_dump"].map(dot)
    # Macro removed from per-row table (now global pill above)
    view["Spike"] = view["gate_spike"].map(dot)
    view["Floor"] = view["gate_floor"].map(dot)

    out = view[
        ["coin", "Price", "15m %", "Dump %", "Vol x", "BTC z", "Signal", "Dump", "Spike", "Floor", "bar_close_utc"]
    ].copy()
    out.rename(columns={"coin": "Coin", "bar_close_utc": "Bar close (UTC)"}, inplace=True)

    st.dataframe(out, use_container_width=True, height=520)
    st.markdown("</div>", unsafe_allow_html=True)


# Render page
pnl_section()

# Positions överst
positions_section()

# Latest scan under
latest_scan_section()
