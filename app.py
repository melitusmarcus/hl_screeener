import os
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from supabase import create_client, Client

from shared import simulate_pnl, TP, SL, HOLD_BARS, INTERVAL

REFRESH_SECONDS = 60
MAX_CALLS_DISPLAY = 30

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

    df["call_time"] = pd.to_datetime(df.get("call_time"), utc=True, errors="coerce")
    df["expiry_time"] = pd.to_datetime(df.get("expiry_time"), utc=True, errors="coerce")

    # fallback om detected_time saknas
    detected_src = df["detected_time"] if "detected_time" in df.columns else df["call_time"]
    df["detected_time"] = pd.to_datetime(detected_src, utc=True, errors="coerce")
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


# Header
st.markdown(
    f"""
    <div class="card">
      <h1 style="margin:0;">HL Whale-Dump Bounce Scanner</h1>
      <div class="small">
        UI reads from Supabase • Worker writes continuously • HL {INTERVAL} • TP +{TP*100:.1f}% • SL -{SL*100:.1f}% • Hold {int(15*HOLD_BARS/60)}h
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Controls row (statiska, behöver inte uppdateras)
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
                mime="text/csv"
            )

with coly:
    st.markdown("<div class='small neutral'>UI can sleep. Worker keeps logging.</div>", unsafe_allow_html=True)


# Live sections (uppdateras var REFRESH_SECONDS, utan custom component)
@st.fragment(run_every=f"{REFRESH_SECONDS}s")
def live_sections():
    calls_all = db_read_calls(limit=200000)

    # PnL Card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("PnL since start (sim)")
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
              Trades simulated: closed <b>{sim["closed_count"]}</b>, open <b>{sim["open_count"]}</b>.
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Calls table
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Recent calls (Since = detected_time)")

    if calls_all.empty:
        st.info("No calls yet.")
    else:
        view = calls_all.sort_values("detected_time", ascending=False).head(MAX_CALLS_DISPLAY).copy()
        view["Since"] = view["detected_time"].apply(human_age)

        def status_tag(s: str) -> str:
            s = str(s)
            if s == "TP":
                return "TP ✅"
            if s == "SL":
                return "SL ❌"
            if s == "EXPIRED":
                return "EXPIRED ⏳"
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
            "detected_time", "Since", "call_time", "coin", "Status", "Chance", "Call px", "Now px",
            "Change %", "Dump %", "BTC vol_z", "Vol spike x"
        ]].copy()
        out.rename(columns={
            "detected_time": "Detected (UTC)",
            "call_time": "Candle close (UTC)",
            "coin": "Coin"
        }, inplace=True)

        st.dataframe(out, use_container_width=True, height=540)

    st.markdown("</div>", unsafe_allow_html=True)


live_sections()
