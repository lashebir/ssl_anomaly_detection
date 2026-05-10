"""
Phishing Detection Dashboard
=============================
Streamlit dashboard for model results, data export, drift detection, and live monitoring.

Install:
    pip install streamlit plotly scipy scikit-learn joblib

Run:
    streamlit run dashboard.py
"""

import io
import json
import math
import warnings
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SSL Phishing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stMetric label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #888;
    }
    .stMetric [data-testid="metric-container"] {
        background: #0f0f0f;
        border: 1px solid #222;
        border-radius: 4px;
        padding: 1rem;
    }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    .stAlert { border-radius: 2px; }
    [data-testid="stSidebar"] {
        background: #0a0a0a;
        border-right: 1px solid #1a1a1a;
    }
    .live-flag {
        display: inline-block;
        background: #ff4444;
        color: white;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        padding: 2px 6px;
        border-radius: 2px;
        letter-spacing: 0.1em;
        margin-left: 8px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

PHISHING_COLOR = "#ff4444"
LEGIT_COLOR    = "#44aaff"
NEUTRAL_COLOR  = "#888888"
WARN_COLOR     = "#ffaa00"

DEFAULT_CERTS_FILE    = "certs_fallback.jsonl"
DEFAULT_LABELS_FILE   = "phishtank_labels.jsonl"
DEFAULT_FEATURES_FILE = "features.parquet"
DEFAULT_MODEL_FILE    = "xgb_model.pkl"
DEFAULT_STATE_FILE    = "poller_state.json"

FEATURE_COLS = ["entropy", "tld_risk", "domain_length", "subdomain_count",
                "brand_distance", "validity_days", "san_count"]

# ── Feature engineering (mirrors feature_engineering.py) ─────────────────────

def domain_entropy(domain: str) -> float:
    s = domain.replace(".", "").lstrip("*")
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


HIGH_RISK_TLDS = {"top","xyz","buzz","click","live","online","site","club",
                  "work","shop","icu","vip","fun","today"}
LOW_RISK_TLDS  = {"com","org","net","edu","gov","co","io","dev"}
TRUST_TLDS     = {"gov","edu","mil"}


def tld_risk(domain: str) -> int:
    tld = domain.rsplit(".", 1)[-1].lower() if "." in domain else ""
    if tld in TRUST_TLDS:  return 0
    if tld in HIGH_RISK_TLDS: return 2
    return 1


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features on a raw cert dataframe. Returns aligned feature df."""
    out = pd.DataFrame(index=df.index)
    domains = df["domain"] if "domain" in df.columns else df["domains"].apply(
        lambda d: d[0] if isinstance(d, list) else d
    )
    domains = domains.str.lower().str.lstrip("*.")
    out["entropy"]    = domains.apply(domain_entropy)
    out["tld_risk"]   = domains.apply(tld_risk)
    out["domain_length"]    = domains.str.len()
    out["subdomain_count"]  = domains.apply(
        lambda d: max(0, len(d.split(".")) - 2)
    )
    return out


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_features(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(path) if p.suffix == ".parquet" else pd.read_json(path, lines=True)


@st.cache_data(ttl=300)
def load_jsonl(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


@st.cache_data(ttl=60)
def load_live_stream(path: str, max_rows: int = 100_000) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True).tail(max_rows)


@st.cache_resource
def load_model(path: str):
    """Load trained model — cached for lifetime of the app."""
    import joblib
    if not Path(path).exists():
        return None
    return joblib.load(path)


def read_new_records(certs_path: str, state_path: str) -> tuple[pd.DataFrame, int]:
    """
    Read only lines written since last poll using poller_state.json.
    Returns (new_records_df, new_last_line).
    """
    if not Path(certs_path).exists():
        return pd.DataFrame(), 0

    # Load last known line
    last_line = 0
    if Path(state_path).exists():
        try:
            state = json.loads(Path(state_path).read_text())
            last_line = state.get("last_line", 0)
        except Exception:
            pass

    with open(certs_path, "r") as f:
        all_lines = f.readlines()

    new_lines = all_lines[last_line:]
    if not new_lines:
        return pd.DataFrame(), last_line

    records = []
    for line in new_lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    new_last_line = last_line + len(new_lines)
    return pd.DataFrame(records) if records else pd.DataFrame(), new_last_line


# ── Session state init ────────────────────────────────────────────────────────

if "live_log"       not in st.session_state: st.session_state.live_log       = []
if "live_flagged"   not in st.session_state: st.session_state.live_flagged   = []
if "live_errors"    not in st.session_state: st.session_state.live_errors    = []
if "live_last_line" not in st.session_state: st.session_state.live_last_line = 0
if "live_running"   not in st.session_state: st.session_state.live_running   = False
if "live_total"     not in st.session_state: st.session_state.live_total     = 0
if "live_phishing"  not in st.session_state: st.session_state.live_phishing  = 0
if "live_errors_n"  not in st.session_state: st.session_state.live_errors_n  = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ SSL Phishing Detection")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Model Results", "Data Explorer", "Drift Detection", "Live Monitor"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Data Sources**")

    features_path = st.text_input("Features file", DEFAULT_FEATURES_FILE)
    certs_path    = st.text_input("Live certs",    DEFAULT_CERTS_FILE)
    labels_path   = st.text_input("Labels file",   DEFAULT_LABELS_FILE)
    model_path    = st.text_input("Model file",    DEFAULT_MODEL_FILE)
    state_path    = st.text_input("Poller state",  DEFAULT_STATE_FILE)

    st.markdown("---")

    for label, path in [
        ("Features", features_path), ("Certs", certs_path),
        ("Labels",   labels_path),   ("Model", model_path),
    ]:
        icon = "🟢" if Path(path).exists() else "🔴"
        st.markdown(f"{icon} `{label}`")

    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

if page == "Model Results":
    st.title("Model Results")

    df = load_features(features_path)
    if df.empty:
        st.warning(f"No features file found at `{features_path}`. Run `feature_engineering.py` first.")
        st.stop()

    total      = len(df)
    phishing   = int(df["y"].sum()) if "y" in df.columns else 0
    legit      = total - phishing
    phish_rate = phishing / total if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Domains",    f"{total:,}")
    c2.metric("Phishing (y=1)",   f"{phishing:,}")
    c3.metric("Legitimate (y=0)", f"{legit:,}")
    c4.metric("Phishing Rate",    f"{phish_rate:.3%}")

    st.markdown("---")

    # ── Feature distributions ─────────────────────────────────────────────────

    st.subheader("Feature Distributions by Class")
    avail_features = [c for c in FEATURE_COLS if c in df.columns]

    if avail_features and "y" in df.columns:
        selected = st.selectbox("Select feature", avail_features)
        fig = go.Figure()
        for label, color, name in [(0, LEGIT_COLOR, "Legitimate"), (1, PHISHING_COLOR, "Phishing")]:
            subset = df[df["y"] == label][selected].dropna()
            fig.add_trace(go.Histogram(x=subset, name=name,
                                       marker_color=color, opacity=0.7, nbinsx=50))
        fig.update_layout(barmode="overlay", template="plotly_dark",
                          paper_bgcolor="#0a0a0a", plot_bgcolor="#0f0f0f",
                          font=dict(family="IBM Plex Mono", size=11),
                          legend=dict(orientation="h", y=1.1),
                          margin=dict(t=30, b=30, l=40, r=20), height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature means table ───────────────────────────────────────────────────

    st.subheader("Feature Means by Class")
    if "y" in df.columns and avail_features:
        means = df.groupby("y")[avail_features].mean().T
        means.columns = ["Legitimate", "Phishing"]
        means["Ratio (Phish/Legit)"] = (
            means["Phishing"] / means["Legitimate"].replace(0, np.nan)
        ).round(2)
        means = means.sort_values("Ratio (Phish/Legit)", ascending=False)
        st.dataframe(
            means.style
                .background_gradient(subset=["Ratio (Phish/Legit)"], cmap="RdYlGn_r")
                .format({"Legitimate": "{:.3f}", "Phishing": "{:.3f}",
                         "Ratio (Phish/Legit)": "{:.2f}x"}),
            use_container_width=True,
        )

    # ── PR curve ──────────────────────────────────────────────────────────────

    st.subheader("Operating Threshold")
    if "y_proba" in df.columns and "y" in df.columns:
        from sklearn.metrics import precision_recall_curve, average_precision_score

        precision, recall, thresholds = precision_recall_curve(df["y"], df["y_proba"])
        ap = average_precision_score(df["y"], df["y_proba"])
        min_precision = st.slider("Minimum acceptable precision", 0.1, 0.9, 0.3, 0.05)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                                  name=f"Model (AP={ap:.3f})",
                                  line=dict(color=PHISHING_COLOR, width=2)))
        fig.add_hline(y=min_precision, line_dash="dash", line_color=WARN_COLOR,
                      annotation_text=f"Min precision = {min_precision}",
                      annotation_font_color=WARN_COLOR)
        fig.add_hline(y=phish_rate, line_dash="dot", line_color=NEUTRAL_COLOR,
                      annotation_text="Random baseline",
                      annotation_font_color=NEUTRAL_COLOR)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0a0a",
                          plot_bgcolor="#0f0f0f",
                          font=dict(family="IBM Plex Mono", size=11),
                          xaxis_title="Recall", yaxis_title="Precision",
                          height=400, margin=dict(t=30, b=40, l=50, r=20))
        st.plotly_chart(fig, use_container_width=True)

        valid = precision[:-1] >= min_precision
        if valid.any():
            best_idx = recall[:-1][valid].argmax()
            st.success(
                f"Optimal threshold: **{thresholds[valid][best_idx]:.3f}** → "
                f"Recall: **{recall[:-1][valid][best_idx]:.3f}** | "
                f"Precision: **{precision[:-1][valid][best_idx]:.3f}**"
            )
    else:
        st.info("Add a `y_proba` column to your features file to enable the PR curve.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DATA EXPLORER + EXPORT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Data Explorer":
    st.title("Data Explorer")

    df = load_features(features_path)
    if df.empty:
        st.warning(f"No features file found at `{features_path}`.")
        st.stop()

    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        label_filter = st.multiselect("Label (y)", [0, 1], default=[0, 1],
                                       format_func=lambda x: "Phishing" if x == 1 else "Legitimate")
    with col2:
        if "data_source" in df.columns:
            source_filter = st.multiselect("Data source",
                                            df["data_source"].unique().tolist(),
                                            default=df["data_source"].unique().tolist())
        else:
            source_filter = None
    with col3:
        if "entropy" in df.columns:
            entropy_range = st.slider("Entropy range",
                                       float(df["entropy"].min()), float(df["entropy"].max()),
                                       (float(df["entropy"].min()), float(df["entropy"].max())))
        else:
            entropy_range = None

    mask = df["y"].isin(label_filter) if "y" in df.columns else pd.Series([True] * len(df))
    if source_filter and "data_source" in df.columns:
        mask &= df["data_source"].isin(source_filter)
    if entropy_range and "entropy" in df.columns:
        mask &= df["entropy"].between(*entropy_range)

    df_filtered = df[mask].reset_index(drop=True)
    st.markdown(f"**{len(df_filtered):,}** records match filters ({len(df_filtered)/len(df):.1%} of total)")

    display_cols = [c for c in ["domain", "y", "entropy", "tld_risk", "data_source",
                                  "label_source", "brand_distance", "validity_days"]
                    if c in df_filtered.columns]
    st.dataframe(df_filtered[display_cols].head(1000), use_container_width=True, height=400)

    if len(df_filtered) > 1000:
        st.caption(f"Showing first 1,000 of {len(df_filtered):,} rows. Export to see all.")

    st.subheader("Export")
    col1, col2, col3 = st.columns(3)

    with col1:
        buf = io.StringIO()
        df_filtered.to_csv(buf, index=False)
        st.download_button("⬇️ Download CSV", data=buf.getvalue(),
                            file_name=f"phishing_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv", use_container_width=True)
    with col2:
        buf = io.StringIO()
        for _, row in df_filtered.iterrows():
            buf.write(json.dumps(row.to_dict(), default=str) + "\n")
        st.download_button("⬇️ Download JSONL", data=buf.getvalue(),
                            file_name=f"phishing_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl",
                            mime="application/jsonl", use_container_width=True)
    with col3:
        buf = io.BytesIO()
        df_filtered.to_parquet(buf, index=False)
        st.download_button("⬇️ Download Parquet", data=buf.getvalue(),
                            file_name=f"phishing_{datetime.now().strftime('%Y%m%d_%H%M')}.parquet",
                            mime="application/octet-stream", use_container_width=True)

    if "y" in df_filtered.columns and "domain" in df_filtered.columns:
        phishing_domains = df_filtered[df_filtered["y"] == 1]["domain"].dropna().unique()
        if len(phishing_domains):
            st.download_button(
                f"⬇️ Phishing Domain List ({len(phishing_domains):,} domains)",
                data="\n".join(phishing_domains),
                file_name=f"phishing_domains_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain", use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Drift Detection":
    st.title("Drift Detection")

    st.info("Compares feature distributions between training data and live stream using KS tests.", icon="ℹ️")

    df_ref  = load_features(features_path)
    df_live = load_live_stream(certs_path)

    if df_ref.empty:
        st.warning("No reference dataset found. Run feature engineering first.")
        st.stop()

    avail_features = [c for c in FEATURE_COLS if c in df_ref.columns]

    col1, col2 = st.columns(2)
    with col1:
        window_hours       = st.slider("Live window (hours)", 1, 48, 24)
        pvalue_threshold   = st.slider("KS p-value threshold", 0.001, 0.1, 0.05, 0.001)
    with col2:
        mean_shift_threshold = st.slider("Mean shift alert (%)", 5, 50, 20)

    st.markdown("---")
    st.subheader("Reference Distribution")
    ref_stats = df_ref[avail_features].describe().T[["mean", "std", "50%"]].rename(columns={"50%": "median"})
    st.dataframe(ref_stats.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Live Stream Drift Analysis")

    if df_live.empty:
        st.warning(f"No live data found at `{certs_path}`. Start your streaming pipeline.")
        st.markdown("When live data is available this panel shows KS tests, mean shift %, and distribution overlays.")
    else:
        if "timestamp" in df_live.columns:
            df_live["timestamp"] = pd.to_datetime(df_live["timestamp"], utc=True)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
            df_window = df_live[df_live["timestamp"] >= cutoff]
        else:
            df_window = df_live

        st.markdown(f"**{len(df_window):,}** records in last {window_hours}h window")

        live_feats = [c for c in avail_features if c in df_window.columns]
        if not live_feats:
            st.warning("Live file missing engineered features. Run feature engineering on live data first.")
        else:
            drift_results = []
            for feature in live_feats:
                ref_vals  = df_ref[feature].dropna().values
                live_vals = df_window[feature].dropna().values
                if len(live_vals) < 30:
                    continue
                ks_stat, p_value = stats.ks_2samp(ref_vals, live_vals)
                ref_mean, live_mean = ref_vals.mean(), live_vals.mean()
                mean_shift_pct = abs(live_mean - ref_mean) / (abs(ref_mean) + 1e-9) * 100
                drift_results.append({
                    "Feature": feature, "Ref Mean": ref_mean, "Live Mean": live_mean,
                    "Mean Shift %": mean_shift_pct, "KS Statistic": ks_stat,
                    "p-value": p_value,
                    "Drift": p_value < pvalue_threshold or mean_shift_pct > mean_shift_threshold,
                })

            df_drift = pd.DataFrame(drift_results)
            if df_drift.empty:
                st.info("Not enough live data yet.")
            else:
                drifted = df_drift[df_drift["Drift"]]
                if len(drifted):
                    st.error(f"⚠️ Drift in {len(drifted)} feature(s): {', '.join(drifted['Feature'].tolist())}")
                else:
                    st.success("✅ No significant drift detected")

                def style_drift(val):
                    return "background-color: #3d1a1a; color: #ff4444;" if val else ""

                st.dataframe(
                    df_drift.style.applymap(style_drift, subset=["Drift"])
                        .format({"Ref Mean": "{:.4f}", "Live Mean": "{:.4f}",
                                  "Mean Shift %": "{:.1f}%", "KS Statistic": "{:.4f}",
                                  "p-value": "{:.4f}"}),
                    use_container_width=True,
                )

                st.subheader("Distribution Overlays")
                for feature in live_feats:
                    row = df_drift[df_drift["Feature"] == feature].iloc[0]
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=df_ref[feature].dropna(), name="Reference",
                                               opacity=0.6, marker_color=LEGIT_COLOR,
                                               nbinsx=40, histnorm="probability density"))
                    fig.add_trace(go.Histogram(x=df_window[feature].dropna(), name="Live",
                                               opacity=0.6, marker_color=WARN_COLOR,
                                               nbinsx=40, histnorm="probability density"))
                    fig.update_layout(
                        barmode="overlay", template="plotly_dark",
                        paper_bgcolor="#0a0a0a", plot_bgcolor="#0f0f0f",
                        font=dict(family="IBM Plex Mono", size=11),
                        title=f"{feature} — {'⚠️ DRIFT' if row['Drift'] else '✅ OK'} "
                              f"(KS={row['KS Statistic']:.3f}, p={row['p-value']:.4f})",
                        height=280, margin=dict(t=40, b=30, l=40, r=20),
                        legend=dict(orientation="h", y=1.15),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if "timestamp" in df_window.columns and len(df_window) > 100:
                    st.subheader("Cert Volume Over Time")
                    df_window["hour"] = df_window["timestamp"].dt.floor("h")
                    hourly = df_window.groupby("hour").size().reset_index(name="count")
                    mean_vol, std_vol = hourly["count"].mean(), hourly["count"].std()
                    spikes = hourly[hourly["count"] > mean_vol + 2 * std_vol]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["count"],
                                             mode="lines+markers", name="Certs/hour",
                                             line=dict(color=LEGIT_COLOR, width=2),
                                             marker=dict(size=4)))
                    fig.add_hline(y=mean_vol, line_dash="dash", line_color=NEUTRAL_COLOR,
                                  annotation_text=f"Mean: {mean_vol:.0f}/hr")
                    if len(spikes):
                        fig.add_trace(go.Scatter(x=spikes["hour"], y=spikes["count"],
                                                  mode="markers", name="Spike",
                                                  marker=dict(color=PHISHING_COLOR, size=10, symbol="x")))
                    fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0a0a",
                                      plot_bgcolor="#0f0f0f",
                                      font=dict(family="IBM Plex Mono", size=11),
                                      xaxis_title="Time", yaxis_title="Certs/hour",
                                      height=350, margin=dict(t=30, b=40, l=50, r=20))
                    st.plotly_chart(fig, use_container_width=True)

                    if len(spikes):
                        st.warning(f"⚠️ {len(spikes)} volume spike(s) — investigate for campaign activity.")

    with st.expander("How drift detection works"):
        st.markdown("""
        **KS Test** — compares full distributions, not just means. Low p-value = significant drift.

        **Mean Shift %** — catches gradual centre drift even if distribution shape is preserved.

        **Volume Spikes** — sudden cert bursts (>2σ) signal coordinated campaign infrastructure.

        **Response playbook:** identify drifted features → score drift window with current model →
        if FN rate up, retrain on rolling window → track newly targeted brands via brand_distance.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Live Monitor":
    st.markdown(
        "## Live Monitor <span class='live-flag'>LIVE</span>",
        unsafe_allow_html=True,
    )

    model = load_model(model_path)

    # ── Config bar ────────────────────────────────────────────────────────────

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        threshold   = st.slider("Phishing threshold", 0.1, 0.9, 0.3, 0.05,
                                  help="Probability above which a domain is flagged")
    with col2:
        poll_secs   = st.slider("Poll interval (s)", 5, 60, 10)
    with col3:
        max_log     = st.slider("Max log rows", 50, 500, 100)
    with col4:
        auto_scroll = st.checkbox("Auto-scroll flagged", value=True)

    if model is None:
        st.warning(
            f"No model found at `{model_path}`. "
            "Save your trained XGBoost model first:\n\n"
            "```python\nimport joblib\njoblib.dump(model, 'xgb_model.pkl')\n```"
        )

    st.markdown("---")

    # ── Session summary metrics ───────────────────────────────────────────────

    m1, m2, m3, m4, m5 = st.columns(5)
    total_placeholder    = m1.empty()
    phishing_placeholder = m2.empty()
    rate_placeholder     = m3.empty()
    errors_placeholder   = m4.empty()
    uptime_placeholder   = m5.empty()

    # ── Stream controls ───────────────────────────────────────────────────────

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        start_btn = st.button("▶ Start Monitoring", use_container_width=True,
                               disabled=st.session_state.live_running)
    with ctrl2:
        stop_btn  = st.button("⏹ Stop Monitoring",  use_container_width=True,
                               disabled=not st.session_state.live_running)

    if start_btn:
        st.session_state.live_running   = True
        st.session_state.live_log       = []
        st.session_state.live_flagged   = []
        st.session_state.live_errors    = []
        st.session_state.live_total     = 0
        st.session_state.live_phishing  = 0
        st.session_state.live_errors_n  = 0
        st.session_state.session_start  = datetime.now(timezone.utc)
        # Reset state to current line so we only score new certs
        if Path(certs_path).exists():
            with open(certs_path) as f:
                current_lines = sum(1 for _ in f)
            Path(state_path).write_text(json.dumps({"last_line": current_lines}))
            st.session_state.live_last_line = current_lines

    if stop_btn:
        st.session_state.live_running = False

    st.markdown("---")

    # ── Live feed panels ──────────────────────────────────────────────────────

    col_feed, col_flags = st.columns([3, 2])

    with col_feed:
        st.subheader("Incoming Certs")
        feed_placeholder = st.empty()

    with col_flags:
        st.subheader("🚨 Flagged Domains")
        flags_placeholder = st.empty()

    # ── Error log ─────────────────────────────────────────────────────────────

    with st.expander("Error log", expanded=False):
        errors_placeholder_exp = st.empty()

    # ── Scoring rate chart ────────────────────────────────────────────────────

    st.subheader("Scoring Rate")
    rate_chart_placeholder = st.empty()

    # ── Poll loop ─────────────────────────────────────────────────────────────

    if st.session_state.live_running:
        import time

        new_df, new_last_line = read_new_records(certs_path, state_path)
        st.session_state.live_last_line = new_last_line

        # Update state file for next poll
        Path(state_path).write_text(json.dumps({"last_line": new_last_line}))

        if not new_df.empty and model is not None:
            # Explode domains
            if "domains" in new_df.columns:
                new_df = new_df.explode("domains").rename(columns={"domains": "domain"})
            new_df = new_df.dropna(subset=["domain"])
            new_df["domain"] = new_df["domain"].str.lower().str.lstrip("*.")

            # Engineer features
            try:
                features = engineer_features(new_df)
                avail    = [c for c in ["entropy", "tld_risk"] if c in features.columns]
                new_df["y_proba"] = model.predict_proba(features[avail])[:, 1]
                new_df["y_pred"]  = (new_df["y_proba"] >= threshold).astype(int)
                new_df["scored_at"] = datetime.now(timezone.utc).isoformat()

                st.session_state.live_total    += len(new_df)
                st.session_state.live_phishing += int(new_df["y_pred"].sum())

                # Append to log
                log_rows = new_df[["domain", "y_proba", "y_pred", "scored_at"]].to_dict("records")
                st.session_state.live_log = (st.session_state.live_log + log_rows)[-max_log:]

                # Collect flagged
                flagged = new_df[new_df["y_pred"] == 1][["domain", "y_proba", "scored_at"]]
                if len(flagged):
                    st.session_state.live_flagged = (
                        st.session_state.live_flagged + flagged.to_dict("records")
                    )[-200:]

            except Exception as e:
                err_msg = f"{datetime.now(timezone.utc).isoformat()} | {str(e)}"
                st.session_state.live_errors.append(err_msg)
                st.session_state.live_errors_n += 1

        elif not new_df.empty and model is None:
            # No model — just log raw certs
            if "domains" in new_df.columns:
                new_df = new_df.explode("domains").rename(columns={"domains": "domain"})
            new_df = new_df.dropna(subset=["domain"])
            st.session_state.live_total += len(new_df)
            log_rows = [{"domain": r, "y_proba": None, "y_pred": None,
                          "scored_at": datetime.now(timezone.utc).isoformat()}
                        for r in new_df["domain"].tolist()]
            st.session_state.live_log = (st.session_state.live_log + log_rows)[-max_log:]

    # ── Render panels ─────────────────────────────────────────────────────────

    total    = st.session_state.live_total
    phishing = st.session_state.live_phishing
    rate     = phishing / total if total else 0
    uptime   = ""
    if "session_start" in st.session_state:
        delta   = datetime.now(timezone.utc) - st.session_state.session_start
        minutes = int(delta.total_seconds() // 60)
        seconds = int(delta.total_seconds() % 60)
        uptime  = f"{minutes}m {seconds}s"

    total_placeholder.metric("Scored",     f"{total:,}")
    phishing_placeholder.metric("Flagged", f"{phishing:,}", delta=None)
    rate_placeholder.metric("Flag Rate",   f"{rate:.3%}")
    errors_placeholder.metric("Errors",    f"{st.session_state.live_errors_n:,}")
    uptime_placeholder.metric("Uptime",    uptime or "—")

    # Feed table
    with feed_placeholder.container():
        if st.session_state.live_log:
            df_log = pd.DataFrame(st.session_state.live_log[::-1])
            if "y_proba" in df_log.columns:
                df_log["y_proba"] = df_log["y_proba"].apply(
                    lambda x: f"{x:.3f}" if x is not None else "—"
                )
                df_log["y_pred"] = df_log["y_pred"].apply(
                    lambda x: "🚨 PHISHING" if x == 1 else ("✅ OK" if x == 0 else "—")
                )
            st.dataframe(df_log, use_container_width=True, height=350)
        else:
            st.info("Waiting for incoming certs…")

    # Flagged domains table
    with flags_placeholder.container():
        if st.session_state.live_flagged:
            df_flags = pd.DataFrame(st.session_state.live_flagged[::-1])
            df_flags["y_proba"] = df_flags["y_proba"].apply(lambda x: f"{x:.3f}")
            st.dataframe(df_flags, use_container_width=True, height=350)

            # Export flagged
            buf = io.StringIO()
            pd.DataFrame(st.session_state.live_flagged).to_csv(buf, index=False)
            st.download_button(
                "⬇️ Export flagged domains",
                data=buf.getvalue(),
                file_name=f"flagged_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.success("No domains flagged yet")

    # Error log
    with errors_placeholder_exp.container():
        if st.session_state.live_errors:
            for err in st.session_state.live_errors[-20:]:
                st.code(err, language=None)
        else:
            st.success("No errors")

    # Scoring rate chart
    with rate_chart_placeholder.container():
        if len(st.session_state.live_log) > 1:
            df_log_ts = pd.DataFrame(st.session_state.live_log)
            df_log_ts["scored_at"] = pd.to_datetime(df_log_ts["scored_at"], utc=True)
            df_log_ts["minute"] = df_log_ts["scored_at"].dt.floor("min")
            per_min = df_log_ts.groupby("minute").size().reset_index(name="count")

            fig = go.Figure()
            fig.add_trace(go.Bar(x=per_min["minute"], y=per_min["count"],
                                  marker_color=LEGIT_COLOR, name="Certs/min"))
            fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0a0a",
                               plot_bgcolor="#0f0f0f",
                               font=dict(family="IBM Plex Mono", size=11),
                               xaxis_title="Time", yaxis_title="Certs scored/min",
                               height=250, margin=dict(t=20, b=40, l=50, r=20),
                               showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Auto-rerun while live
    if st.session_state.live_running:
        import time
        time.sleep(poll_secs)
        st.rerun()
    else:
        if total > 0:
            st.info(f"Session ended. Scored {total:,} domains, flagged {phishing:,} ({rate:.3%}).")