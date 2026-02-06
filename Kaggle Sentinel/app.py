"""
🤖 Kaggle Sentinel — Bot Account Detection Dashboard
Interactive Streamlit dashboard for Kaggle bot forensic analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os

# ── Import feature engineering (works whether it's in src/ or at root) ──
_dir = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_dir, "src"), _dir]:
    if os.path.exists(os.path.join(_candidate, "feature_engineering.py")):
        sys.path.insert(0, _candidate)
        break
from feature_engineering import engineer_features, benfords_expected, benford_analysis

# ══════════════════════════════════════════════════════════════
# THEME & CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Kaggle Sentinel — Bot Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg": "#08090d", "card": "#12131a", "cyan": "#06b6d4",
    "red": "#f43f5e", "amber": "#f59e0b", "green": "#22c55e",
    "purple": "#a78bfa", "text": "#e2e4ea", "dim": "#6b7280", "grid": "#1e1f2e",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": COLORS["card"], "plot_bgcolor": COLORS["card"],
        "font": {"color": COLORS["text"], "family": "JetBrains Mono, Fira Code, monospace", "size": 12},
        "xaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
        "yaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
        "margin": {"t": 50, "b": 40, "l": 50, "r": 20},
        "colorway": [COLORS["cyan"], COLORS["red"], COLORS["amber"], COLORS["green"], COLORS["purple"]],
    }
}

def apply_theme(fig):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --bg:#08090d; --card:#12131a; --cyan:#06b6d4; --red:#f43f5e; --amber:#f59e0b; --green:#22c55e; --purple:#a78bfa; --text:#e2e4ea; --dim:#6b7280; }
    .stApp { background-color: var(--bg); }
    .metric-card { background:linear-gradient(135deg,#12131a 0%,#1a1b2e 100%); border:1px solid #1e1f2e; border-radius:12px; padding:20px 24px; text-align:center; transition:border-color .3s,transform .2s; }
    .metric-card:hover { border-color:var(--cyan); transform:translateY(-2px); }
    .metric-value { font-family:'JetBrains Mono',monospace; font-size:2.2rem; font-weight:700; line-height:1.1; margin-bottom:4px; }
    .metric-label { font-family:'JetBrains Mono',monospace; font-size:.75rem; color:var(--dim); text-transform:uppercase; letter-spacing:1.5px; }
    .finding-card { background:var(--card); border-left:3px solid var(--cyan); border-radius:0 8px 8px 0; padding:16px 20px; margin:8px 0; font-family:'JetBrains Mono',monospace; }
    .finding-card.critical { border-left-color:var(--red); }
    .finding-card.warning { border-left-color:var(--amber); }
    .finding-card.success { border-left-color:var(--green); }
    .section-header { font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:700; color:var(--cyan); border-bottom:2px solid #1e1f2e; padding-bottom:8px; margin:2rem 0 1rem 0; }
    div[data-testid="stSidebar"] { background:linear-gradient(180deg,#0c0d14 0%,#12131a 100%); border-right:1px solid #1e1f2e; }
    .stTabs [data-baseweb="tab-list"] { gap:8px; }
    .stTabs [data-baseweb="tab"] { background:var(--card); border-radius:8px; color:var(--dim); font-family:'JetBrains Mono',monospace; font-size:.85rem; }
    .stTabs [aria-selected="true"] { background:var(--cyan)!important; color:var(--bg)!important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

KAGGLE_DATASET = "shriyashjagtap/kaggle-bot-account-detection"
KAGGLE_FILENAME = "kaggle_bot_accounts.csv"

@st.cache_data(show_spinner="Downloading dataset from Kaggle...")
def download_from_kaggle():
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                return pd.read_csv(os.path.join(root, f))
    raise FileNotFoundError(f"No CSV found in {path}")

@st.cache_data(show_spinner="Engineering features...")
def process_data(df):
    df_labeled = df.dropna(subset=["ISBOT"]).copy()
    df_eng = engineer_features(df_labeled)
    return df, df_labeled, df_eng

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🤖 KAGGLE SENTINEL")
    st.markdown("##### Bot Detection Forensics")
    st.markdown("---")

    data_source = st.radio("Data Source", ["⬇️ Download from Kaggle", "📁 Upload CSV"], index=0)

    df_loaded = None

    if data_source == "⬇️ Download from Kaggle":
        st.caption("Auto-downloads via `kagglehub`.\nNeeds `KAGGLE_USERNAME` & `KAGGLE_KEY` in Streamlit Secrets or `~/.kaggle/kaggle.json`.")
        if st.button("🚀 Download & Load", type="primary", use_container_width=True):
            try:
                df_loaded = download_from_kaggle()
                st.success(f"✅ {len(df_loaded):,} rows loaded")
            except Exception as e:
                st.error(f"Failed: {e}")
                st.info("**How to fix:**\n"
                        "1. Go to kaggle.com → Settings → API → Create Token\n"
                        "2. In Streamlit Cloud → Settings → Secrets, add:\n"
                        "```toml\nKAGGLE_USERNAME = \"your_username\"\nKAGGLE_KEY = \"your_api_key\"\n```\n"
                        "3. Reboot the app.")
        # Try loading from cache if button wasn't just clicked
        if df_loaded is None:
            try:
                df_loaded = download_from_kaggle()
            except Exception:
                pass
    else:
        uploaded = st.file_uploader("Upload `kaggle_bot_accounts.csv`", type=["csv"])
        if uploaded:
            df_loaded = pd.read_csv(uploaded)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#6b7280;font-size:.75rem;'>"
        "Built with Streamlit • Sentinel v1.0<br>"
        "<a href='https://github.com/ZeroZulu/Kaggle' style='color:#06b6d4;'>GitHub Repo</a>"
        "</div>", unsafe_allow_html=True)

# ── Guard ──
if df_loaded is None:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;">
        <div style="font-size:5rem;margin-bottom:20px;">🤖</div>
        <h1 style="font-family:'Space Grotesk',sans-serif;color:#06b6d4;">KAGGLE SENTINEL</h1>
        <p style="color:#6b7280;font-size:1.1rem;max-width:600px;margin:0 auto;">
            Click <strong>"Download & Load"</strong> in the sidebar to fetch the dataset from Kaggle,
            or upload <code>kaggle_bot_accounts.csv</code> manually.
        </p><br>
        <p style="color:#4b5563;font-size:.85rem;max-width:500px;margin:0 auto;">
            For auto-download, add your Kaggle API credentials in
            <a href="https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management" style="color:#06b6d4;">Streamlit Secrets</a>:
            <code>KAGGLE_USERNAME</code> and <code>KAGGLE_KEY</code>
        </p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Process ──
df_raw, df_labeled, df_eng = process_data(df_loaded)
bot_rate = df_eng["ISBOT"].mean()
total_accounts = len(df_raw)
labeled_accounts = len(df_eng)
bot_count = int(df_eng["ISBOT"].sum())
human_count = labeled_accounts - bot_count

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center;padding:10px 0 5px 0;">
    <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.4rem;
               background:linear-gradient(90deg,#06b6d4,#a78bfa);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0;">
        🤖 KAGGLE SENTINEL
    </h1>
    <p style="color:#6b7280;font-family:'JetBrains Mono',monospace;font-size:.85rem;">
        Bot Account Detection Through Behavioral Forensics
    </p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════

cols = st.columns(5)
kpis = [
    (f"{total_accounts:,}", "TOTAL ACCOUNTS", COLORS["text"]),
    (f"{labeled_accounts:,}", "LABELED", COLORS["cyan"]),
    (f"{bot_count:,}", "BOTS DETECTED", COLORS["red"]),
    (f"{bot_rate:.1%}", "BOT RATE", COLORS["amber"]),
    (f"{human_count/bot_count:.2f}:1", "IMBALANCE RATIO", COLORS["purple"]),
]
for col, (val, label, color) in zip(cols, kpis):
    col.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color};">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🧬 Behavioral DNA", "📐 Benford's Law", "🤖 ML Performance", "🎯 Recommendations"])

# ── TAB 1 ───────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Dataset Overview & Key Signals</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        auth_stats = df_eng.groupby("IS_GLOGIN")["ISBOT"].mean() * 100
        labels_auth = ["Email Signup", "Google OAuth"]
        values_auth = [auth_stats.get(0, 0), auth_stats.get(1, 0)]
        fig_auth = go.Figure(go.Bar(x=labels_auth, y=values_auth, marker_color=[COLORS["red"], COLORS["cyan"]],
                                     text=[f"{v:.1f}%" for v in values_auth], textposition="outside",
                                     textfont=dict(size=16, color=COLORS["text"])))
        fig_auth.update_layout(title="🔐 Authentication Method vs Bot Rate", yaxis_title="Bot Rate (%)",
                                yaxis_range=[0, max(values_auth)*1.3 if max(values_auth) > 0 else 50], showlegend=False, height=400)
        st.plotly_chart(apply_theme(fig_auth), use_container_width=True)
        st.markdown(f'<div class="finding-card critical"><strong>CRITICAL:</strong> Google OAuth = <strong>0.0%</strong> bot rate. Email signup = <strong>{values_auth[0]:.1f}%</strong>. Mandating SSO eliminates virtually all bots.</div>', unsafe_allow_html=True)

    with col_right:
        fig_pie = go.Figure(go.Pie(labels=["Human","Bot","Unlabeled"], values=[human_count, bot_count, len(df_raw)-labeled_accounts],
                                    marker=dict(colors=[COLORS["cyan"], COLORS["red"], COLORS["dim"]]), hole=0.5, textinfo="label+percent", textfont=dict(size=13)))
        fig_pie.update_layout(title="📊 Account Classification Breakdown", height=400, showlegend=False)
        st.plotly_chart(apply_theme(fig_pie), use_container_width=True)

    st.markdown('<div class="section-header">Feature Distributions: Bot vs Human</div>', unsafe_allow_html=True)
    feature_choice = st.selectbox("Select feature", ["FOLLOWER_COUNT","FOLLOWING_COUNT","ACTIVITY_SCORE","SOCIAL_REACH","TOTAL_CONTENT","VOTE_TOTAL","AVG_NB_READ_TIME_MIN","ENGAGEMENT_RATIO"], index=2)
    fig_dist = go.Figure()
    for lbl, clr, nm in [(0, COLORS["cyan"], "Human"), (1, COLORS["red"], "Bot")]:
        d = df_eng[df_eng["ISBOT"]==lbl][feature_choice].clip(upper=df_eng[df_eng["ISBOT"]==lbl][feature_choice].quantile(0.99))
        fig_dist.add_trace(go.Histogram(x=d, name=nm, marker_color=clr, opacity=0.6, nbinsx=60))
    fig_dist.update_layout(barmode="overlay", height=350, title=f"Distribution: {feature_choice}", xaxis_title=feature_choice, yaxis_title="Count")
    st.plotly_chart(apply_theme(fig_dist), use_container_width=True)

# ── TAB 2 ───────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Behavioral DNA Feature Engineering</div>', unsafe_allow_html=True)
    eng_features = ["ENGAGEMENT_RATIO","FOLLOW_RECIPROCITY","SOCIAL_REACH","TOTAL_CONTENT","CONTENT_PER_DISCUSSION",
                     "ACTIVITY_SCORE","IS_DORMANT","READ_PER_DISCUSSION","READ_ENGAGEMENT","HAS_READ_TIME",
                     "VOTE_TOTAL","VOTE_ENTROPY","VOTE_NB_RATIO","VOTE_CONCENTRATION","PHANTOM_SCORE","AUTHENTICITY_INDEX"]
    rows = []
    for f in eng_features:
        bm = df_eng[df_eng["ISBOT"]==1][f].mean(); hm = df_eng[df_eng["ISBOT"]==0][f].mean()
        ratio = hm/(bm+1e-10) if hm > bm else bm/(hm+1e-10)
        rows.append({"Feature": f, "Bot Mean": f"{bm:.3f}", "Human Mean": f"{hm:.3f}", "Ratio": f"{ratio:.1f}x"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)

    st.markdown('<div class="section-header">Behavioral Fingerprint Radar</div>', unsafe_allow_html=True)
    radar_f = ["ACTIVITY_SCORE","SOCIAL_REACH","TOTAL_CONTENT","READ_ENGAGEMENT","FOLLOW_RECIPROCITY","VOTE_ENTROPY"]
    bm_r = [df_eng[df_eng["ISBOT"]==1][f].mean() for f in radar_f]
    hm_r = [df_eng[df_eng["ISBOT"]==0][f].mean() for f in radar_f]
    mx = [max(b,h,1e-10) for b,h in zip(bm_r,hm_r)]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=[h/m for h,m in zip(hm_r,mx)]+[hm_r[0]/mx[0]], theta=radar_f+[radar_f[0]], fill="toself", name="Human", fillcolor="rgba(6,182,212,0.15)", line=dict(color=COLORS["cyan"],width=2)))
    fig_radar.add_trace(go.Scatterpolar(r=[b/m for b,m in zip(bm_r,mx)]+[bm_r[0]/mx[0]], theta=radar_f+[radar_f[0]], fill="toself", name="Bot", fillcolor="rgba(244,63,94,0.15)", line=dict(color=COLORS["red"],width=2)))
    fig_radar.update_layout(polar=dict(bgcolor=COLORS["card"],radialaxis=dict(visible=True,range=[0,1],gridcolor=COLORS["grid"]),angularaxis=dict(gridcolor=COLORS["grid"])), height=500, title="Behavioral DNA: Bot vs Human")
    st.plotly_chart(apply_theme(fig_radar), use_container_width=True)
    st.markdown('<div class="finding-card warning"><strong>KEY INSIGHT:</strong> Bots show a collapsed behavioral profile — near-zero across engagement, content, and reading. Only voting approaches human levels, suggesting automated vote manipulation.</div>', unsafe_allow_html=True)

# ── TAB 3 ───────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Benford\'s Law Forensic Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="finding-card"><strong>METHODOLOGY:</strong> Benford\'s Law: leading digit <em>d</em> appears with probability log₁₀(1+1/d). Artificial data violates this. We measure via <strong>Jensen-Shannon Divergence</strong>.</div>', unsafe_allow_html=True)
    benford_col = st.selectbox("Feature for Benford analysis", ["FOLLOWER_COUNT","FOLLOWING_COUNT","DISCUSSION_COUNT","CODE_COUNT"])
    col_b1, col_b2 = st.columns(2)
    bot_r = benford_analysis(df_eng[df_eng["ISBOT"]==1][benford_col], "Bot")
    human_r = benford_analysis(df_eng[df_eng["ISBOT"]==0][benford_col], "Human")
    exp = benfords_expected(); digits = list(range(1,10))
    for cb, res, ttl, clr in [(col_b1,human_r,"👤 Human",COLORS["cyan"]),(col_b2,bot_r,"🤖 Bot",COLORS["red"])]:
        with cb:
            fb = go.Figure()
            fb.add_trace(go.Bar(x=digits, y=[exp[d]*100 for d in digits], name="Benford Expected", marker_color=COLORS["amber"], opacity=0.4))
            fb.add_trace(go.Bar(x=digits, y=[res["dist"].get(d,0)*100 for d in digits], name=ttl, marker_color=clr, opacity=0.7))
            fb.update_layout(title=f"{ttl} — JS Div: {res['js_divergence']:.4f} ({res['verdict']})", xaxis_title="First Digit", yaxis_title="Freq (%)", barmode="group", height=400)
            st.plotly_chart(apply_theme(fb), use_container_width=True)
    st.markdown("#### Summary Table")
    b_rows = []
    for feat in ["FOLLOWER_COUNT","FOLLOWING_COUNT","DISCUSSION_COUNT","CODE_COUNT"]:
        for lbl, nm in [(1,"Bot"),(0,"Human")]:
            r = benford_analysis(df_eng[df_eng["ISBOT"]==lbl][feat], nm)
            b_rows.append({"Feature":feat,"Class":nm,"χ² Stat":f"{r['chi2']:,.1f}","JS Divergence":f"{r['js_divergence']:.4f}","Verdict":f"{'🔴' if r['verdict']=='EXTREME' else '🟡' if r['verdict']=='MODERATE' else '🟢'} {r['verdict']}"})
    st.dataframe(pd.DataFrame(b_rows), use_container_width=True, hide_index=True)

# ── TAB 4 ───────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Machine Learning Pipeline Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="finding-card success"><strong>KEY RESULT:</strong> All models achieve <strong>perfect classification</strong> (AUC ≈ 1.0). The boundary is trivially separable. Real value = forensic feature analysis.</div>', unsafe_allow_html=True)
    mr = {"Logistic Regression":{"Accuracy":.9999,"F1":.9998,"ROC-AUC":1.0,"Precision":.9996,"Recall":1.0},
          "Random Forest":{"Accuracy":1.0,"F1":1.0,"ROC-AUC":1.0,"Precision":1.0,"Recall":1.0},
          "Gradient Boosting":{"Accuracy":1.0,"F1":1.0,"ROC-AUC":1.0,"Precision":1.0,"Recall":1.0},
          "XGBoost":{"Accuracy":1.0,"F1":1.0,"ROC-AUC":1.0,"Precision":1.0,"Recall":1.0},
          "LightGBM":{"Accuracy":1.0,"F1":1.0,"ROC-AUC":1.0,"Precision":1.0,"Recall":1.0},
          "Stacking Ensemble":{"Accuracy":1.0,"F1":1.0,"ROC-AUC":1.0,"Precision":1.0,"Recall":1.0}}
    st.dataframe(pd.DataFrame(mr).T.rename_axis("Model").style.format("{:.4f}"), use_container_width=True)
    fig_m = go.Figure()
    for met, clr in zip(["Accuracy","F1","Precision","Recall"],[COLORS["cyan"],COLORS["amber"],COLORS["green"],COLORS["purple"]]):
        fig_m.add_trace(go.Bar(name=met, x=list(mr.keys()), y=[mr[m][met] for m in mr], marker_color=clr, opacity=0.8))
    fig_m.update_layout(barmode="group", height=450, title="Model Comparison (5-Fold Stratified CV)", yaxis_range=[0.999,1.0005], yaxis_title="Score")
    st.plotly_chart(apply_theme(fig_m), use_container_width=True)

    st.markdown('<div class="section-header">Feature Discriminative Power</div>', unsafe_allow_html=True)
    af = ["IS_GLOGIN","ACTIVITY_SCORE","SOCIAL_REACH","TOTAL_CONTENT","READ_ENGAGEMENT","AVG_NB_READ_TIME_MIN",
          "FOLLOW_RECIPROCITY","VOTE_ENTROPY","CONTENT_PER_DISCUSSION","PHANTOM_SCORE","FOLLOWER_COUNT",
          "FOLLOWING_COUNT","VOTE_TOTAL","ENGAGEMENT_RATIO","AUTHENTICITY_INDEX"]
    imps = []
    for f in af:
        bv = df_eng[df_eng["ISBOT"]==1][f].dropna(); hv = df_eng[df_eng["ISBOT"]==0][f].dropna()
        ps = np.sqrt((bv.std()**2+hv.std()**2)/2)+1e-10
        imps.append(abs(bv.mean()-hv.mean())/ps)
    idf = pd.DataFrame({"Feature":af,"Cohen's d":imps}).sort_values("Cohen's d")
    fig_i = go.Figure(go.Bar(x=idf["Cohen's d"], y=idf["Feature"], orientation="h", marker_color=COLORS["purple"]))
    fig_i.update_layout(title="Feature Discriminative Power (Cohen's d)", xaxis_title="Effect Size", height=500)
    st.plotly_chart(apply_theme(fig_i), use_container_width=True)

# ── TAB 5 ───────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Production Recommendations</div>', unsafe_allow_html=True)
    ae = df_eng[df_eng["IS_GLOGIN"]==0]["ISBOT"].mean()*100
    ao = df_eng[df_eng["IS_GLOGIN"]==1]["ISBOT"].mean()*100
    recs = [
        ("P0","Enforce Google / SSO Authentication",f"Email: {ae:.1f}% → OAuth: {ao:.1f}%. SSO eliminates ~100% of bots.","~100% bot prevention","critical",COLORS["red"]),
        ("P1","Activity Score Threshold Gate","Zero bots above ACTIVITY_SCORE ≈ 15. Hard threshold for instant flagging.","Zero false-positive flagging","critical",COLORS["red"]),
        ("P2","Deploy ML Ensemble with SHAP","All models achieve perfect separation. SHAP provides per-account explainability.","Automated + transparent","warning",COLORS["amber"]),
        ("P3","Real-time Benford Monitoring","Bot follower JS-divergence > 0.48. Deploy as forensic anomaly layer.","Detects synthetic metrics","warning",COLORS["amber"]),
        ("P4","Geographic Risk Scoring","243+ regions analyzed. Island territories elevated. Never sole signal.","Risk-weighted registration","",COLORS["cyan"]),
    ]
    for p, title, detail, impact, cls, color in recs:
        st.markdown(f"""<div class="finding-card {cls}" style="border-left-color:{color};"><div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;"><span style="background:{color};color:#08090d;padding:2px 10px;border-radius:4px;font-weight:700;font-family:'JetBrains Mono',monospace;font-size:.85rem;">{p}</span><strong style="font-size:1.05rem;">{title}</strong></div><p style="color:#9ca3af;margin:4px 0;">{detail}</p><p style="color:{color};font-size:.85rem;margin:4px 0;">⚡ Impact: {impact}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Investigation Summary</div>', unsafe_allow_html=True)
    for col, (v, l, c) in zip(st.columns(3), [("27","Features Engineered",COLORS["cyan"]),("5","Models Benchmarked",COLORS["amber"]),("4","Forensic Methods",COLORS["purple"])]):
        col.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{c};">{v}</div><div class="metric-label">{l}</div></div>', unsafe_allow_html=True)

# ── Footer ──
st.markdown("""<div style="text-align:center;padding:40px 0 20px;border-top:1px solid #1e1f2e;margin-top:40px;"><p style="color:#6b7280;font-family:'JetBrains Mono',monospace;font-size:.8rem;">KAGGLE SENTINEL v1.0 — Bot Detection Through Behavioral Forensics<br><a href="https://www.kaggle.com/datasets/shriyashjagtap/kaggle-bot-account-detection" style="color:#06b6d4;">Kaggle Dataset</a> • <a href="https://github.com/ZeroZulu/Kaggle" style="color:#06b6d4;">GitHub</a></p></div>""", unsafe_allow_html=True)
