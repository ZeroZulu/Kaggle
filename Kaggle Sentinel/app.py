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

# ── Add src to path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
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

# Sentinel color palette
COLORS = {
    "bg": "#08090d",
    "card": "#12131a",
    "cyan": "#06b6d4",
    "red": "#f43f5e",
    "amber": "#f59e0b",
    "green": "#22c55e",
    "purple": "#a78bfa",
    "text": "#e2e4ea",
    "dim": "#6b7280",
    "grid": "#1e1f2e",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": COLORS["card"],
        "plot_bgcolor": COLORS["card"],
        "font": {"color": COLORS["text"], "family": "JetBrains Mono, Fira Code, monospace", "size": 12},
        "xaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
        "yaxis": {"gridcolor": COLORS["grid"], "zerolinecolor": COLORS["grid"]},
        "margin": {"t": 50, "b": 40, "l": 50, "r": 20},
        "colorway": [COLORS["cyan"], COLORS["red"], COLORS["amber"], COLORS["green"], COLORS["purple"]],
    }
}


def apply_theme(fig):
    """Apply Sentinel dark theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig


# ══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #08090d;
        --card: #12131a;
        --cyan: #06b6d4;
        --red: #f43f5e;
        --amber: #f59e0b;
        --green: #22c55e;
        --purple: #a78bfa;
        --text: #e2e4ea;
        --dim: #6b7280;
    }

    .stApp { background-color: var(--bg); }

    .metric-card {
        background: linear-gradient(135deg, #12131a 0%, #1a1b2e 100%);
        border: 1px solid #1e1f2e;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: border-color 0.3s ease, transform 0.2s ease;
    }
    .metric-card:hover {
        border-color: var(--cyan);
        transform: translateY(-2px);
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--dim);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .finding-card {
        background: var(--card);
        border-left: 3px solid var(--cyan);
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 8px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    .finding-card.critical { border-left-color: var(--red); }
    .finding-card.warning { border-left-color: var(--amber); }
    .finding-card.success { border-left-color: var(--green); }

    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--cyan);
        border-bottom: 2px solid #1e1f2e;
        padding-bottom: 8px;
        margin: 2rem 0 1rem 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0d14 0%, #12131a 100%);
        border-right: 1px solid #1e1f2e;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--card);
        border-radius: 8px;
        color: var(--dim);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: var(--cyan) !important;
        color: var(--bg) !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading dataset...")
def load_data(path):
    df = pd.read_csv(path)
    df_labeled = df.dropna(subset=["ISBOT"]).copy()
    df_eng = engineer_features(df_labeled)
    return df, df_labeled, df_eng


# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🤖 KAGGLE SENTINEL")
    st.markdown("##### Bot Detection Forensics")
    st.markdown("---")

    data_source = st.radio(
        "Data Source",
        ["Upload CSV", "Use sample path"],
        index=0,
    )

    data_path = None
    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload `kaggle_bot_accounts.csv`", type=["csv"])
        if uploaded:
            data_path = uploaded
    else:
        path_input = st.text_input(
            "CSV Path",
            value="data/kaggle_bot_accounts.csv",
        )
        if os.path.exists(path_input):
            data_path = path_input
        else:
            st.warning(f"File not found: `{path_input}`")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color: #6b7280; font-size: 0.75rem;'>"
        "Built with Streamlit • Sentinel v1.0<br>"
        "<a href='https://github.com/YOUR_USERNAME/kaggle-sentinel' style='color: #06b6d4;'>GitHub Repo</a>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Guard: no data yet ──
if data_path is None:
    st.markdown(
        """
        <div style="text-align: center; padding: 80px 20px;">
            <div style="font-size: 5rem; margin-bottom: 20px;">🤖</div>
            <h1 style="font-family: 'Space Grotesk', sans-serif; color: #06b6d4;">
                KAGGLE SENTINEL
            </h1>
            <p style="color: #6b7280; font-size: 1.1rem; max-width: 500px; margin: 0 auto;">
                Upload the <code>kaggle_bot_accounts.csv</code> dataset via the sidebar to begin the investigation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ── Load ──
df_raw, df_labeled, df_eng = load_data(data_path)
bot_rate = df_eng["ISBOT"].mean()
total_accounts = len(df_raw)
labeled_accounts = len(df_eng)
bot_count = int(df_eng["ISBOT"].sum())
human_count = labeled_accounts - bot_count


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style="text-align: center; padding: 10px 0 5px 0;">
        <h1 style="font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem;
                   background: linear-gradient(90deg, #06b6d4, #a78bfa);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   margin-bottom: 0;">
            🤖 KAGGLE SENTINEL
        </h1>
        <p style="color: #6b7280; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
            Bot Account Detection Through Behavioral Forensics
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════

cols = st.columns(5)

kpis = [
    (f"{total_accounts:,}", "TOTAL ACCOUNTS", COLORS["text"]),
    (f"{labeled_accounts:,}", "LABELED", COLORS["cyan"]),
    (f"{bot_count:,}", "BOTS DETECTED", COLORS["red"]),
    (f"{bot_rate:.1%}", "BOT RATE", COLORS["amber"]),
    (f"{human_count / bot_count:.2f}:1", "IMBALANCE RATIO", COLORS["purple"]),
]

for col, (val, label, color) in zip(cols, kpis):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {color};">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TABBED DASHBOARD
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🧬 Behavioral DNA", "📐 Benford's Law",
    "🤖 ML Performance", "🎯 Recommendations"
])


# ── TAB 1: Overview ─────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Dataset Overview & Key Signals</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # Authentication impact
        auth_stats = df_eng.groupby("IS_GLOGIN")["ISBOT"].mean() * 100
        fig_auth = go.Figure()
        labels = ["Email Signup", "Google OAuth"]
        values = [auth_stats.get(0, 0), auth_stats.get(1, 0)]
        fig_auth.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=[COLORS["red"], COLORS["cyan"]],
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(size=16, color=COLORS["text"]),
        ))
        fig_auth.update_layout(
            title="🔐 Authentication Method vs Bot Rate",
            yaxis_title="Bot Rate (%)",
            yaxis_range=[0, max(values) * 1.3],
            showlegend=False,
            height=400,
        )
        st.plotly_chart(apply_theme(fig_auth), use_container_width=True)

        st.markdown(
            '<div class="finding-card critical">'
            '<strong>CRITICAL FINDING:</strong> Google OAuth has a <strong>0.0%</strong> bot rate. '
            f'Email signup has <strong>{values[0]:.1f}%</strong>. '
            'Mandating SSO eliminates virtually all bot registrations.'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_right:
        # Target distribution
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Human", "Bot", "Unlabeled"],
            values=[human_count, bot_count, len(df_raw) - labeled_accounts],
            marker=dict(colors=[COLORS["cyan"], COLORS["red"], COLORS["dim"]]),
            hole=0.5,
            textinfo="label+percent",
            textfont=dict(size=13),
        )])
        fig_pie.update_layout(
            title="📊 Account Classification Breakdown",
            height=400,
            showlegend=False,
        )
        st.plotly_chart(apply_theme(fig_pie), use_container_width=True)

    # Distribution comparison
    st.markdown('<div class="section-header">Feature Distributions: Bot vs Human</div>', unsafe_allow_html=True)

    feature_choice = st.selectbox(
        "Select feature to compare",
        ["FOLLOWER_COUNT", "FOLLOWING_COUNT", "ACTIVITY_SCORE", "SOCIAL_REACH",
         "TOTAL_CONTENT", "VOTE_TOTAL", "AVG_NB_READ_TIME_MIN", "ENGAGEMENT_RATIO"],
        index=2,
    )

    fig_dist = go.Figure()
    for label, color, name in [(0, COLORS["cyan"], "Human"), (1, COLORS["red"], "Bot")]:
        data = df_eng[df_eng["ISBOT"] == label][feature_choice].clip(
            upper=df_eng[df_eng["ISBOT"] == label][feature_choice].quantile(0.99)
        )
        fig_dist.add_trace(go.Histogram(
            x=data, name=name, marker_color=color, opacity=0.6,
            nbinsx=60,
        ))
    fig_dist.update_layout(
        barmode="overlay", height=350,
        title=f"Distribution: {feature_choice}",
        xaxis_title=feature_choice, yaxis_title="Count",
    )
    st.plotly_chart(apply_theme(fig_dist), use_container_width=True)


# ── TAB 2: Behavioral DNA ──────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Behavioral DNA Feature Engineering</div>', unsafe_allow_html=True)

    eng_features = [
        "ENGAGEMENT_RATIO", "FOLLOW_RECIPROCITY", "SOCIAL_REACH",
        "TOTAL_CONTENT", "CONTENT_PER_DISCUSSION", "ACTIVITY_SCORE",
        "IS_DORMANT", "READ_PER_DISCUSSION", "READ_ENGAGEMENT",
        "HAS_READ_TIME", "VOTE_TOTAL", "VOTE_ENTROPY",
        "VOTE_NB_RATIO", "VOTE_CONCENTRATION", "PHANTOM_SCORE", "AUTHENTICITY_INDEX",
    ]

    # Feature comparison table
    rows = []
    for f in eng_features:
        bm = df_eng[df_eng["ISBOT"] == 1][f].mean()
        hm = df_eng[df_eng["ISBOT"] == 0][f].mean()
        ratio = hm / (bm + 1e-10) if hm > bm else bm / (hm + 1e-10)
        rows.append({"Feature": f, "Bot Mean": f"{bm:.3f}", "Human Mean": f"{hm:.3f}", "Ratio": f"{ratio:.1f}x"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)

    # Radar chart
    st.markdown('<div class="section-header">Behavioral Fingerprint Radar</div>', unsafe_allow_html=True)

    radar_features = ["ACTIVITY_SCORE", "SOCIAL_REACH", "TOTAL_CONTENT",
                       "READ_ENGAGEMENT", "FOLLOW_RECIPROCITY", "VOTE_ENTROPY"]

    bot_means = [df_eng[df_eng["ISBOT"] == 1][f].mean() for f in radar_features]
    human_means = [df_eng[df_eng["ISBOT"] == 0][f].mean() for f in radar_features]

    # Normalize to 0-1 for radar
    max_vals = [max(b, h, 1e-10) for b, h in zip(bot_means, human_means)]
    bot_norm = [b / m for b, m in zip(bot_means, max_vals)]
    human_norm = [h / m for h, m in zip(human_means, max_vals)]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=human_norm + [human_norm[0]], theta=radar_features + [radar_features[0]],
        fill="toself", name="Human", fillcolor="rgba(6,182,212,0.15)",
        line=dict(color=COLORS["cyan"], width=2),
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=bot_norm + [bot_norm[0]], theta=radar_features + [radar_features[0]],
        fill="toself", name="Bot", fillcolor="rgba(244,63,94,0.15)",
        line=dict(color=COLORS["red"], width=2),
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor=COLORS["card"],
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["grid"]),
            angularaxis=dict(gridcolor=COLORS["grid"]),
        ),
        height=500,
        title="Behavioral DNA: Bot vs Human Profile",
    )
    st.plotly_chart(apply_theme(fig_radar), use_container_width=True)

    st.markdown(
        '<div class="finding-card warning">'
        '<strong>KEY INSIGHT:</strong> Bots exhibit a dramatically collapsed behavioral profile — '
        'near-zero across all engagement, content production, and reading dimensions. '
        'The only metric where bots approach humans is voting, suggesting automated vote manipulation.'
        '</div>',
        unsafe_allow_html=True,
    )


# ── TAB 3: Benford's Law ───────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Benford\'s Law Forensic Analysis</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="finding-card">
        <strong>METHODOLOGY:</strong> Benford's Law states that in naturally occurring datasets,
        the leading digit <em>d</em> appears with probability log₁₀(1 + 1/d).
        Artificial data (e.g., bot-generated metrics) violates this law.
        We measure deviation using <strong>Jensen-Shannon Divergence</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    benford_col = st.selectbox(
        "Select feature for Benford analysis",
        ["FOLLOWER_COUNT", "FOLLOWING_COUNT", "DISCUSSION_COUNT", "CODE_COUNT"],
    )

    col_b1, col_b2 = st.columns(2)

    bot_result = benford_analysis(df_eng[df_eng["ISBOT"] == 1][benford_col], "Bot")
    human_result = benford_analysis(df_eng[df_eng["ISBOT"] == 0][benford_col], "Human")
    expected = benfords_expected()

    digits = list(range(1, 10))

    for col_b, result, title, color in [
        (col_b1, human_result, "👤 Human", COLORS["cyan"]),
        (col_b2, bot_result, "🤖 Bot", COLORS["red"]),
    ]:
        with col_b:
            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                x=digits, y=[expected[d] * 100 for d in digits],
                name="Benford Expected", marker_color=COLORS["amber"], opacity=0.4,
            ))
            fig_b.add_trace(go.Bar(
                x=digits, y=[result["dist"].get(d, 0) * 100 for d in digits],
                name=title, marker_color=color, opacity=0.7,
            ))
            fig_b.update_layout(
                title=f"{title} — JS Divergence: {result['js_divergence']:.4f} ({result['verdict']})",
                xaxis_title="First Digit", yaxis_title="Frequency (%)",
                barmode="group", height=400,
            )
            st.plotly_chart(apply_theme(fig_b), use_container_width=True)

    # Comparison table
    st.markdown("#### Analysis Summary")
    benford_features = ["FOLLOWER_COUNT", "FOLLOWING_COUNT", "DISCUSSION_COUNT", "CODE_COUNT"]
    benford_rows = []
    for feat in benford_features:
        for label, name in [(1, "Bot"), (0, "Human")]:
            r = benford_analysis(df_eng[df_eng["ISBOT"] == label][feat], name)
            benford_rows.append({
                "Feature": feat, "Class": name,
                "χ² Statistic": f"{r['chi2']:,.1f}",
                "JS Divergence": f"{r['js_divergence']:.4f}",
                "Verdict": f"{'🔴' if r['verdict'] == 'EXTREME' else '🟡' if r['verdict'] == 'MODERATE' else '🟢'} {r['verdict']}",
            })
    st.dataframe(pd.DataFrame(benford_rows), use_container_width=True, hide_index=True)


# ── TAB 4: ML Performance ──────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Machine Learning Pipeline Results</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="finding-card success">'
        '<strong>KEY RESULT:</strong> All models achieve <strong>perfect or near-perfect classification</strong> '
        '(AUC ≈ 1.0000). The bot-vs-human boundary is trivially separable in this dataset. '
        'The real value lies in the forensic feature analysis, not model complexity.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Model results (from notebook outputs)
    model_results = {
        "Logistic Regression": {"Accuracy": 0.9999, "F1": 0.9998, "ROC-AUC": 1.0000, "Precision": 0.9996, "Recall": 1.0000},
        "Random Forest": {"Accuracy": 1.0000, "F1": 1.0000, "ROC-AUC": 1.0000, "Precision": 1.0000, "Recall": 1.0000},
        "Gradient Boosting": {"Accuracy": 1.0000, "F1": 1.0000, "ROC-AUC": 1.0000, "Precision": 1.0000, "Recall": 1.0000},
        "XGBoost": {"Accuracy": 1.0000, "F1": 1.0000, "ROC-AUC": 1.0000, "Precision": 1.0000, "Recall": 1.0000},
        "LightGBM": {"Accuracy": 1.0000, "F1": 1.0000, "ROC-AUC": 1.0000, "Precision": 1.0000, "Recall": 1.0000},
        "Stacking Ensemble": {"Accuracy": 1.0000, "F1": 1.0000, "ROC-AUC": 1.0000, "Precision": 1.0000, "Recall": 1.0000},
    }

    # Performance table
    df_results = pd.DataFrame(model_results).T
    df_results.index.name = "Model"
    st.dataframe(df_results.style.format("{:.4f}").background_gradient(cmap="YlGn", axis=None), use_container_width=True)

    # Bar chart
    fig_models = go.Figure()
    models = list(model_results.keys())
    metrics = ["Accuracy", "F1", "Precision", "Recall"]
    colors = [COLORS["cyan"], COLORS["amber"], COLORS["green"], COLORS["purple"]]

    for metric, color in zip(metrics, colors):
        fig_models.add_trace(go.Bar(
            name=metric,
            x=models,
            y=[model_results[m][metric] for m in models],
            marker_color=color,
            opacity=0.8,
        ))
    fig_models.update_layout(
        barmode="group", height=450,
        title="Model Comparison (5-Fold Stratified CV)",
        yaxis_range=[0.999, 1.0005],
        yaxis_title="Score",
    )
    st.plotly_chart(apply_theme(fig_models), use_container_width=True)

    # SHAP-like feature importance (computed from data separation)
    st.markdown('<div class="section-header">Feature Discriminative Power</div>', unsafe_allow_html=True)

    all_features = [
        "IS_GLOGIN", "ACTIVITY_SCORE", "SOCIAL_REACH", "TOTAL_CONTENT",
        "READ_ENGAGEMENT", "AVG_NB_READ_TIME_MIN", "FOLLOW_RECIPROCITY",
        "VOTE_ENTROPY", "CONTENT_PER_DISCUSSION", "PHANTOM_SCORE",
        "FOLLOWER_COUNT", "FOLLOWING_COUNT", "VOTE_TOTAL",
        "ENGAGEMENT_RATIO", "AUTHENTICITY_INDEX",
    ]

    # Use effect size (Cohen's d) as proxy for discriminative power
    importances = []
    for f in all_features:
        bot_vals = df_eng[df_eng["ISBOT"] == 1][f].dropna()
        human_vals = df_eng[df_eng["ISBOT"] == 0][f].dropna()
        pooled_std = np.sqrt((bot_vals.std() ** 2 + human_vals.std() ** 2) / 2) + 1e-10
        d = abs(bot_vals.mean() - human_vals.mean()) / pooled_std
        importances.append(d)

    imp_df = pd.DataFrame({"Feature": all_features, "Effect Size (Cohen's d)": importances})
    imp_df = imp_df.sort_values("Effect Size (Cohen's d)", ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=imp_df["Effect Size (Cohen's d)"],
        y=imp_df["Feature"],
        orientation="h",
        marker_color=COLORS["purple"],
    ))
    fig_imp.update_layout(
        title="Feature Discriminative Power (Cohen's d Effect Size)",
        xaxis_title="Effect Size",
        height=500,
    )
    st.plotly_chart(apply_theme(fig_imp), use_container_width=True)


# ── TAB 5: Recommendations ─────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Production Recommendations</div>', unsafe_allow_html=True)

    auth_email_rate = df_eng[df_eng["IS_GLOGIN"] == 0]["ISBOT"].mean() * 100
    auth_oauth_rate = df_eng[df_eng["IS_GLOGIN"] == 1]["ISBOT"].mean() * 100

    recommendations = [
        {
            "priority": "P0",
            "title": "Enforce Google / SSO Authentication",
            "detail": f"Email bot rate: {auth_email_rate:.1f}% → OAuth: {auth_oauth_rate:.1f}%. "
                      "Mandating SSO at registration eliminates virtually all bots.",
            "impact": "~100% bot prevention at signup",
            "class": "critical",
            "color": COLORS["red"],
        },
        {
            "priority": "P1",
            "title": "Activity Score Threshold Gate",
            "detail": "Zero bots observed above ACTIVITY_SCORE ≈ 15. "
                      "Hard threshold on low-activity + non-OAuth accounts for immediate flagging.",
            "impact": "Instant bot flagging with zero false positives",
            "class": "critical",
            "color": COLORS["red"],
        },
        {
            "priority": "P2",
            "title": "Deploy ML Ensemble with SHAP",
            "detail": "XGBoost / LightGBM / Stacking all achieve perfect separation. "
                      "SHAP provides per-account explainability for human review.",
            "impact": "Automated detection + transparent decisions",
            "class": "warning",
            "color": COLORS["amber"],
        },
        {
            "priority": "P3",
            "title": "Real-time Benford's Law Monitoring",
            "detail": "Bot follower counts show extreme Benford violation (JS divergence > 0.48). "
                      "Deploy as forensic anomaly detection layer.",
            "impact": "Detects synthetic metric generation",
            "class": "warning",
            "color": COLORS["amber"],
        },
        {
            "priority": "P4",
            "title": "Geographic Risk Scoring",
            "detail": f"{len(df_eng['REGISTRATION_LOCATION'].dropna().unique())} regions analyzed. "
                      "Small island territories show elevated bot rates (VPN / proxy pattern). "
                      "Never use as sole signal — ensemble weight only.",
            "impact": "Risk-weighted registration scoring",
            "class": "",
            "color": COLORS["cyan"],
        },
    ]

    for rec in recommendations:
        st.markdown(
            f"""
            <div class="finding-card {rec['class']}" style="border-left-color: {rec['color']};">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <span style="background: {rec['color']}; color: #08090d; padding: 2px 10px;
                                 border-radius: 4px; font-weight: 700; font-family: 'JetBrains Mono', monospace;
                                 font-size: 0.85rem;">{rec['priority']}</span>
                    <strong style="font-size: 1.05rem;">{rec['title']}</strong>
                </div>
                <p style="color: #9ca3af; margin: 4px 0;">{rec['detail']}</p>
                <p style="color: {rec['color']}; font-size: 0.85rem; margin: 4px 0;">
                    ⚡ Impact: {rec['impact']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Summary metrics
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Investigation Summary</div>', unsafe_allow_html=True)

    summary_cols = st.columns(3)
    summaries = [
        ("27", "Behavioral Features Engineered", COLORS["cyan"]),
        ("5", "ML Models Benchmarked", COLORS["amber"]),
        ("4", "Forensic Methods Applied", COLORS["purple"]),
    ]
    for col, (val, label, color) in zip(summary_cols, summaries):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color: {color};">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style="text-align: center; padding: 40px 0 20px 0; border-top: 1px solid #1e1f2e; margin-top: 40px;">
        <p style="color: #6b7280; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">
            KAGGLE SENTINEL v1.0 — Bot Detection Through Behavioral Forensics<br>
            <a href="https://kaggle.com" style="color: #06b6d4;">Kaggle Dataset</a> •
            <a href="https://github.com/YOUR_USERNAME/kaggle-sentinel" style="color: #06b6d4;">GitHub</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
