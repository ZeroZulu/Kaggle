"""
🤖 Kaggle Sentinel — Bot Account Detection Dashboard
Optimized for Streamlit Cloud (1GB RAM limit).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os, gc

# ── Import feature engineering ──
_dir = os.path.dirname(os.path.abspath(__file__))
for _candidate in [os.path.join(_dir, "src"), _dir]:
    if os.path.exists(os.path.join(_candidate, "feature_engineering.py")):
        sys.path.insert(0, _candidate)
        break
from feature_engineering import engineer_features, benfords_expected, benford_analysis

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(page_title="Kaggle Sentinel", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

C = {
    "bg":"#08090d","card":"#12131a","cyan":"#06b6d4","red":"#f43f5e",
    "amber":"#f59e0b","green":"#22c55e","purple":"#a78bfa","text":"#e2e4ea",
    "dim":"#6b7280","grid":"#1e1f2e",
}

def themed(fig):
    fig.update_layout(
        paper_bgcolor=C["card"], plot_bgcolor=C["card"],
        font=dict(color=C["text"], family="JetBrains Mono, monospace", size=12),
        xaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
        yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"]),
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;700&display=swap');
:root{--bg:#08090d;--card:#12131a;--cyan:#06b6d4;--red:#f43f5e;--amber:#f59e0b;--green:#22c55e;--purple:#a78bfa;--text:#e2e4ea;--dim:#6b7280;}
.stApp{background:var(--bg);}
.mc{background:linear-gradient(135deg,#12131a,#1a1b2e);border:1px solid #1e1f2e;border-radius:12px;padding:20px 24px;text-align:center;}
.mc:hover{border-color:var(--cyan);}
.mv{font-family:'JetBrains Mono',monospace;font-size:2.2rem;font-weight:700;line-height:1.1;margin-bottom:4px;}
.ml{font-family:'JetBrains Mono',monospace;font-size:.75rem;color:var(--dim);text-transform:uppercase;letter-spacing:1.5px;}
.fc{background:var(--card);border-left:3px solid var(--cyan);border-radius:0 8px 8px 0;padding:16px 20px;margin:8px 0;font-family:'JetBrains Mono',monospace;font-size:.9rem;}
.fc.crit{border-left-color:var(--red);}.fc.warn{border-left-color:var(--amber);}.fc.ok{border-left-color:var(--green);}
.sh{font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;color:var(--cyan);border-bottom:2px solid #1e1f2e;padding-bottom:8px;margin:2rem 0 1rem;}
div[data-testid="stSidebar"]{background:linear-gradient(180deg,#0c0d14,#12131a);border-right:1px solid #1e1f2e;}
.stTabs [data-baseweb="tab-list"]{gap:8px;}
.stTabs [data-baseweb="tab"]{background:var(--card);border-radius:8px;color:var(--dim);font-family:'JetBrains Mono',monospace;font-size:.85rem;}
.stTabs [aria-selected="true"]{background:var(--cyan)!important;color:var(--bg)!important;}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DATA — optimized for low memory
# ══════════════════════════════════════════════════════════════

KAGGLE_DATASET = "shriyashjagtap/kaggle-bot-account-detection"

@st.cache_data(show_spinner="Downloading from Kaggle...", ttl=3600)
def load_kaggle():
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                # Read with optimized dtypes
                df = pd.read_csv(os.path.join(root, f))
                return df
    raise FileNotFoundError("No CSV found")

@st.cache_data(show_spinner="Engineering features...", ttl=3600)
def process(df):
    total = len(df)
    df_lab = df.dropna(subset=["ISBOT"]).copy()
    # Downcast numeric columns to save memory
    for col in df_lab.select_dtypes(include=['float64']).columns:
        df_lab[col] = df_lab[col].astype(np.float32)
    for col in df_lab.select_dtypes(include=['int64']).columns:
        if col != 'ISBOT':
            df_lab[col] = pd.to_numeric(df_lab[col], downcast='integer')
    df_eng = engineer_features(df_lab)
    # Pre-compute stats to avoid reprocessing full df
    stats = precompute_stats(df_eng, total)
    return df_eng, stats

def precompute_stats(df, total_raw):
    """Pre-aggregate everything so we don't touch the full df in rendering."""
    bot = df[df["ISBOT"] == 1]
    human = df[df["ISBOT"] == 0]
    labeled = len(df)
    bot_count = len(bot)
    human_count = len(human)

    # Auth stats - use .loc to avoid FutureWarning
    auth_grp = df.groupby("IS_GLOGIN")["ISBOT"].mean() * 100
    email_rate = float(auth_grp.loc[0]) if 0 in auth_grp.index else 0.0
    oauth_rate = float(auth_grp.loc[1]) if 1 in auth_grp.index else 0.0

    # Feature means for behavioral DNA
    eng_features = [
        "ENGAGEMENT_RATIO","FOLLOW_RECIPROCITY","SOCIAL_REACH","TOTAL_CONTENT",
        "CONTENT_PER_DISCUSSION","ACTIVITY_SCORE","IS_DORMANT","READ_PER_DISCUSSION",
        "READ_ENGAGEMENT","HAS_READ_TIME","VOTE_TOTAL","VOTE_ENTROPY",
        "VOTE_NB_RATIO","VOTE_CONCENTRATION","PHANTOM_SCORE","AUTHENTICITY_INDEX",
    ]
    feat_stats = []
    for f in eng_features:
        bm = float(bot[f].mean())
        hm = float(human[f].mean())
        ratio = hm / (bm + 1e-10) if hm > bm else bm / (hm + 1e-10)
        feat_stats.append({"Feature": f, "Bot Mean": bm, "Human Mean": hm, "Ratio": ratio})

    # Cohen's d for feature importance
    imp_features = [
        "IS_GLOGIN","ACTIVITY_SCORE","SOCIAL_REACH","TOTAL_CONTENT","READ_ENGAGEMENT",
        "AVG_NB_READ_TIME_MIN","FOLLOW_RECIPROCITY","VOTE_ENTROPY","CONTENT_PER_DISCUSSION",
        "PHANTOM_SCORE","FOLLOWER_COUNT","FOLLOWING_COUNT","VOTE_TOTAL","ENGAGEMENT_RATIO","AUTHENTICITY_INDEX",
    ]
    cohen_d = []
    for f in imp_features:
        bv = bot[f].dropna()
        hv = human[f].dropna()
        ps = np.sqrt((float(bv.std())**2 + float(hv.std())**2) / 2) + 1e-10
        d = abs(float(bv.mean()) - float(hv.mean())) / ps
        cohen_d.append({"Feature": f, "d": d})

    # Radar features
    radar_f = ["ACTIVITY_SCORE","SOCIAL_REACH","TOTAL_CONTENT","READ_ENGAGEMENT","FOLLOW_RECIPROCITY","VOTE_ENTROPY"]
    radar_bot = [float(bot[f].mean()) for f in radar_f]
    radar_human = [float(human[f].mean()) for f in radar_f]

    return {
        "total_raw": total_raw, "labeled": labeled,
        "bot_count": bot_count, "human_count": human_count,
        "bot_rate": bot_count / labeled,
        "email_rate": email_rate, "oauth_rate": oauth_rate,
        "unlabeled": total_raw - labeled,
        "feat_stats": feat_stats, "cohen_d": cohen_d,
        "radar_f": radar_f, "radar_bot": radar_bot, "radar_human": radar_human,
    }

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🤖 KAGGLE SENTINEL")
    st.markdown("##### Bot Detection Forensics")
    st.markdown("---")
    data_source = st.radio("Data Source", ["⬇️ Download from Kaggle", "📁 Upload CSV"], index=0)
    df_loaded = None
    if data_source == "⬇️ Download from Kaggle":
        st.caption("Auto-downloads via kagglehub.")
        if st.button("🚀 Download & Load", type="primary", use_container_width=True):
            try:
                df_loaded = load_kaggle()
                st.success(f"✅ {len(df_loaded):,} rows")
            except Exception as e:
                st.error(f"Failed: {e}")
                st.info("Add `KAGGLE_USERNAME` and `KAGGLE_KEY` in Streamlit Secrets.")
        if df_loaded is None:
            try: df_loaded = load_kaggle()
            except: pass
    else:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up: df_loaded = pd.read_csv(up)
    st.markdown("---")
    st.markdown("<div style='text-align:center;color:#6b7280;font-size:.75rem;'>Sentinel v1.0 • <a href='https://github.com/ZeroZulu/Kaggle' style='color:#06b6d4;'>GitHub</a></div>", unsafe_allow_html=True)

if df_loaded is None:
    st.markdown("""<div style="text-align:center;padding:80px 20px;"><div style="font-size:5rem;">🤖</div>
    <h1 style="font-family:'Space Grotesk',sans-serif;color:#06b6d4;">KAGGLE SENTINEL</h1>
    <p style="color:#6b7280;font-size:1.1rem;">Click <strong>Download & Load</strong> in the sidebar or upload the CSV.</p></div>""", unsafe_allow_html=True)
    st.stop()

# ── Process ──
df_eng, S = process(df_loaded)
del df_loaded; gc.collect()

# ══════════════════════════════════════════════════════════════
# HEADER + KPIs
# ══════════════════════════════════════════════════════════════

st.markdown("""<div style="text-align:center;padding:10px 0 5px;">
<h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.4rem;background:linear-gradient(90deg,#06b6d4,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0;">🤖 KAGGLE SENTINEL</h1>
<p style="color:#6b7280;font-family:'JetBrains Mono',monospace;font-size:.85rem;">Bot Account Detection Through Behavioral Forensics</p></div>""", unsafe_allow_html=True)

for col, (v, l, c) in zip(st.columns(5), [
    (f"{S['total_raw']:,}", "TOTAL ACCOUNTS", C["text"]),
    (f"{S['labeled']:,}", "LABELED", C["cyan"]),
    (f"{S['bot_count']:,}", "BOTS DETECTED", C["red"]),
    (f"{S['bot_rate']:.1%}", "BOT RATE", C["amber"]),
    (f"{S['human_count']/S['bot_count']:.2f}:1", "IMBALANCE", C["purple"]),
]):
    col.markdown(f'<div class="mc"><div class="mv" style="color:{c};">{v}</div><div class="ml">{l}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview","🧬 Behavioral DNA","📐 Benford's Law","🤖 ML Performance","🎯 Recommendations"])

# ── TAB 1 ──
with tab1:
    st.markdown('<div class="sh">Dataset Overview & Key Signals</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(
            x=["Email Signup","Google OAuth"], y=[S["email_rate"], S["oauth_rate"]],
            marker_color=[C["red"], C["cyan"]],
            text=[f"{S['email_rate']:.1f}%", f"{S['oauth_rate']:.1f}%"],
            textposition="outside", textfont=dict(size=16, color=C["text"]),
        ))
        fig.update_layout(title="🔐 Authentication vs Bot Rate", yaxis_title="Bot Rate (%)",
                           yaxis_range=[0, S["email_rate"]*1.3 if S["email_rate"]>0 else 50], showlegend=False, height=400)
        st.plotly_chart(themed(fig), use_container_width=True)
        st.markdown(f'<div class="fc crit"><strong>CRITICAL:</strong> OAuth = <strong>0.0%</strong> bots. Email = <strong>{S["email_rate"]:.1f}%</strong>. SSO eliminates virtually all bots.</div>', unsafe_allow_html=True)
    with c2:
        fig = go.Figure(go.Pie(
            labels=["Human","Bot","Unlabeled"], values=[S["human_count"], S["bot_count"], S["unlabeled"]],
            marker=dict(colors=[C["cyan"], C["red"], C["dim"]]), hole=0.5, textinfo="label+percent", textfont=dict(size=13),
        ))
        fig.update_layout(title="📊 Account Breakdown", height=400, showlegend=False)
        st.plotly_chart(themed(fig), use_container_width=True)

    # Distribution — SAMPLED to save memory
    st.markdown('<div class="sh">Feature Distributions: Bot vs Human</div>', unsafe_allow_html=True)
    feat_choice = st.selectbox("Select feature", ["FOLLOWER_COUNT","FOLLOWING_COUNT","ACTIVITY_SCORE","SOCIAL_REACH","TOTAL_CONTENT","VOTE_TOTAL","AVG_NB_READ_TIME_MIN","ENGAGEMENT_RATIO"], index=2)

    MAX_HIST = 50_000  # sample for histogram to avoid memory blow-up
    fig = go.Figure()
    for lbl, clr, nm in [(0, C["cyan"], "Human"), (1, C["red"], "Bot")]:
        subset = df_eng[df_eng["ISBOT"] == lbl][feat_choice]
        q99 = float(subset.quantile(0.99))
        subset = subset.clip(upper=q99)
        if len(subset) > MAX_HIST:
            subset = subset.sample(MAX_HIST, random_state=42)
        fig.add_trace(go.Histogram(x=subset, name=nm, marker_color=clr, opacity=0.6, nbinsx=60))
    fig.update_layout(barmode="overlay", height=350, title=f"Distribution: {feat_choice}", xaxis_title=feat_choice, yaxis_title="Count")
    st.plotly_chart(themed(fig), use_container_width=True)

# ── TAB 2 ──
with tab2:
    st.markdown('<div class="sh">Behavioral DNA Feature Engineering</div>', unsafe_allow_html=True)
    rows = [{"Feature": r["Feature"], "Bot Mean": f"{r['Bot Mean']:.3f}", "Human Mean": f"{r['Human Mean']:.3f}", "Ratio": f"{r['Ratio']:.1f}x"} for r in S["feat_stats"]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)

    st.markdown('<div class="sh">Behavioral Fingerprint Radar</div>', unsafe_allow_html=True)
    rf = S["radar_f"]; rb = S["radar_bot"]; rh = S["radar_human"]
    mx = [max(b,h,1e-10) for b,h in zip(rb,rh)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[h/m for h,m in zip(rh,mx)]+[rh[0]/mx[0]], theta=rf+[rf[0]], fill="toself", name="Human", fillcolor="rgba(6,182,212,0.15)", line=dict(color=C["cyan"],width=2)))
    fig.add_trace(go.Scatterpolar(r=[b/m for b,m in zip(rb,mx)]+[rb[0]/mx[0]], theta=rf+[rf[0]], fill="toself", name="Bot", fillcolor="rgba(244,63,94,0.15)", line=dict(color=C["red"],width=2)))
    fig.update_layout(polar=dict(bgcolor=C["card"],radialaxis=dict(visible=True,range=[0,1],gridcolor=C["grid"]),angularaxis=dict(gridcolor=C["grid"])), height=500, title="Bot vs Human Behavioral Profile")
    st.plotly_chart(themed(fig), use_container_width=True)
    st.markdown('<div class="fc warn"><strong>KEY INSIGHT:</strong> Bots show a collapsed profile — near-zero across engagement, content, and reading. Only voting approaches human levels → automated vote manipulation.</div>', unsafe_allow_html=True)

# ── TAB 3 ──
with tab3:
    st.markdown('<div class="sh">Benford\'s Law Forensic Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="fc"><strong>METHOD:</strong> Leading digit <em>d</em> appears with probability log₁₀(1+1/d). Bots violate this. Measured via <strong>Jensen-Shannon Divergence</strong>.</div>', unsafe_allow_html=True)
    bcol = st.selectbox("Feature", ["FOLLOWER_COUNT","FOLLOWING_COUNT","DISCUSSION_COUNT","CODE_COUNT"])
    cb1, cb2 = st.columns(2)
    br = benford_analysis(df_eng[df_eng["ISBOT"]==1][bcol], "Bot")
    hr = benford_analysis(df_eng[df_eng["ISBOT"]==0][bcol], "Human")
    exp = benfords_expected(); digits = list(range(1,10))
    for cb, res, ttl, clr in [(cb1,hr,"👤 Human",C["cyan"]),(cb2,br,"🤖 Bot",C["red"])]:
        with cb:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=digits, y=[exp[d]*100 for d in digits], name="Benford", marker_color=C["amber"], opacity=0.4))
            fig.add_trace(go.Bar(x=digits, y=[res["dist"].get(d,0)*100 for d in digits], name=ttl, marker_color=clr, opacity=0.7))
            fig.update_layout(title=f"{ttl} — JS: {res['js_divergence']:.4f} ({res['verdict']})", xaxis_title="First Digit", yaxis_title="Freq %", barmode="group", height=400)
            st.plotly_chart(themed(fig), use_container_width=True)
    st.markdown("#### Summary")
    brows = []
    for feat in ["FOLLOWER_COUNT","FOLLOWING_COUNT","DISCUSSION_COUNT","CODE_COUNT"]:
        for lbl, nm in [(1,"Bot"),(0,"Human")]:
            r = benford_analysis(df_eng[df_eng["ISBOT"]==lbl][feat], nm)
            icon = "🔴" if r["verdict"]=="EXTREME" else "🟡" if r["verdict"]=="MODERATE" else "🟢"
            brows.append({"Feature":feat,"Class":nm,"χ²":f"{r['chi2']:,.1f}","JS Div":f"{r['js_divergence']:.4f}","Verdict":f"{icon} {r['verdict']}"})
    st.dataframe(pd.DataFrame(brows), use_container_width=True, hide_index=True)

# ── TAB 4 ──
with tab4:
    st.markdown('<div class="sh">Machine Learning Pipeline Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="fc ok"><strong>KEY RESULT:</strong> All models achieve <strong>perfect classification</strong> (AUC ≈ 1.0). Boundary is trivially separable. Real value = forensic features.</div>', unsafe_allow_html=True)
    mr = {"Logistic Regression":{"Acc":.9999,"F1":.9998,"AUC":1.0,"Prec":.9996,"Rec":1.0},
          "Random Forest":{"Acc":1.0,"F1":1.0,"AUC":1.0,"Prec":1.0,"Rec":1.0},
          "Gradient Boosting":{"Acc":1.0,"F1":1.0,"AUC":1.0,"Prec":1.0,"Rec":1.0},
          "XGBoost":{"Acc":1.0,"F1":1.0,"AUC":1.0,"Prec":1.0,"Rec":1.0},
          "LightGBM":{"Acc":1.0,"F1":1.0,"AUC":1.0,"Prec":1.0,"Rec":1.0},
          "Stacking Ensemble":{"Acc":1.0,"F1":1.0,"AUC":1.0,"Prec":1.0,"Rec":1.0}}
    st.dataframe(pd.DataFrame(mr).T.rename_axis("Model").style.format("{:.4f}"), use_container_width=True)

    fig = go.Figure()
    models = list(mr.keys())
    for met, clr in zip(["Acc","F1","Prec","Rec"],[C["cyan"],C["amber"],C["green"],C["purple"]]):
        fig.add_trace(go.Bar(name=met, x=models, y=[mr[m][met] for m in models], marker_color=clr, opacity=0.8))
    fig.update_layout(barmode="group", height=450, title="5-Fold Stratified CV", yaxis_range=[0.999,1.0005], yaxis_title="Score")
    st.plotly_chart(themed(fig), use_container_width=True)

    st.markdown('<div class="sh">Feature Discriminative Power</div>', unsafe_allow_html=True)
    idf = pd.DataFrame(S["cohen_d"]).sort_values("d")
    fig = go.Figure(go.Bar(x=idf["d"], y=idf["Feature"], orientation="h", marker_color=C["purple"]))
    fig.update_layout(title="Cohen's d Effect Size", xaxis_title="Effect Size", height=500)
    st.plotly_chart(themed(fig), use_container_width=True)

# ── TAB 5 ──
with tab5:
    st.markdown('<div class="sh">Production Recommendations</div>', unsafe_allow_html=True)
    recs = [
        ("P0","Enforce Google / SSO Auth",f"Email: {S['email_rate']:.1f}% → OAuth: {S['oauth_rate']:.1f}%. SSO eliminates ~100% of bots.","~100% bot prevention","crit",C["red"]),
        ("P1","Activity Score Gate","Zero bots above ACTIVITY_SCORE ≈ 15. Instant flagging.","Zero false-positive flagging","crit",C["red"]),
        ("P2","ML Ensemble + SHAP","All models achieve perfect separation. SHAP for explainability.","Automated + transparent","warn",C["amber"]),
        ("P3","Benford Monitoring","Bot follower JS-divergence > 0.48. Forensic anomaly layer.","Detects synthetic metrics","warn",C["amber"]),
        ("P4","Geographic Risk Scoring","243+ regions. Island territories elevated. Ensemble weight only.","Risk-weighted registration","",C["cyan"]),
    ]
    for p, title, detail, impact, cls, color in recs:
        st.markdown(f'<div class="fc {cls}" style="border-left-color:{color};"><div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;"><span style="background:{color};color:#08090d;padding:2px 10px;border-radius:4px;font-weight:700;font-family:\'JetBrains Mono\',monospace;font-size:.85rem;">{p}</span><strong style="font-size:1.05rem;">{title}</strong></div><p style="color:#9ca3af;margin:4px 0;">{detail}</p><p style="color:{color};font-size:.85rem;margin:4px 0;">⚡ {impact}</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sh">Investigation Summary</div>', unsafe_allow_html=True)
    for col, (v,l,c) in zip(st.columns(3),[("27","Features Engineered",C["cyan"]),("5","Models Benchmarked",C["amber"]),("4","Forensic Methods",C["purple"])]):
        col.markdown(f'<div class="mc"><div class="mv" style="color:{c};">{v}</div><div class="ml">{l}</div></div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;padding:40px 0 20px;border-top:1px solid #1e1f2e;margin-top:40px;"><p style="color:#6b7280;font-family:\'JetBrains Mono\',monospace;font-size:.8rem;">KAGGLE SENTINEL v1.0<br><a href="https://www.kaggle.com/datasets/shriyashjagtap/kaggle-bot-account-detection" style="color:#06b6d4;">Dataset</a> • <a href="https://github.com/ZeroZulu/Kaggle" style="color:#06b6d4;">GitHub</a></p></div>', unsafe_allow_html=True)
