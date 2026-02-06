<div align="center">

# 🤖 KAGGLE SENTINEL

### Bot Account Detection Through Behavioral Forensics

[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-06b6d4?style=for-the-badge&logo=streamlit&logoColor=white)](https://kaggle-bn59nk2nujbu7agu6sadrw.streamlit.app)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/zerol0l/kaggle-sentinel)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge)](#license)

*Beyond classification — a multi-lens forensic investigation into platform manipulation on Kaggle.*

---

<img src="assets/dashboard_preview.png" alt="Sentinel Dashboard Preview" width="90%">

</div>

---

## 🔬 What Makes This Different?

Most bot detection projects stop at "train XGBoost, get 99% accuracy." Sentinel goes further — treating bot detection as a **forensic investigation** using techniques from financial fraud detection, information theory, and network analysis.

| Method | Technique | Key Finding |
|--------|-----------|-------------|
| **Benford's Law** | First-digit distribution analysis | Bot follower counts show catastrophic Benford violation (JS divergence > 0.48) |
| **Information Theory** | Shannon entropy of voting patterns | Bots concentrate votes in single categories |
| **Behavioral DNA** | 16 engineered ratio-based features | 221× gap in READ_ENGAGEMENT between bots and humans |
| **Unsupervised Detection** | Isolation Forest + K-Means | IsoForest inverts — bots are "too uniform" to be anomalous |
| **Ensemble ML** | XGBoost, LightGBM, RF + Stacking | Perfect separation (AUC = 1.0000) across all models |
| **Explainability** | SHAP TreeExplainer | Per-account forensic explanations |

---

## 📊 Key Results

```
┌─────────────────────────────────────────────────────┐
│  Google OAuth bot rate:   0.0%     (vs 42.9% email) │
│  Activity cliff:          Score ~15 (zero bots above│
│  Benford JS divergence:   0.4837   (EXTREME)        │
│  Best model AUC:          1.0000   (trivially sep.)  │
│  Engineered features:     16       (behavioral DNA)  │
│  Total features:          27       (raw + engineered)│
└─────────────────────────────────────────────────────┘
```

## 📐 Methodology

### Feature Engineering (Behavioral DNA)

16 forensic features engineered from raw account data:

| Category | Features | Insight |
|----------|----------|---------|
| **Engagement** | `ENGAGEMENT_RATIO`, `FOLLOW_RECIPROCITY`, `SOCIAL_REACH` | Bots have asymmetric follower/following patterns |
| **Content** | `TOTAL_CONTENT`, `CONTENT_PER_DISCUSSION`, `ACTIVITY_SCORE`, `IS_DORMANT` | Bots produce zero datasets/code |
| **Reading** | `READ_PER_DISCUSSION`, `READ_ENGAGEMENT`, `HAS_READ_TIME` | 221× gap in reading engagement |
| **Voting** | `VOTE_TOTAL`, `VOTE_ENTROPY`, `VOTE_NB_RATIO`, `VOTE_CONCENTRATION` | Bots out-vote humans despite no activity |
| **Composite** | `PHANTOM_SCORE`, `AUTHENTICITY_INDEX` | High votes + low activity = phantom behavior |

### ML Pipeline

- 5 models benchmarked with 5-fold stratified CV
- Stacking ensemble (RF + XGBoost + LightGBM → Logistic meta-learner)
- SHAP global + local explainability
- Feature ablation study with log loss differentiation

---

## 🖥️ Dashboard Features

The Streamlit dashboard provides five interactive panels:

| Tab | Contents |
|-----|----------|
| **📊 Overview** | KPI cards, authentication analysis, interactive feature distributions |
| **🧬 Behavioral DNA** | Feature comparison table, radar chart fingerprint |
| **📐 Benford's Law** | Interactive first-digit analysis, JS divergence comparison |
| **🤖 ML Performance** | Model comparison, Cohen's d feature importance |
| **🎯 Recommendations** | Prioritized production deployment strategy (P0-P4) |

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io]([https://share.streamlit.io](https://kaggle-bn59nk2nujbu7agu6sadrw.streamlit.app))
3. Connect your repo → set main file to `app.py`
4. **Note:** The app requires users to upload their own CSV (the dataset is not bundled due to Kaggle terms)

---

## 🏭 Production Recommendations

| Priority | Action | Impact |
|----------|--------|--------|
| **P0** | Enforce Google/SSO authentication | Eliminates ~100% of bots at registration |
| **P1** | Activity score threshold gate (< 15) | Instant flagging with zero false positives |
| **P2** | Deploy ML ensemble with SHAP | Automated detection + transparent decisions |
| **P3** | Real-time Benford's Law monitoring | Detects synthetic metric generation |
| **P4** | Geographic risk scoring | Risk-weighted registration (never sole signal) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**If you found this analysis useful, consider giving it a ⭐ on GitHub and an upvote on Kaggle!**

*Built with Python, scikit-learn, XGBoost, LightGBM, SHAP, Plotly, and Streamlit.*

</div>
