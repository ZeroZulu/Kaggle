# 🎵 Beats & Bytes: Predictive Analytics for Spotify

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Regression-green.svg)

Analyzing what audio features predict streaming success using statistical rigor and honest evaluation.

## 🎯 Problem Statement

Understanding what makes a song successful is the holy grail of the music industry. This project analyzes **950+ top-streamed Spotify tracks** to identify patterns in audio features and streaming performance.

**Goals:**
- Identify which audio features correlate with streaming success
- Build predictive models and honestly evaluate their limitations
- Provide evidence-based recommendations for music strategy

## 🔍 Key Findings

| Insight | Details |
|---------|---------|
| **Playlist Correlation** | r = 0.78 with streams — but likely reverse causation |
| **Audio Predictive Power** | Only ~8% of variance explained (R² = 0.08) |
| **Seasonal Effect** | Small impact (η² = 0.03) — timing < quality |
| **Mood Clusters** | 4 distinct audio profiles identified |

## 🚀 Project Highlights

- **Statistical Testing:** Kruskal-Wallis, Mann-Whitney U, Bonferroni correction
- **Two-Model Comparison:** Separates correlation (all features) from prediction (audio-only)
- **Interpretability:** SHAP analysis for feature importance
- **Clustering:** K-Means + t-SNE for mood-based segmentation

## 📈 Results

**Best Audio-Only Model: Random Forest**
- R² = 0.08 (5-fold CV)
- Top features: Speechiness, Instrumentalness, Acousticness

**Key Insight:** High R² models using playlist counts are misleading — playlists are an *effect* of success, not a cause.

## 📊 View the Analysis

🔗 **[View on Kaggle](https://www.kaggle.com/code/zerol0l/beats-and-bytes-predictive-analytics-for-spotify)**
🔗 **Interactive Dashboard:** [View on Tableau Public](https://public.tableau.com/views/BeatsBytesSpotifyStreamingAnalytics/ExecutiveOverview?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## 🛠 Technologies

Python • Pandas • Scikit-learn • SHAP • Matplotlib • Seaborn • SciPy • Statsmodels

## 💡 Business Recommendations

1. **Playlist Strategy** — Diversify pitching across platforms; focus on editorial placements
2. **Audio Guidelines** — Use danceability 54-75% as a soft benchmark, not a rule
3. **Release Timing** — Don't delay great tracks for "perfect" timing
4. **Mood Curation** — Leverage cluster analysis for playlist personalization

## ⚠️ Limitations

- **Survivorship bias** — Dataset contains only top-streamed songs
- **Correlation ≠ Causation** — Observational data cannot prove causal relationships
- **Low predictability** — Audio features alone don't predict hits (and that's an honest finding)

## 🔮 Future Work

- Integrate Spotify API for real-time analysis
- A/B test recommendations with playlist curators
- Expand to genre-specific models
