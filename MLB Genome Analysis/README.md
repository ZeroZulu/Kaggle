<div align="center">

# ⚾ Baseball Genome Map

**Is Today's Game Even the Same Sport They Played in 1901?**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://kaggle.com)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)](https://plotly.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

124 MLB seasons mapped into a 2D "genome" using t-SNE dimensionality reduction, revealing how baseball's statistical DNA has shifted across eras.

## Key Findings
- **Eras form distinct "islands"** in strategy space — transitions are phase shifts, not gradual drift
- **Run scoring has swung ±30%** between the Deadball Era and the Steroid Era explosion
- **Run differential is the tightest predictor in sports** — R² > 0.85 for predicting winning percentage
- **Modern baseball echoes the 1990s** — cosine similarity shows today's run environment is closest to the pre-Steroid era
- **Competitive balance has fluctuated** — some eras bred dynasties, others gave every team a shot

## Unique Contributions
- **t-SNE Genome Map** — Every season projected into 2D space, colored by era, revealing cluster structure
- **Cosine Similarity Matrix** — Season-vs-season heatmap showing which years "play alike"
- **Competitive Balance Index** — Standard deviation of win% over time measures dynasty vs. parity eras
- **Era Statistical Fingerprints** — Normalized heatmap comparing run scoring, win distribution, and balance metrics

## Dataset
[MLB Statistics 1901–Present](https://www.kaggle.com/datasets/diazk2/mlb-statistics-1901-present/data) — 2,690 team-seasons × 16 columns

## Quick Start
Upload `mlb-genome-map.ipynb` to Kaggle with the dataset attached → Run All.
