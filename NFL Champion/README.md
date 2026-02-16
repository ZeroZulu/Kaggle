<div align="center">

# 🏈 Anatomy of a Super Bowl Champion

**What Does It Actually Take to Win the Biggest Game in Sports?**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://kaggle.com)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)](https://plotly.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

6,499 NFL games across 23 seasons (2002–2024) — unpivoted from game-level ESPN data into team-perspective rows — to build a data-driven championship profile.

## Key Findings
- **SB Losers outscore Winners** (28.1 vs 26.6 PPG) — raw offensive firepower doesn't win rings
- **Defense is the separator** — Winners allow 18.5 PPG vs 22.6 league average (4.1-point gap)
- **Turnover margin is the #1 predictor** — Winners: +0.60/game vs Others: −0.05
- **93.2% AUC** — Logistic Regression identifies champions from regular-season stats alone
- **You don't need perfection** — 2011 Giants won it all at just 65% (9-7)

## Unique Contributions
- **Champion DNA Fingerprint** — Normalized radar chart comparing Winners, Losers, and the field across 8 dimensions (defense inverted)
- **Game-Level Unpivot Pipeline** — Transforms ESPN's `_away`/`_home` format into team-perspective rows with opponent stats
- **Multi-Factor Analysis** — Offense, defense, turnovers, point differential, and 3rd-down efficiency
- **3-Model Comparison** — Logistic Regression, Random Forest, and Gradient Boosting with feature importance

## Dataset
[NFL Team Stats 2002–2025 (ESPN)](https://www.kaggle.com/datasets/cviaxmiwnptr/nfl-team-stats-20022019-espn/data) — 6,499 games × 61 columns

## Quick Start
Upload `nfl-superbowl-champion-anatomy.ipynb` to Kaggle with the dataset attached → Run All.
