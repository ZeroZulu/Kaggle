<div align="center">

# 🏀 NBA Player Archetypes

**Are Traditional Basketball Positions Still Meaningful?**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://kaggle.com)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)](https://plotly.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

K-Means clustering on 9 advanced stats discovers natural player archetypes — with zero position labels — across 27 NBA seasons (1996–2022).

## Key Findings
- **6 natural archetypes emerge** from stats alone — Star Playmakers, Paint Beasts, 3-and-D Wings, Energy Bigs, Combo Guards, and Deep Bench
- **The NBA is getting more specialized, not positionless** — Entropy is DOWN 6.6%, meaning roles are concentrating, not dissolving
- **True Shooting is up +5.3pp** across every archetype — the real revolution is efficiency, not positions
- **Unicorn players have doubled** — Outlier rate grew from 4.7% to 8.1% (Jokic, Giannis, Westbrook)
- **Usage distribution is remarkably stable** — despite "hero ball" narratives, offensive burden hasn't shifted

## Unique Contributions
- **Score-Based Archetype Naming** — Greedy assignment system matches cluster profiles to basketball-meaningful labels
- **Positional Entropy Score** — Shannon entropy quantifies how evenly players spread across archetypes each season
- **Unicorn Index** — Distance-to-centroid outlier detection identifies players who defy all categories (min 5 PPG filter)
- **Efficiency Revolution Charts** — TS% and USG% trends by archetype over 3-year bins

## Dataset
[NBA Players Data](https://www.kaggle.com/datasets/justinas/nba-players-data) — 12,844 player-seasons × 22 columns (10,720 after filtering)

## Quick Start
Upload `nba-player-archetypes.ipynb` to Kaggle with the dataset attached → Run All.
