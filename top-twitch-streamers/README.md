# 🎮 Top Twitch Streamers Analysis

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-444876?style=for-the-badge&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)

This repository contains an exploratory data analysis (EDA) and machine learning project on the **Top Twitch Streamers** dataset. The analysis uncovers patterns in viewer count, follower growth, language preferences, and streamer success factors on the Twitch platform.

![Twitch Banner](https://www.dexerto.com/cdn-image/wp-content/uploads/2024/11/05/twitch-clarifies-sensitive-content-label-lived-experiences.jpg?width=1200&quality=75&format=auto)

---

## 📁 Dataset Overview

The dataset is sourced from [Kaggle: Top Twitch Streamers](https://www.kaggle.com/datasets/aayushmishra1512/twitchdata) and contains **1,000 top streamers** with the following features:

| Feature | Description |
|---------|-------------|
| `Channel` | Streamer's channel name |
| `Watch time(Minutes)` | Total watch time accumulated |
| `Stream time(minutes)` | Total time spent streaming |
| `Peak viewers` | Maximum concurrent viewers |
| `Average viewers` | Average viewers per stream |
| `Followers` | Total follower count |
| `Followers gained` | New followers acquired |
| `Views gained` | New views acquired |
| `Partnered` | Twitch Partner status (True/False) |
| `Mature` | Mature content flag (True/False) |
| `Language` | Primary streaming language |

---

## 🔍 Key Questions Explored

- Who are the most followed and most-watched Twitch streamers?
- What is the relationship between stream time and follower growth?
- How does language affect reach and viewer engagement?
- Are Twitch Partners more successful in terms of views or followers?
- Which factors best predict follower growth?
- How do different streamer tiers (by follower count) compare in engagement?

---

## 📊 Visualizations Included

| Type | Description |
|------|-------------|
| **Bar Charts** | Top streamers by followers, watch time, and views |
| **Scatter Plots** | Stream time vs followers gained relationship |
| **Correlation Heatmap** | Relationships between all numeric features |
| **Distribution Plots** | Histogram/KDE of key metrics (log-transformed) |
| **Pie Charts** | Language and partner status distribution |
| **Word Clouds** | Streamer names and languages |
| **Feature Importance** | ML model feature rankings |

---

## 🧠 Machine Learning

The project includes predictive modeling to forecast **follower growth**:

### Models Compared
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Evaluation Metrics
- R² Score
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

### Key Findings
- **Random Forest** achieved the best predictive performance
- Top predictive features: `Views gained`, `Followers`, `Watch time`
- Viral content (views) is the strongest predictor of growth

---

## ⚙️ Feature Engineering

New calculated features created for deeper analysis:

| Feature | Formula | Insight |
|---------|---------|---------|
| `Watch_Stream_Ratio` | Watch time / Stream time | Content engagement level |
| `Followers_per_Stream_Hour` | Followers gained / Stream hours | Growth efficiency |
| `Views_per_Follower` | Views gained / Followers | Content virality |
| `Engagement_Score` | (Avg viewers / Peak viewers) × 100 | Viewer retention |
| `Streamer_Tier` | Categorical (by follower count) | Emerging → Elite classification |

---

## 🏆 Key Insights

### Top Performers
| Category | Streamer | Value |
|----------|----------|-------|
| Most Watch Time | xQcOW | 6.2B minutes |
| Most Followers | Tfue | ~9M followers |
| Highest Avg Viewers | Dota2ti | 147,643 viewers |
| Most Followers Gained | auronplay | ~4M gained |
| Most Views Gained | Fextralife | 670M views |

### Statistical Findings
- **93%+** of top 1,000 streamers are Twitch Partners
- **English** dominates with the most streamers, followed by Spanish and Portuguese
- Moderate correlation (~0.35) between stream time and follower growth
- Partnered streamers average significantly more followers than non-partners

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **Matplotlib & Seaborn** — Statistical visualization
- **Plotly** — Interactive charts
- **Scikit-learn** — Machine learning models
- **WordCloud** — Text visualization
- **Jupyter Notebook** — Interactive development

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn wordcloud
```

### Run the Notebook
```bash
# Clone the repository
git clone https://github.com/yourusername/top-twitch-streamers-analysis.git
cd top-twitch-streamers-analysis

# Launch Jupyter
jupyter notebook top-twitch-streamers.ipynb
```

### Kaggle
You can also run this notebook directly on [Kaggle](https://www.kaggle.com/) with the dataset pre-loaded.

---

## 📂 Project Structure

```
top-twitch-streamers-analysis/
├── README.md
├── top-twitch-streamers.ipynb    # Main analysis notebook
├── twitchdata-update.csv         # Dataset
└── images/                       # Visualization exports (optional)
```

---

## 🎯 Recommendations for Aspiring Streamers

Based on the analysis:

1. **Focus on viral content** — Views gained is the #1 predictor of follower growth
2. **Quality over quantity** — Stream time helps, but engagement matters more
3. **Work toward Partnership** — Partners show significantly better metrics
4. **Consider emerging markets** — Non-English languages may have less competition

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

---

## ⭐ Show Your Support

Give a ⭐ if this project helped you!

---

*Built with 💜 for the Twitch community*
