# 🛒 Store Sales Forecasting

> **Time series demand prediction across 54 stores × 33 product families, achieving 66% cost reduction vs. baseline methods.**

[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit)](https://storesaleforecasting.streamlit.app)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/code/YOUR_USERNAME/store-sales-forecasting)

---

## 📊 Results

| Model | RMSLE | Business Impact |
|-------|-------|-----------------|
| 🥇 **XGBoost** | **0.4510** | **$12.9M projected annual savings** |
| 🥈 LightGBM | 0.5540 | — |
| 🥉 Seasonal Naive | 0.6565 | Baseline |

**66.4% cost reduction** by reducing stockouts ($1.1M/month) while managing overstock losses.

---

## 🔧 Approach

**The Challenge:** 1,782 individual time series with 3M+ records, external factors (oil prices, holidays, 2016 earthquake), and Ecuador-specific economic dynamics.

**Solution:** XGBoost with 43 engineered features including:
- **Lag features** (1, 7, 14, 21, 28 days) — most predictive
- **Rolling statistics** (7/14/28-day mean, std)
- **External factors** (oil prices, holidays, promotions)
- **Domain features** (earthquake impact, paydays, store clusters)

**Key Insight:** Proper time-based validation and iterative lag computation for test data prevents data leakage.

---

## 🛠️ Tech Stack

Python · Pandas · XGBoost · LightGBM · Prophet · Statsmodels · Streamlit

---

## 🔗 Links

- 🌐 [Live Demo](https://storesaleforecasting.streamlit.app) — Interactive sales trends & forecasts
- 📓 [Kaggle Notebook](https://www.kaggle.com/code/YOUR_USERNAME/store-sales-forecasting) — Full analysis

---

*Built by [Shril Patel](https://github.com/ZeroZulu)*
