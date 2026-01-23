# 🌍 Life Expectancy Analysis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)
![R²](https://img.shields.io/badge/R²_Score-0.9763-success.svg)

Predicting global life expectancy using machine learning with **97.6% accuracy** (R² = 0.9763).

## 📊 Overview

Analysis of WHO life expectancy data (2000-2015) across 193 countries using conventional and advanced ML techniques.

## 🔑 Key Findings

| Insight | Value |
|---------|-------|
| Best Model | XGBoost (R² = 0.9763) |
| Top Predictor | HIV/AIDS prevalence |
| Dev vs Developing Gap | 12.1 years |
| Countries Analyzed | 193 |

### Top 5 Predictors (SHAP)
1. HIV/AIDS (2.10)
2. Adult Mortality (1.66)
3. Income Composition (1.54)
4. Mortality Score (0.83)
5. Year-over-Year Change (0.46)

## 🛠️ Methods

**Conventional ML:** Linear Regression, Ridge, Lasso, KNN, Decision Tree, Random Forest, Gradient Boosting, SVR

**Advanced ML:** XGBoost, LightGBM, CatBoost, Stacking Ensemble

**Interpretability:** SHAP analysis for feature importance

## 📈 Model Performance

| Model | Test R² | RMSE |
|-------|---------|------|
| XGBoost | 0.9763 | 1.42 |
| LightGBM | 0.9751 | 1.45 |
| CatBoost | 0.9743 | 1.48 |
| Gradient Boosting | 0.9674 | 1.66 |
| Random Forest | 0.9605 | 1.83 |

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/life-expectancy-analysis.git

# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm catboost shap plotly

# Run notebook
jupyter notebook life-expectancy.ipynb
```

## 📁 Project Structure

```
├── life-expectancy.ipynb    # Main analysis notebook
├── README.md                # This file
└── data/
    └── Life Expectancy Data.csv
```

## 📊 Dataset

- **Source:** [Kaggle - Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- **Records:** 2,938 observations
- **Features:** 22 columns
- **Period:** 2000-2015

## 💡 Policy Recommendations

1. **Education Investment** - Strong correlation with life expectancy
2. **HIV/AIDS Prevention** - Highest impact factor
3. **Child Health Programs** - Under-5 mortality is critical
4. **Economic Development** - Income composition matters

## 📜 License

MIT License

## 🙏 Acknowledgments

- WHO for the dataset
- Kaggle community for inspiration
