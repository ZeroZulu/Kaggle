# 📊 Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Tableau](https://img.shields.io/badge/Tableau-Dashboard-orange.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Classification-green.svg)

Predicting customer churn in the telecom industry using machine learning to enable proactive retention strategies.

## 🎯 Problem Statement

Customer churn is a critical issue for telecom companies — acquiring new customers costs **5-7x more** than retaining existing ones. This project analyzes **7,043 customers** to identify churn patterns and build predictive models.

**Goals:**
- Understand key drivers behind customer churn
- Build and evaluate machine learning models to predict churn
- Provide actionable insights for retention strategies

## 🔍 Key Findings

| Insight | Details |
|---------|---------|
| **Churn Rate** | 26.5% of customers churned |
| **Highest Risk** | Month-to-month contracts (42% churn vs 3% for 2-year) |
| **Critical Period** | First 12 months — highest churn risk |
| **Service Impact** | Customers without add-ons churn 2x more |

## 🚀 Project Highlights

- **Data Preprocessing:** Cleaned data, engineered features, handled class imbalance
- **EDA:** Analyzed churn across demographics, services, and tenure
- **Modeling:** Compared Logistic Regression, Random Forest, XGBoost, and more
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

## 📈 Results

**Best Model: XGBoost**
- AUC-ROC: 0.84
- Best precision-recall trade-off for business application

**Top Predictors:** Tenure, Monthly Charges, Contract Type, Internet Service, Payment Method

## 📊 View the Analysis

🔗 **[View on Kaggle](https://www.kaggle.com/code/zerol0l/telecom-churn)**

🔗 **[View on Tableau Public](https://public.tableau.com/views/TelecomChurnv2/ExecutiveOverview?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)**

## 🛠 Technologies

Python • Pandas • Scikit-learn • XGBoost • Matplotlib • Seaborn • Tableau

## 💡 Business Recommendations

1. **Early Intervention** — Target customers in first 6 months
2. **Contract Conversion** — Incentivize annual plan upgrades
3. **Service Bundles** — Promote add-on services to increase stickiness
4. **Proactive Outreach** — Use model to identify and retain at-risk customers

## 🔮 Future Work

- Deploy real-time prediction API with FastAPI
- Build Streamlit dashboard for stakeholders
- Implement automated model retraining

---

⭐ If this project helped you, please give it a star!
