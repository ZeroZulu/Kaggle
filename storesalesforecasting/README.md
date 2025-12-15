## 🎯 Project Overview

### The Business Problem

Accurate demand forecasting is critical for retail operations. Poor forecasts lead to:
- **Overstocking** → Tied-up capital, spoilage (especially perishables), storage costs
- **Understocking** → Lost sales, unhappy customers, damaged brand reputation

### The Technical Challenge

This is a **hierarchical forecasting problem** with:
- **54 stores × 33 product families = 1,782 individual time series**
- **3+ million historical records** spanning 2013-2017
- External factors: holidays, oil prices, promotions, economic conditions
- Ecuador-specific dynamics (oil-dependent economy, regional holidays, 2016 earthquake impact)

### My Approach

1. **Proper time series methodology** — No data leakage, time-based validation
2. **Multiple model comparison** — Statistical (SARIMA) vs. Modern (Prophet) vs. ML (XGBoost, LightGBM)
3. **Domain-aware feature engineering** — 43 features including lag features, rolling statistics, holiday effects
4. **Business impact quantification** — Translated model performance into dollar savings

---

## 📈 Model Performance Comparison

| Model | RMSLE | RMSE | MAE | MAPE |
|-------|-------|------|-----|------|
| 🥇 **XGBoost** | **0.4510** | 196.20 | 56.49 | 36.83% |
| 🥈 LightGBM | 0.5540 | 198.23 | 59.10 | 40.87% |
| 🥉 Seasonal Naive | 0.6565 | 488.61 | 130.10 | 47.12% |
| Store-Family Mean | 0.6844 | 536.99 | 148.39 | 47.80% |
| SARIMA (single series) | 0.1454 | 1,638.71 | 1,072.40 | 10.57% |
| Global Mean | 3.4020 | 1,285.59 | 587.39 | 4,085% |

---

## 💰 Business Impact

```
┌─────────────────────────────────────────────────────────────┐
│                    COST ANALYSIS                            │
├─────────────────────────────────────────────────────────────┤
│  XGBoost Model:                                             │
│     Overstock losses:  $154,429                             │
│     Stockout losses:   $394,113                             │
│     Total:             $548,542                             │
├─────────────────────────────────────────────────────────────┤
│  Seasonal Naive Baseline:                                   │
│     Overstock losses:  $110,935                             │
│     Stockout losses:   $1,519,426                           │
│     Total:             $1,630,361                           │
├─────────────────────────────────────────────────────────────┤
│  💰 MONTHLY SAVINGS:        $1,081,818                      │
│  💰 PROJECTED ANNUAL:       $12,981,820                     │
│  📉 COST REDUCTION:         66.4%                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Feature Engineering

### Features Created (43 total)

| Category | Features | Description |
|----------|----------|-------------|
| **Date** | 12 | year, month, day, dayofweek, quarter, is_weekend, is_month_start/end, cyclical encoding (sin/cos) |
| **Lag** | 5 | sales_lag_1, sales_lag_7, sales_lag_14, sales_lag_21, sales_lag_28 |
| **Rolling** | 8 | 7/14/28-day rolling mean, std, min, max |
| **Holiday** | 4 | is_national_holiday, is_regional_holiday, is_local_holiday, days_to_holiday |
| **Oil** | 4 | dcoilwtico, oil_ma7, oil_ma30, oil_change |
| **Store** | 2 | store_type_encoded, cluster |
| **Earthquake** | 2 | post_earthquake, days_after_earthquake |
| **Other** | 6 | onpromotion, family_encoded, is_payday, expanding_mean |

### Top 10 Most Important Features

1. `sales_lag_1` — Previous day sales
2. `sales_lag_7` — Same day last week
3. `sales_rolling_mean_7` — 7-day rolling average
4. `dayofweek` — Day of week (strong weekly pattern)
5. `sales_lag_14` — Two weeks ago
6. `sales_rolling_mean_14` — 14-day rolling average
7. `family_encoded` — Product family
8. `store_nbr` — Store identity
9. `onpromotion` — Promotion flag
10. `days_to_holiday` — Proximity to holidays

---
## 🔬 Methodology

### Data Splitting Strategy

```
Training Data: 2013-01-01 to 2017-07-15 (1,626 days)
Validation Data: 2017-07-16 to 2017-08-15 (31 days)
Test Data: 2017-08-16 to 2017-08-31 (16 days)
```

**Why this matters:** Time series requires temporal splitting to prevent data leakage. We never use future data to predict the past.

### Handling Test Data Lag Features

A critical challenge: test data doesn't have sales values, so how do we compute lag features?

**Solution:** Iterative approach using only historical data

```python
for test_date in test_dates:
    # For each test date, compute lags from available historical data
    for lag in [1, 7, 14, 21, 28]:
        lag_date = test_date - pd.Timedelta(days=lag)
        lag_values = historical_data[historical_data['date'] == lag_date]
        # Merge lag values into test data
```

### Model Training

- **XGBoost**: 500 estimators, early stopping at round 366
- **LightGBM**: 1000 estimators, early stopping at round 605
- **Hyperparameters**: Learning rate 0.05, max_depth 8, subsample 0.8

---

## 📝 Key Findings

### 1. Seasonality Patterns
- **Weekly**: Strong pattern with Saturday peaks, Sunday dips
- **Monthly**: Paydays (15th, end of month) show increased sales
- **Annual**: December holidays drive significant spikes

### 2. External Factors
- **Oil prices**: Moderate negative correlation (-0.22) — Ecuador is oil-dependent
- **Holidays**: 15-20% sales increase on national holidays
- **Promotions**: Significant positive impact on sales

### 3. Store Characteristics
- **Type A stores**: Highest sales volume
- **Clusters**: Geographic clustering affects demand patterns
- **Earthquake impact**: 2016 Ecuador earthquake caused lasting disruption

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | XGBoost, LightGBM, Scikit-learn |
| **Time Series** | Statsmodels (SARIMA), Prophet |
| **Model Persistence** | Joblib, Pickle |

---

## 📚 What I Learned

1. **Lag features are king** in time series forecasting — they capture the most signal
2. **Proper test data handling** is critical — you can't use future data for lag features
3. **Business context matters** — Ecuador's oil dependency affects retail patterns
4. **Multiple baselines** help contextualize model performance
5. **Early stopping** prevents overfitting in gradient boosting

---

## 🔮 Future Improvements

- [ ] **Ensemble methods**: Combine XGBoost + LightGBM + Prophet
- [ ] **Hierarchical reconciliation**: Ensure store/family/total predictions are consistent
- [ ] **Neural networks**: Experiment with LSTM, Transformer architectures
- [ ] **Hyperparameter tuning**: Use Optuna for systematic optimization
- [ ] **External data**: Add weather data, competitor pricing
- [ ] **Real-time pipeline**: Deploy with FastAPI + automated retraining

---

