# 💳 Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Fraud%20Detection-red.svg)](#)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Autoencoder-orange.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF.svg)](https://www.kaggle.com/code)

A comprehensive machine learning project for detecting fraudulent credit card transactions using **10 different approaches** — from traditional ML methods to advanced deep learning and anomaly detection techniques.

![Fraud Detection Banner](https://img.shields.io/badge/🎯_Best_Model-Time--Aware_LightGBM_(F1:_0.876)-success)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Results](#-results)
- [Methods Implemented](#-methods-implemented)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Key Findings](#-key-findings)
- [Visualizations](#-visualizations)
- [Future Improvements](#-future-improvements)
- [References](#-references)
- [License](#-license)

---

## 🎯 Overview

Credit card fraud is a significant global problem, with losses exceeding **$30 billion annually**. This project tackles the challenge of detecting fraudulent transactions using multiple machine learning approaches, addressing key challenges such as:

- **Extreme Class Imbalance**: Only 0.17% of transactions are fraudulent (492 out of 284,807)
- **Real-time Requirements**: Models must be fast enough for production deployment
- **Cost Asymmetry**: Missing fraud is far more costly than false alarms
- **Evolving Patterns**: Fraudsters constantly adapt their techniques

### Objectives

1. Explore and compare **10 different ML/DL approaches**
2. Handle extreme class imbalance effectively
3. Identify the most important features for fraud detection
4. Provide actionable deployment recommendations

---

## 📊 Dataset

**Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

| Attribute | Value |
|-----------|-------|
| Total Transactions | 284,807 |
| Fraudulent | 492 (0.17%) |
| Legitimate | 284,315 (99.83%) |
| Features | 30 (V1-V28 PCA + Time + Amount) |
| Target | Class (0 = Legitimate, 1 = Fraud) |

> **Note**: Features V1-V28 are PCA-transformed to protect confidential information. Only `Time` and `Amount` are original features.

---

## 🏆 Results

### Final Model Rankings

| Rank | Model | Precision | Recall | F1-Score | PR-AUC | Composite Score |
|:----:|-------|:---------:|:------:|:--------:|:------:|:---------------:|
| 🥇 | **Time-Aware LightGBM** | 0.885 | 0.867 | **0.876** | 0.879 | **0.876** |
| 🥈 | XGBoost | 0.854 | 0.837 | 0.845 | 0.879 | 0.855 |
| 🥉 | SMOTE + XGBoost | 0.770 | **0.888** | 0.825 | 0.883 | 0.849 |
| 4 | Random Forest | **0.898** | 0.806 | 0.849 | 0.859 | 0.848 |
| 5 | Self-Training | 0.769 | 0.847 | 0.806 | 0.860 | 0.827 |
| 6 | Stacking Ensemble | 0.225 | 0.908 | 0.360 | 0.784 | 0.583 |
| 7 | Logistic Regression | 0.061 | 0.918 | 0.115 | 0.716 | 0.450 |
| 8 | Autoencoder | 0.029 | 0.898 | 0.056 | 0.239 | 0.277 |
| 9 | Hybrid Anomaly Ensemble | 0.369 | 0.245 | 0.294 | 0.167 | 0.254 |
| 10 | Isolation Forest | 0.206 | 0.265 | 0.232 | 0.128 | 0.205 |

> **Composite Score** = 0.4×F1 + 0.3×PR-AUC + 0.2×Recall + 0.1×Precision

### Key Metrics Achieved

| Metric | Best Value | Model |
|--------|:----------:|-------|
| **F1-Score** | 0.876 | Time-Aware LightGBM |
| **Recall** | 0.918 | Logistic Regression |
| **Precision** | 0.898 | Random Forest |
| **PR-AUC** | 0.883 | SMOTE + XGBoost |

---

## 🔬 Methods Implemented

### Basic Methods

| Method | Description | Key Hyperparameters |
|--------|-------------|---------------------|
| **Logistic Regression** | Baseline linear classifier | `class_weight='balanced'` |
| **Random Forest** | Ensemble of decision trees | `n_estimators=200`, `max_depth=15` |
| **XGBoost** | Gradient boosting with class weights | `scale_pos_weight=577`, `max_depth=6` |
| **SMOTE + XGBoost** | Synthetic oversampling + XGBoost | `sampling_strategy=0.5` |

### Advanced Methods (Unique Approaches) 🌟

| Method | Concept | Innovation |
|--------|---------|------------|
| **Isolation Forest** | Unsupervised anomaly detection | Trains only on legitimate transactions |
| **Autoencoder** | Neural network reconstruction | High reconstruction error = fraud |
| **Hybrid Anomaly Ensemble** | IF + Autoencoder combination | Weighted score fusion |
| **Self-Training** | Semi-supervised learning | Iteratively expands labeled data |
| **Stacking Ensemble** | XGB + LGBM + MLP + meta-learner | Combines diverse model strengths |
| **Time-Aware LightGBM** | Feature engineering + LGBM | 15 engineered features including interactions |

### Engineered Features (Time-Aware Model)

```python
# Temporal features
'Hour_Sin', 'Hour_Cos'    # Cyclical time encoding
'Is_Night', 'Is_Peak'     # Time-based flags

# Interaction features  
'V14_V12', 'V14_V10'      # Top predictor interactions
'V4_V11', 'V3_V14'

# Ratio features
'V14_Amount_Ratio'
'V12_Amount_Ratio'

# Polynomial features
'V14_Squared', 'V4_Squared', 'V12_Squared'
```

---

## 🛠 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
imbalanced-learn>=0.9.0
tensorflow>=2.8.0
scipy>=1.7.0
```

---

## 🚀 Usage

### Option 1: Run on Kaggle (Recommended)

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Create a new notebook
3. Add the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
4. Upload or copy the notebook code
5. Run all cells

### Option 2: Run Locally

```bash
# Download dataset from Kaggle
# Place creditcard.csv in the project directory

# Run the notebook
jupyter notebook credit-card-fraud-detection.ipynb

# Or run as Python script
python fraud_detection.py
```

### Quick Start Example

```python
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train_res, y_train_res)

# Predict
y_pred = model.predict(X_test)
```

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── 📓 credit-card-fraud-detection.ipynb   # Main Kaggle notebook
├── 📄 README.md                            # This file
├── 📄 requirements.txt                     # Python dependencies
├── 📄 LICENSE                              # MIT License
│
├── 📂 data/                                # Dataset (not included)
│   └── creditcard.csv
│
├── 📂 models/                              # Saved models (optional)
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   └── autoencoder.h5
│
├── 📂 figures/                             # Generated visualizations
│   ├── class_distribution.png
│   ├── feature_importance.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   └── model_comparison.png
│
└── 📂 src/                                 # Source code (optional)
    ├── preprocessing.py
    ├── models.py
    └── evaluation.py
```

---

## 💡 Key Findings

### 1. Feature Importance

The most discriminative features for fraud detection (via KS-test):

| Rank | Feature | KS Statistic | Direction |
|:----:|---------|:------------:|:---------:|
| 1 | V14 | 0.88 | ↓ Lower = Fraud |
| 2 | V4 | 0.81 | ↑ Higher = Fraud |
| 3 | V12 | 0.78 | ↓ Lower = Fraud |
| 4 | V10 | 0.76 | ↓ Lower = Fraud |
| 5 | V11 | 0.73 | ↑ Higher = Fraud |

### 2. Class Imbalance Strategies

| Strategy | Impact |
|----------|--------|
| **SMOTE** | Best recall improvement (+15%) |
| **Class Weights** | Good balance, no data augmentation |
| **Threshold Tuning** | Fine-tune precision/recall trade-off |

### 3. Model Selection Guidelines

| Priority | Recommended Model | Why |
|----------|-------------------|-----|
| **Balanced Performance** | Time-Aware LightGBM | Best F1-Score (0.876) |
| **Maximize Fraud Detection** | SMOTE + XGBoost | Highest Recall (0.888) |
| **Minimize False Alarms** | Random Forest | Highest Precision (0.898) |
| **Detect Novel Patterns** | Autoencoder | Unsupervised, adapts to new fraud |
| **Production Robustness** | Stacking Ensemble | Combines multiple model strengths |

### 4. Deployment Recommendations

```
┌─────────────────────────────────────────────────────────────┐
│                 PRODUCTION DEPLOYMENT                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PRIMARY SYSTEM: SMOTE + XGBoost                           │
│  ├── Real-time transaction scoring                          │
│  ├── Threshold: 0.5 (adjust based on business needs)       │
│  └── Latency: <10ms per prediction                         │
│                                                             │
│  SECONDARY LAYER: Autoencoder                               │
│  ├── Detect novel/unseen fraud patterns                    │
│  ├── Run in batch mode (daily/hourly)                      │
│  └── Alert on high reconstruction error                    │
│                                                             │
│  MONITORING:                                                │
│  ├── Precision: Target > 80%                               │
│  ├── Recall: Target > 85%                                  │
│  ├── FPR: Target < 0.1%                                    │
│  └── Model drift detection                                  │
│                                                             │
│  RETRAINING: Monthly with new fraud patterns               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Visualizations

### Class Distribution
The dataset is highly imbalanced with only 0.17% fraudulent transactions.

### Feature Correlations with Fraud
- **V14, V12, V10** → Strong negative correlation (lower values indicate fraud)
- **V4, V11** → Positive correlation (higher values indicate fraud)

### Model Comparison
The performance heatmap shows all 10 models across 5 metrics, enabling easy comparison.

### ROC & Precision-Recall Curves
PR-AUC is more informative than ROC-AUC for imbalanced datasets.

---

## 🔮 Future Improvements

1. **Real-time Feature Engineering**
   - Implement velocity features (transactions per hour/day)
   - Merchant category risk scores
   - Geographic anomaly detection

2. **Advanced Models**
   - Graph Neural Networks for transaction networks
   - Temporal Convolutional Networks
   - Transformer-based models

3. **Production Enhancements**
   - Model serving with FastAPI/Flask
   - A/B testing framework
   - Automated retraining pipeline
   - Explainability dashboard (SHAP values)

4. **Ensemble Optimization**
   - Bayesian model averaging
   - Online learning with concept drift detection

---

## 📚 References

1. **Dataset**: Dal Pozzolo, A., et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. IEEE SSCI.

2. **SMOTE**: Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.

3. **Isolation Forest**: Liu, F. T., et al. (2008). Isolation Forest. IEEE ICDM.

4. **XGBoost**: Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. ACM KDD.

5. **Autoencoders**: Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks. Science.

---

## 🙏 Acknowledgments

- **ULB Machine Learning Group** for providing the dataset
- **Kaggle** for hosting the competition and dataset
- **Worldline and Machine Learning Group** for data collection

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@ZeroZulu](https://github.com/ZeroZulu)
- Kaggle: [@zerol0l](https://www.kaggle.com/zerol0l)
- LinkedIn: [Shril Patel](https://www.linkedin.com/in/shril-patel-020504284/)

---

## ⭐ Star History

If you found this project helpful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/credit-card-fraud-detection&type=Date)](https://star-history.com/#yourusername/credit-card-fraud-detection&Date)

---

<p align="center">
  <b>If you found this project useful, please ⭐ star the repository!</b>
</p>
