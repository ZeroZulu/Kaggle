# ❤️ Heart Disease Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/neurocipher/heartdisease)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep-dive analysis of the Cleveland Heart Disease dataset with ML prediction models and an interactive risk calculator.

![Key Insights](figures/12_key_insights.png)

## 🚨 Key Discovery

**Asymptomatic patients have the HIGHEST disease rate (70.5%)** — not the lowest as expected. This "silent ischemia" finding suggests symptom-based screening may miss the majority of at-risk patients.

| Chest Pain Type | Disease Rate |
|-----------------|--------------|
| Typical Angina | 25.0% |
| Atypical Angina | 16.7% |
| Non-Anginal Pain | 21.5% |
| **Asymptomatic** | **70.5%** ⚠️ |

## 📊 Results

| Model | Accuracy |
|-------|----------|
| **Logistic Regression** | **85.2%** ✅ |
| Random Forest | 81.5% |
| Gradient Boosting | 81.5% |
| SVM | 81.5% |
| KNN | 79.6% |

**Other Findings:**
- Males have 2.4x higher risk than females
- Peak risk age: 55-65 years (60.8%)
- Lower max heart rate = higher risk

## 🛠️ Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt

# Run analysis
python analysis.py

# Or open notebook
jupyter notebook heart-disease-analysis.ipynb
```

## ⚠️ Disclaimer

For educational purposes only. Not intended for medical diagnosis or treatment.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">⭐ Star this repo if you found it helpful!</p>
