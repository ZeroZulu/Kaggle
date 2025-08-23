**📉 Telecom Customer Churn Prediction**
-------
This project explores customer churn behavior in the telecom industry using machine learning. The goal is to identify patterns that contribute to customer churn and build a predictive model that can help telecom companies proactively reduce attrition.

**📊 Problem Statement**
-------------
Customer churn is a critical issue for telecom companies, where acquiring new customers is significantly more expensive than retaining existing ones. This project aims to:

* Understand key drivers behind customer churn.
* Build and evaluate machine learning models to predict churn.
* Provide actionable insights for customer retention strategies.

**🔍 Project Highlights**
-----
* **Data Preprocessing:** Cleaned and encoded the dataset for machine learning workflows.  
* **Exploratory Data Analysis (EDA):** Uncovered insights about churn distribution, customer demographics, service usage, and tenure.  
  🔗 **Interactive Dashboard:** [View on Tableau Public](https://public.tableau.com/shared/4DF7R79TS?:display_count=n&:origin=viz_share_link)  
* **Modeling:**
  * Trained and compared multiple classification models: Logistic Regression, Decision Tree, Random Forest, and XGBoost.
  * Performed hyperparameter tuning and cross-validation for performance optimization.
  * **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

**📈 Results**
--------
The **XGBoost model** emerged as the top performer with the best trade-off between precision and recall, providing a solid foundation for churn prediction with measurable business impact.

**📌 Future Work**
-----
* Integrate a real-time prediction API.
* Improve feature engineering with domain-specific attributes.
* Deploy the model using FastAPI or Flask.
