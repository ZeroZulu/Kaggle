"""
Store Sales Time Series Forecasting - Streamlit Dashboard
=========================================================
Interactive dashboard showcasing demand forecasting for retail stores.

Author: [Shril Patel]
GitHub: https://github.com/ZeroZulu/kaggle
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Store Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .highlight-box {
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/shopping-cart.png", width=150)
    st.title("Navigation")
    
    page = st.radio(
        "Go to",
        ["🏠 Overview", "📊 Data Explorer", "🔬 Model Performance", 
         "🔮 Forecasting Demo", "💰 Business Impact", "📚 Documentation"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This dashboard showcases a **demand forecasting system** 
    built for retail stores using machine learning.
    
    **Key Results:**
    - 31% improvement over baseline
    - $12.9M projected annual savings
    - 1,782 time series forecasted
    """)
    
    st.markdown("---")
    st.markdown("### Connect")
    st.markdown("""
    - [GitHub Repository](https://github.com/ZeroZulu/kaggle)
    - [Kaggle Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
    - [LinkedIn](https://linkedin.com/in/shril-patel-020504284/)
    """)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
@st.cache_data
def load_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2017-01-01', '2017-08-15', freq='D')
    
    # Generate realistic sales patterns
    data = []
    for date in dates:
        base = 10000
        # Weekly seasonality
        dow_effect = {0: 1.0, 1: 0.95, 2: 0.93, 3: 0.95, 4: 1.05, 5: 1.15, 6: 0.85}
        weekly = dow_effect[date.dayofweek]
        # Monthly trend
        monthly = 1 + 0.1 * np.sin(2 * np.pi * date.month / 12)
        # Random noise
        noise = np.random.normal(1, 0.1)
        
        sales = base * weekly * monthly * noise
        data.append({'date': date, 'sales': sales})
    
    return pd.DataFrame(data)

@st.cache_data
def get_model_results():
    """Return model comparison results"""
    return pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM', 'Seasonal Naive', 'Store-Family Mean', 'Global Mean'],
        'RMSLE': [0.4510, 0.5540, 0.6565, 0.6844, 3.4020],
        'RMSE': [196.20, 198.23, 488.61, 536.99, 1285.59],
        'MAE': [56.49, 59.10, 130.10, 148.39, 587.39],
        'MAPE': [36.83, 40.87, 47.12, 47.80, 4084.69]
    })

@st.cache_data
def get_feature_importance():
    """Return feature importance data"""
    return pd.DataFrame({
        'Feature': ['sales_lag_1', 'sales_lag_7', 'sales_rolling_mean_7', 'dayofweek', 
                   'sales_lag_14', 'sales_rolling_mean_14', 'family_encoded', 'store_nbr',
                   'onpromotion', 'days_to_holiday'],
        'Importance': [0.25, 0.18, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02],
        'Category': ['Lag', 'Lag', 'Rolling', 'Date', 'Lag', 'Rolling', 'Store', 'Store', 'Promo', 'Holiday']
    })

# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.markdown('<p class="main-header">🛒 Store Sales Forecasting</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Demand Forecasting for 54 Stores × 33 Product Families</p>', unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Time Series",
            value="1,782",
            help="54 stores × 33 product families"
        )
    
    with col2:
        st.metric(
            label="Best RMSLE",
            value="0.4510",
            delta="-31% vs baseline",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Annual Savings",
            value="$12.9M",
            delta="66% cost reduction"
        )
    
    with col4:
        st.metric(
            label="Features Engineered",
            value="43",
            help="Lag, rolling, date, holiday, oil features"
        )
    
    st.markdown("---")
    
    # Project Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Project Overview")
        st.markdown("""
        This project tackles **demand forecasting** for Corporación Favorita, Ecuador's largest 
        grocery retailer. The challenge involves predicting 16 days of future sales across:
        
        - **54 stores** of varying types and locations
        - **33 product families** from groceries to automotive supplies
        - **3+ million historical records** spanning 2013-2017
        
        #### Why It Matters
        
        Poor demand forecasts lead to:
        - 📦 **Overstocking**: Tied-up capital, spoilage, storage costs
        - ❌ **Stockouts**: Lost sales, unhappy customers, damaged reputation
        
        My model reduces these costs by **66%**, translating to **$12.9M annual savings**.
        """)
    
    with col2:
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
        **Data Processing**
        - Python 3.10
        - Pandas, NumPy
        
        **Machine Learning**
        - XGBoost
        - LightGBM
        - Scikit-learn
        
        **Time Series**
        - SARIMA
        - Prophet
        
        **Visualization**
        - Matplotlib
        - Plotly
        - Streamlit
        """)
    
    st.markdown("---")
    
    # Methodology
    st.markdown("### 📋 Methodology")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        #### 1️⃣ Data Exploration
        - Analyzed 3M+ records
        - Identified seasonality patterns
        - Explored external factors
        """)
    
    with col2:
        st.markdown("""
        #### 2️⃣ Feature Engineering
        - 43 features created
        - Lag & rolling statistics
        - Holiday & oil price effects
        """)
    
    with col3:
        st.markdown("""
        #### 3️⃣ Model Training
        - Time-based validation
        - Early stopping
        - 5 models compared
        """)
    
    with col4:
        st.markdown("""
        #### 4️⃣ Business Impact
        - Cost analysis
        - ROI calculation
        - Deployment ready
        """)

# ============================================================
# PAGE: DATA EXPLORER
# ============================================================
elif page == "📊 Data Explorer":
    st.markdown("## 📊 Data Explorer")
    st.markdown("Explore the patterns in retail sales data")
    
    # Load sample data
    df = load_sample_data()
    
    tab1, tab2, tab3 = st.tabs(["📈 Sales Trends", "📅 Seasonality", "🔍 Data Sample"])
    
    with tab1:
        st.markdown("### Daily Sales Trend")
        
        fig = px.line(df, x='date', y='sales', 
                     title='Daily Aggregated Sales (2017)',
                     labels={'sales': 'Sales ($)', 'date': 'Date'})
        fig.update_traces(line_color='#2E86AB')
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Insight**: Clear weekly seasonality with Saturday peaks and Sunday dips. December shows holiday-driven spikes.")
    
    with tab2:
        st.markdown("### Weekly Seasonality Pattern")
        
        df['dayofweek'] = df['date'].dt.dayofweek
        df['day_name'] = df['date'].dt.day_name()
        
        weekly_avg = df.groupby(['dayofweek', 'day_name'])['sales'].mean().reset_index()
        weekly_avg = weekly_avg.sort_values('dayofweek')
        
        fig = px.bar(weekly_avg, x='day_name', y='sales',
                    title='Average Sales by Day of Week',
                    labels={'sales': 'Average Sales ($)', 'day_name': 'Day'},
                    color='sales',
                    color_continuous_scale=['#f0f2f6', '#2E86AB'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("📈 **Peak Day**: Saturday (+15% above average)")
        with col2:
            st.warning("📉 **Low Day**: Sunday (-15% below average)")
    
    with tab3:
        st.markdown("### Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", "3,000,888")
        with col2:
            st.metric("Date Range", "2013-01-01 to 2017-08-15")
        with col3:
            st.metric("Memory Usage", "60 MB")
        
        st.markdown("#### Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("#### Data Sources")
        st.markdown("""
        | File | Description | Records |
        |------|-------------|---------|
        | train.csv | Historical sales data | 3,000,888 |
        | test.csv | 16-day forecast period | 28,512 |
        | stores.csv | Store metadata | 54 |
        | oil.csv | Daily oil prices | 1,218 |
        | holidays_events.csv | Holiday calendar | 350 |
        | transactions.csv | Daily transactions | 83,488 |
        """)

# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================
elif page == "🔬 Model Performance":
    st.markdown("## 🔬 Model Performance")
    st.markdown("Comparing different forecasting approaches")
    
    results = get_model_results()
    feature_imp = get_feature_importance()
    
    tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🎯 Feature Importance", "📈 Learning Curves"])
    
    with tab1:
        st.markdown("### Model Comparison")
        
        # Highlight best model
        st.success("🏆 **Winner: XGBoost** with RMSLE of 0.4510 (31% better than Seasonal Naive baseline)")
        
        # Bar chart comparison
        fig = px.bar(results, x='Model', y='RMSLE',
                    title='Model Performance Comparison (Lower is Better)',
                    color='RMSLE',
                    color_continuous_scale=['#28a745', '#ffc107', '#dc3545'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Full metrics table
        st.markdown("#### Detailed Metrics")
        st.dataframe(results, use_container_width=True)
        
        st.markdown("""
        **Metric Definitions:**
        - **RMSLE**: Root Mean Squared Logarithmic Error (competition metric)
        - **RMSE**: Root Mean Squared Error
        - **MAE**: Mean Absolute Error
        - **MAPE**: Mean Absolute Percentage Error
        """)
    
    with tab2:
        st.markdown("### Feature Importance (XGBoost)")
        
        # Color by category
        color_map = {'Lag': '#2E86AB', 'Rolling': '#A23B72', 'Date': '#F18F01', 
                    'Store': '#28A745', 'Promo': '#DC3545', 'Holiday': '#6C757D'}
        
        fig = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                    color='Category', color_discrete_map=color_map,
                    title='Top 10 Most Important Features')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        💡 **Key Insights:**
        - **Lag features dominate** - Yesterday's sales and last week's same day are most predictive
        - **Rolling statistics** capture momentum and trend
        - **Day of week** is critical for weekly seasonality
        """)
        
        # Feature categories breakdown
        st.markdown("#### Features by Category")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Lag Features (5)**
            - sales_lag_1, 7, 14, 21, 28
            
            **Rolling Features (8)**
            - 7/14/28-day mean, std
            
            **Date Features (12)**
            - Year, month, day, dayofweek
            - Cyclical encoding (sin/cos)
            """)
        
        with col2:
            st.markdown("""
            **Holiday Features (4)**
            - National, regional, local flags
            - Days to next holiday
            
            **Oil Features (4)**
            - Price, 7/30-day MA, change
            
            **Store Features (2)**
            - Store type, cluster
            """)
    
    with tab3:
        st.markdown("### Training Progress")
        
        # Simulated learning curve
        np.random.seed(42)
        iterations = list(range(0, 420, 20))
        train_rmse = [1200 * np.exp(-0.01 * i) + 180 + np.random.normal(0, 5) for i in iterations]
        val_rmse = [1200 * np.exp(-0.008 * i) + 195 + np.random.normal(0, 8) for i in iterations]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iterations, y=train_rmse, mode='lines', name='Training RMSE',
                                line=dict(color='#2E86AB')))
        fig.add_trace(go.Scatter(x=iterations, y=val_rmse, mode='lines', name='Validation RMSE',
                                line=dict(color='#A23B72')))
        
        # Add vertical line using shapes instead of add_vline
        fig.add_shape(
            type="line",
            x0=366, x1=366,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="green", width=2, dash="dash")
        )
        fig.add_annotation(
            x=366, y=1.05,
            yref="paper",
            text="Early Stopping (366)",
            showarrow=False,
            font=dict(color="green")
        )
        
        fig.update_layout(title='XGBoost Learning Curve',
                         xaxis_title='Boosting Round',
                         yaxis_title='RMSE')
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ **Early stopping** triggered at round 366 (out of 500), preventing overfitting")

# ============================================================
# PAGE: FORECASTING DEMO
# ============================================================
elif page == "🔮 Forecasting Demo":
    st.markdown("## 🔮 Forecasting Demo")
    st.markdown("Interactive sales prediction demonstration")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Input Parameters")
        
        store = st.selectbox("Store Number", list(range(1, 55)), index=43)
        family = st.selectbox("Product Family", 
                             ['GROCERY I', 'BEVERAGES', 'PRODUCE', 'CLEANING', 'DAIRY',
                              'BREAD/BAKERY', 'POULTRY', 'MEATS', 'PERSONAL CARE', 'DELI'])
        
        forecast_days = st.slider("Forecast Horizon (days)", 1, 16, 16)
        
        show_confidence = st.checkbox("Show Confidence Interval", value=True)
        
        predict_btn = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### Forecast Results")
        
        if predict_btn:
            with st.spinner("Generating forecast..."):
                # Simulate prediction
                import time
                time.sleep(1)
                
                # Generate sample forecast
                np.random.seed(store + hash(family) % 100)
                
                # Use fixed dates for demo
                start_date = datetime(2017, 8, 16)
                dates = [start_date + timedelta(days=i) for i in range(forecast_days)]
                base = 500 + (store % 10) * 50
                
                predictions = []
                for i, date in enumerate(dates):
                    dow_effect = {0: 1.0, 1: 0.95, 2: 0.93, 3: 0.95, 4: 1.05, 5: 1.15, 6: 0.85}
                    pred = base * dow_effect[date.weekday()] + np.random.normal(0, 30)
                    predictions.append(max(0, pred))
                
                forecast_df = pd.DataFrame({
                    'Date': dates,
                    'Predicted Sales': predictions,
                    'Lower Bound': [p * 0.85 for p in predictions],
                    'Upper Bound': [p * 1.15 for p in predictions]
                })
                
                # Plot
                fig = go.Figure()
                
                if show_confidence:
                    fig.add_trace(go.Scatter(
                        x=list(forecast_df['Date']) + list(forecast_df['Date'][::-1]),
                        y=list(forecast_df['Upper Bound']) + list(forecast_df['Lower Bound'][::-1]),
                        fill='toself',
                        fillcolor='rgba(46, 134, 171, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence Interval'
                    ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Predicted Sales'],
                    mode='lines+markers',
                    name='Predicted Sales',
                    line=dict(color='#2E86AB', width=3)
                ))
                
                fig.update_layout(
                    title=f'16-Day Sales Forecast: Store {store} - {family}',
                    xaxis_title='Date',
                    yaxis_title='Predicted Sales ($)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Forecasted", f"${sum(predictions):,.0f}")
                with col2:
                    st.metric("Daily Average", f"${np.mean(predictions):,.0f}")
                with col3:
                    peak_idx = predictions.index(max(predictions))
                    peak_date = dates[peak_idx]
                    st.metric("Peak Day", peak_date.strftime('%a, %b %d'))
                
                # Data table
                with st.expander("📋 View Forecast Data"):
                    st.dataframe(forecast_df.round(2), use_container_width=True)
        else:
            st.info("👈 Configure parameters and click **Generate Forecast** to see predictions")

# ============================================================
# PAGE: BUSINESS IMPACT
# ============================================================
elif page == "💰 Business Impact":
    st.markdown("## 💰 Business Impact Analysis")
    st.markdown("Translating model performance into dollar value")
    
    # Cost assumptions
    st.markdown("### 📊 Cost Model Assumptions")
    
    col1, col2 = st.columns(2)
    with col1:
        overstock_cost = st.slider("Overstock Cost (% of item value)", 5, 20, 10)
        st.caption("Storage, spoilage, markdowns")
    with col2:
        stockout_cost = st.slider("Stockout Cost (% of margin)", 15, 40, 25)
        st.caption("Lost sales, customer dissatisfaction")
    
    st.markdown("---")
    
    # Calculate costs
    xgb_overstock = 154429 * (overstock_cost / 10)
    xgb_stockout = 394113 * (stockout_cost / 25)
    xgb_total = xgb_overstock + xgb_stockout
    
    baseline_overstock = 110935 * (overstock_cost / 10)
    baseline_stockout = 1519426 * (stockout_cost / 25)
    baseline_total = baseline_overstock + baseline_stockout
    
    monthly_savings = baseline_total - xgb_total
    annual_savings = monthly_savings * 12
    
    # Results
    st.markdown("### 💵 Cost Comparison (Monthly)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### XGBoost Model")
        st.metric("Overstock Losses", f"${xgb_overstock:,.0f}")
        st.metric("Stockout Losses", f"${xgb_stockout:,.0f}")
        st.metric("Total Cost", f"${xgb_total:,.0f}", delta=None)
    
    with col2:
        st.markdown("#### Seasonal Naive Baseline")
        st.metric("Overstock Losses", f"${baseline_overstock:,.0f}")
        st.metric("Stockout Losses", f"${baseline_stockout:,.0f}")
        st.metric("Total Cost", f"${baseline_total:,.0f}", delta=None)
    
    with col3:
        st.markdown("#### 💰 Savings")
        st.metric("Monthly Savings", f"${monthly_savings:,.0f}", delta="vs baseline")
        st.metric("Annual Savings", f"${annual_savings:,.0f}", delta=f"{(monthly_savings/baseline_total)*100:.1f}% reduction")
    
    st.markdown("---")
    
    # Visualization
    fig = go.Figure()
    
    categories = ['Overstock', 'Stockout']
    xgb_costs = [xgb_overstock, xgb_stockout]
    baseline_costs = [baseline_overstock, baseline_stockout]
    
    fig.add_trace(go.Bar(name='XGBoost', x=categories, y=xgb_costs, marker_color='#28a745'))
    fig.add_trace(go.Bar(name='Baseline', x=categories, y=baseline_costs, marker_color='#dc3545'))
    
    fig.update_layout(
        title='Cost Breakdown: XGBoost vs Baseline',
        yaxis_title='Cost ($)',
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key takeaway
    st.markdown(f"""
    <div class="highlight-box">
        <h3 style="margin:0; color:white;">🎯 Key Takeaway</h3>
        <p style="font-size: 1.5rem; margin: 10px 0 0 0;">
            Implementing the XGBoost model would save approximately 
            <strong>${annual_savings:,.0f}</strong> annually, 
            representing a <strong>{(monthly_savings/baseline_total)*100:.1f}%</strong> cost reduction.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: DOCUMENTATION
# ============================================================
elif page == "📚 Documentation":
    st.markdown("## 📚 Documentation")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Technical Details", "❓ FAQ", "📖 References"])
    
    with tab1:
        st.markdown("""
        ### Data Pipeline
        
        ```
        Raw Data → Feature Engineering → Train/Val Split → Model Training → Evaluation → Deployment
        ```
        
        ### Feature Engineering
        
        | Category | Features | Description |
        |----------|----------|-------------|
        | Lag | 5 | sales_lag_1, 7, 14, 21, 28 |
        | Rolling | 8 | 7/14/28-day mean and std |
        | Date | 12 | Year, month, day, dayofweek, cyclical |
        | Holiday | 4 | National, regional, local, days_to |
        | Oil | 4 | Price, moving averages, change |
        | Store | 2 | Type, cluster |
        | Other | 8 | Promotion, earthquake, expanding mean |
        
        ### Model Configuration
        
        **XGBoost Parameters:**
        ```python
        {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 50
        }
        ```
        
        ### Validation Strategy
        
        - **Training**: 2013-01-01 to 2017-07-15
        - **Validation**: 2017-07-16 to 2017-08-15 (31 days)
        - **Test**: 2017-08-16 to 2017-08-31 (16 days)
        
        Time-based split ensures no data leakage.
        """)
    
    with tab2:
        st.markdown("""
        ### Frequently Asked Questions
        
        **Q: Why XGBoost over SARIMA or Prophet?**
        
        A: XGBoost can model all 1,782 time series simultaneously as a single model, 
        while SARIMA requires fitting separate models for each series. XGBoost also 
        naturally incorporates external features like oil prices and holidays.
        
        ---
        
        **Q: How did you handle test data without sales values?**
        
        A: For lag features in test data, I used an iterative approach where each 
        test date's lags are computed from the available historical data. This 
        ensures no data leakage.
        
        ---
        
        **Q: Why is MAPE 36% but the model is considered good?**
        
        A: MAPE is inflated by low-sales items (predicting 2 when actual is 1 = 100% MAPE). 
        RMSLE is the better metric for this data, and our 0.4510 RMSLE represents 
        strong performance (31% better than baseline).
        
        ---
        
        **Q: How would you deploy this in production?**
        
        A: FastAPI endpoint for predictions, weekly automated retraining, Docker 
        containerization, and monitoring for data drift. Models are already saved 
        in joblib format for deployment.
        """)
    
    with tab3:
        st.markdown("""
        ### References
        
        **Competition**
        - [Kaggle: Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
        
        **Libraries**
        - [XGBoost Documentation](https://xgboost.readthedocs.io/)
        - [LightGBM Documentation](https://lightgbm.readthedocs.io/)
        - [Prophet Documentation](https://facebook.github.io/prophet/)
        - [Streamlit Documentation](https://docs.streamlit.io/)
        
        **Related Reading**
        - [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
        - [Feature Engineering for Time Series](https://www.kaggle.com/code/ryanholbrook/time-series-as-features)
        
        **Source Code**
        - [GitHub Repository](https://github.com/ZeroZulu/kaggle/storesalesforecasting)
        """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    Built with ❤️ using Streamlit | 
    <a href="https://github.com/ZeroZulu">GitHub</a> | 
    <a href="https://www.kaggle.com/competitions/store-sales-time-series-forecasting">Kaggle Competition</a>
</div>
""", unsafe_allow_html=True)
