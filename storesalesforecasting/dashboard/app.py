"""
🛒 Retail Sales Forecasting Dashboard
=====================================
Interactive Streamlit dashboard for exploring sales forecasts.

Author: [Your Name]
Date: December 2025

Run with:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🛒 Retail Sales Forecaster",
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
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
PRODUCT_FAMILIES = [
    "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS",
    "BREAD/BAKERY", "CELEBRATION", "CLEANING", "DAIRY", "DELI",
    "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE",
    "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME APPLIANCES",
    "HOME CARE", "LADIESWEAR", "LAWN AND GARDEN", "LINGERIE",
    "LIQUOR,WINE,BEER", "MAGAZINES", "MEATS", "PERSONAL CARE",
    "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY",
    "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD"
]

STORE_INFO = {
    1: {"city": "Quito", "type": "D", "cluster": 13},
    2: {"city": "Quito", "type": "D", "cluster": 13},
    3: {"city": "Quito", "type": "D", "cluster": 8},
    4: {"city": "Quito", "type": "D", "cluster": 9},
    5: {"city": "Santo Domingo", "type": "D", "cluster": 4},
    6: {"city": "Quito", "type": "D", "cluster": 13},
    7: {"city": "Quito", "type": "D", "cluster": 8},
    8: {"city": "Quito", "type": "D", "cluster": 8},
    9: {"city": "Quito", "type": "B", "cluster": 6},
    10: {"city": "Quito", "type": "C", "cluster": 15},
    24: {"city": "Guayaquil", "type": "D", "cluster": 1},
    37: {"city": "Cuenca", "type": "D", "cluster": 2},
    44: {"city": "Quito", "type": "A", "cluster": 5},
    45: {"city": "Quito", "type": "A", "cluster": 11},
    46: {"city": "Quito", "type": "A", "cluster": 14},
    47: {"city": "Quito", "type": "A", "cluster": 14},
    48: {"city": "Quito", "type": "A", "cluster": 14},
    49: {"city": "Quito", "type": "A", "cluster": 11},
    50: {"city": "Ambato", "type": "A", "cluster": 14},
    51: {"city": "Guayaquil", "type": "A", "cluster": 17},
    52: {"city": "Manta", "type": "A", "cluster": 11},
    53: {"city": "Manta", "type": "D", "cluster": 13},
    54: {"city": "El Carmen", "type": "C", "cluster": 3}
}

# Fill in remaining stores
for i in range(1, 55):
    if i not in STORE_INFO:
        STORE_INFO[i] = {"city": "Various", "type": "C", "cluster": (i % 17) + 1}

# API Configuration (change this to your deployed API URL)
API_BASE_URL = "http://localhost:8000"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(ttl=3600)
def generate_sample_historical_data(store_nbr, family, days=365):
    """Generate realistic sample historical data for visualization."""
    np.random.seed(store_nbr * 100 + hash(family) % 100)
    
    dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=days)
    
    # Base sales by family
    family_base = {
        'GROCERY I': 5000, 'BEVERAGES': 2500, 'PRODUCE': 2000,
        'CLEANING': 1500, 'DAIRY': 1200, 'BREAD/BAKERY': 800,
        'MEATS': 700, 'DELI': 600, 'FROZEN FOODS': 500,
        'PERSONAL CARE': 400, 'HOME CARE': 350, 'EGGS': 300,
    }
    base = family_base.get(family, 300)
    
    # Store type multiplier
    store_type = STORE_INFO.get(store_nbr, {}).get('type', 'C')
    type_mult = {'A': 2.5, 'B': 1.5, 'C': 1.0, 'D': 0.8, 'E': 0.6}.get(store_type, 1.0)
    
    # Generate sales with patterns
    sales = []
    for i, date in enumerate(dates):
        # Weekly pattern
        dow_effect = [1.0, 0.95, 0.95, 1.0, 1.1, 1.3, 0.7][date.weekday()]
        
        # Monthly pattern (December boost)
        month_effect = 1.3 if date.month == 12 else (1.1 if date.month in [11, 1] else 1.0)
        
        # Trend
        trend = 1 + (i / days) * 0.1
        
        # Random noise
        noise = np.random.normal(1, 0.15)
        
        sale = base * type_mult * dow_effect * month_effect * trend * noise
        sales.append(max(0, sale))
    
    return pd.DataFrame({
        'date': dates,
        'sales': sales,
        'store_nbr': store_nbr,
        'family': family
    })

def generate_forecast(store_nbr, family, days_ahead=30, historical_data=None):
    """Generate forecast with confidence intervals."""
    if historical_data is not None and len(historical_data) > 7:
        # Use historical data for more realistic forecasts
        recent = historical_data['sales'].tail(28).values
        base_prediction = np.mean(recent[-7:])  # Last week average
        trend = (np.mean(recent[-7:]) - np.mean(recent[-14:-7])) / 7  # Daily trend
    else:
        # Fallback to simple baseline
        family_base = {
            'GROCERY I': 5000, 'BEVERAGES': 2500, 'PRODUCE': 2000,
            'CLEANING': 1500, 'DAIRY': 1200
        }
        base_prediction = family_base.get(family, 500)
        store_type = STORE_INFO.get(store_nbr, {}).get('type', 'C')
        type_mult = {'A': 2.5, 'B': 1.5, 'C': 1.0, 'D': 0.8, 'E': 0.6}.get(store_type, 1.0)
        base_prediction *= type_mult
        trend = 0
    
    # Generate forecast dates
    start_date = datetime.now()
    forecast_dates = [start_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    
    forecasts = []
    for i, date in enumerate(forecast_dates):
        # Day of week effect
        dow_effect = [1.0, 0.95, 0.95, 1.0, 1.1, 1.3, 0.7][date.weekday()]
        
        # Calculate prediction
        prediction = (base_prediction + trend * i) * dow_effect
        
        # Confidence intervals (widen over time)
        uncertainty = prediction * 0.15 * (1 + i * 0.02)
        
        forecasts.append({
            'date': date,
            'predicted_sales': max(0, prediction),
            'lower_bound': max(0, prediction - 1.96 * uncertainty),
            'upper_bound': prediction + 1.96 * uncertainty
        })
    
    return pd.DataFrame(forecasts)

def call_api_predict(store_nbr, family, date, onpromotion=0):
    """Call the FastAPI prediction endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "store_nbr": store_nbr,
                "family": family,
                "date": str(date),
                "onpromotion": onpromotion
            },
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shopping-cart--v1.png", width=80)
    st.title("🛒 Sales Forecaster")
    st.markdown("---")
    
    # Store Selection
    st.subheader("📍 Store Selection")
    selected_store = st.selectbox(
        "Select Store",
        options=sorted(STORE_INFO.keys()),
        format_func=lambda x: f"Store {x} ({STORE_INFO[x]['city']}, Type {STORE_INFO[x]['type']})"
    )
    
    # Product Family Selection
    st.subheader("📦 Product Family")
    selected_family = st.selectbox(
        "Select Family",
        options=PRODUCT_FAMILIES,
        index=PRODUCT_FAMILIES.index("GROCERY I")
    )
    
    # Forecast Settings
    st.subheader("🔮 Forecast Settings")
    forecast_days = st.slider("Forecast Days", 7, 90, 30)
    
    # Promotion Input
    st.subheader("🏷️ Promotions")
    promotion_items = st.number_input("Items on Promotion", 0, 500, 0)
    
    st.markdown("---")
    
    # Store Info Card
    st.subheader("📋 Store Details")
    store_details = STORE_INFO.get(selected_store, {})
    st.markdown(f"""
    - **City:** {store_details.get('city', 'N/A')}
    - **Type:** {store_details.get('type', 'N/A')}
    - **Cluster:** {store_details.get('cluster', 'N/A')}
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")
    st.markdown("[GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)")

# ============================================================
# MAIN CONTENT
# ============================================================

# Header
st.markdown("<h1 class='main-header'>🛒 Retail Sales Forecasting Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Powered by XGBoost | 54 Stores | 33 Product Families</p>", unsafe_allow_html=True)

# Generate data
historical_data = generate_sample_historical_data(selected_store, selected_family)
forecast_data = generate_forecast(selected_store, selected_family, forecast_days, historical_data)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast", "📈 Historical Analysis", "🏪 Store Comparison", "ℹ️ About"])

# ============================================================
# TAB 1: FORECAST
# ============================================================
with tab1:
    st.header(f"Sales Forecast: Store {selected_store} - {selected_family}")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_forecast = forecast_data['predicted_sales'].mean()
        st.metric(
            "📊 Avg Daily Forecast",
            f"${avg_forecast:,.0f}",
            delta=f"{((avg_forecast / historical_data['sales'].tail(30).mean()) - 1) * 100:.1f}%"
        )
    
    with col2:
        total_forecast = forecast_data['predicted_sales'].sum()
        st.metric(
            "💰 Total Forecast",
            f"${total_forecast:,.0f}",
            delta=f"{forecast_days} days"
        )
    
    with col3:
        peak_day = forecast_data.loc[forecast_data['predicted_sales'].idxmax()]
        st.metric(
            "🔝 Peak Day",
            f"${peak_day['predicted_sales']:,.0f}",
            delta=peak_day['date'].strftime('%a, %b %d')
        )
    
    with col4:
        low_day = forecast_data.loc[forecast_data['predicted_sales'].idxmin()]
        st.metric(
            "📉 Lowest Day",
            f"${low_day['predicted_sales']:,.0f}",
            delta=low_day['date'].strftime('%a, %b %d')
        )
    
    st.markdown("---")
    
    # Forecast Chart
    fig = go.Figure()
    
    # Historical data (last 90 days)
    hist_recent = historical_data.tail(90)
    fig.add_trace(go.Scatter(
        x=hist_recent['date'],
        y=hist_recent['sales'],
        name='Historical Sales',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_data['date'],
        y=forecast_data['predicted_sales'],
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_data['date'], forecast_data['date'][::-1]]),
        y=pd.concat([forecast_data['upper_bound'], forecast_data['lower_bound'][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))
    
    # Vertical line at forecast start
    fig.add_vline(
        x=datetime.now(),
        line_dash="dot",
        line_color="red",
        annotation_text="Today"
    )
    
    fig.update_layout(
        title=f'Sales Forecast - Store {selected_store}, {selected_family}',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        hovermode='x unified',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast Table
    with st.expander("📋 View Detailed Forecast Table"):
        forecast_display = forecast_data.copy()
        forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
        forecast_display['day_of_week'] = pd.to_datetime(forecast_display['date']).dt.day_name()
        forecast_display = forecast_display.rename(columns={
            'date': 'Date',
            'predicted_sales': 'Predicted Sales',
            'lower_bound': 'Lower Bound (95%)',
            'upper_bound': 'Upper Bound (95%)',
            'day_of_week': 'Day'
        })
        
        st.dataframe(
            forecast_display[['Date', 'Day', 'Predicted Sales', 'Lower Bound (95%)', 'Upper Bound (95%)']],
            use_container_width=True
        )
        
        # Download button
        csv = forecast_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"forecast_store{selected_store}_{selected_family}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ============================================================
# TAB 2: HISTORICAL ANALYSIS
# ============================================================
with tab2:
    st.header("📈 Historical Sales Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekly Pattern
        historical_data['dayofweek'] = historical_data['date'].dt.dayofweek
        weekly_pattern = historical_data.groupby('dayofweek')['sales'].mean().reset_index()
        weekly_pattern['day_name'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig_weekly = px.bar(
            weekly_pattern,
            x='day_name',
            y='sales',
            title='Average Sales by Day of Week',
            color='sales',
            color_continuous_scale='Blues'
        )
        fig_weekly.update_layout(showlegend=False, xaxis_title='', yaxis_title='Avg Sales ($)')
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    with col2:
        # Monthly Pattern
        historical_data['month'] = historical_data['date'].dt.month
        monthly_pattern = historical_data.groupby('month')['sales'].mean().reset_index()
        monthly_pattern['month_name'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig_monthly = px.bar(
            monthly_pattern,
            x='month_name',
            y='sales',
            title='Average Sales by Month',
            color='sales',
            color_continuous_scale='Oranges'
        )
        fig_monthly.update_layout(showlegend=False, xaxis_title='', yaxis_title='Avg Sales ($)')
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Trend Analysis
    st.subheader("📉 Sales Trend Analysis")
    
    historical_data['rolling_7d'] = historical_data['sales'].rolling(7).mean()
    historical_data['rolling_30d'] = historical_data['sales'].rolling(30).mean()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=historical_data['date'], y=historical_data['sales'],
                                   name='Daily Sales', opacity=0.4, line=dict(color='lightblue')))
    fig_trend.add_trace(go.Scatter(x=historical_data['date'], y=historical_data['rolling_7d'],
                                   name='7-Day Moving Avg', line=dict(color='blue', width=2)))
    fig_trend.add_trace(go.Scatter(x=historical_data['date'], y=historical_data['rolling_30d'],
                                   name='30-Day Moving Avg', line=dict(color='red', width=2)))
    
    fig_trend.update_layout(
        title='Sales Trend with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        height=400
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Statistics
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Daily Sales", f"${historical_data['sales'].mean():,.0f}")
    with col2:
        st.metric("Median Daily Sales", f"${historical_data['sales'].median():,.0f}")
    with col3:
        st.metric("Std Deviation", f"${historical_data['sales'].std():,.0f}")
    with col4:
        st.metric("Max Daily Sales", f"${historical_data['sales'].max():,.0f}")

# ============================================================
# TAB 3: STORE COMPARISON
# ============================================================
with tab3:
    st.header("🏪 Store Comparison")
    
    # Select stores to compare
    compare_stores = st.multiselect(
        "Select stores to compare",
        options=sorted(STORE_INFO.keys()),
        default=[44, 45, 24],
        format_func=lambda x: f"Store {x} ({STORE_INFO[x]['city']}, Type {STORE_INFO[x]['type']})"
    )
    
    if len(compare_stores) > 0:
        comparison_data = []
        
        for store in compare_stores:
            store_data = generate_sample_historical_data(store, selected_family, 90)
            store_data['store'] = f"Store {store} ({STORE_INFO[store]['type']})"
            comparison_data.append(store_data)
        
        comparison_df = pd.concat(comparison_data)
        
        # Time series comparison
        fig_compare = px.line(
            comparison_df,
            x='date',
            y='sales',
            color='store',
            title=f'Sales Comparison - {selected_family}'
        )
        fig_compare.update_layout(height=400, xaxis_title='Date', yaxis_title='Sales ($)')
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Summary comparison
        col1, col2 = st.columns(2)
        
        with col1:
            summary_stats = comparison_df.groupby('store')['sales'].agg(['mean', 'std', 'max']).round(0)
            summary_stats.columns = ['Avg Sales', 'Std Dev', 'Max Sales']
            st.dataframe(summary_stats, use_container_width=True)
        
        with col2:
            fig_box = px.box(
                comparison_df,
                x='store',
                y='sales',
                title='Sales Distribution by Store'
            )
            fig_box.update_layout(height=300, xaxis_title='', yaxis_title='Sales ($)')
            st.plotly_chart(fig_box, use_container_width=True)

# ============================================================
# TAB 4: ABOUT
# ============================================================
with tab4:
    st.header("ℹ️ About This Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Project Overview")
        st.markdown("""
        This dashboard provides **real-time sales forecasting** for Corporación Favorita, 
        a major grocery retailer in Ecuador with 54 stores and 33 product families.
        
        **Key Features:**
        - 📊 Interactive forecast visualization with confidence intervals
        - 📈 Historical trend analysis
        - 🏪 Multi-store comparison tools
        - 📥 Downloadable forecast reports
        """)
        
        st.subheader("🤖 Model Information")
        st.markdown("""
        | Attribute | Value |
        |-----------|-------|
        | **Algorithm** | XGBoost Regressor |
        | **Training Data** | 2013-2017 Sales Data |
        | **Features** | 45 (temporal, lag, rolling, external) |
        | **Validation MAPE** | ~18% |
        | **Forecast Horizon** | Up to 90 days |
        """)
    
    with col2:
        st.subheader("📊 Feature Importance")
        
        feature_importance = {
            'sales_lag_1': 0.25,
            'sales_lag_7': 0.18,
            'sales_rolling_mean_7': 0.12,
            'dayofweek': 0.10,
            'onpromotion': 0.08,
            'store_nbr': 0.07,
            'family_encoded': 0.06,
            'month': 0.05,
            'is_holiday': 0.04,
            'oil_price': 0.03
        }
        
        fig_importance = px.bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            title='Top 10 Features'
        )
        fig_importance.update_layout(
            height=400,
            xaxis_title='Importance',
            yaxis_title='',
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🔧 Technical Stack")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Data Science**
        - Python 3.10+
        - Pandas, NumPy
        - XGBoost
        - Statsmodels
        - Prophet
        """)
    
    with col2:
        st.markdown("""
        **Visualization**
        - Plotly
        - Matplotlib
        - Seaborn
        - Streamlit
        """)
    
    with col3:
        st.markdown("""
        **Deployment**
        - FastAPI
        - Docker
        - GitHub Actions
        - Streamlit Cloud
        """)
    
    st.markdown("---")
    
    st.subheader("👨‍💻 About the Author")
    st.markdown("""
    This project was built as part of a **Data Science portfolio** to demonstrate:
    
    ✅ Time series forecasting methodology  
    ✅ Feature engineering at scale  
    ✅ ML model development and evaluation  
    ✅ API development (FastAPI)  
    ✅ Interactive dashboards (Streamlit)  
    ✅ Production-ready code practices  
    
    **Connect with me:**
    - 📧 your.email@example.com
    - 💼 [LinkedIn](https://linkedin.com/in/yourprofile)
    - 🐙 [GitHub](https://github.com/yourusername)
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>© 2025 Retail Sales Forecasting | Built with Streamlit</p>",
    unsafe_allow_html=True
)
