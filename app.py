import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crime Intelligence Pro | AI for Bharat",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PREMIUM PROFESSIONAL CSS ----------------
st.markdown("""
<style>
/* Main Background - Professional Gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 20%, #1e3c72 100%);
    color: #FFFFFF;
}

/* Sidebar - Dark Professional */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    border-right: 2px solid #e74c3c;
}

/* Sidebar Content */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #ecf0f1;
}

/* Make ALL text visible */
label, .stMarkdown, .stText, .stSelectbox label,
.stNumberInput label, .stRadio label, p, span, div {
    color: #ecf0f1 !important;
}

/* Main Title */
h1 {
    color: #ffffff !important;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    padding: 20px 0;
    text-align: center;
    background: linear-gradient(90deg, #e74c3c, #f39c12, #e74c3c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glow 3s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

/* Subheaders */
h2, h3 {
    color: #f39c12 !important;
    font-weight: 700 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    margin-top: 20px !important;
}

/* Metric Cards - Enhanced */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(231, 76, 60, 0.15), rgba(192, 57, 43, 0.1));
    border: 2px solid #e74c3c;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(231, 76, 60, 0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(231, 76, 60, 0.5);
}

[data-testid="stMetricValue"] {
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    color: #e74c3c !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}

[data-testid="stMetricLabel"] {
    color: #ecf0f1 !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricDelta"] {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
}

/* Buttons - Premium Style */
.stButton>button {
    background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
    color: white !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    border-radius: 15px !important;
    padding: 15px 40px !important;
    border: 3px solid #ffffff;
    box-shadow: 0 8px 20px rgba(231, 76, 60, 0.4);
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton>button:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 12px 30px rgba(231, 76, 60, 0.6);
    background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%) !important;
}

/* Download Button */
.stDownloadButton>button {
    background: linear-gradient(135deg, #27ae60 0%, #229954 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 12px 30px !important;
    border: 2px solid #ffffff;
    box-shadow: 0 6px 15px rgba(39, 174, 96, 0.4);
}

.stDownloadButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(39, 174, 96, 0.6);
}

/* Dataframe Styling */
[data-testid="stDataFrame"] {
    background: rgba(44, 62, 80, 0.6);
    border-radius: 15px;
    border: 2px solid #34495e;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}

/* Select Box & Input Fields */
.stSelectbox, .stNumberInput {
    background: rgba(52, 73, 94, 0.5);
    border-radius: 10px;
    padding: 5px;
}

/* Radio Buttons */
.stRadio > div {
    background: rgba(52, 73, 94, 0.3);
    border-radius: 12px;
    padding: 15px;
}

/* Divider */
hr {
    border: 2px solid #e74c3c;
    margin: 30px 0;
    box-shadow: 0 2px 10px rgba(231, 76, 60, 0.5);
}

/* Info/Warning/Success Boxes */
.stAlert {
    background: rgba(52, 73, 94, 0.8);
    border-radius: 12px;
    border-left: 5px solid #e74c3c;
    padding: 15px;
    color: #ecf0f1;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(44, 62, 80, 0.5);
    border-radius: 15px;
    padding: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(52, 73, 94, 0.6);
    border-radius: 10px;
    color: #ecf0f1;
    font-weight: 600;
    padding: 12px 24px;
    border: 2px solid transparent;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    color: white;
    border: 2px solid #ffffff;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(52, 73, 94, 0.4);
    border-radius: 12px;
    padding: 20px;
    border: 2px dashed #f39c12;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(52, 73, 94, 0.6);
    border-radius: 10px;
    color: #ecf0f1;
    font-weight: 600;
}

/* Progress Bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #e74c3c, #f39c12);
}

/* Success message animation */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.element-container {
    animation: slideIn 0.5s ease-out;
}

</style>
""", unsafe_allow_html=True)

# ---------------- ANIMATED TITLE WITH BADGE ----------------
st.markdown("""
<div style='text-align: center; margin-bottom: 30px;'>
    <h1 style='margin-bottom: 10px;'>🚨 CRIME INTELLIGENCE PRO</h1>
    <p style='font-size: 1.4rem; color: #f39c12; font-weight: 600; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
        AI-Powered Predictive Analytics for Strategic Crime Prevention
    </p>
    <div style='display: inline-block; background: linear-gradient(135deg, #e74c3c, #c0392b); 
                padding: 10px 25px; border-radius: 25px; margin-top: 15px; 
                box-shadow: 0 5px 15px rgba(231,76,60,0.5);'>
        <span style='color: white; font-weight: 700; font-size: 1rem;'>
            🏆 AI FOR BHARAT HACKATHON 2024
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- ENHANCED SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎯 CONTROL PANEL")
    st.markdown("---")
    
    # Navigation with icons
    menu = st.radio(
        "📊 Select Analysis Module",
        ["🔍 Crime Analysis & Insights", "🤖 AI Crime Prediction Engine"],
        index=0
    )
    
    st.markdown("---")
    
    # File Upload Section
    st.markdown("### 📁 Data Management")
    uploaded_file = st.file_uploader(
        "Upload Crime Dataset (CSV)", 
        type=["csv"],
        help="Upload your custom crime data or use sample data"
    )
    
    if uploaded_file:
        st.success("✅ File uploaded successfully!")
    else:
        st.info("📊 Using sample dataset")
    
    st.markdown("---")
    
    # Stats Summary in Sidebar
    st.markdown("### 📈 Quick Stats")

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data(file):
    if file:
        return pd.read_csv(file)
    else:
        years = np.arange(2015, 2024)
        return pd.DataFrame({
            "Year": years,
            "Theft": [400, 450, 480, 500, 550, 600, 650, 700, 720],
            "Assault": [300, 320, 350, 370, 400, 420, 450, 470, 500],
            "Burglary": [200, 210, 230, 240, 260, 280, 300, 320, 340],
            "Robbery": [150, 160, 175, 185, 200, 215, 230, 245, 260],
            "Fraud": [100, 120, 140, 160, 180, 200, 220, 240, 255]
        })

df = load_data(uploaded_file)

# Update sidebar stats
with st.sidebar:
    total_crimes = df.iloc[:, 1:].sum().sum()
    st.metric("Total Crimes Recorded", f"{int(total_crimes):,}")
    st.metric("Crime Categories", len(df.columns) - 1)
    st.metric("Years Analyzed", len(df))

st.markdown("---")

# =====================================================
# ============== CRIME ANALYSIS MODULE ================
# =====================================================
if "Crime Analysis" in menu:
    
    # Header Section
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(231,76,60,0.2), rgba(192,57,43,0.2)); 
                border-left: 6px solid #e74c3c; padding: 20px; border-radius: 15px; margin-bottom: 30px;'>
        <h2 style='color: #f39c12; margin: 0;'>📊 COMPREHENSIVE CRIME TREND ANALYSIS</h2>
        <p style='color: #ecf0f1; margin-top: 10px; font-size: 1.1rem;'>
            Advanced statistical analysis and visualization of crime patterns
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Crime Type Selection
    col_select1, col_select2 = st.columns([2, 1])
    with col_select1:
        crime_type = st.selectbox(
            "🎯 Select Crime Category for Analysis",
            df.columns[1:],
            help="Choose the crime type you want to analyze in detail"
        )
    
    # Calculate comprehensive metrics
    total_cases = int(df[crime_type].sum())
    avg_cases = int(df[crime_type].mean())
    max_year = int(df.loc[df[crime_type].idxmax(), "Year"])
    min_year = int(df.loc[df[crime_type].idxmin(), "Year"])
    growth_rate = round((df[crime_type].iloc[-1] - df[crime_type].iloc[0]) / df[crime_type].iloc[0] * 100, 2)
    recent_trend = "Increasing 📈" if df[crime_type].iloc[-1] > df[crime_type].iloc[-3] else "Decreasing 📉"
    
    # KEY METRICS DASHBOARD
    st.markdown("### 🎯 KEY PERFORMANCE INDICATORS")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Total Cases", 
            f"{total_cases:,}",
            delta=f"{growth_rate}% growth",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "📅 Peak Year", 
            max_year,
            delta=f"{int(df.loc[df[crime_type].idxmax(), crime_type])} cases"
        )
    
    with col3:
        st.metric(
            "📉 Lowest Year", 
            min_year,
            delta=f"{int(df.loc[df[crime_type].idxmin(), crime_type])} cases"
        )
    
    with col4:
        st.metric(
            "📈 Trend Status", 
            recent_trend.split()[0],
            delta=f"Avg: {avg_cases}"
        )
    
    st.markdown("---")
    
    # TABS FOR DIFFERENT VIEWS
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend Visualization", "📊 Comparative Analysis", "📉 Statistical Insights", "🗂️ Data Table"])
    
    with tab1:
        st.markdown("### 📈 Interactive Crime Trend Chart")
        
        # Create professional Plotly chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df["Year"],
            y=df[crime_type],
            mode='lines+markers',
            name=crime_type,
            line=dict(color='#e74c3c', width=4),
            marker=dict(size=12, color='#c0392b', 
                       line=dict(color='white', width=2)),
            hovertemplate='<b>Year: %{x}</b><br>Cases: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{crime_type} Cases Over Time",
            xaxis_title="Year",
            yaxis_title="Number of Cases",
            hovermode='x unified',
            plot_bgcolor='rgba(44, 62, 80, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ecf0f1', size=14),
            title_font=dict(size=22, color='#f39c12'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 All Crime Categories Comparison")
        
        # Multi-line chart for all crime types
        fig2 = go.Figure()
        
        colors = ['#e74c3c', '#3498db', '#f39c12', '#27ae60', '#9b59b6']
        
        for idx, col in enumerate(df.columns[1:]):
            fig2.add_trace(go.Scatter(
                x=df["Year"],
                y=df[col],
                mode='lines+markers',
                name=col,
                line=dict(width=3, color=colors[idx % len(colors)]),
                marker=dict(size=8)
            ))
        
        fig2.update_layout(
            title="Comprehensive Crime Comparison",
            xaxis_title="Year",
            yaxis_title="Number of Cases",
            hovermode='x unified',
            plot_bgcolor='rgba(44, 62, 80, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ecf0f1', size=14),
            title_font=dict(size=22, color='#f39c12'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(bgcolor='rgba(44, 62, 80, 0.6)'),
            height=500
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Bar chart for latest year
        st.markdown("### 📊 Crime Distribution (Latest Year)")
        latest_data = df.iloc[-1, 1:]
        
        fig3 = go.Figure(data=[
            go.Bar(
                x=latest_data.index,
                y=latest_data.values,
                marker=dict(
                    color=latest_data.values,
                    colorscale='Reds',
                    line=dict(color='white', width=2)
                ),
                text=latest_data.values,
                textposition='outside'
            )
        ])
        
        fig3.update_layout(
            title=f"Crime Statistics - {int(df.iloc[-1, 0])}",
            xaxis_title="Crime Type",
            yaxis_title="Cases",
            plot_bgcolor='rgba(44, 62, 80, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ecf0f1', size=14),
            title_font=dict(size=20, color='#f39c12'),
            height=400
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.markdown("### 📉 Statistical Analysis & Insights")
        
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.markdown("""
            <div style='background: rgba(52,73,94,0.5); padding: 20px; border-radius: 12px; 
                        border-left: 5px solid #e74c3c;'>
                <h4 style='color: #f39c12;'>📊 Descriptive Statistics</h4>
            </div>
            """, unsafe_allow_html=True)
            
            stats_df = pd.DataFrame({
                "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Range"],
                "Value": [
                    f"{df[crime_type].mean():.2f}",
                    f"{df[crime_type].median():.2f}",
                    f"{df[crime_type].std():.2f}",
                    f"{df[crime_type].min():.0f}",
                    f"{df[crime_type].max():.0f}",
                    f"{df[crime_type].max() - df[crime_type].min():.0f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col_stat2:
            st.markdown("""
            <div style='background: rgba(52,73,94,0.5); padding: 20px; border-radius: 12px; 
                        border-left: 5px solid #f39c12;'>
                <h4 style='color: #f39c12;'>🎯 Key Insights</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Year-over-year change
            yoy_change = df[crime_type].iloc[-1] - df[crime_type].iloc[-2]
            yoy_percent = (yoy_change / df[crime_type].iloc[-2]) * 100
            
            st.markdown(f"""
            - **Trend Direction:** {recent_trend}
            - **YoY Change:** {yoy_change:+.0f} cases ({yoy_percent:+.2f}%)
            - **Volatility:** {df[crime_type].std():.2f} (Std Dev)
            - **Total Growth:** {growth_rate:+.2f}% since {int(df['Year'].iloc[0])}
            """)
            
            # Risk Level
            if growth_rate > 30:
                risk = "🔴 HIGH RISK - Significant Increase"
            elif growth_rate > 10:
                risk = "🟡 MODERATE RISK - Growing Trend"
            else:
                risk = "🟢 LOW RISK - Stable/Decreasing"
            
            st.info(f"**Risk Assessment:** {risk}")
    
    with tab4:
        st.markdown("### 🗂️ Complete Crime Dataset")
        st.dataframe(
            df.style.background_gradient(cmap='Reds', subset=df.columns[1:]),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Dataset",
            data=csv,
            file_name=f"crime_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# =====================================================
# ============ AI CRIME PREDICTION MODULE =============
# =====================================================
if "AI Crime Prediction" in menu:
    
    # Header Section
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(52,152,219,0.2), rgba(41,128,185,0.2)); 
                border-left: 6px solid #3498db; padding: 20px; border-radius: 15px; margin-bottom: 30px;'>
        <h2 style='color: #f39c12; margin: 0;'>🤖 AI-POWERED CRIME PREDICTION ENGINE</h2>
        <p style='color: #ecf0f1; margin-top: 10px; font-size: 1.1rem;'>
            Advanced Machine Learning for Future Crime Rate Forecasting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Crime Type Selection for Prediction
    col_pred1, col_pred2 = st.columns([2, 1])
    with col_pred1:
        crime_type = st.selectbox(
            "🎯 Select Crime Category for Prediction",
            df.columns[1:],
            help="Choose the crime type you want to predict"
        )
    
    with col_pred2:
        st.markdown("""
        <div style='background: rgba(52,152,219,0.2); padding: 15px; border-radius: 10px; 
                    border: 2px solid #3498db; margin-top: 25px;'>
            <p style='margin: 0; text-align: center; font-weight: 700;'>
                🧠 Random Forest ML Model
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Prepare data
    X = df[["Year"]]
    y = df[crime_type]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model with progress
    with st.spinner("🔄 Training AI Model..."):
        model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
    
    # Model Evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # MODEL PERFORMANCE METRICS
    st.markdown("### 🎯 MODEL PERFORMANCE METRICS")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric(
            "🎯 R² Score (Test)", 
            f"{r2_test:.3f}",
            delta=f"Train: {r2_train:.3f}"
        )
    
    with col_m2:
        st.metric(
            "📊 Mean Abs Error", 
            f"{mae:.2f}",
            delta="cases"
        )
    
    with col_m3:
        st.metric(
            "📈 RMSE", 
            f"{rmse:.2f}",
            delta="Root MSE"
        )
    
    with col_m4:
        accuracy_label = "Excellent" if r2_test > 0.9 else "Good" if r2_test > 0.7 else "Fair"
        st.metric(
            "✅ Model Quality", 
            accuracy_label,
            delta=f"{r2_test*100:.1f}%"
        )
    
    st.markdown("---")
    
    # PREDICTION INTERFACE
    st.markdown("### 🔮 MAKE PREDICTIONS")
    
    col_input1, col_input2, col_input3 = st.columns([2, 1, 1])
    
    with col_input1:
        future_year = st.number_input(
            "📅 Select Future Year for Prediction",
            min_value=int(df["Year"].max()) + 1,
            max_value=2040,
            value=int(df["Year"].max()) + 1,
            step=1,
            help="Enter the year you want to predict crime rates for"
        )
    
    with col_input2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🚀 PREDICT NOW", use_container_width=True)
    
    with col_input3:
        st.markdown("<br>", unsafe_allow_html=True)
        years_ahead = future_year - int(df["Year"].max())
        st.info(f"📊 {years_ahead} year(s) ahead")
    
    # PREDICTION RESULTS
    if predict_button:
        with st.spinner("🔮 Generating prediction..."):
            prediction = model.predict([[future_year]])
            predicted_value = int(prediction[0])
            
            # Confidence interval (simplified)
            confidence = predicted_value * 0.1  # ±10% confidence
            lower_bound = int(predicted_value - confidence)
            upper_bound = int(predicted_value + confidence)
        
        st.balloons()
        
        # RESULTS DISPLAY
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(39,174,96,0.2), rgba(34,153,84,0.2)); 
                    border: 3px solid #27ae60; padding: 30px; border-radius: 20px; margin: 30px 0;
                    box-shadow: 0 10px 30px rgba(39,174,96,0.3);'>
            <h2 style='text-align: center; color: #27ae60; margin-bottom: 20px;'>
                🎯 PREDICTION RESULTS
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Main Prediction Metric
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        
        with col_res2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #e74c3c, #c0392b); 
                        padding: 40px; border-radius: 20px; text-align: center;
                        box-shadow: 0 15px 40px rgba(231,76,60,0.5); border: 3px solid white;'>
                <h3 style='color: white; margin: 0; font-size: 1.2rem;'>
                    Predicted {crime_type} Cases
                </h3>
                <h1 style='color: white; margin: 20px 0; font-size: 4rem; font-weight: 900;'>
                    {predicted_value:,}
                </h1>
                <p style='color: #ecf0f1; margin: 0; font-size: 1.1rem;'>
                    Year: {future_year}
                </p>
                <p style='color: #f39c12; margin-top: 15px; font-size: 1rem; font-weight: 600;'>
                    Confidence Range: {lower_bound:,} - {upper_bound:,} cases
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualization Tabs
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📈 Forecast Visualization", "📊 Prediction Analysis", "🎯 Model Insights"])
        
        with viz_tab1:
            st.markdown("### 📈 Historical Data + Future Prediction")
            
            # Create forecast chart
            fig_pred = go.Figure()
            
            # Historical data
            fig_pred.add_trace(go.Scatter(
                x=df["Year"],
                y=df[crime_type],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='#3498db', width=4),
                marker=dict(size=12, color='#2980b9')
            ))
            
            # Prediction point
            fig_pred.add_trace(go.Scatter(
                x=[future_year],
                y=[predicted_value],
                mode='markers',
                name='Prediction',
                marker=dict(
                    size=25,
                    color='#e74c3c',
                    symbol='star',
                    line=dict(color='white', width=3)
                )
            ))
            
            # Prediction line
            fig_pred.add_trace(go.Scatter(
                x=[df["Year"].iloc[-1], future_year],
                y=[df[crime_type].iloc[-1], predicted_value],
                mode='lines',
                name='Forecast Trend',
                line=dict(color='#e74c3c', width=3, dash='dash')
            ))
            
            # Confidence interval
            fig_pred.add_trace(go.Scatter(
                x=[future_year, future_year],
                y=[lower_bound, upper_bound],
                mode='lines',
                name='Confidence Range',
                line=dict(color='#f39c12', width=0),
                fill='toself',
                fillcolor='rgba(243, 156, 18, 0.2)'
            ))
            
            fig_pred.update_layout(
                title=f"{crime_type} Forecast Analysis",
                xaxis_title="Year",
                yaxis_title="Number of Cases",
                hovermode='x unified',
                plot_bgcolor='rgba(44, 62, 80, 0.3)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ecf0f1', size=14),
                title_font=dict(size=22, color='#f39c12'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                legend=dict(bgcolor='rgba(44, 62, 80, 0.6)'),
                height=550
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with viz_tab2:
            st.markdown("### 📊 Comparative Prediction Analysis")
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                # Comparison with historical average
                hist_avg = df[crime_type].mean()
                diff_from_avg = predicted_value - hist_avg
                percent_diff = (diff_from_avg / hist_avg) * 100
                
                st.markdown("""
                <div style='background: rgba(52,73,94,0.5); padding: 20px; border-radius: 12px; 
                            border-left: 5px solid #3498db;'>
                    <h4 style='color: #f39c12;'>📊 Comparison Metrics</h4>
                </div>
                """, unsafe_allow_html=True)
                
                comparison_df = pd.DataFrame({
                    "Metric": [
                        "Historical Average",
                        "Predicted Value",
                        "Difference",
                        "Percentage Change",
                        "Latest Year Value"
                    ],
                    "Value": [
                        f"{hist_avg:.0f} cases",
                        f"{predicted_value:,} cases",
                        f"{diff_from_avg:+.0f} cases",
                        f"{percent_diff:+.2f}%",
                        f"{df[crime_type].iloc[-1]:.0f} cases"
                    ]
                })
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            with col_comp2:
                # Risk Assessment
                st.markdown("""
                <div style='background: rgba(52,73,94,0.5); padding: 20px; border-radius: 12px; 
                            border-left: 5px solid #e74c3c;'>
                    <h4 style='color: #f39c12;'>🚨 Risk Assessment</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if percent_diff > 20:
                    risk_level = "🔴 HIGH RISK"
                    risk_desc = "Significant increase predicted. Immediate action recommended."
                    risk_color = "#e74c3c"
                elif percent_diff > 5:
                    risk_level = "🟡 MODERATE RISK"
                    risk_desc = "Notable increase expected. Enhanced monitoring advised."
                    risk_color = "#f39c12"
                else:
                    risk_level = "🟢 LOW RISK"
                    risk_desc = "Stable or decreasing trend. Continue current strategies."
                    risk_color = "#27ae60"
                
                st.markdown(f"""
                <div style='background: rgba(52,73,94,0.3); padding: 20px; border-radius: 10px; 
                            border: 2px solid {risk_color};'>
                    <h3 style='color: {risk_color}; text-align: center; margin-bottom: 15px;'>
                        {risk_level}
                    </h3>
                    <p style='color: #ecf0f1; text-align: center; margin: 0;'>
                        {risk_desc}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("##### 💡 Recommendations:")
                if percent_diff > 20:
                    st.markdown("""
                    - 🚨 Increase patrol frequency
                    - 📊 Allocate additional resources
                    - 🎯 Implement preventive programs
                    - 📱 Enhance community awareness
                    """)
                elif percent_diff > 5:
                    st.markdown("""
                    - 👁️ Monitor hotspot areas
                    - 📈 Review current strategies
                    - 🤝 Strengthen community engagement
                    """)
                else:
                    st.markdown("""
                    - ✅ Maintain current approach
                    - 📊 Continue data monitoring
                    - 🎯 Optimize resource allocation
                    """)
        
        with viz_tab3:
            st.markdown("### 🎯 Model Performance & Insights")
            
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                # Feature Importance
                st.markdown("#### 📊 Feature Importance")
                importance = model.feature_importances_
                
                fig_imp = go.Figure(data=[
                    go.Bar(
                        x=["Year"],
                        y=importance,
                        marker=dict(color='#e74c3c'),
                        text=[f"{imp:.2%}" for imp in importance],
                        textposition='outside'
                    )
                ])
                
                fig_imp.update_layout(
                    title="Model Feature Importance",
                    yaxis_title="Importance Score",
                    plot_bgcolor='rgba(44, 62, 80, 0.3)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ecf0f1'),
                    height=350
                )
                
                st.plotly_chart(fig_imp, use_container_width=True)
            
            with col_insight2:
                # Model Details
                st.markdown("#### 🧠 Model Configuration")
                st.markdown(f"""
                - **Algorithm:** Random Forest Regressor
                - **Estimators:** 200 trees
                - **Max Depth:** 10
                - **Training Size:** {len(X_train)} samples
                - **Test Size:** {len(X_test)} samples
                - **Train R² Score:** {r2_train:.4f}
                - **Test R² Score:** {r2_test:.4f}
                - **Mean Absolute Error:** {mae:.2f}
                - **Root Mean Squared Error:** {rmse:.2f}
                """)
        
        st.markdown("---")
        
        # DOWNLOAD SECTION
        st.markdown("### 📥 EXPORT PREDICTION REPORT")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            # Detailed Report
            report = pd.DataFrame({
                "Metric": [
                    "Crime Type",
                    "Prediction Year",
                    "Predicted Cases",
                    "Confidence Range (Lower)",
                    "Confidence Range (Upper)",
                    "Historical Average",
                    "Change from Average (%)",
                    "Model R² Score",
                    "Mean Absolute Error",
                    "RMSE",
                    "Risk Level"
                ],
                "Value": [
                    crime_type,
                    future_year,
                    predicted_value,
                    lower_bound,
                    upper_bound,
                    f"{hist_avg:.2f}",
                    f"{percent_diff:.2f}%",
                    f"{r2_test:.4f}",
                    f"{mae:.2f}",
                    f"{rmse:.2f}",
                    risk_level
                ]
            })
            
            csv = report.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Download Detailed Report (CSV)",
                data=csv,
                file_name=f"crime_prediction_{crime_type}_{future_year}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_dl2:
            # Summary Report
            summary = f"""
CRIME PREDICTION REPORT
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PREDICTION DETAILS
------------------
Crime Type: {crime_type}
Prediction Year: {future_year}
Predicted Cases: {predicted_value:,}
Confidence Range: {lower_bound:,} - {upper_bound:,}

MODEL PERFORMANCE
-----------------
R² Score: {r2_test:.4f}
MAE: {mae:.2f}
RMSE: {rmse:.2f}

RISK ASSESSMENT
---------------
Level: {risk_level}
Change from Avg: {percent_diff:+.2f}%

Generated by Crime Intelligence Pro
AI for Bharat Hackathon 2024
"""
            st.download_button(
                "📋 Download Summary (TXT)",
                data=summary,
                file_name=f"prediction_summary_{future_year}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_dl3:
            # Historical + Prediction Data
            forecast_df = pd.concat([
                df[["Year", crime_type]],
                pd.DataFrame({
                    "Year": [future_year],
                    crime_type: [predicted_value]
                })
            ], ignore_index=True)
            
            forecast_csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📊 Download Forecast Data (CSV)",
                data=forecast_csv,
                file_name=f"forecast_data_{crime_type}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ---------------- PREMIUM FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(231,76,60,0.2), rgba(192,57,43,0.2)); 
            border-radius: 20px; margin-top: 40px;'>
    <h3 style='color: #f39c12; margin-bottom: 15px;'>🏆 CRIME INTELLIGENCE PRO</h3>
    <p style='color: #ecf0f1; font-size: 1.1rem; margin-bottom: 10px;'>
        AI-Powered Crime Analytics & Prediction Platform
    </p>
    <p style='color: #95a5a6; font-size: 0.9rem;'>
        Developed for AI for Bharat Hackathon 2024 | Powered by Machine Learning & Advanced Analytics
    </p>
    <div style='margin-top: 20px;'>
        <span style='background: #e74c3c; padding: 8px 20px; border-radius: 20px; 
                     color: white; font-weight: 700; margin: 0 5px;'>
            🚀 Innovation
        </span>
        <span style='background: #3498db; padding: 8px 20px; border-radius: 20px; 
                     color: white; font-weight: 700; margin: 0 5px;'>
            🧠 AI/ML
        </span>
        <span style='background: #27ae60; padding: 8px 20px; border-radius: 20px; 
                     color: white; font-weight: 700; margin: 0 5px;'>
            📊 Data Science
        </span>
    </div>
</div>
""", unsafe_allow_html=True)