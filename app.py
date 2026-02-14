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
    page_title="Crime Intelligence Pro",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- PREMIUM PROFESSIONAL CSS ----------------
st.markdown("""
<style>
/* Main Background with Pattern */
[data-testid="stAppViewContainer"] {
    background: 
        linear-gradient(135deg, rgba(15, 12, 41, 0.95) 0%, rgba(48, 43, 99, 0.95) 50%, rgba(36, 36, 62, 0.95) 100%),
        repeating-linear-gradient(45deg, transparent, transparent 10px, rgba(255,255,255,.03) 10px, rgba(255,255,255,.03) 20px);
    color: #FFFFFF;
}

/* Sidebar - Dark Professional */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 3px solid #dc143c;
}

/* Sidebar Content - Enhanced Visibility */
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Make ALL text highly visible */
label, .stMarkdown, .stText, .stSelectbox label,
.stNumberInput label, .stRadio label, p, span, div {
    color: #ffffff !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
}

/* Select box text visibility */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: rgba(30, 60, 114, 0.8) !important;
    color: #ffffff !important;
    border: 2px solid #dc143c !important;
}

/* Select box options */
.stSelectbox div[data-baseweb="select"] span {
    color: #ffffff !important;
}

/* Number input visibility */
.stNumberInput input {
    background-color: rgba(30, 60, 114, 0.8) !important;
    color: #ffffff !important;
    border: 2px solid #dc143c !important;
    font-weight: 600 !important;
}

/* Main Title */
h1 {
    color: #ffffff !important;
    text-shadow: 4px 4px 8px rgba(0,0,0,0.8), 0 0 30px rgba(220, 20, 60, 0.5);
    font-size: 3.5rem !important;
    font-weight: 900 !important;
    padding: 25px 0;
    text-align: center;
    letter-spacing: 2px;
}

/* Subheaders - High Visibility */
h2, h3 {
    color: #ffd700 !important;
    font-weight: 800 !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.8);
    margin-top: 20px !important;
}

h4 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Metric Cards - Enhanced */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(220, 20, 60, 0.25), rgba(139, 0, 0, 0.2));
    border: 3px solid #dc143c;
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 12px 35px rgba(220, 20, 60, 0.4), inset 0 1px 3px rgba(255,255,255,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    backdrop-filter: blur(10px);
}

[data-testid="metric-container"]:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 18px 45px rgba(220, 20, 60, 0.6), inset 0 1px 3px rgba(255,255,255,0.2);
}

[data-testid="stMetricValue"] {
    font-size: 3rem !important;
    font-weight: 900 !important;
    color: #ff4444 !important;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.8), 0 0 20px rgba(255, 68, 68, 0.5);
}

[data-testid="stMetricLabel"] {
    color: #ffffff !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
}

[data-testid="stMetricDelta"] {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
}

/* Buttons - Premium Style */
.stButton>button {
    background: linear-gradient(135deg, #dc143c 0%, #8b0000 100%) !important;
    color: white !important;
    font-weight: 900 !important;
    font-size: 1.2rem !important;
    border-radius: 15px !important;
    padding: 18px 45px !important;
    border: 3px solid #ffffff;
    box-shadow: 0 10px 25px rgba(220, 20, 60, 0.5), inset 0 1px 3px rgba(255,255,255,0.2);
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.stButton>button:hover {
    transform: translateY(-4px) scale(1.05);
    box-shadow: 0 15px 35px rgba(220, 20, 60, 0.7), inset 0 1px 3px rgba(255,255,255,0.3);
    background: linear-gradient(135deg, #ff1744 0%, #dc143c 100%) !important;
}

/* Download Button */
.stDownloadButton>button {
    background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 12px 30px !important;
    border: 2px solid #ffffff;
    box-shadow: 0 8px 20px rgba(30, 136, 229, 0.4);
}

.stDownloadButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(30, 136, 229, 0.6);
}

/* Dataframe Styling */
[data-testid="stDataFrame"] {
    background: rgba(26, 26, 46, 0.7);
    border-radius: 15px;
    border: 2px solid #dc143c;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
}

/* Divider */
hr {
    border: 2px solid #dc143c;
    margin: 30px 0;
    box-shadow: 0 2px 10px rgba(220, 20, 60, 0.5);
}

/* Info/Warning/Success Boxes - High Visibility */
.stAlert {
    background: rgba(26, 26, 46, 0.9) !important;
    border-radius: 12px;
    border-left: 5px solid #dc143c;
    padding: 20px;
    color: #ffffff !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
}

.stSuccess {
    background: rgba(27, 94, 32, 0.9) !important;
    border-left: 5px solid #4caf50 !important;
}

.stInfo {
    background: rgba(13, 71, 161, 0.9) !important;
    border-left: 5px solid #2196f3 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(26, 26, 46, 0.7);
    border-radius: 15px;
    padding: 15px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(30, 60, 114, 0.6);
    border-radius: 12px;
    color: #ffffff;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 15px 30px;
    border: 2px solid transparent;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #dc143c, #8b0000);
    color: white;
    border: 2px solid #ffffff;
    box-shadow: 0 5px 15px rgba(220, 20, 60, 0.5);
}

/* File Uploader - High Visibility */
[data-testid="stFileUploader"] {
    background: rgba(30, 60, 114, 0.6);
    border-radius: 12px;
    padding: 25px;
    border: 3px dashed #ffd700;
}

[data-testid="stFileUploader"] label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}

/* Radio Buttons - High Visibility */
.stRadio > div {
    background: rgba(30, 60, 114, 0.5);
    border-radius: 12px;
    padding: 20px;
    border: 2px solid #dc143c;
}

.stRadio label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}

/* Success message animation - Professional pulse */
@keyframes professionalPulse {
    0%, 100% { 
        transform: scale(1);
        box-shadow: 0 0 20px rgba(220, 20, 60, 0.3);
    }
    50% { 
        transform: scale(1.02);
        box-shadow: 0 0 40px rgba(220, 20, 60, 0.6);
    }
}

.prediction-complete {
    animation: professionalPulse 2s ease-in-out 3;
}

/* Smooth entrance animation */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.element-container {
    animation: slideIn 0.4s ease-out;
}

</style>
""", unsafe_allow_html=True)

# ---------------- PROFESSIONAL TITLE ----------------
st.markdown("""
<div style='text-align: center; margin-bottom: 40px;'>
    <h1 style='margin-bottom: 15px; letter-spacing: 3px;'>🚨 CRIME INTELLIGENCE PRO 🚨</h1>
    <p style='font-size: 1.6rem; color: #ffd700; font-weight: 700; text-shadow: 2px 2px 6px rgba(0,0,0,0.8);'>
        Advanced AI-Powered Crime Analytics & Prediction System
    </p>
    <div style='display: inline-block; background: linear-gradient(135deg, #dc143c, #8b0000); 
                padding: 12px 30px; border-radius: 30px; margin-top: 20px; 
                box-shadow: 0 8px 20px rgba(220,20,60,0.6); border: 2px solid #ffd700;'>
        <span style='color: white; font-weight: 800; font-size: 1.1rem; letter-spacing: 1px;'>
            🎯 PROFESSIONAL CRIME ANALYSIS PLATFORM
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
    <div style='background: linear-gradient(135deg, rgba(220,20,60,0.3), rgba(139,0,0,0.3)); 
                border-left: 6px solid #dc143c; padding: 25px; border-radius: 15px; margin-bottom: 30px;
                box-shadow: 0 8px 20px rgba(220,20,60,0.3); border: 2px solid rgba(220,20,60,0.5);'>
        <h2 style='color: #ffd700; margin: 0; text-shadow: 2px 2px 6px rgba(0,0,0,0.8);'>📊 COMPREHENSIVE CRIME TREND ANALYSIS</h2>
        <p style='color: #ffffff; margin-top: 15px; font-size: 1.2rem; font-weight: 600; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);'>
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
            line=dict(color='#dc143c', width=4),
            marker=dict(size=14, color='#ff1744', 
                       line=dict(color='white', width=2)),
            hovertemplate='<b>Year: %{x}</b><br>Cases: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{crime_type} Cases Over Time",
            xaxis_title="Year",
            yaxis_title="Number of Cases",
            hovermode='x unified',
            plot_bgcolor='rgba(26, 26, 46, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=14),
            title_font=dict(size=22, color='#ffd700', family='Arial Black'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.15)', color='#ffffff'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.15)', color='#ffffff'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 All Crime Categories Comparison")
        
        # Multi-line chart for all crime types
        fig2 = go.Figure()
        
        colors = ['#dc143c', '#ff9800', '#ffd700', '#4caf50', '#2196f3']
        
        for idx, col in enumerate(df.columns[1:]):
            fig2.add_trace(go.Scatter(
                x=df["Year"],
                y=df[col],
                mode='lines+markers',
                name=col,
                line=dict(width=3, color=colors[idx % len(colors)]),
                marker=dict(size=10)
            ))
        
        fig2.update_layout(
            title="Comprehensive Crime Comparison",
            xaxis_title="Year",
            yaxis_title="Number of Cases",
            hovermode='x unified',
            plot_bgcolor='rgba(26, 26, 46, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=14),
            title_font=dict(size=22, color='#ffd700', family='Arial Black'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.15)', color='#ffffff'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.15)', color='#ffffff'),
            legend=dict(bgcolor='rgba(26, 26, 46, 0.8)', font=dict(color='#ffffff')),
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
                textposition='outside',
                textfont=dict(color='#ffffff', size=14, family='Arial Black')
            )
        ])
        
        fig3.update_layout(
            title=f"Crime Statistics - {int(df.iloc[-1, 0])}",
            xaxis_title="Crime Type",
            yaxis_title="Cases",
            plot_bgcolor='rgba(26, 26, 46, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=14),
            title_font=dict(size=20, color='#ffd700', family='Arial Black'),
            xaxis=dict(color='#ffffff'),
            yaxis=dict(color='#ffffff'),
            height=400
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.markdown("### 📉 Statistical Analysis & Insights")
        
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.markdown("""
            <div style='background: rgba(30,60,114,0.6); padding: 25px; border-radius: 12px; 
                        border-left: 5px solid #dc143c; box-shadow: 0 8px 20px rgba(0,0,0,0.4);'>
                <h4 style='color: #ffd700; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);'>📊 Descriptive Statistics</h4>
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
            <div style='background: rgba(30,60,114,0.6); padding: 25px; border-radius: 12px; 
                        border-left: 5px solid #ffd700; box-shadow: 0 8px 20px rgba(0,0,0,0.4);'>
                <h4 style='color: #ffd700; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);'>🎯 Key Insights</h4>
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
    <div style='background: linear-gradient(135deg, rgba(30,136,229,0.3), rgba(21,101,192,0.3)); 
                border-left: 6px solid #1e88e5; padding: 25px; border-radius: 15px; margin-bottom: 30px;
                box-shadow: 0 8px 20px rgba(30,136,229,0.3); border: 2px solid rgba(30,136,229,0.5);'>
        <h2 style='color: #ffd700; margin: 0; text-shadow: 2px 2px 6px rgba(0,0,0,0.8);'>🤖 AI-POWERED CRIME PREDICTION ENGINE</h2>
        <p style='color: #ffffff; margin-top: 15px; font-size: 1.2rem; font-weight: 600; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);'>
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
        <div style='background: rgba(30,136,229,0.3); padding: 18px; border-radius: 12px; 
                    border: 3px solid #1e88e5; margin-top: 25px; box-shadow: 0 5px 15px rgba(30,136,229,0.4);'>
            <p style='margin: 0; text-align: center; font-weight: 800; font-size: 1.1rem; color: #ffffff;
                      text-shadow: 2px 2px 4px rgba(0,0,0,0.8);'>
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
        with st.spinner("🔮 Analyzing data and generating prediction..."):
            prediction = model.predict([[future_year]])
            predicted_value = int(prediction[0])
            
            # Confidence interval (simplified)
            confidence = predicted_value * 0.1  # ±10% confidence
            lower_bound = int(predicted_value - confidence)
            upper_bound = int(predicted_value + confidence)
        
        # Professional success message instead of balloons
        st.success("✅ Prediction completed successfully!")
        
        # RESULTS DISPLAY
        st.markdown("""
        <div class='prediction-complete' style='background: linear-gradient(135deg, rgba(76,175,80,0.3), rgba(56,142,60,0.3)); 
                    border: 3px solid #4caf50; padding: 30px; border-radius: 20px; margin: 30px 0;
                    box-shadow: 0 12px 35px rgba(76,175,80,0.4);'>
            <h2 style='text-align: center; color: #4caf50; margin-bottom: 20px; text-shadow: 2px 2px 6px rgba(0,0,0,0.8);'>
                🎯 PREDICTION RESULTS
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Main Prediction Metric
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        
        with col_res2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #dc143c, #8b0000); 
                        padding: 45px; border-radius: 20px; text-align: center;
                        box-shadow: 0 18px 45px rgba(220,20,60,0.6); border: 4px solid #ffd700;'>
                <h3 style='color: #ffd700; margin: 0; font-size: 1.3rem; text-shadow: 2px 2px 6px rgba(0,0,0,0.8);'>
                    Predicted {crime_type} Cases
                </h3>
                <h1 style='color: white; margin: 25px 0; font-size: 4.5rem; font-weight: 900; text-shadow: 3px 3px 8px rgba(0,0,0,0.8);'>
                    {predicted_value:,}
                </h1>
                <p style='color: #ffffff; margin: 0; font-size: 1.2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);'>
                    Year: {future_year}
                </p>
                <p style='color: #ffd700; margin-top: 20px; font-size: 1.1rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);'>
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
                line=dict(color='#1e88e5', width=4),
                marker=dict(size=12, color='#1565c0')
            ))
            
            # Prediction point
            fig_pred.add_trace(go.Scatter(
                x=[future_year],
                y=[predicted_value],
                mode='markers',
                name='Prediction',
                marker=dict(
                    size=28,
                    color='#dc143c',
                    symbol='star',
                    line=dict(color='#ffd700', width=3)
                )
            ))
            
            # Prediction line
            fig_pred.add_trace(go.Scatter(
                x=[df["Year"].iloc[-1], future_year],
                y=[df[crime_type].iloc[-1], predicted_value],
                mode='lines',
                name='Forecast Trend',
                line=dict(color='#dc143c', width=4, dash='dash')
            ))
            
            # Confidence interval
            fig_pred.add_trace(go.Scatter(
                x=[future_year, future_year],
                y=[lower_bound, upper_bound],
                mode='lines',
                name='Confidence Range',
                line=dict(color='#ffd700', width=0),
                fill='toself',
                fillcolor='rgba(255, 215, 0, 0.2)'
            ))
            
            fig_pred.update_layout(
                title=f"{crime_type} Forecast Analysis",
                xaxis_title="Year",
                yaxis_title="Number of Cases",
                hovermode='x unified',
                plot_bgcolor='rgba(26, 26, 46, 0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', size=14),
                title_font=dict(size=22, color='#ffd700', family='Arial Black'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.15)', color='#ffffff'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.15)', color='#ffffff'),
                legend=dict(bgcolor='rgba(26, 26, 46, 0.8)', font=dict(color='#ffffff')),
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
                <div style='background: rgba(30,60,114,0.6); padding: 25px; border-radius: 12px; 
                            border-left: 5px solid #1e88e5; box-shadow: 0 8px 20px rgba(0,0,0,0.4);'>
                    <h4 style='color: #ffd700; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);'>📊 Comparison Metrics</h4>
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
                <div style='background: rgba(30,60,114,0.6); padding: 25px; border-radius: 12px; 
                            border-left: 5px solid #dc143c; box-shadow: 0 8px 20px rgba(0,0,0,0.4);'>
                    <h4 style='color: #ffd700; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);'>🚨 Risk Assessment</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if percent_diff > 20:
                    risk_level = "🔴 HIGH RISK"
                    risk_desc = "Significant increase predicted. Immediate action recommended."
                    risk_color = "#dc143c"
                elif percent_diff > 5:
                    risk_level = "🟡 MODERATE RISK"
                    risk_desc = "Notable increase expected. Enhanced monitoring advised."
                    risk_color = "#ffd700"
                else:
                    risk_level = "🟢 LOW RISK"
                    risk_desc = "Stable or decreasing trend. Continue current strategies."
                    risk_color = "#4caf50"
                
                st.markdown(f"""
                <div style='background: rgba(26,26,46,0.6); padding: 25px; border-radius: 12px; 
                            border: 3px solid {risk_color}; box-shadow: 0 8px 20px rgba(0,0,0,0.4);'>
                    <h3 style='color: {risk_color}; text-align: center; margin-bottom: 20px; text-shadow: 2px 2px 6px rgba(0,0,0,0.8);'>
                        {risk_level}
                    </h3>
                    <p style='color: #ffffff; text-align: center; margin: 0; font-size: 1.1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);'>
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
                        marker=dict(color='#dc143c'),
                        text=[f"{imp:.2%}" for imp in importance],
                        textposition='outside',
                        textfont=dict(color='#ffffff', size=14)
                    )
                ])
                
                fig_imp.update_layout(
                    title="Model Feature Importance",
                    yaxis_title="Importance Score",
                    plot_bgcolor='rgba(26, 26, 46, 0.5)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff'),
                    title_font=dict(color='#ffd700'),
                    xaxis=dict(color='#ffffff'),
                    yaxis=dict(color='#ffffff'),
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
            summary = f"""CRIME PREDICTION REPORT
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
Professional Crime Analysis Platform"""
            
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

# ---------------- PROFESSIONAL FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 35px; background: linear-gradient(135deg, rgba(220,20,60,0.3), rgba(139,0,0,0.3)); 
            border-radius: 20px; margin-top: 50px; border: 2px solid #dc143c; box-shadow: 0 10px 30px rgba(220,20,60,0.4);'>
    <h3 style='color: #ffd700; margin-bottom: 20px; text-shadow: 2px 2px 6px rgba(0,0,0,0.8);'>🚨 CRIME INTELLIGENCE PRO 🚨</h3>
    <p style='color: #ffffff; font-size: 1.2rem; margin-bottom: 15px; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);'>
        Professional AI-Powered Crime Analytics & Prediction Platform
    </p>
    <p style='color: #ffffff; font-size: 1rem; opacity: 0.9; text-shadow: 1px 1px 3px rgba(0,0,0,0.8);'>
        Advanced Machine Learning | Data Science | Strategic Intelligence
    </p>
    <div style='margin-top: 25px;'>
        <span style='background: #dc143c; padding: 10px 25px; border-radius: 25px; 
                     color: white; font-weight: 800; margin: 0 8px; box-shadow: 0 5px 15px rgba(220,20,60,0.4);'>
            🚀 Innovation
        </span>
        <span style='background: #1e88e5; padding: 10px 25px; border-radius: 25px; 
                     color: white; font-weight: 800; margin: 0 8px; box-shadow: 0 5px 15px rgba(30,136,229,0.4);'>
            🧠 AI/ML
        </span>
        <span style='background: #4caf50; padding: 10px 25px; border-radius: 25px; 
                     color: white; font-weight: 800; margin: 0 8px; box-shadow: 0 5px 15px rgba(76,175,80,0.4);'>
            📊 Analytics
        </span>
    </div>
</div>
""", unsafe_allow_html=True)