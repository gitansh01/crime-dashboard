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

# ---------------- PROFESSIONAL LAW ENFORCEMENT CSS ----------------
st.markdown("""
<style>
/* Main Background - Serious Law Enforcement Theme */
[data-testid="stAppViewContainer"] {
    background: 
        linear-gradient(135deg, rgba(10, 10, 30, 0.98) 0%, rgba(25, 25, 60, 0.98) 25%, rgba(15, 15, 40, 0.98) 50%, rgba(30, 30, 70, 0.98) 75%, rgba(10, 10, 30, 0.98) 100%),
        repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(139, 0, 0, 0.05) 2px, rgba(139, 0, 0, 0.05) 4px),
        repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(139, 0, 0, 0.05) 2px, rgba(139, 0, 0, 0.05) 4px);
    color: #FFFFFF;
}

/* Sidebar - Dark Command Center */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1e 0%, #1a1a3c 50%, #0f0f28 100%);
    border-right: 4px solid #8b0000;
    box-shadow: 5px 0 20px rgba(139, 0, 0, 0.3);
}

/* Sidebar Content - Maximum Visibility */
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
}

[data-testid="stSidebar"] .stMarkdown {
    color: #ffffff !important;
}

/* ALL TEXT - Maximum Visibility */
label, .stMarkdown, .stText, .stSelectbox label,
.stNumberInput label, .stRadio label, p, span, div {
    color: #ffffff !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
}

/* Select Box - HIGH CONTRAST */
.stSelectbox div[data-baseweb="select"] > div {
    background: linear-gradient(135deg, #1a1a3c 0%, #0f0f28 100%) !important;
    color: #ffffff !important;
    border: 3px solid #8b0000 !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    box-shadow: 0 5px 15px rgba(139, 0, 0, 0.4), inset 0 1px 3px rgba(255, 255, 255, 0.1);
}

/* Select Box Dropdown Menu */
[data-baseweb="popover"] {
    background: #0a0a1e !important;
    border: 3px solid #8b0000 !important;
}

/* Select Box Options */
.stSelectbox li {
    background: #0a0a1e !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    padding: 12px 20px !important;
}

.stSelectbox li:hover {
    background: linear-gradient(135deg, #8b0000, #dc143c) !important;
    color: #ffffff !important;
}

/* Select Box Selected Text */
.stSelectbox div[data-baseweb="select"] span {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Number Input - HIGH CONTRAST */
.stNumberInput input {
    background: linear-gradient(135deg, #1a1a3c 0%, #0f0f28 100%) !important;
    color: #ffffff !important;
    border: 3px solid #8b0000 !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    box-shadow: 0 5px 15px rgba(139, 0, 0, 0.4);
}

/* Main Title */
h1 {
    color: #ffffff !important;
    text-shadow: 5px 5px 10px rgba(0,0,0,0.9), 0 0 40px rgba(139, 0, 0, 0.6);
    font-size: 4rem !important;
    font-weight: 900 !important;
    padding: 30px 0;
    text-align: center;
    letter-spacing: 4px;
    text-transform: uppercase;
}

/* Subheaders - Maximum Visibility */
h2, h3 {
    color: #ff4444 !important;
    font-weight: 900 !important;
    text-shadow: 3px 3px 8px rgba(0,0,0,0.9);
    margin-top: 20px !important;
    letter-spacing: 2px;
}

h4 {
    color: #ffffff !important;
    font-weight: 800 !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.8);
}

/* Metric Cards - Enhanced */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(139, 0, 0, 0.3), rgba(80, 0, 0, 0.25));
    border: 4px solid #8b0000;
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 15px 40px rgba(139, 0, 0, 0.5), inset 0 2px 5px rgba(255, 255, 255, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    backdrop-filter: blur(15px);
    position: relative;
}

[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, transparent, #ff0000, transparent);
}

[data-testid="metric-container"]:hover {
    transform: translateY(-10px) scale(1.03);
    box-shadow: 0 20px 50px rgba(139, 0, 0, 0.7), inset 0 2px 5px rgba(255, 255, 255, 0.2);
}

[data-testid="stMetricValue"] {
    font-size: 3.5rem !important;
    font-weight: 900 !important;
    color: #ff0000 !important;
    text-shadow: 4px 4px 8px rgba(0,0,0,0.9), 0 0 30px rgba(255, 0, 0, 0.6);
}

[data-testid="stMetricLabel"] {
    color: #ffffff !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
    text-transform: uppercase;
    letter-spacing: 1px;
}

[data-testid="stMetricDelta"] {
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
}

/* Buttons - Professional Command Style */
.stButton>button {
    background: linear-gradient(135deg, #8b0000 0%, #5a0000 100%) !important;
    color: white !important;
    font-weight: 900 !important;
    font-size: 1.3rem !important;
    border-radius: 12px !important;
    padding: 20px 50px !important;
    border: 4px solid #ff0000;
    box-shadow: 0 12px 30px rgba(139, 0, 0, 0.6), inset 0 2px 5px rgba(255, 255, 255, 0.2);
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 3px;
}

.stButton>button:hover {
    transform: translateY(-5px) scale(1.08);
    box-shadow: 0 18px 40px rgba(139, 0, 0, 0.8), inset 0 2px 5px rgba(255, 255, 255, 0.3);
    background: linear-gradient(135deg, #dc143c 0%, #8b0000 100%) !important;
    border-color: #ffffff;
}

/* Download Button */
.stDownloadButton>button {
    background: linear-gradient(135deg, #003d82 0%, #001f42 100%) !important;
    color: white !important;
    font-weight: 800 !important;
    border-radius: 10px !important;
    padding: 15px 35px !important;
    border: 3px solid #0066cc;
    box-shadow: 0 10px 25px rgba(0, 61, 130, 0.5);
}

.stDownloadButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(0, 61, 130, 0.7);
}

/* Dataframe Styling */
[data-testid="stDataFrame"] {
    background: rgba(10, 10, 30, 0.8);
    border-radius: 15px;
    border: 3px solid #8b0000;
    box-shadow: 0 12px 30px rgba(0,0,0,0.6);
}

/* Divider */
hr {
    border: 3px solid #8b0000;
    margin: 40px 0;
    box-shadow: 0 3px 15px rgba(139, 0, 0, 0.6);
}

/* Alert Boxes - High Visibility */
.stAlert {
    background: rgba(10, 10, 30, 0.95) !important;
    border-radius: 12px;
    border-left: 6px solid #8b0000;
    padding: 20px;
    color: #ffffff !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    font-weight: 700 !important;
}

.stSuccess {
    background: rgba(0, 50, 0, 0.95) !important;
    border-left: 6px solid #00ff00 !important;
}

.stInfo {
    background: rgba(0, 20, 60, 0.95) !important;
    border-left: 6px solid #0066ff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 15px;
    background: rgba(10, 10, 30, 0.8);
    border-radius: 15px;
    padding: 20px;
    border: 2px solid #8b0000;
}

.stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, rgba(139, 0, 0, 0.3), rgba(80, 0, 0, 0.3));
    border-radius: 12px;
    color: #ffffff;
    font-weight: 800;
    font-size: 1.2rem;
    padding: 18px 35px;
    border: 3px solid transparent;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #8b0000, #dc143c);
    color: white;
    border: 3px solid #ff0000;
    box-shadow: 0 8px 20px rgba(139, 0, 0, 0.6);
}

/* Radio Buttons - High Visibility */
.stRadio > div {
    background: rgba(26, 26, 60, 0.6);
    border-radius: 12px;
    padding: 25px;
    border: 3px solid #8b0000;
}

.stRadio label {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1.2rem !important;
}

/* Professional Alert Animation */
@keyframes criticalPulse {
    0%, 100% { 
        transform: scale(1);
        box-shadow: 0 0 30px rgba(139, 0, 0, 0.4);
    }
    50% { 
        transform: scale(1.02);
        box-shadow: 0 0 60px rgba(139, 0, 0, 0.8);
    }
}

.prediction-complete {
    animation: criticalPulse 2.5s ease-in-out 3;
}

/* Professional Entrance */
@keyframes commandDeploy {
    from {
        opacity: 0;
        transform: translateY(-15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.element-container {
    animation: commandDeploy 0.5s ease-out;
}

</style>
""", unsafe_allow_html=True)

# ---------------- PROFESSIONAL LAW ENFORCEMENT TITLE ----------------
st.markdown("""
<div style='text-align: center; margin-bottom: 50px;'>
    <h1 style='margin-bottom: 20px; letter-spacing: 5px;'>🚨 CRIME INTELLIGENCE PRO 🚨</h1>
    <p style='font-size: 1.8rem; color: #ff4444; font-weight: 900; text-shadow: 3px 3px 8px rgba(0,0,0,0.9);
              letter-spacing: 2px; text-transform: uppercase;'>
        Advanced AI-Powered Crime Analytics & Prediction System
    </p>
    <div style='display: inline-block; background: linear-gradient(135deg, #8b0000, #5a0000); 
                padding: 15px 40px; border-radius: 30px; margin-top: 25px; 
                box-shadow: 0 10px 25px rgba(139,0,0,0.7); border: 3px solid #ff0000;'>
        <span style='color: white; font-weight: 900; font-size: 1.2rem; letter-spacing: 2px; text-transform: uppercase;'>
            🎯 PROFESSIONAL LAW ENFORCEMENT PLATFORM
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- ENHANCED SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎯 COMMAND CENTER")
    st.markdown("---")
    
    # Navigation with icons
    menu = st.radio(
        "📊 Select Analysis Module",
        ["🔍 Crime Analysis & Insights", "🤖 AI Crime Prediction Engine"],
        index=0
    )
    
    st.markdown("---")
    
    # Stats Summary in Sidebar
    st.markdown("### 📈 System Status")

# ---------------- DATA LOADING (NO FILE UPLOAD) ----------------
@st.cache_data
def load_data():
    years = np.arange(2015, 2024)
    return pd.DataFrame({
        "Year": years,
        "Theft": [400, 450, 480, 500, 550, 600, 650, 700, 720],
        "Assault": [300, 320, 350, 370, 400, 420, 450, 470, 500],
        "Burglary": [200, 210, 230, 240, 260, 280, 300, 320, 340],
        "Robbery": [150, 160, 175, 185, 200, 215, 230, 245, 260],
        "Fraud": [100, 120, 140, 160, 180, 200, 220, 240, 255]
    })

df = load_data()

# Update sidebar stats
with st.sidebar:
    total_crimes = df.iloc[:, 1:].sum().sum()
    st.metric("Total Crimes Recorded", f"{int(total_crimes):,}")
    st.metric("Crime Categories", len(df.columns) - 1)
    st.metric("Years Analyzed", len(df))
    st.metric("Database Status", "🟢 ONLINE")

st.markdown("---")

# =====================================================
# ============== CRIME ANALYSIS MODULE ================
# =====================================================
if "Crime Analysis" in menu:
    
    # Header Section
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(139,0,0,0.4), rgba(80,0,0,0.4)); 
                border-left: 8px solid #8b0000; padding: 30px; border-radius: 15px; margin-bottom: 35px;
                box-shadow: 0 10px 25px rgba(139,0,0,0.4); border: 3px solid rgba(139,0,0,0.6);'>
        <h2 style='color: #ff4444; margin: 0; text-shadow: 3px 3px 8px rgba(0,0,0,0.9); letter-spacing: 2px;'>
            📊 COMPREHENSIVE CRIME TREND ANALYSIS
        </h2>
        <p style='color: #ffffff; margin-top: 20px; font-size: 1.3rem; font-weight: 700; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);'>
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
            line=dict(color='#ff0000', width=5),
            marker=dict(size=16, color='#dc143c', 
                       line=dict(color='white', width=3)),
            hovertemplate='<b>Year: %{x}</b><br>Cases: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{crime_type} Cases Over Time",
            xaxis_title="Year",
            yaxis_title="Number of Cases",
            hovermode='x unified',
            plot_bgcolor='rgba(10, 10, 30, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=15, family='Arial Black'),
            title_font=dict(size=24, color='#ff4444', family='Arial Black'),
            xaxis=dict(gridcolor='rgba(139, 0, 0, 0.3)', color='#ffffff'),
            yaxis=dict(gridcolor='rgba(139, 0, 0, 0.3)', color='#ffffff'),
            height=550
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 All Crime Categories Comparison")
        
        # Multi-line chart for all crime types
        fig2 = go.Figure()
        
        colors = ['#ff0000', '#ff6600', '#ffcc00', '#00ff00', '#0099ff']
        
        for idx, col in enumerate(df.columns[1:]):
            fig2.add_trace(go.Scatter(
                x=df["Year"],
                y=df[col],
                mode='lines+markers',
                name=col,
                line=dict(width=4, color=colors[idx % len(colors)]),
                marker=dict(size=12)
            ))
        
        fig2.update_layout(
            title="Comprehensive Crime Comparison",
            xaxis_title="Year",
            yaxis_title="Number of Cases",
            hovermode='x unified',
            plot_bgcolor='rgba(10, 10, 30, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=15, family='Arial Black'),
            title_font=dict(size=24, color='#ff4444', family='Arial Black'),
            xaxis=dict(gridcolor='rgba(139, 0, 0, 0.3)', color='#ffffff'),
            yaxis=dict(gridcolor='rgba(139, 0, 0, 0.3)', color='#ffffff'),
            legend=dict(bgcolor='rgba(10, 10, 30, 0.9)', font=dict(color='#ffffff', size=13)),
            height=550
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
                    line=dict(color='white', width=3)
                ),
                text=latest_data.values,
                textposition='outside',
                textfont=dict(color='#ffffff', size=16, family='Arial Black')
            )
        ])
        
        fig3.update_layout(
            title=f"Crime Statistics - {int(df.iloc[-1, 0])}",
            xaxis_title="Crime Type",
            yaxis_title="Cases",
            plot_bgcolor='rgba(10, 10, 30, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=15, family='Arial Black'),
            title_font=dict(size=22, color='#ff4444', family='Arial Black'),
            xaxis=dict(color='#ffffff'),
            yaxis=dict(color='#ffffff'),
            height=450
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.markdown("### 📉 Statistical Analysis & Insights")
        
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.markdown("""
            <div style='background: rgba(26,26,60,0.7); padding: 30px; border-radius: 15px; 
                        border-left: 6px solid #8b0000; box-shadow: 0 10px 25px rgba(0,0,0,0.5);'>
                <h4 style='color: #ff4444; text-shadow: 2px 2px 6px rgba(0,0,0,0.9); letter-spacing: 1px;'>
                    📊 DESCRIPTIVE STATISTICS
                </h4>
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
            <div style='background: rgba(26,26,60,0.7); padding: 30px; border-radius: 15px; 
                        border-left: 6px solid #ff6600; box-shadow: 0 10px 25px rgba(0,0,0,0.5);'>
                <h4 style='color: #ff4444; text-shadow: 2px 2px 6px rgba(0,0,0,0.9); letter-spacing: 1px;'>
                    🎯 KEY INSIGHTS
                </h4>
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
            height=450
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
    <div style='background: linear-gradient(135deg, rgba(0,61,130,0.4), rgba(0,31,66,0.4)); 
                border-left: 8px solid #0066cc; padding: 30px; border-radius: 15px; margin-bottom: 35px;
                box-shadow: 0 10px 25px rgba(0,61,130,0.4); border: 3px solid rgba(0,102,204,0.6);'>
        <h2 style='color: #ff4444; margin: 0; text-shadow: 3px 3px 8px rgba(0,0,0,0.9); letter-spacing: 2px;'>
            🤖 AI-POWERED CRIME PREDICTION ENGINE
        </h2>
        <p style='color: #ffffff; margin-top: 20px; font-size: 1.3rem; font-weight: 700; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);'>
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
            help="Choose the crime type you want to predict",
            key="prediction_select"
        )
    
    with col_pred2:
        st.markdown("""
        <div style='background: rgba(0,61,130,0.4); padding: 20px; border-radius: 15px; 
                    border: 4px solid #0066cc; margin-top: 25px; box-shadow: 0 8px 20px rgba(0,61,130,0.5);'>
            <p style='margin: 0; text-align: center; font-weight: 900; font-size: 1.2rem; color: #ffffff;
                      text-shadow: 2px 2px 6px rgba(0,0,0,0.9); letter-spacing: 1px;'>
                🧠 RANDOM FOREST ML
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
            confidence = predicted_value * 0.1
            lower_bound = int(predicted_value - confidence)
            upper_bound = int(predicted_value + confidence)
        
        # Professional success message
        st.success("✅ PREDICTION ANALYSIS COMPLETE")
        
        # RESULTS DISPLAY
        st.markdown("""
        <div class='prediction-complete' style='background: linear-gradient(135deg, rgba(0,100,0,0.4), rgba(0,50,0,0.4)); 
                    border: 4px solid #00ff00; padding: 35px; border-radius: 20px; margin: 35px 0;
                    box-shadow: 0 15px 40px rgba(0,255,0,0.3);'>
            <h2 style='text-align: center; color: #00ff00; margin-bottom: 20px; text-shadow: 3px 3px 8px rgba(0,0,0,0.9);
                       letter-spacing: 2px; text-transform: uppercase;'>
                🎯 PREDICTION RESULTS
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Main Prediction Metric
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        
        with col_res2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #8b0000, #5a0000); 
                        padding: 50px; border-radius: 20px; text-align: center;
                        box-shadow: 0 20px 50px rgba(139,0,0,0.7); border: 5px solid #ff0000;'>
                <h3 style='color: #ffffff; margin: 0; font-size: 1.4rem; text-shadow: 3px 3px 8px rgba(0,0,0,0.9);
                           letter-spacing: 2px; text-transform: uppercase;'>
                    PREDICTED {crime_type.upper()} CASES
                </h3>
                <h1 style='color: #ff0000; margin: 30px 0; font-size: 5rem; font-weight: 900; 
                           text-shadow: 5px 5px 10px rgba(0,0,0,0.9);'>
                    {predicted_value:,}
                </h1>
                <p style='color: #ffffff; margin: 0; font-size: 1.3rem; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
                          font-weight: 700;'>
                    TARGET YEAR: {future_year}
                </p>
                <p style='color: #ffcc00; margin-top: 25px; font-size: 1.2rem; font-weight: 800; 
                          text-shadow: 2px 2px 6px rgba(0,0,0,0.9);'>
                    CONFIDENCE RANGE: {lower_bound:,} - {upper_bound:,} CASES
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
                line=dict(color='#0099ff', width=5),
                marker=dict(size=14, color='#0066cc')
            ))
            
            # Prediction point
            fig_pred.add_trace(go.Scatter(
                x=[future_year],
                y=[predicted_value],
                mode='markers',
                name='Prediction',
                marker=dict(
                    size=32,
                    color='#ff0000',
                    symbol='star',
                    line=dict(color='#ffff00', width=4)
                )
            ))
            
            # Prediction line
            fig_pred.add_trace(go.Scatter(
                x=[df["Year"].iloc[-1], future_year],
                y=[df[crime_type].iloc[-1], predicted_value],
                mode='lines',
                name='Forecast Trend',
                line=dict(color='#ff0000', width=5, dash='dash')
            ))
            
            # Confidence interval
            fig_pred.add_trace(go.Scatter(
                x=[future_year, future_year],
                y=[lower_bound, upper_bound],
                mode='lines',
                name='Confidence Range',
                line=dict(color='#ffcc00', width=0),
                fill='toself',
                fillcolor='rgba(255, 204, 0, 0.25)'
            ))
            
            fig_pred.update_layout(
                title=f"{crime_type} Forecast Analysis",
                xaxis_title="Year",
                yaxis_title="Number of Cases",
                hovermode='x unified',
                plot_bgcolor='rgba(10, 10, 30, 0.6)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', size=15, family='Arial Black'),
                title_font=dict(size=24, color='#ff4444', family='Arial Black'),
                xaxis=dict(gridcolor='rgba(139, 0, 0, 0.3)', color='#ffffff'),
                yaxis=dict(gridcolor='rgba(139, 0, 0, 0.3)', color='#ffffff'),
                legend=dict(bgcolor='rgba(10, 10, 30, 0.9)', font=dict(color='#ffffff', size=13)),
                height=600
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
                <div style='background: rgba(26,26,60,0.7); padding: 30px; border-radius: 15px; 
                            border-left: 6px solid #0066cc; box-shadow: 0 10px 25px rgba(0,0,0,0.5);'>
                    <h4 style='color: #ff4444; text-shadow: 2px 2px 6px rgba(0,0,0,0.9); letter-spacing: 1px;'>
                        📊 COMPARISON METRICS
                    </h4>
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
                <div style='background: rgba(26,26,60,0.7); padding: 30px; border-radius: 15px; 
                            border-left: 6px solid #ff0000; box-shadow: 0 10px 25px rgba(0,0,0,0.5);'>
                    <h4 style='color: #ff4444; text-shadow: 2px 2px 6px rgba(0,0,0,0.9); letter-spacing: 1px;'>
                        🚨 RISK ASSESSMENT
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                
                if percent_diff > 20:
                    risk_level = "🔴 HIGH RISK"
                    risk_desc = "Significant increase predicted. Immediate action recommended."
                    risk_color = "#ff0000"
                elif percent_diff > 5:
                    risk_level = "🟡 MODERATE RISK"
                    risk_desc = "Notable increase expected. Enhanced monitoring advised."
                    risk_color = "#ffcc00"
                else:
                    risk_level = "🟢 LOW RISK"
                    risk_desc = "Stable or decreasing trend. Continue current strategies."
                    risk_color = "#00ff00"
                
                st.markdown(f"""
                <div style='background: rgba(10,10,30,0.8); padding: 30px; border-radius: 15px; 
                            border: 4px solid {risk_color}; box-shadow: 0 10px 25px rgba(0,0,0,0.5);'>
                    <h3 style='color: {risk_color}; text-align: center; margin-bottom: 25px; text-shadow: 3px 3px 8px rgba(0,0,0,0.9);
                               font-size: 1.5rem; letter-spacing: 2px;'>
                        {risk_level}
                    </h3>
                    <p style='color: #ffffff; text-align: center; margin: 0; font-size: 1.2rem; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
                              font-weight: 700;'>
                        {risk_desc}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("##### 💡 RECOMMENDATIONS:")
                if percent_diff > 20:
                    st.markdown("""
                    - 🚨 **Increase patrol frequency**
                    - 📊 **Allocate additional resources**
                    - 🎯 **Implement preventive programs**
                    - 📱 **Enhance community awareness**
                    """)
                elif percent_diff > 5:
                    st.markdown("""
                    - 👁️ **Monitor hotspot areas**
                    - 📈 **Review current strategies**
                    - 🤝 **Strengthen community engagement**
                    """)
                else:
                    st.markdown("""
                    - ✅ **Maintain current approach**
                    - 📊 **Continue data monitoring**
                    - 🎯 **Optimize resource allocation**
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
                        marker=dict(color='#ff0000'),
                        text=[f"{imp:.2%}" for imp in importance],
                        textposition='outside',
                        textfont=dict(color='#ffffff', size=16, family='Arial Black')
                    )
                ])
                
                fig_imp.update_layout(
                    title="Model Feature Importance",
                    yaxis_title="Importance Score",
                    plot_bgcolor='rgba(10, 10, 30, 0.6)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff', family='Arial Black'),
                    title_font=dict(color='#ff4444'),
                    xaxis=dict(color='#ffffff'),
                    yaxis=dict(color='#ffffff'),
                    height=400
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
                "📄 Download Detailed Report",
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
Professional Law Enforcement Platform"""
            
            st.download_button(
                "📋 Download Summary",
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
                "📊 Download Forecast Data",
                data=forecast_csv,
                file_name=f"forecast_data_{crime_type}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ---------------- PROFESSIONAL LAW ENFORCEMENT FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 40px; background: linear-gradient(135deg, rgba(139,0,0,0.4), rgba(80,0,0,0.4)); 
            border-radius: 20px; margin-top: 60px; border: 4px solid #8b0000; box-shadow: 0 15px 35px rgba(139,0,0,0.5);'>
    <h3 style='color: #ff0000; margin-bottom: 25px; text-shadow: 3px 3px 8px rgba(0,0,0,0.9); 
               font-size: 2rem; letter-spacing: 3px; text-transform: uppercase;'>
        🚨 CRIME INTELLIGENCE PRO 🚨
    </h3>
    <p style='color: #ffffff; font-size: 1.3rem; margin-bottom: 20px; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
              font-weight: 800; letter-spacing: 1px;'>
        Professional AI-Powered Crime Analytics & Prediction Platform
    </p>
    <p style='color: #ffffff; font-size: 1.1rem; opacity: 0.95; text-shadow: 2px 2px 4px rgba(0,0,0,0.9);
              font-weight: 700;'>
        Advanced Machine Learning | Strategic Intelligence | Data Science
    </p>
    <div style='margin-top: 30px;'>
        <span style='background: linear-gradient(135deg, #8b0000, #5a0000); padding: 12px 30px; border-radius: 30px; 
                     color: white; font-weight: 900; margin: 0 10px; box-shadow: 0 8px 20px rgba(139,0,0,0.5);
                     border: 2px solid #ff0000;'>
            🚀 INNOVATION
        </span>
        <span style='background: linear-gradient(135deg, #003d82, #001f42); padding: 12px 30px; border-radius: 30px; 
                     color: white; font-weight: 900; margin: 0 10px; box-shadow: 0 8px 20px rgba(0,61,130,0.5);
                     border: 2px solid #0066cc;'>
            🧠 AI/ML
        </span>
        <span style='background: linear-gradient(135deg, #005a00, #003300); padding: 12px 30px; border-radius: 30px; 
                     color: white; font-weight: 900; margin: 0 10px; box-shadow: 0 8px 20px rgba(0,90,0,0.5);
                     border: 2px solid #00ff00;'>
            📊 ANALYTICS
        </span>
    </div>
</div>
""", unsafe_allow_html=True)