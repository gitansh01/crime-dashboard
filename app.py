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

# ---------------- OPTIMIZED PROFESSIONAL CSS ----------------
st.markdown("""
<style>
/* Main Background */
[data-testid="stAppViewContainer"] {
    background: 
        linear-gradient(135deg, rgba(10, 10, 30, 0.98) 0%, rgba(25, 25, 60, 0.98) 25%, rgba(15, 15, 40, 0.98) 50%, rgba(30, 30, 70, 0.98) 75%, rgba(10, 10, 30, 0.98) 100%),
        repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(139, 0, 0, 0.05) 2px, rgba(139, 0, 0, 0.05) 4px),
        repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(139, 0, 0, 0.05) 2px, rgba(139, 0, 0, 0.05) 4px);
    color: #FFFFFF;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1e 0%, #1a1a3c 50%, #0f0f28 100%);
    border-right: 4px solid #8b0000;
    box-shadow: 5px 0 20px rgba(139, 0, 0, 0.3);
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
}

/* ALL TEXT - Reduced Size */
label, .stMarkdown, .stText {
    color: #ffffff !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    font-size: 0.95rem !important;
}

p, span, div {
    color: #ffffff !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
}

/* Select Box - MAXIMUM CONTRAST */
.stSelectbox div[data-baseweb="select"] > div {
    background: #000000 !important;
    color: #ffffff !important;
    border: 3px solid #ff0000 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    box-shadow: 0 5px 15px rgba(255, 0, 0, 0.5), inset 0 1px 3px rgba(255, 255, 255, 0.2);
}

/* Select Box Dropdown Menu - BLACK BACKGROUND */
[data-baseweb="popover"] {
    background: #000000 !important;
    border: 3px solid #ff0000 !important;
}

/* Select Box Options - BLACK WITH WHITE TEXT */
.stSelectbox li {
    background: #000000 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px 20px !important;
    border-bottom: 1px solid #333333 !important;
}

.stSelectbox li:hover {
    background: linear-gradient(135deg, #dc143c, #ff0000) !important;
    color: #ffffff !important;
}

/* Select Box Selected Text - WHITE ON BLACK */
.stSelectbox div[data-baseweb="select"] span {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Number Input */
.stNumberInput input {
    background: #000000 !important;
    color: #ffffff !important;
    border: 3px solid #ff0000 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    box-shadow: 0 5px 15px rgba(255, 0, 0, 0.5);
}

.stNumberInput label {
    font-size: 0.95rem !important;
}

/* Main Title - Reduced Size */
h1 {
    color: #ffffff !important;
    text-shadow: 5px 5px 10px rgba(0,0,0,0.9), 0 0 40px rgba(139, 0, 0, 0.6);
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    padding: 20px 0;
    text-align: center;
    letter-spacing: 3px;
    text-transform: uppercase;
}

/* Subheaders - Reduced Size */
h2 {
    color: #ff4444 !important;
    font-weight: 800 !important;
    text-shadow: 3px 3px 8px rgba(0,0,0,0.9);
    margin-top: 15px !important;
    letter-spacing: 1px;
    font-size: 1.6rem !important;
}

h3 {
    color: #ff4444 !important;
    font-weight: 700 !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
    font-size: 1.3rem !important;
}

h4 {
    color: #ffffff !important;
    font-weight: 700 !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.8);
    font-size: 1.1rem !important;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(139, 0, 0, 0.3), rgba(80, 0, 0, 0.25));
    border: 4px solid #8b0000;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 15px 40px rgba(139, 0, 0, 0.5), inset 0 2px 5px rgba(255, 255, 255, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    backdrop-filter: blur(15px);
}

[data-testid="metric-container"]:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 50px rgba(139, 0, 0, 0.7);
}

[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    font-weight: 900 !important;
    color: #ff0000 !important;
    text-shadow: 4px 4px 8px rgba(0,0,0,0.9), 0 0 30px rgba(255, 0, 0, 0.6);
}

[data-testid="stMetricLabel"] {
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

[data-testid="stMetricDelta"] {
    font-size: 0.9rem !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #8b0000 0%, #5a0000 100%) !important;
    color: white !important;
    font-weight: 900 !important;
    font-size: 1rem !important;
    border-radius: 12px !important;
    padding: 16px 40px !important;
    border: 4px solid #ff0000;
    box-shadow: 0 12px 30px rgba(139, 0, 0, 0.6);
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.stButton>button:hover {
    transform: translateY(-4px) scale(1.05);
    box-shadow: 0 18px 40px rgba(139, 0, 0, 0.8);
    background: linear-gradient(135deg, #dc143c 0%, #8b0000 100%) !important;
}

/* Download Button */
.stDownloadButton>button {
    background: linear-gradient(135deg, #003d82 0%, #001f42 100%) !important;
    color: white !important;
    font-weight: 800 !important;
    font-size: 0.9rem !important;
    border-radius: 10px !important;
    padding: 12px 30px !important;
    border: 3px solid #0066cc;
    box-shadow: 0 10px 25px rgba(0, 61, 130, 0.5);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: rgba(10, 10, 30, 0.8);
    border-radius: 15px;
    border: 3px solid #8b0000;
    box-shadow: 0 12px 30px rgba(0,0,0,0.6);
}

/* Divider */
hr {
    border: 3px solid #8b0000;
    margin: 30px 0;
    box-shadow: 0 3px 15px rgba(139, 0, 0, 0.6);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: rgba(10, 10, 30, 0.8);
    border-radius: 15px;
    padding: 15px;
    border: 2px solid #8b0000;
}

.stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, rgba(139, 0, 0, 0.3), rgba(80, 0, 0, 0.3));
    border-radius: 12px;
    color: #ffffff;
    font-weight: 800;
    font-size: 1rem;
    padding: 14px 28px;
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

/* Radio Buttons */
.stRadio > div {
    background: rgba(26, 26, 60, 0.6);
    border-radius: 12px;
    padding: 20px;
    border: 3px solid #8b0000;
}

.stRadio label {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
}

/* Alert Boxes */
.stAlert {
    background: rgba(10, 10, 30, 0.95) !important;
    border-radius: 12px;
    border-left: 6px solid #8b0000;
    padding: 15px;
    color: #ffffff !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}

.stSuccess {
    background: rgba(0, 50, 0, 0.95) !important;
    border-left: 6px solid #00ff00 !important;
}

.stInfo {
    background: rgba(0, 20, 60, 0.95) !important;
    border-left: 6px solid #0066ff !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE - Reduced Size ----------------
st.markdown("""
<div style='text-align: center; margin-bottom: 35px;'>
    <h1 style='margin-bottom: 15px; letter-spacing: 3px;'>🚨 CRIME INTELLIGENCE PRO 🚨</h1>
    <p style='font-size: 1.3rem; color: #ff4444; font-weight: 800; text-shadow: 3px 3px 8px rgba(0,0,0,0.9);
              letter-spacing: 1px; text-transform: uppercase;'>
        Advanced AI-Powered Crime Analytics & Prediction System
    </p>
    <div style='display: inline-block; background: linear-gradient(135deg, #8b0000, #5a0000); 
                padding: 10px 30px; border-radius: 25px; margin-top: 20px; 
                box-shadow: 0 10px 25px rgba(139,0,0,0.7); border: 3px solid #ff0000;'>
        <span style='color: white; font-weight: 900; font-size: 1rem; letter-spacing: 2px; text-transform: uppercase;'>
            🎯 PROFESSIONAL LAW ENFORCEMENT PLATFORM
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎯 COMMAND CENTER")
    st.markdown("---")
    
    menu = st.radio(
        "📊 Select Module",
        ["🔍 Crime Analysis & Insights", "🤖 AI Crime Prediction Engine"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📈 System Status")

# ---------------- DATA LOADING ----------------
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

with st.sidebar:
    total_crimes = df.iloc[:, 1:].sum().sum()
    st.metric("Total Crimes", f"{int(total_crimes):,}")
    st.metric("Categories", len(df.columns) - 1)
    st.metric("Years Analyzed", len(df))
    st.metric("Status", "🟢 ONLINE")

st.markdown("---")

# =====================================================
# ============== CRIME ANALYSIS MODULE ================
# =====================================================
if "Crime Analysis" in menu:
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(139,0,0,0.4), rgba(80,0,0,0.4)); 
                border-left: 6px solid #8b0000; padding: 20px; border-radius: 15px; margin-bottom: 25px;
                box-shadow: 0 10px 25px rgba(139,0,0,0.4); border: 3px solid rgba(139,0,0,0.6);'>
        <h2 style='color: #ff4444; margin: 0; text-shadow: 3px 3px 8px rgba(0,0,0,0.9); letter-spacing: 1px;'>
            📊 COMPREHENSIVE CRIME TREND ANALYSIS
        </h2>
        <p style='color: #ffffff; margin-top: 10px; font-size: 1.1rem; font-weight: 700; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);'>
            Advanced statistical analysis and visualization of crime patterns
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_select1, col_select2 = st.columns([2, 1])
    with col_select1:
        crime_type = st.selectbox(
            "🎯 Select Crime Category",
            df.columns[1:],
            help="Choose crime type to analyze"
        )
    
    # Calculate metrics
    total_cases = int(df[crime_type].sum())
    avg_cases = int(df[crime_type].mean())
    max_year = int(df.loc[df[crime_type].idxmax(), "Year"])
    min_year = int(df.loc[df[crime_type].idxmin(), "Year"])
    growth_rate = round((df[crime_type].iloc[-1] - df[crime_type].iloc[0]) / df[crime_type].iloc[0] * 100, 2)
    recent_trend = "Increasing 📈" if df[crime_type].iloc[-1] > df[crime_type].iloc[-3] else "Decreasing 📉"
    
    st.markdown("### 🎯 KEY PERFORMANCE INDICATORS")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Cases", f"{total_cases:,}", delta=f"{growth_rate}% growth", delta_color="inverse")
    with col2:
        st.metric("📅 Peak Year", max_year, delta=f"{int(df.loc[df[crime_type].idxmax(), crime_type])} cases")
    with col3:
        st.metric("📉 Lowest Year", min_year, delta=f"{int(df.loc[df[crime_type].idxmin(), crime_type])} cases")
    with col4:
        st.metric("📈 Trend", recent_trend.split()[0], delta=f"Avg: {avg_cases}")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend", "📊 Comparison", "📉 Statistics", "🗂️ Data"])
    
    with tab1:
        st.markdown("### 📈 Crime Trend Chart")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df["Year"],
            y=df[crime_type],
            mode='lines+markers',
            name=crime_type,
            line=dict(color='#ff0000', width=5),
            marker=dict(size=16, color='#dc143c', line=dict(color='white', width=3)),
            hovertemplate='<b>Year: %{x}</b><br>Cases: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=f"{crime_type} Cases Over Time", font=dict(size=20, color='#ffffff', family='Arial Black')),
            xaxis_title="Year",
            yaxis_title="Cases",
            hovermode='x unified',
            plot_bgcolor='rgba(10, 10, 30, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=14, family='Arial'),
            xaxis=dict(gridcolor='rgba(139, 0, 0, 0.3)', color='#ffffff', title_font=dict(size=14)),
            yaxis=dict(gridcolor='rgba(139, 0, 0, 0.3)', color='#ffffff', title_font=dict(size=14)),
            height=500,
            hoverlabel=dict(bgcolor='#000000', font_size=14, font_family='Arial', font_color='#ffffff')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 All Categories Comparison")
        
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
            title=dict(text="Crime Comparison", font=dict(size=20, color='#ffffff', family='Arial Black')),
            xaxis_title="Year",
            yaxis_title="Cases",
            hovermode='x unified',
            plot_bgcolor='rgba(10, 10, 30, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=14, family='Arial'),
            xaxis=dict(gridcolor='rgba(139, 0, 0, 0.3)', color='#ffffff', title_font=dict(size=14)),
            yaxis=dict(gridcolor='rgba(139, 0, 0, 0.3)', color='#ffffff', title_font=dict(size=14)),
            legend=dict(bgcolor='rgba(0, 0, 0, 0.9)', font=dict(color='#ffffff', size=13), bordercolor='#ff0000', borderwidth=2),
            height=500,
            hoverlabel=dict(bgcolor='#000000', font_size=14, font_family='Arial', font_color='#ffffff')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### 📊 Latest Year Distribution")
        latest_data = df.iloc[-1, 1:]
        
        fig3 = go.Figure(data=[
            go.Bar(
                x=latest_data.index,
                y=latest_data.values,
                marker=dict(color=latest_data.values, colorscale='Reds', line=dict(color='white', width=3)),
                text=latest_data.values,
                textposition='outside',
                textfont=dict(color='#ffffff', size=15, family='Arial Black')
            )
        ])
        
        fig3.update_layout(
            title=dict(text=f"Crime Statistics - {int(df.iloc[-1, 0])}", font=dict(size=18, color='#ffffff', family='Arial Black')),
            xaxis_title="Crime Type",
            yaxis_title="Cases",
            plot_bgcolor='rgba(10, 10, 30, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=14, family='Arial'),
            xaxis=dict(color='#ffffff', title_font=dict(size=14)),
            yaxis=dict(color='#ffffff', title_font=dict(size=14)),
            height=400
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.markdown("### 📉 Statistical Analysis")
        
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.markdown("""
            <div style='background: rgba(26,26,60,0.7); padding: 20px; border-radius: 12px; 
                        border-left: 5px solid #8b0000; box-shadow: 0 8px 20px rgba(0,0,0,0.5);'>
                <h4 style='color: #ff4444; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);'>📊 STATISTICS</h4>
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
            <div style='background: rgba(26,26,60,0.7); padding: 20px; border-radius: 12px; 
                        border-left: 5px solid #ff6600; box-shadow: 0 8px 20px rgba(0,0,0,0.5);'>
                <h4 style='color: #ff4444; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);'>🎯 INSIGHTS</h4>
            </div>
            """, unsafe_allow_html=True)
            
            yoy_change = df[crime_type].iloc[-1] - df[crime_type].iloc[-2]
            yoy_percent = (yoy_change / df[crime_type].iloc[-2]) * 100
            
            st.markdown(f"""
            - **Trend:** {recent_trend}
            - **YoY Change:** {yoy_change:+.0f} cases ({yoy_percent:+.2f}%)
            - **Volatility:** {df[crime_type].std():.2f}
            - **Growth:** {growth_rate:+.2f}% since {int(df['Year'].iloc[0])}
            """)
            
            if growth_rate > 30:
                risk = "🔴 HIGH RISK"
            elif growth_rate > 10:
                risk = "🟡 MODERATE RISK"
            else:
                risk = "🟢 LOW RISK"
            
            st.info(f"**Risk:** {risk}")
    
    with tab4:
        st.markdown("### 🗂️ Complete Dataset")
        st.dataframe(df.style.background_gradient(cmap='Reds', subset=df.columns[1:]), use_container_width=True, height=400)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Dataset", data=csv, file_name=f"crime_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# =====================================================
# ============ AI PREDICTION MODULE ===================
# =====================================================
if "AI Crime Prediction" in menu:
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(0,61,130,0.4), rgba(0,31,66,0.4)); 
                border-left: 6px solid #0066cc; padding: 20px; border-radius: 15px; margin-bottom: 25px;
                box-shadow: 0 10px 25px rgba(0,61,130,0.4); border: 3px solid rgba(0,102,204,0.6);'>
        <h2 style='color: #ff4444; margin: 0; text-shadow: 3px 3px 8px rgba(0,0,0,0.9); letter-spacing: 1px;'>
            🤖 AI-POWERED CRIME PREDICTION ENGINE
        </h2>
        <p style='color: #ffffff; margin-top: 10px; font-size: 1.1rem; font-weight: 700; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);'>
            Machine Learning for Future Crime Rate Forecasting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_pred1, col_pred2 = st.columns([2, 1])
    with col_pred1:
        crime_type = st.selectbox(
            "🎯 Select Crime Category",
            df.columns[1:],
            help="Choose crime type to predict",
            key="pred_select"
        )
    
    with col_pred2:
        st.markdown("""
        <div style='background: rgba(0,61,130,0.4); padding: 15px; border-radius: 12px; 
                    border: 3px solid #0066cc; margin-top: 25px; box-shadow: 0 6px 15px rgba(0,61,130,0.5);'>
            <p style='margin: 0; text-align: center; font-weight: 900; font-size: 1rem; color: #ffffff;
                      text-shadow: 2px 2px 6px rgba(0,0,0,0.9);'>
                🧠 RANDOM FOREST ML
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    X = df[["Year"]]
    y = df[crime_type]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with st.spinner("🔄 Training Model..."):
        model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    st.markdown("### 🎯 MODEL PERFORMANCE")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric("🎯 R² Score", f"{r2_test:.3f}", delta=f"Train: {r2_train:.3f}")
    with col_m2:
        st.metric("📊 MAE", f"{mae:.2f}", delta="cases")
    with col_m3:
        st.metric("📈 RMSE", f"{rmse:.2f}")
    with col_m4:
        accuracy_label = "Excellent" if r2_test > 0.9 else "Good" if r2_test > 0.7 else "Fair"
        st.metric("✅ Quality", accuracy_label, delta=f"{r2_test*100:.1f}%")
    
    st.markdown("---")
    st.markdown("### 🔮 MAKE PREDICTION")
    
    col_input1, col_input2, col_input3 = st.columns([2, 1, 1])
    
    with col_input1:
        future_year = st.number_input("📅 Future Year", min_value=int(df["Year"].max()) + 1, max_value=2040, value=int(df["Year"].max()) + 1, step=1)
    
    with col_input2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🚀 PREDICT", use_container_width=True)
    
    with col_input3:
        st.markdown("<br>", unsafe_allow_html=True)
        years_ahead = future_year - int(df["Year"].max())
        st.info(f"📊 {years_ahead} year(s)")
    
    if predict_button:
        with st.spinner("🔮 Analyzing..."):
            prediction = model.predict([[future_year]])
            predicted_value = int(prediction[0])
            confidence = predicted_value * 0.1
            lower_bound = int(predicted_value - confidence)
            upper_bound = int(predicted_value + confidence)
        
        st.success("✅ PREDICTION COMPLETE")
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(0,100,0,0.4), rgba(0,50,0,0.4)); 
                    border: 4px solid #00ff00; padding: 25px; border-radius: 20px; margin: 25px 0;
                    box-shadow: 0 12px 35px rgba(0,255,0,0.3);'>
            <h2 style='text-align: center; color: #00ff00; margin: 0; text-shadow: 3px 3px 8px rgba(0,0,0,0.9);
                       letter-spacing: 1px; text-transform: uppercase; font-size: 1.5rem;'>
                🎯 PREDICTION RESULTS
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
        
        with col_res2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #8b0000, #5a0000); 
                        padding: 40px; border-radius: 20px; text-align: center;
                        box-shadow: 0 18px 45px rgba(139,0,0,0.7); border: 4px solid #ff0000;'>
                <h3 style='color: #ffffff; margin: 0; font-size: 1.2rem; text-shadow: 3px 3px 8px rgba(0,0,0,0.9);
                           letter-spacing: 1px; text-transform: uppercase;'>
                    PREDICTED {crime_type.upper()} CASES
                </h3>
                <h1 style='color: #ff0000; margin: 25px 0; font-size: 4rem; font-weight: 900; 
                           text-shadow: 5px 5px 10px rgba(0,0,0,0.9);'>
                    {predicted_value:,}
                </h1>
                <p style='color: #ffffff; margin: 0; font-size: 1.1rem; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
                          font-weight: 700;'>
                    YEAR: {future_year}
                </p>
                <p style='color: #ffcc00; margin-top: 20px; font-size: 1rem; font-weight: 800; 
                          text-shadow: 2px 2px 6px rgba(0,0,0,0.9);'>
                    RANGE: {lower_bound:,} - {upper_bound:,} CASES
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📈 Forecast", "📊 Analysis", "🎯 Model Info"])
        
        with viz_tab1:
            st.markdown("### 📈 Forecast Visualization")
            
            fig_pred = go.Figure()
            
            # Historical data - BRIGHT BLUE
            fig_pred.add_trace(go.Scatter(
                x=df["Year"],
                y=df[crime_type],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='#00ccff', width=5),
                marker=dict(size=14, color='#0099ff'),
                hovertemplate='<b>Year: %{x}</b><br>Cases: %{y}<extra></extra>'
            ))
            
            # Prediction point - YELLOW STAR
            fig_pred.add_trace(go.Scatter(
                x=[future_year],
                y=[predicted_value],
                mode='markers',
                name='Prediction',
                marker=dict(size=35, color='#ffff00', symbol='star', line=dict(color='#ff0000', width=4)),
                hovertemplate='<b>Predicted</b><br>Year: %{x}<br>Cases: %{y}<extra></extra>'
            ))
            
            # Forecast line - RED DASHED
            fig_pred.add_trace(go.Scatter(
                x=[df["Year"].iloc[-1], future_year],
                y=[df[crime_type].iloc[-1], predicted_value],
                mode='lines',
                name='Forecast Trend',
                line=dict(color='#ff0000', width=5, dash='dash'),
                hovertemplate='<b>Forecast</b><br>Year: %{x}<br>Cases: %{y}<extra></extra>'
            ))
            
            # Confidence interval - BRIGHT YELLOW with opacity
            fig_pred.add_trace(go.Scatter(
                x=[future_year, future_year, future_year],
                y=[lower_bound, predicted_value, upper_bound],
                mode='lines',
                name='Confidence Range',
                fill='toself',
                fillcolor='rgba(255, 255, 0, 0.3)',
                line=dict(color='#ffff00', width=3),
                hovertemplate='<b>Confidence Range</b><br>Lower: ' + f'{lower_bound:,}' + '<br>Upper: ' + f'{upper_bound:,}' + '<extra></extra>'
            ))
            
            fig_pred.update_layout(
                title=dict(text=f"{crime_type} Forecast Analysis", font=dict(size=20, color='#ffffff', family='Arial Black')),
                xaxis_title="Year",
                yaxis_title="Cases",
                hovermode='closest',
                plot_bgcolor='rgba(10, 10, 30, 0.6)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', size=14, family='Arial'),
                xaxis=dict(gridcolor='rgba(139, 0, 0, 0.3)', color='#ffffff', title_font=dict(size=14)),
                yaxis=dict(gridcolor='rgba(139, 0, 0, 0.3)', color='#ffffff', title_font=dict(size=14)),
                legend=dict(
                    bgcolor='rgba(0, 0, 0, 0.95)', 
                    font=dict(color='#ffffff', size=13, family='Arial Black'), 
                    bordercolor='#ff0000', 
                    borderwidth=2,
                    x=0.02,
                    y=0.98
                ),
                height=550,
                hoverlabel=dict(
                    bgcolor='#000000', 
                    font_size=15, 
                    font_family='Arial Black', 
                    font_color='#ffffff',
                    bordercolor='#ff0000'
                )
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with viz_tab2:
            st.markdown("### 📊 Analysis")
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                hist_avg = df[crime_type].mean()
                diff_from_avg = predicted_value - hist_avg
                percent_diff = (diff_from_avg / hist_avg) * 100
                
                st.markdown("""
                <div style='background: rgba(26,26,60,0.7); padding: 20px; border-radius: 12px; 
                            border-left: 5px solid #0066cc; box-shadow: 0 8px 20px rgba(0,0,0,0.5);'>
                    <h4 style='color: #ff4444; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);'>📊 METRICS</h4>
                </div>
                """, unsafe_allow_html=True)
                
                comparison_df = pd.DataFrame({
                    "Metric": ["Historical Avg", "Predicted Value", "Difference", "% Change", "Latest Year"],
                    "Value": [f"{hist_avg:.0f}", f"{predicted_value:,}", f"{diff_from_avg:+.0f}", f"{percent_diff:+.2f}%", f"{df[crime_type].iloc[-1]:.0f}"]
                })
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            with col_comp2:
                st.markdown("""
                <div style='background: rgba(26,26,60,0.7); padding: 20px; border-radius: 12px; 
                            border-left: 5px solid #ff0000; box-shadow: 0 8px 20px rgba(0,0,0,0.5);'>
                    <h4 style='color: #ff4444; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);'>🚨 RISK</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if percent_diff > 20:
                    risk_level = "🔴 HIGH RISK"
                    risk_desc = "Significant increase predicted."
                    risk_color = "#ff0000"
                elif percent_diff > 5:
                    risk_level = "🟡 MODERATE RISK"
                    risk_desc = "Notable increase expected."
                    risk_color = "#ffcc00"
                else:
                    risk_level = "🟢 LOW RISK"
                    risk_desc = "Stable or decreasing trend."
                    risk_color = "#00ff00"
                
                st.markdown(f"""
                <div style='background: rgba(10,10,30,0.8); padding: 20px; border-radius: 12px; 
                            border: 3px solid {risk_color}; box-shadow: 0 8px 20px rgba(0,0,0,0.5);'>
                    <h3 style='color: {risk_color}; text-align: center; margin-bottom: 15px; text-shadow: 3px 3px 8px rgba(0,0,0,0.9);
                               font-size: 1.3rem;'>
                        {risk_level}
                    </h3>
                    <p style='color: #ffffff; text-align: center; margin: 0; font-size: 1rem; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
                              font-weight: 700;'>
                        {risk_desc}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with viz_tab3:
            st.markdown("### 🎯 Model Details")
            
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.markdown("#### 📊 Feature Importance")
                importance = model.feature_importances_
                
                fig_imp = go.Figure(data=[
                    go.Bar(
                        x=["Year"],
                        y=importance,
                        marker=dict(color='#ff0000'),
                        text=[f"{imp:.2%}" for imp in importance],
                        textposition='outside',
                        textfont=dict(color='#ffffff', size=15, family='Arial Black')
                    )
                ])
                
                fig_imp.update_layout(
                    plot_bgcolor='rgba(10, 10, 30, 0.6)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff', family='Arial'),
                    xaxis=dict(color='#ffffff'),
                    yaxis=dict(color='#ffffff', title='Importance'),
                    height=350
                )
                
                st.plotly_chart(fig_imp, use_container_width=True)
            
            with col_insight2:
                st.markdown("#### 🧠 Configuration")
                st.markdown(f"""
                - **Algorithm:** Random Forest
                - **Trees:** 200
                - **Max Depth:** 10
                - **Training:** {len(X_train)} samples
                - **Testing:** {len(X_test)} samples
                - **R² Train:** {r2_train:.4f}
                - **R² Test:** {r2_test:.4f}
                - **MAE:** {mae:.2f}
                - **RMSE:** {rmse:.2f}
                """)
        
        st.markdown("---")
        st.markdown("### 📥 EXPORT REPORTS")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            report = pd.DataFrame({
                "Metric": ["Crime Type", "Year", "Predicted", "Lower Bound", "Upper Bound", "Hist Avg", "Change %", "R² Score", "MAE", "RMSE"],
                "Value": [crime_type, future_year, predicted_value, lower_bound, upper_bound, f"{hist_avg:.2f}", f"{percent_diff:.2f}%", f"{r2_test:.4f}", f"{mae:.2f}", f"{rmse:.2f}"]
            })
            csv = report.to_csv(index=False).encode("utf-8")
            st.download_button("📄 Detailed Report", data=csv, file_name=f"prediction_{crime_type}_{future_year}.csv", mime="text/csv", use_container_width=True)
        
        with col_dl2:
            summary = f"""PREDICTION REPORT
=================
Crime: {crime_type}
Year: {future_year}
Predicted: {predicted_value:,}
Range: {lower_bound:,} - {upper_bound:,}
R²: {r2_test:.4f}
MAE: {mae:.2f}
Change: {percent_diff:+.2f}%"""
            st.download_button("📋 Summary", data=summary, file_name=f"summary_{future_year}.txt", mime="text/plain", use_container_width=True)
        
        with col_dl3:
            forecast_df = pd.concat([df[["Year", crime_type]], pd.DataFrame({"Year": [future_year], crime_type: [predicted_value]})], ignore_index=True)
            forecast_csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button("📊 Forecast Data", data=forecast_csv, file_name=f"forecast_{crime_type}.csv", mime="text/csv", use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(139,0,0,0.4), rgba(80,0,0,0.4)); 
            border-radius: 20px; margin-top: 40px; border: 3px solid #8b0000; box-shadow: 0 12px 30px rgba(139,0,0,0.5);'>
    <h3 style='color: #ff0000; margin-bottom: 15px; text-shadow: 3px 3px 8px rgba(0,0,0,0.9); 
               font-size: 1.5rem; letter-spacing: 2px; text-transform: uppercase;'>
        🚨 CRIME INTELLIGENCE PRO 🚨
    </h3>
    <p style='color: #ffffff; font-size: 1.1rem; margin-bottom: 12px; text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
              font-weight: 800;'>
        Professional Crime Analytics & Prediction Platform
    </p>
    <p style='color: #ffffff; font-size: 0.95rem; opacity: 0.95; text-shadow: 2px 2px 4px rgba(0,0,0,0.9);'>
        AI/ML | Data Science | Strategic Intelligence
    </p>
</div>
""", unsafe_allow_html=True)