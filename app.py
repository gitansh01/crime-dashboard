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

st.set_page_config(
    page_title="Crime Intelligence Pro",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Main Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0a1e 0%, #1a1a3c 25%, #0f0f28 50%, #1e1e46 75%, #0a0a1e 100%);
    color: #FFFFFF;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1e 0%, #1a1a3c 50%, #0f0f28 100%);
    border-right: 4px solid #dc143c;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* All Text - White and Visible */
label, .stMarkdown, .stText, p, span, div {
    color: #ffffff !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.9);
}

/* Dropdown Label */
.stSelectbox label {
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 1.1rem !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
}

/* Dropdown Selected Box */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #1a0a0a !important;
    color: #ffffff !important;
    border: 4px solid #dc143c !important;
    font-weight: 900 !important;
    font-size: 1.3rem !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
}

/* Dropdown Menu */
[data-baseweb="popover"] {
    background-color: #1a0a0a !important;
    border: 4px solid #dc143c !important;
}

/* Dropdown Options */
.stSelectbox li {
    background-color: #1a0a0a !important;
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 1.3rem !important;
    padding: 20px 28px !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
    border-bottom: 2px solid #333333 !important;
}

.stSelectbox li:hover {
    background: linear-gradient(135deg, #dc143c, #ff0000) !important;
}

/* Headers */
h1 {
    color: #ffffff !important;
    text-shadow: 5px 5px 10px rgba(0,0,0,0.9);
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    text-align: center;
}

h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 900 !important;
    text-shadow: 3px 3px 8px rgba(0,0,0,0.9);
}

h2 {
    font-size: 1.8rem !important;
}

h3 {
    font-size: 1.5rem !important;
}

h4 {
    font-size: 1.3rem !important;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(220, 20, 60, 0.4), rgba(139, 0, 0, 0.3));
    border: 4px solid #dc143c;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 15px 40px rgba(220, 20, 60, 0.6);
}

[data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    font-weight: 900 !important;
    color: #ff0000 !important;
    text-shadow: 4px 4px 8px rgba(0,0,0,0.9);
}

[data-testid="stMetricLabel"] {
    color: #ffffff !important;
    font-weight: 800 !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
}

/* Regular Buttons */
.stButton>button {
    background: linear-gradient(135deg, #dc143c, #8b0000) !important;
    color: white !important;
    font-weight: 900 !important;
    font-size: 1.1rem !important;
    border: 4px solid #ff0000 !important;
    border-radius: 12px !important;
    padding: 16px 40px !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.9);
    box-shadow: 0 8px 20px rgba(220, 20, 60, 0.6);
}

.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(220, 20, 60, 0.8);
}

/* DOWNLOAD BUTTONS - MAXIMUM VISIBILITY */
.stDownloadButton>button {
    background: linear-gradient(135deg, #0066cc, #003d82) !important;
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 1.2rem !important;
    border: 4px solid #ffffff !important;
    border-radius: 12px !important;
    padding: 18px 40px !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
    box-shadow: 0 10px 25px rgba(0, 102, 204, 0.7);
    letter-spacing: 1px;
}

.stDownloadButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(0, 102, 204, 0.9);
    background: linear-gradient(135deg, #0080ff, #0066cc) !important;
}

/* Download Button Text - Force Visibility */
.stDownloadButton>button span {
    color: #ffffff !important;
    font-weight: 900 !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
}

/* FULLSCREEN BUTTONS - WHITE AND VISIBLE */
button[title="View fullscreen"], 
button[title*="fullscreen"],
button[kind="header"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 3px solid #dc143c !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    font-weight: 900 !important;
    font-size: 1rem !important;
    box-shadow: 0 5px 15px rgba(255, 255, 255, 0.6) !important;
}

button[title="View fullscreen"]:hover {
    background-color: #dc143c !important;
    color: #ffffff !important;
    box-shadow: 0 8px 20px rgba(220, 20, 60, 0.8) !important;
}

/* All Toolbar Buttons */
[data-testid="stElementToolbar"] button,
[data-testid="StyledFullScreenButton"] button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 3px solid #dc143c !important;
    font-weight: 900 !important;
}

/* Dataframe Container - Extra Space for Buttons */
[data-testid="stDataFrame"] {
    background: rgba(10, 10, 30, 0.9);
    border-radius: 15px;
    border: 3px solid #dc143c;
    box-shadow: 0 12px 30px rgba(0,0,0,0.7);
    padding-top: 60px !important;
    position: relative;
}

/* Dataframe Headers */
[data-testid="stDataFrame"] th {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 1.2rem !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
    padding: 15px !important;
}

/* Dataframe Cells */
[data-testid="stDataFrame"] td {
    color: #000000 !important;
    font-weight: 700 !important;
}

/* Expander Headers */
.streamlit-expanderHeader {
    background-color: #1a1a3c !important;
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 1.3rem !important;
    text-shadow: 3px 3px 8px rgba(0,0,0,0.9);
    border: 4px solid #dc143c !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: rgba(10, 10, 30, 0.9);
    border-radius: 15px;
    padding: 15px;
    border: 3px solid #dc143c;
}

.stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, rgba(220, 20, 60, 0.3), rgba(139, 0, 0, 0.3));
    border-radius: 12px;
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 1.1rem !important;
    padding: 16px 30px;
    border: 3px solid transparent;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.9);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #dc143c, #ff0000);
    color: white !important;
    border: 3px solid #ffffff;
    box-shadow: 0 8px 20px rgba(220, 20, 60, 0.7);
}

/* Radio Buttons */
.stRadio > div {
    background: rgba(26, 26, 60, 0.7);
    border-radius: 12px;
    padding: 20px;
    border: 3px solid #dc143c;
}

.stRadio label {
    color: #ffffff !important;
    font-weight: 900 !important;
    font-size: 1.1rem !important;
}

/* Alert Boxes */
.stAlert {
    background: rgba(10, 10, 30, 0.95) !important;
    border-radius: 12px;
    padding: 20px;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.6);
}

.stSuccess {
    background: rgba(0, 100, 0, 0.95) !important;
    border-left: 6px solid #00ff00 !important;
}

.stInfo {
    background: rgba(0, 50, 150, 0.95) !important;
    border-left: 6px solid #0099ff !important;
}

.stWarning {
    background: rgba(200, 100, 0, 0.95) !important;
    border-left: 6px solid #ffcc00 !important;
}

.stError {
    background: rgba(150, 0, 0, 0.95) !important;
    border-left: 6px solid #ff0000 !important;
}

/* Number Input */
.stNumberInput input {
    background-color: #000000 !important;
    color: #ffffff !important;
    border: 3px solid #dc143c !important;
    font-weight: 900 !important;
    font-size: 1.1rem !important;
}

/* Divider */
hr {
    border: 3px solid #dc143c;
    margin: 30px 0;
    box-shadow: 0 3px 15px rgba(220, 20, 60, 0.7);
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🚨 CRIME INTELLIGENCE PRO 🚨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.4rem; color: #ff4444; font-weight: 800;'>AI-Powered Crime Analytics & Prediction System</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 COMMAND CENTER")
    st.markdown("---")
    menu = st.radio("📊 Select Module", ["🔍 Crime Analysis & Insights", "🤖 AI Crime Prediction Engine"], index=0)
    st.markdown("---")
    st.markdown("### 📈 System Status")

# Data Loading
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
    st.metric("Years", len(df))
    st.metric("Status", "🟢 ONLINE")

st.markdown("---")

# ========== CRIME ANALYSIS MODULE ==========
if "Crime Analysis" in menu:
    st.markdown("<h2>📊 COMPREHENSIVE CRIME TREND ANALYSIS</h2>", unsafe_allow_html=True)
    
    crime_type = st.selectbox("🎯 Select Crime Category", df.columns[1:])
    
    # Calculate metrics
    total_cases = int(df[crime_type].sum())
    max_year = int(df.loc[df[crime_type].idxmax(), "Year"])
    min_year = int(df.loc[df[crime_type].idxmin(), "Year"])
    growth_rate = round((df[crime_type].iloc[-1] - df[crime_type].iloc[0]) / df[crime_type].iloc[0] * 100, 2)
    avg_cases = int(df[crime_type].mean())
    
    st.markdown("### 🎯 KEY PERFORMANCE INDICATORS")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Cases", f"{total_cases:,}", f"{growth_rate}%")
    with col2:
        st.metric("📅 Peak Year", max_year, f"{int(df.loc[df[crime_type].idxmax(), crime_type])} cases")
    with col3:
        st.metric("📉 Lowest Year", min_year, f"{int(df.loc[df[crime_type].idxmin(), crime_type])} cases")
    with col4:
        st.metric("📈 Average", avg_cases)
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend", "📊 Compare", "📉 Stats", "🗂️ Data"])
    
    with tab1:
        st.markdown("<h3>📈 Crime Trend Chart</h3>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Year"], y=df[crime_type], mode='lines+markers',
            line=dict(color='#ff0000', width=5),
            marker=dict(size=16, color='#dc143c', line=dict(color='white', width=3)),
            hovertemplate='<b style="font-size:18px">Year: %{x}<br>Cases: %{y}</b><extra></extra>'
        ))
        fig.update_layout(
            title=dict(text=f"<b>{crime_type} Cases Over Time</b>", font=dict(size=24, color='#ffffff', family='Arial Black')),
            plot_bgcolor='rgba(10, 10, 30, 0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=16, family='Arial Black'),
            xaxis=dict(title="<b>Year</b>", gridcolor='rgba(220, 20, 60, 0.3)', color='#ffffff', title_font=dict(size=16)),
            yaxis=dict(title="<b>Cases</b>", gridcolor='rgba(220, 20, 60, 0.3)', color='#ffffff', title_font=dict(size=16)),
            height=520,
            hoverlabel=dict(bgcolor='#000000', font_size=18, font_family='Arial Black', font_color='#ffffff', bordercolor='#dc143c')
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("<h3>📊 All Crime Categories</h3>", unsafe_allow_html=True)
        fig2 = go.Figure()
        colors = ['#ff0000', '#ff6600', '#ffcc00', '#00ff00', '#0099ff']
        for idx, col in enumerate(df.columns[1:]):
            fig2.add_trace(go.Scatter(
                x=df["Year"], y=df[col], mode='lines+markers', name=col,
                line=dict(width=4, color=colors[idx]), marker=dict(size=12)
            ))
        fig2.update_layout(
            title=dict(text="<b>Crime Comparison</b>", font=dict(size=24, color='#ffffff', family='Arial Black')),
            plot_bgcolor='rgba(10, 10, 30, 0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=16, family='Arial Black'),
            legend=dict(bgcolor='rgba(0, 0, 0, 0.95)', font=dict(color='#ffffff', size=14), bordercolor='#dc143c', borderwidth=2),
            height=520
        ))
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.markdown("<h3>📉 Statistical Analysis</h3>", unsafe_allow_html=True)
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.markdown("<h4>📊 STATISTICS</h4>", unsafe_allow_html=True)
            stats_df = pd.DataFrame({
                "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Range"],
                "Value": [
                    f"{df[crime_type].mean():.2f}", f"{df[crime_type].median():.2f}",
                    f"{df[crime_type].std():.2f}", f"{df[crime_type].min():.0f}",
                    f"{df[crime_type].max():.0f}", f"{df[crime_type].max() - df[crime_type].min():.0f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        with col_stat2:
            st.markdown("<h4>🎯 INSIGHTS</h4>", unsafe_allow_html=True)
            yoy_change = df[crime_type].iloc[-1] - df[crime_type].iloc[-2]
            yoy_percent = (yoy_change / df[crime_type].iloc[-2]) * 100
            st.markdown(f"""
            - **Growth Rate:** {growth_rate:+.2f}%
            - **YoY Change:** {yoy_change:+.0f} ({yoy_percent:+.2f}%)
            - **Average:** {df[crime_type].mean():.2f}
            - **Volatility:** {df[crime_type].std():.2f}
            """)
    
    with tab4:
        st.markdown("<h3>🗂️ Complete Dataset</h3>", unsafe_allow_html=True)
        st.dataframe(df.style.background_gradient(cmap='Reds', subset=df.columns[1:]), use_container_width=True, height=420)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Complete Dataset", data=csv, file_name=f"crime_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# ========== AI PREDICTION MODULE ==========
if "AI Crime Prediction" in menu:
    st.markdown("<h2>🤖 AI-POWERED CRIME PREDICTION ENGINE</h2>", unsafe_allow_html=True)
    
    crime_type = st.selectbox("🎯 Select Crime Category", df.columns[1:], key="pred")
    
    # Train Model
    X = df[["Year"]]
    y = df[crime_type]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Model Metrics
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    
    st.markdown("### 🎯 MODEL PERFORMANCE METRICS")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("🎯 R² Score", f"{r2_test:.3f}", f"Train: {r2_train:.3f}")
    with col_m2:
        st.metric("📊 MAE", f"{mae:.2f}", "cases")
    with col_m3:
        st.metric("📈 RMSE", f"{rmse:.2f}")
    with col_m4:
        quality = "Excellent" if r2_test > 0.9 else "Good" if r2_test > 0.7 else "Fair"
        st.metric("✅ Quality", quality, f"{r2_test*100:.1f}%")
    
    st.markdown("---")
    st.markdown("### 🔮 MAKE PREDICTION")
    
    col_input1, col_input2, col_input3 = st.columns([2, 1, 1])
    with col_input1:
        future_year = st.number_input("📅 Future Year", min_value=int(df["Year"].max()) + 1, max_value=2040, value=2025, step=1)
    with col_input2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🚀 PREDICT NOW", use_container_width=True)
    with col_input3:
        st.markdown("<br>", unsafe_allow_html=True)
        years_ahead = future_year - int(df["Year"].max())
        st.info(f"📊 {years_ahead} year(s) ahead")
    
    if predict_button:
        predicted_value = int(model.predict([[future_year]])[0])
        lower_bound = int(predicted_value * 0.9)
        upper_bound = int(predicted_value * 1.1)
        
        st.success("✅ PREDICTION ANALYSIS COMPLETE")
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #dc143c, #8b0000); padding: 40px; border-radius: 20px; 
                    text-align: center; border: 4px solid #ff0000; box-shadow: 0 15px 40px rgba(220,20,60,0.8);'>
            <h3 style='color: #ffffff; font-size: 1.3rem; margin-bottom: 20px;'>PREDICTED {crime_type.upper()} CASES</h3>
            <h1 style='color: #ffffff; font-size: 4rem; margin: 20px 0; text-shadow: 4px 4px 10px rgba(0,0,0,0.9);'>{predicted_value:,}</h1>
            <p style='color: #ffffff; font-size: 1.2rem; margin-bottom: 10px;'>YEAR: {future_year}</p>
            <p style='color: #ffff00; font-size: 1.1rem; font-weight: 800;'>CONFIDENCE RANGE: {lower_bound:,} - {upper_bound:,} CASES</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📈 Forecast", "📊 Analysis", "🎯 Model Info"])
        
        with viz_tab1:
            st.markdown("<h3>📈 Forecast Visualization</h3>", unsafe_allow_html=True)
            fig_pred = go.Figure()
            
            # Historical
            fig_pred.add_trace(go.Scatter(
                x=df["Year"], y=df[crime_type], mode='lines+markers', name='Historical Data',
                line=dict(color='#00ccff', width=5), marker=dict(size=14, color='#0099ff')
            ))
            
            # Prediction
            fig_pred.add_trace(go.Scatter(
                x=[future_year], y=[predicted_value], mode='markers', name='Prediction',
                marker=dict(size=40, color='#ffff00', symbol='star', line=dict(color='#ff0000', width=5))
            ))
            
            # Forecast line
            fig_pred.add_trace(go.Scatter(
                x=[df["Year"].iloc[-1], future_year], y=[df[crime_type].iloc[-1], predicted_value],
                mode='lines', name='Forecast Trend', line=dict(color='#ff0000', width=5, dash='dash')
            ))
            
            # Confidence
            fig_pred.add_trace(go.Scatter(
                x=[future_year]*3, y=[lower_bound, predicted_value, upper_bound],
                fill='toself', fillcolor='rgba(255, 255, 0, 0.4)', line=dict(color='#ffff00', width=5),
                name='Confidence Range'
            ))
            
            fig_pred.update_layout(
                title=dict(text=f"<b>{crime_type} Forecast Analysis</b>", font=dict(size=24, color='#ffffff', family='Arial Black')),
                plot_bgcolor='rgba(10, 10, 30, 0.7)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', size=16, family='Arial Black'),
                xaxis=dict(title="<b>Year</b>", gridcolor='rgba(220, 20, 60, 0.3)', color='#ffffff'),
                yaxis=dict(title="<b>Cases</b>", gridcolor='rgba(220, 20, 60, 0.3)', color='#ffffff'),
                legend=dict(bgcolor='rgba(0, 0, 0, 0.95)', font=dict(color='#ffffff', size=14), bordercolor='#dc143c', borderwidth=3),
                height=560,
                hoverlabel=dict(bgcolor='#000000', font_size=18, font_family='Arial Black', font_color='#ffffff')
            ))
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with viz_tab2:
            st.markdown("<h3>📊 Comparative Analysis</h3>", unsafe_allow_html=True)
            
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                st.markdown("<h4>📊 METRICS</h4>", unsafe_allow_html=True)
                hist_avg = df[crime_type].mean()
                diff_from_avg = predicted_value - hist_avg
                percent_diff = (diff_from_avg / hist_avg) * 100
                
                comparison_df = pd.DataFrame({
                    "Metric": ["Historical Avg", "Predicted Value", "Difference", "% Change", "Latest Year"],
                    "Value": [f"{hist_avg:.0f}", f"{predicted_value:,}", f"{diff_from_avg:+.0f}", f"{percent_diff:+.2f}%", f"{df[crime_type].iloc[-1]:.0f}"]
                })
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            with col_comp2:
                st.markdown("<h4>🚨 RISK ASSESSMENT</h4>", unsafe_allow_html=True)
                if percent_diff > 20:
                    st.error("🔴 HIGH RISK - Significant increase predicted")
                elif percent_diff > 5:
                    st.warning("🟡 MODERATE RISK - Notable increase expected")
                else:
                    st.success("🟢 LOW RISK - Stable or decreasing trend")
        
        with viz_tab3:
            st.markdown("<h3>🎯 Model Performance Details</h3>", unsafe_allow_html=True)
            
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.markdown("<h4>📊 Feature Importance</h4>", unsafe_allow_html=True)
                importance = model.feature_importances_
                fig_imp = go.Figure(data=[
                    go.Bar(x=["Year"], y=importance, marker=dict(color='#ff0000'),
                           text=[f"{imp:.2%}" for imp in importance], textposition='outside',
                           textfont=dict(color='#ffffff', size=16, family='Arial Black'))
                ])
                fig_imp.update_layout(
                    plot_bgcolor='rgba(10, 10, 30, 0.7)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff', family='Arial Black'),
                    yaxis=dict(title='<b>Importance</b>', color='#ffffff'),
                    height=360
                ))
                st.plotly_chart(fig_imp, use_container_width=True)
            
            with col_insight2:
                st.markdown("<h4>🧠 Model Configuration</h4>", unsafe_allow_html=True)
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
        st.markdown("### 📥 EXPORT PREDICTION REPORTS")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            report = pd.DataFrame({
                "Metric": ["Crime Type", "Year", "Predicted", "Lower Bound", "Upper Bound", "Change %"],
                "Value": [crime_type, future_year, predicted_value, lower_bound, upper_bound, f"{percent_diff:+.2f}%"]
            })
            csv_report = report.to_csv(index=False).encode("utf-8")
            st.download_button("📄 Download Detailed Report (CSV)", data=csv_report, file_name=f"prediction_{crime_type}_{future_year}.csv", mime="text/csv", use_container_width=True)
        
        with col_dl2:
            summary = f"""PREDICTION REPORT
=================
Crime: {crime_type}
Year: {future_year}
Predicted: {predicted_value:,}
Range: {lower_bound:,} - {upper_bound:,}
Change: {percent_diff:+.2f}%
R²: {r2_test:.4f}"""
            st.download_button("📋 Download Summary (TXT)", data=summary, file_name=f"summary_{future_year}.txt", mime="text/plain", use_container_width=True)
        
        with col_dl3:
            forecast_df = pd.concat([df[["Year", crime_type]], pd.DataFrame({"Year": [future_year], crime_type: [predicted_value]})], ignore_index=True)
            forecast_csv = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button("📊 Download Forecast Data (CSV)", data=forecast_csv, file_name=f"forecast_{crime_type}.csv", mime="text/csv", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(220,20,60,0.4), rgba(139,0,0,0.4)); 
            border-radius: 20px; border: 3px solid #dc143c;'>
    <h3 style='color: #ff0000; margin-bottom: 15px; font-size: 1.5rem;'>🚨 CRIME INTELLIGENCE PRO 🚨</h3>
    <p style='color: #ffffff; font-size: 1.1rem; font-weight: 800;'>Professional Crime Analytics & Prediction Platform</p>
    <p style='color: #ffffff; font-size: 0.95rem;'>AI/ML | Data Science | Strategic Intelligence</p>
</div>
""", unsafe_allow_html=True)