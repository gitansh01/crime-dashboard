import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from datetime import datetime
import json

st.set_page_config(
    page_title="Crime Intelligence Pro",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0a1e 0%, #1a1a3c 25%, #0f0f28 50%, #1e1e46 75%, #0a0a1e 100%);
    color: #FFFFFF;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1e 0%, #1a1a3c 50%, #0f0f28 100%);
    border-right: 4px solid #dc143c;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
label, .stMarkdown, .stText, p, span, div {
    color: #ffffff !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
}
.stSelectbox label {
    color: #ffffff !important; font-weight: 900 !important;
    font-size: 1.1rem !important;
}
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #1a0a0a !important; color: #ffffff !important;
    border: 4px solid #dc143c !important; font-weight: 900 !important;
    font-size: 1.2rem !important; text-shadow: none !important; min-height: 52px !important;
}
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] div[class*="singleValue"],
.stSelectbox [class*="ValueContainer"] *,
.stSelectbox [class*="singleValue"],
.stSelectbox [class*="placeholder"],
.stSelectbox input {
    color: #ffffff !important; font-weight: 900 !important;
    text-shadow: none !important; -webkit-text-fill-color: #ffffff !important;
}
.stSelectbox svg { fill: #dc143c !important; }
[data-baseweb="popover"] { background-color: #1a0a0a !important; border: 4px solid #dc143c !important; }
.stSelectbox li, [data-baseweb="menu"] li, [role="option"] {
    background-color: #1a0a0a !important; color: #ffffff !important;
    font-weight: 900 !important; font-size: 1.1rem !important;
    padding: 16px 24px !important; border-bottom: 2px solid #333333 !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stSelectbox li:hover, [role="option"]:hover {
    background: linear-gradient(135deg, #dc143c, #ff0000) !important;
}
h1 { color: #ffffff !important; text-shadow: 5px 5px 10px rgba(0,0,0,0.9);
     font-size: 2.8rem !important; font-weight: 900 !important; text-align: center; }
h2, h3, h4 { color: #ffffff !important; font-weight: 900 !important; text-shadow: 3px 3px 8px rgba(0,0,0,0.9); }
h2 { font-size: 1.8rem !important; } h3 { font-size: 1.5rem !important; } h4 { font-size: 1.3rem !important; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(220,20,60,0.4), rgba(139,0,0,0.3));
    border: 4px solid #dc143c; border-radius: 15px; padding: 25px;
    box-shadow: 0 15px 40px rgba(220,20,60,0.6);
}
[data-testid="stMetricValue"] { font-size: 2.5rem !important; font-weight: 900 !important;
    color: #ff0000 !important; text-shadow: 4px 4px 8px rgba(0,0,0,0.9); }
[data-testid="stMetricLabel"] { color: #ffffff !important; font-weight: 800 !important; }
.stButton>button {
    background: linear-gradient(135deg, #dc143c, #8b0000) !important; color: white !important;
    font-weight: 900 !important; font-size: 1.1rem !important; border: 4px solid #ff0000 !important;
    border-radius: 12px !important; padding: 16px 40px !important;
    box-shadow: 0 8px 20px rgba(220,20,60,0.6);
}
.stButton>button:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(220,20,60,0.8); }
.stDownloadButton>button {
    background: linear-gradient(135deg, #0066cc, #003d82) !important; color: #ffffff !important;
    font-weight: 900 !important; font-size: 1.1rem !important; border: 4px solid #ffffff !important;
    border-radius: 12px !important; padding: 16px 36px !important;
    box-shadow: 0 10px 25px rgba(0,102,204,0.7);
}
.stDownloadButton>button:hover { transform: translateY(-3px); box-shadow: 0 15px 35px rgba(0,102,204,0.9); background: linear-gradient(135deg, #0080ff, #0066cc) !important; }
.stDownloadButton>button span { color: #ffffff !important; font-weight: 900 !important; }
button[title="View fullscreen"] { background-color: #ffffff !important; color: #000000 !important;
    border: 3px solid #dc143c !important; border-radius: 8px !important; font-weight: 900 !important; }
[data-testid="stDataFrame"] {
    background: rgba(10,10,30,0.9); border-radius: 12px; border: 3px solid #dc143c;
    box-shadow: 0 12px 30px rgba(0,0,0,0.7); padding-top: 8px !important; margin-bottom: 0 !important;
}
[data-testid="stDataFrame"] th { background-color: #1a0020 !important; color: #ffffff !important;
    font-weight: 900 !important; font-size: 1rem !important; padding: 12px !important; }
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] [role="gridcell"] {
    color: #ffffff !important; font-weight: 700 !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 10px; background: rgba(10,10,30,0.9);
    border-radius: 15px; padding: 12px; border: 3px solid #dc143c; }
.stTabs [data-baseweb="tab"] { background: linear-gradient(135deg, rgba(220,20,60,0.3), rgba(139,0,0,0.3));
    border-radius: 12px; color: #ffffff !important; font-weight: 900 !important;
    font-size: 1rem !important; padding: 14px 24px; border: 2px solid transparent; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #dc143c, #ff0000);
    color: white !important; border: 2px solid #ffffff; box-shadow: 0 8px 20px rgba(220,20,60,0.7); }
.stRadio > div { background: rgba(26,26,60,0.7); border-radius: 12px; padding: 20px; border: 3px solid #dc143c; }
.stRadio label { color: #ffffff !important; font-weight: 900 !important; font-size: 1.1rem !important; }
.stAlert { background: rgba(10,10,30,0.95) !important; border-radius: 12px; padding: 20px;
    color: #ffffff !important; font-weight: 700 !important; }
.stSuccess { background: rgba(0,100,0,0.95) !important; border-left: 6px solid #00ff00 !important; }
.stInfo { background: rgba(0,50,150,0.95) !important; border-left: 6px solid #0099ff !important; }
.stWarning { background: rgba(200,100,0,0.95) !important; border-left: 6px solid #ffcc00 !important; }
.stError { background: rgba(150,0,0,0.95) !important; border-left: 6px solid #ff0000 !important; }
.stNumberInput input { background-color: #000000 !important; color: #ffffff !important;
    border: 3px solid #dc143c !important; font-weight: 900 !important; font-size: 1.1rem !important; }
hr { border: 3px solid #dc143c; margin: 30px 0; box-shadow: 0 3px 15px rgba(220,20,60,0.7); }
.streamlit-expanderHeader { background-color: #1a1a3c !important; color: #ffffff !important;
    font-weight: 900 !important; font-size: 1.2rem !important; border: 4px solid #dc143c !important;
    border-radius: 12px !important; padding: 14px !important; }
.kpi-card {
    background: linear-gradient(135deg, rgba(220,20,60,0.25), rgba(10,10,30,0.9));
    border: 2px solid #dc143c; border-radius: 14px; padding: 20px 16px;
    text-align: center; box-shadow: 0 8px 24px rgba(220,20,60,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 14px 36px rgba(220,20,60,0.65); }
.kpi-title { font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase;
    color: rgba(255,255,255,0.7) !important; margin-bottom: 6px; font-weight: 700; }
.kpi-value { font-size: 2rem; font-weight: 900; color: #ff4444 !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.7); line-height: 1.1; }
.kpi-sub { font-size: 0.78rem; color: rgba(255,255,255,0.6) !important; margin-top: 4px; }
.state-rank-card {
    background: linear-gradient(135deg, rgba(220,20,60,0.2), rgba(10,10,30,0.95));
    border-left: 5px solid #dc143c; border-radius: 10px; padding: 14px 18px;
    margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data
def load_national_data():
    years = np.arange(2015, 2024)
    return pd.DataFrame({
        "Year": years,
        "Theft":   [400,450,480,500,550,600,650,700,720],
        "Assault": [300,320,350,370,400,420,450,470,500],
        "Burglary":[200,210,230,240,260,280,300,320,340],
        "Robbery": [150,160,175,185,200,215,230,245,260],
        "Fraud":   [100,120,140,160,180,200,220,240,255]
    })

@st.cache_data
def load_state_data():
    states = [
        "ANDHRA PRADESH","ARUNACHAL PRADESH","ASSAM","BIHAR","CHHATTISGARH",
        "GOA","GUJARAT","HARYANA","HIMACHAL PRADESH","JAMMU & KASHMIR",
        "JHARKHAND","KARNATAKA","KERALA","MADHYA PRADESH","MAHARASHTRA",
        "MANIPUR","MEGHALAYA","MIZORAM","NAGALAND","ODISHA",
        "PUNJAB","RAJASTHAN","SIKKIM","TAMIL NADU","TRIPURA",
        "UTTAR PRADESH","UTTARAKHAND","WEST BENGAL",
        "A & N ISLANDS","CHANDIGARH","D & N HAVELI","DAMAN & DIU","LAKSHADWEEP","PUDUCHERRY"
    ]
    # Realistic base crime indices per state (relative scale)
    base = {
        "ANDHRA PRADESH":4500,"ARUNACHAL PRADESH":320,"ASSAM":3800,"BIHAR":6200,
        "CHHATTISGARH":3100,"GOA":580,"GUJARAT":5800,"HARYANA":4900,
        "HIMACHAL PRADESH":890,"JAMMU & KASHMIR":2100,"JHARKHAND":3400,
        "KARNATAKA":6100,"KERALA":4200,"MADHYA PRADESH":7800,"MAHARASHTRA":9200,
        "MANIPUR":410,"MEGHALAYA":380,"MIZORAM":190,"NAGALAND":220,"ODISHA":4100,
        "PUNJAB":4400,"RAJASTHAN":7100,"SIKKIM":95,"TAMIL NADU":6800,
        "TRIPURA":620,"UTTAR PRADESH":12500,"UTTARAKHAND":1100,"WEST BENGAL":7200,
        "A & N ISLANDS":110,"CHANDIGARH":820,"D & N HAVELI":88,"DAMAN & DIU":72,
        "LAKSHADWEEP":18,"PUDUCHERRY":340
    }
    # State coordinates (lat, lon) for map
    coords = {
        "ANDHRA PRADESH":(15.9129,79.7400),"ARUNACHAL PRADESH":(28.2180,94.7278),
        "ASSAM":(26.2006,92.9376),"BIHAR":(25.0961,85.3131),
        "CHHATTISGARH":(21.2787,81.8661),"GOA":(15.2993,74.1240),
        "GUJARAT":(22.2587,71.1924),"HARYANA":(29.0588,76.0856),
        "HIMACHAL PRADESH":(31.1048,77.1734),"JAMMU & KASHMIR":(33.7782,76.5762),
        "JHARKHAND":(23.6102,85.2799),"KARNATAKA":(15.3173,75.7139),
        "KERALA":(10.8505,76.2711),"MADHYA PRADESH":(22.9734,78.6569),
        "MAHARASHTRA":(19.7515,75.7139),"MANIPUR":(24.6637,93.9063),
        "MEGHALAYA":(25.4670,91.3662),"MIZORAM":(23.1645,92.9376),
        "NAGALAND":(26.1584,94.5624),"ODISHA":(20.9517,85.0985),
        "PUNJAB":(31.1471,75.3412),"RAJASTHAN":(27.0238,74.2179),
        "SIKKIM":(27.5330,88.5122),"TAMIL NADU":(11.1271,78.6569),
        "TRIPURA":(23.9408,91.9882),"UTTAR PRADESH":(26.8467,80.9462),
        "UTTARAKHAND":(30.0668,79.0193),"WEST BENGAL":(22.9868,87.8550),
        "A & N ISLANDS":(11.7401,92.6586),"CHANDIGARH":(30.7333,76.7794),
        "D & N HAVELI":(20.1809,73.0169),"DAMAN & DIU":(20.4283,72.8397),
        "LAKSHADWEEP":(10.5667,72.6417),"PUDUCHERRY":(11.9416,79.8083)
    }
    years = list(range(2015, 2024))
    rows = []
    np.random.seed(42)
    for state in states:
        b = base[state]
        lat, lon = coords[state]
        for i, yr in enumerate(years):
            growth = 1 + (i * 0.04) + np.random.uniform(-0.02, 0.03)
            total = int(b * growth)
            theft    = int(total * np.random.uniform(0.28, 0.36))
            assault  = int(total * np.random.uniform(0.20, 0.26))
            burglary = int(total * np.random.uniform(0.14, 0.18))
            robbery  = int(total * np.random.uniform(0.10, 0.14))
            fraud    = int(total * np.random.uniform(0.08, 0.13))
            murder   = int(total * np.random.uniform(0.03, 0.06))
            kidnap   = int(total * np.random.uniform(0.04, 0.08))
            rows.append({
                "State": state, "Year": yr,
                "Theft": theft, "Assault": assault, "Burglary": burglary,
                "Robbery": robbery, "Fraud": fraud, "Murder": murder, "Kidnapping": kidnap,
                "Total_Crimes": theft+assault+burglary+robbery+fraud+murder+kidnap,
                "Latitude": lat, "Longitude": lon
            })
    return pd.DataFrame(rows)

df_national = load_national_data()
df_state    = load_state_data()

CRIME_TYPES  = ["Theft","Assault","Burglary","Robbery","Fraud","Murder","Kidnapping"]
STATE_LIST   = sorted(df_state["State"].unique().tolist())
YEARS        = sorted(df_state["Year"].unique().tolist())

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>🎯 COMMAND CENTER</h2>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("📊 Select Module", [
        "🏠 National Overview",
        "🗺️ India Crime Map",
        "📊 State-wise Analysis",
        "🔍 National Trend Analysis",
        "🤖 AI Prediction Engine"
    ], index=0)
    st.markdown("---")
    st.markdown("### 📈 System Status")
    total_records = len(df_state)
    st.metric("Total Records", f"{total_records:,}")
    st.metric("States/UTs", df_state["State"].nunique())
    st.metric("Years Covered", f"{YEARS[0]}–{YEARS[-1]}")
    st.metric("Crime Categories", len(CRIME_TYPES))
    st.metric("Status", "🟢 ONLINE")
    st.markdown("---")
    st.markdown("<p style='text-align:center; font-size:0.8rem; color:#aaaaaa;'>Crime Intelligence Pro v3.0<br>Hackathon Edition 🏆</p>", unsafe_allow_html=True)

st.markdown("<h1>🚨 CRIME INTELLIGENCE PRO 🚨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.3rem; color:#ff6666; font-weight:800;'>AI-Powered National Crime Analytics & Prediction System | India</p>", unsafe_allow_html=True)
st.markdown("---")

# ═══════════════════════════════════════════════════════
# MODULE 1 — NATIONAL OVERVIEW
# ═══════════════════════════════════════════════════════
if "National Overview" in menu:
    st.markdown("<h2>🏠 NATIONAL CRIME OVERVIEW — INDIA</h2>", unsafe_allow_html=True)

    latest = df_state[df_state["Year"] == 2023]
    prev   = df_state[df_state["Year"] == 2022]
    total_2023  = latest["Total_Crimes"].sum()
    total_2022  = prev["Total_Crimes"].sum()
    yoy_pct     = ((total_2023 - total_2022) / total_2022) * 100
    worst_state = latest.loc[latest["Total_Crimes"].idxmax(), "State"]
    safest_state= latest.loc[latest["Total_Crimes"].idxmin(), "State"]
    highest_cat = latest[CRIME_TYPES].sum().idxmax()

    # KPI Cards
    kpi_data = [
        ("🇮🇳 Total Crimes 2023", f"{total_2023:,}", f"YoY {yoy_pct:+.1f}%"),
        ("⚠️ Highest Crime State", worst_state.title(), f"{latest['Total_Crimes'].max():,} cases"),
        ("✅ Safest State/UT",     safest_state.title(), f"{latest['Total_Crimes'].min():,} cases"),
        ("🔺 Top Crime Category",  highest_cat, f"{int(latest[highest_cat].sum()):,} cases"),
        ("📅 States/UTs Tracked",  "34", "All India Coverage"),
        ("📈 Avg Growth Rate",     f"{yoy_pct:.1f}%", "2022 → 2023"),
    ]
    cols = st.columns(6)
    for col, (title, val, sub) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("<h3>📊 Top 10 States by Total Crimes (2023)</h3>", unsafe_allow_html=True)
        top10 = latest.nlargest(10, "Total_Crimes")[["State","Total_Crimes"]].reset_index(drop=True)
        top10["State"] = top10["State"].str.title()
        fig_top = go.Figure(go.Bar(
            x=top10["Total_Crimes"], y=top10["State"],
            orientation='h',
            marker=dict(
                color=top10["Total_Crimes"],
                colorscale=[[0,"#8b0000"],[0.5,"#dc143c"],[1,"#ff6666"]],
                line=dict(color="#ffffff", width=0.5)
            ),
            text=[f"{v:,}" for v in top10["Total_Crimes"]],
            textposition="outside",
            textfont=dict(color="#ffffff", size=12, family="Arial Black")
        ))
        fig_top.update_layout(
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=13, family='Arial Black'),
            yaxis=dict(color='#ffffff', autorange="reversed"),
            xaxis=dict(color='#ffffff', gridcolor='rgba(220,20,60,0.2)'),
            height=420, margin=dict(l=10, r=80, t=10, b=10)
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col_right:
        st.markdown("<h3>🥧 Crime Category Distribution (2023)</h3>", unsafe_allow_html=True)
        cat_totals = latest[CRIME_TYPES].sum()
        fig_pie = go.Figure(go.Pie(
            labels=cat_totals.index, values=cat_totals.values,
            hole=0.45,
            marker=dict(colors=['#ff0000','#ff6600','#ffcc00','#00ccff','#cc00ff','#00ff88','#ff66cc'],
                        line=dict(color='#0a0a1e', width=2)),
            textfont=dict(color='#ffffff', size=13, family='Arial Black'),
            hovertemplate='<b>%{label}</b><br>Cases: %{value:,}<br>Share: %{percent}<extra></extra>'
        ))
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Arial Black'),
            legend=dict(font=dict(color='#ffffff', size=12)),
            height=380
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.markdown("<h3>📈 All-India Crime Trend (2015–2023)</h3>", unsafe_allow_html=True)
    nat_trend = df_state.groupby("Year")[CRIME_TYPES + ["Total_Crimes"]].sum().reset_index()
    fig_trend = go.Figure()
    colors_t = ['#ff0000','#ff6600','#ffcc00','#00ccff','#cc00ff','#00ff88','#ff66cc']
    for i, ct in enumerate(CRIME_TYPES):
        fig_trend.add_trace(go.Scatter(
            x=nat_trend["Year"], y=nat_trend[ct], mode='lines+markers', name=ct,
            line=dict(width=3, color=colors_t[i]), marker=dict(size=8),
            hovertemplate=f'<b>{ct}</b><br>Year: %{{x}}<br>Cases: %{{y:,}}<extra></extra>'
        ))
    fig_trend.update_layout(
        plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff', size=14, family='Arial Black'),
        xaxis=dict(color='#ffffff', gridcolor='rgba(220,20,60,0.2)', title="Year"),
        yaxis=dict(color='#ffffff', gridcolor='rgba(220,20,60,0.2)', title="Cases"),
        legend=dict(bgcolor='rgba(0,0,0,0.8)', font=dict(color='#ffffff', size=12), bordercolor='#dc143c', borderwidth=2),
        height=400, hovermode='x unified'
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    st.markdown("<h3>🏆 State Rankings (2023)</h3>", unsafe_allow_html=True)
    rank_df = latest[["State","Total_Crimes","Theft","Assault","Burglary","Robbery","Fraud","Murder","Kidnapping"]].copy()
    rank_df = rank_df.sort_values("Total_Crimes", ascending=False).reset_index(drop=True)
    rank_df.index = rank_df.index + 1
    rank_df["State"] = rank_df["State"].str.title()
    rank_df.columns = ["State","Total","Theft","Assault","Burglary","Robbery","Fraud","Murder","Kidnapping"]
    styled_rank = rank_df.style.background_gradient(cmap='Reds', subset=["Total","Theft","Assault","Burglary","Robbery","Fraud","Murder","Kidnapping"]) \
        .set_properties(**{'color':'white','font-weight':'bold'})
    st.dataframe(styled_rank, use_container_width=True, height=400)

# ═══════════════════════════════════════════════════════
# MODULE 2 — INDIA CRIME MAP
# ═══════════════════════════════════════════════════════
elif "India Crime Map" in menu:
    st.markdown("<h2>🗺️ INDIA CRIME HEATMAP</h2>", unsafe_allow_html=True)

    col_mc1, col_mc2, col_mc3 = st.columns(3)
    with col_mc1:
        map_year = st.selectbox("📅 Select Year", YEARS, index=len(YEARS)-1, key="map_year")
    with col_mc2:
        map_crime = st.selectbox("🔍 Select Crime Type", ["Total_Crimes"] + CRIME_TYPES, key="map_crime")
    with col_mc3:
        map_size = st.selectbox("🔵 Bubble Size", ["Proportional","Equal"], key="map_size")

    map_df = df_state[df_state["Year"] == map_year].copy()
    map_df["State_Title"] = map_df["State"].str.title()
    crime_col = map_crime
    max_val = map_df[crime_col].max()
    map_df["bubble_size"] = (map_df[crime_col] / max_val * 50 + 5) if map_size == "Proportional" else 20

    fig_map = go.Figure()

    # India outline placeholder (scatter geo)
    fig_map.add_trace(go.Scattergeo(
        lat=map_df["Latitude"], lon=map_df["Longitude"],
        mode='markers+text',
        marker=dict(
            size=map_df["bubble_size"],
            color=map_df[crime_col],
            colorscale=[[0,"#1a0020"],[0.3,"#8b0000"],[0.6,"#dc143c"],[1,"#ff6666"]],
            colorbar=dict(
                title=dict(text=f"<b>{crime_col}</b>", font=dict(color="#ffffff", size=13)),
                tickfont=dict(color="#ffffff"), thickness=16, len=0.7,
                bgcolor="rgba(10,10,30,0.8)", bordercolor="#dc143c", borderwidth=2
            ),
            cmin=0, cmax=max_val,
            line=dict(color='#ffffff', width=1),
            opacity=0.85,
        ),
        text=map_df["State_Title"],
        textfont=dict(color='#ffffff', size=9, family='Arial Black'),
        textposition="top center",
        customdata=np.stack([map_df["State_Title"], map_df[crime_col],
                             map_df["Total_Crimes"], map_df["Murder"]], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            f"{crime_col}: %{{customdata[1]:,}}<br>"
            "Total Crimes: %{customdata[2]:,}<br>"
            "Murder: %{customdata[3]:,}<extra></extra>"
        )
    ))

    fig_map.update_layout(
        geo=dict(
            scope='asia',
            center=dict(lat=22.5, lon=82.5),
            projection_scale=4.5,
            showland=True, landcolor='rgba(20,20,50,0.95)',
            showocean=True, oceancolor='rgba(5,5,20,0.95)',
            showcountries=True, countrycolor='rgba(220,20,60,0.6)',
            showcoastlines=True, coastlinecolor='rgba(220,20,60,0.8)',
            showsubunits=True, subunitcolor='rgba(200,200,200,0.3)',
            bgcolor='rgba(0,0,0,0)',
            lataxis=dict(range=[6,38]), lonaxis=dict(range=[66,100])
        ),
        paper_bgcolor='rgba(10,10,30,0.95)',
        height=640,
        margin=dict(l=0, r=0, t=10, b=10),
        title=dict(
            text=f"<b>India Crime Map — {crime_col} ({map_year})</b>",
            font=dict(size=18, color='#ffffff', family='Arial Black'),
            x=0.5
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    col_ml, col_mr = st.columns(2)

    with col_ml:
        st.markdown(f"<h3>🔴 Top 5 Most Affected States ({map_year})</h3>", unsafe_allow_html=True)
        top5 = map_df.nlargest(5, crime_col)[["State_Title", crime_col, "Total_Crimes"]]
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][i-1]
            st.markdown(f"""
            <div class="state-rank-card">
                <div><b>{medal} {row['State_Title']}</b></div>
                <div style='color:#ff4444; font-weight:900; font-size:1.1rem;'>{int(row[crime_col]):,}</div>
            </div>""", unsafe_allow_html=True)

    with col_mr:
        st.markdown(f"<h3>🟢 Top 5 Safest States ({map_year})</h3>", unsafe_allow_html=True)
        bot5 = map_df.nsmallest(5, crime_col)[["State_Title", crime_col, "Total_Crimes"]]
        for i, (_, row) in enumerate(bot5.iterrows(), 1):
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,rgba(0,100,0,0.2),rgba(10,10,30,0.95));
                        border-left:5px solid #00cc44; border-radius:10px; padding:14px 18px;
                        margin-bottom:10px; display:flex; justify-content:space-between; align-items:center;'>
                <div><b>✅ {row['State_Title']}</b></div>
                <div style='color:#44ff88; font-weight:900; font-size:1.1rem;'>{int(row[crime_col]):,}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h3>📊 Crime Distribution Across States (Choropleth-style Heatmap)</h3>", unsafe_allow_html=True)
    heat_df = map_df[["State_Title"] + CRIME_TYPES].set_index("State_Title")
    fig_heat = go.Figure(go.Heatmap(
        z=heat_df.values, x=CRIME_TYPES, y=heat_df.index,
        colorscale=[[0,"#0a0a1e"],[0.3,"#8b0000"],[0.7,"#dc143c"],[1,"#ff6666"]],
        hovertemplate='State: %{y}<br>Crime: %{x}<br>Cases: %{z:,}<extra></extra>',
        colorbar=dict(title=dict(text="Cases", font=dict(color="#ffffff")),
                      tickfont=dict(color="#ffffff"), bgcolor="rgba(10,10,30,0.8)",
                      bordercolor="#dc143c", borderwidth=2)
    ))
    fig_heat.update_layout(
        plot_bgcolor='rgba(10,10,30,0.9)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff', size=11, family='Arial Black'),
        xaxis=dict(color='#ffffff'), yaxis=dict(color='#ffffff', autorange='reversed'),
        height=900, margin=dict(l=220, r=20, t=20, b=40)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════
# MODULE 3 — STATE-WISE ANALYSIS
# ═══════════════════════════════════════════════════════
elif "State-wise Analysis" in menu:
    st.markdown("<h2>📊 STATE-WISE DEEP ANALYSIS</h2>", unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sel_state = st.selectbox("🏛️ Select State/UT", STATE_LIST, key="state_sel")
    with col_s2:
        sel_crime = st.selectbox("🔍 Select Crime Type", ["Total_Crimes"] + CRIME_TYPES, key="state_crime")

    sdf = df_state[df_state["State"] == sel_state].sort_values("Year")

    # State KPIs
    latest_s = sdf[sdf["Year"] == 2023].iloc[0]
    prev_s   = sdf[sdf["Year"] == 2022].iloc[0]
    growth_s = ((latest_s[sel_crime] - prev_s[sel_crime]) / prev_s[sel_crime]) * 100
    nat_rank = df_state[df_state["Year"]==2023].sort_values(sel_crime, ascending=False)["State"].tolist().index(sel_state) + 1

    st.markdown(f"<h3>📍 {sel_state.title()} — {sel_crime} Analysis</h3>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis_s = [
        ("2023 Cases",    f"{int(latest_s[sel_crime]):,}", f"YoY {growth_s:+.1f}%"),
        ("National Rank", f"#{nat_rank}", f"out of 34"),
        ("9-Yr Total",    f"{int(sdf[sel_crime].sum()):,}", "2015–2023"),
        ("Peak Year",     str(int(sdf.loc[sdf[sel_crime].idxmax(),"Year"])), f"{int(sdf[sel_crime].max()):,} cases"),
        ("Avg/Year",      f"{int(sdf[sel_crime].mean()):,}", "2015–2023"),
    ]
    for col, (t, v, s) in zip([k1,k2,k3,k4,k5], kpis_s):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">{t}</div>'
                        f'<div class="kpi-value">{v}</div><div class="kpi-sub">{s}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("---")

    tab_s1, tab_s2, tab_s3, tab_s4, tab_s5 = st.tabs(
        ["📈 Trend","📊 Category Mix","📉 YoY Change","🆚 Compare States","🗂️ Raw Data"])

    with tab_s1:
        fig_st = go.Figure()
        fig_st.add_trace(go.Scatter(
            x=sdf["Year"], y=sdf[sel_crime], mode='lines+markers',
            line=dict(color='#ff0000', width=5),
            marker=dict(size=14, color='#dc143c', line=dict(color='white', width=3)),
            fill='tozeroy', fillcolor='rgba(220,20,60,0.12)',
            hovertemplate='<b>Year: %{x}</b><br>Cases: %{y:,}<extra></extra>'
        ))
        # Trend line
        z = np.polyfit(sdf["Year"], sdf[sel_crime], 1)
        p = np.poly1d(z)
        fig_st.add_trace(go.Scatter(
            x=sdf["Year"], y=p(sdf["Year"]), mode='lines', name='Trend',
            line=dict(color='#ffff00', width=2, dash='dash')
        ))
        fig_st.update_layout(
            title=dict(text=f"<b>{sel_state.title()} — {sel_crime} Trend</b>",
                       font=dict(size=20, color='#ffffff', family='Arial Black')),
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=14, family='Arial Black'),
            xaxis=dict(color='#ffffff', gridcolor='rgba(220,20,60,0.2)', title="Year"),
            yaxis=dict(color='#ffffff', gridcolor='rgba(220,20,60,0.2)', title="Cases"),
            height=460, showlegend=True,
            legend=dict(bgcolor='rgba(0,0,0,0.7)', bordercolor='#dc143c', borderwidth=1,
                        font=dict(color='#ffffff'))
        )
        st.plotly_chart(fig_st, use_container_width=True)

    with tab_s2:
        cat_sums = latest_s[CRIME_TYPES]
        fig_cat = go.Figure()
        colors_c = ['#ff0000','#ff6600','#ffcc00','#00ccff','#cc00ff','#00ff88','#ff66cc']
        fig_cat.add_trace(go.Bar(
            x=CRIME_TYPES, y=cat_sums.values,
            marker=dict(color=colors_c, line=dict(color='#ffffff', width=1)),
            text=[f"{int(v):,}" for v in cat_sums.values], textposition='outside',
            textfont=dict(color='#ffffff', size=12, family='Arial Black'),
            hovertemplate='<b>%{x}</b><br>Cases: %{y:,}<extra></extra>'
        ))
        fig_cat.update_layout(
            title=dict(text=f"<b>{sel_state.title()} — Category Breakdown (2023)</b>",
                       font=dict(size=18, color='#ffffff', family='Arial Black')),
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Arial Black'),
            xaxis=dict(color='#ffffff'), yaxis=dict(color='#ffffff', gridcolor='rgba(220,20,60,0.2)'),
            height=420
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        fig_pie_s = go.Figure(go.Pie(
            labels=CRIME_TYPES, values=cat_sums.values, hole=0.4,
            marker=dict(colors=colors_c, line=dict(color='#0a0a1e', width=2)),
            textfont=dict(color='#ffffff', size=12, family='Arial Black'),
            hovertemplate='<b>%{label}</b><br>Cases: %{value:,}<br>%{percent}<extra></extra>'
        ))
        fig_pie_s.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff', family='Arial Black'),
            legend=dict(font=dict(color='#ffffff', size=11)), height=360
        )
        st.plotly_chart(fig_pie_s, use_container_width=True)

    with tab_s3:
        sdf2 = sdf.copy()
        sdf2[f"{sel_crime}_YoY"] = sdf2[sel_crime].pct_change() * 100
        sdf2 = sdf2.dropna(subset=[f"{sel_crime}_YoY"])
        bar_colors = ['#ff4444' if v > 0 else '#44ff88' for v in sdf2[f"{sel_crime}_YoY"]]
        fig_yoy = go.Figure(go.Bar(
            x=sdf2["Year"], y=sdf2[f"{sel_crime}_YoY"],
            marker=dict(color=bar_colors, line=dict(color='#ffffff', width=1)),
            text=[f"{v:+.1f}%" for v in sdf2[f"{sel_crime}_YoY"]],
            textposition='outside', textfont=dict(color='#ffffff', size=12, family='Arial Black'),
            hovertemplate='Year: %{x}<br>YoY Change: %{y:.2f}%<extra></extra>'
        ))
        fig_yoy.add_hline(y=0, line_color='#ffffff', line_width=2, line_dash='dash')
        fig_yoy.update_layout(
            title=dict(text=f"<b>Year-on-Year % Change — {sel_crime}</b>",
                       font=dict(size=18, color='#ffffff', family='Arial Black')),
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Arial Black'),
            xaxis=dict(color='#ffffff', dtick=1), yaxis=dict(color='#ffffff', gridcolor='rgba(220,20,60,0.2)', title="% Change"),
            height=420
        )
        st.plotly_chart(fig_yoy, use_container_width=True)

    with tab_s4:
        compare_states = st.multiselect("🆚 Add States to Compare", [s for s in STATE_LIST if s != sel_state],
                                        default=STATE_LIST[:3], max_selections=5, key="cmp_states")
        states_to_plot = [sel_state] + compare_states
        fig_cmp = go.Figure()
        cmp_colors = ['#ff0000','#00ccff','#ffcc00','#00ff88','#cc00ff','#ff6600']
        for ci, st_name in enumerate(states_to_plot):
            cdf = df_state[df_state["State"] == st_name].sort_values("Year")
            fig_cmp.add_trace(go.Scatter(
                x=cdf["Year"], y=cdf[sel_crime], mode='lines+markers',
                name=st_name.title(), line=dict(width=3, color=cmp_colors[ci % len(cmp_colors)]),
                marker=dict(size=8),
                hovertemplate=f'<b>{st_name.title()}</b><br>Year: %{{x}}<br>Cases: %{{y:,}}<extra></extra>'
            ))
        fig_cmp.update_layout(
            title=dict(text=f"<b>Multi-State Comparison — {sel_crime}</b>",
                       font=dict(size=18, color='#ffffff', family='Arial Black')),
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Arial Black'),
            xaxis=dict(color='#ffffff', title="Year", dtick=1),
            yaxis=dict(color='#ffffff', title="Cases", gridcolor='rgba(220,20,60,0.2)'),
            legend=dict(bgcolor='rgba(0,0,0,0.8)', bordercolor='#dc143c', borderwidth=2,
                        font=dict(color='#ffffff', size=12)),
            height=480, hovermode='x unified'
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    with tab_s5:
        disp_cols = ["Year"] + CRIME_TYPES + ["Total_Crimes"]
        styled_s = sdf[disp_cols].style.background_gradient(cmap='Reds', subset=CRIME_TYPES+["Total_Crimes"]) \
            .set_properties(**{'color':'white','font-weight':'bold'})
        st.dataframe(styled_s, use_container_width=True, height=380)
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        csv_s = sdf[disp_cols].to_csv(index=False).encode('utf-8')
        st.download_button(f"📥 Download {sel_state.title()} Data", data=csv_s,
                           file_name=f"{sel_state.replace(' ','_')}_crime_data.csv", mime="text/csv")

# ═══════════════════════════════════════════════════════
# MODULE 4 — NATIONAL TREND ANALYSIS
# ═══════════════════════════════════════════════════════
elif "National Trend" in menu:
    st.markdown("<h2>🔍 NATIONAL CRIME TREND ANALYSIS</h2>", unsafe_allow_html=True)

    crime_type = st.selectbox("🎯 Select Crime Category", df_national.columns[1:])

    total_cases  = int(df_national[crime_type].sum())
    max_year     = int(df_national.loc[df_national[crime_type].idxmax(), "Year"])
    min_year     = int(df_national.loc[df_national[crime_type].idxmin(), "Year"])
    growth_rate  = round((df_national[crime_type].iloc[-1] - df_national[crime_type].iloc[0]) / df_national[crime_type].iloc[0] * 100, 2)

    st.markdown("### 🎯 KEY PERFORMANCE INDICATORS")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("📊 Total Cases", f"{total_cases:,}", f"{growth_rate}%")
    with c2: st.metric("📅 Peak Year", max_year, f"{int(df_national.loc[df_national[crime_type].idxmax(), crime_type])} cases")
    with c3: st.metric("📉 Lowest Year", min_year, f"{int(df_national.loc[df_national[crime_type].idxmin(), crime_type])} cases")
    with c4: st.metric("📈 Average/Year", int(df_national[crime_type].mean()))

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend","📊 Compare","📉 Stats","🗂️ Data"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_national["Year"], y=df_national[crime_type], mode='lines+markers',
            line=dict(color='#ff0000', width=5),
            marker=dict(size=16, color='#dc143c', line=dict(color='white', width=3)),
            fill='tozeroy', fillcolor='rgba(220,20,60,0.1)',
            hovertemplate='<b>Year: %{x}</b><br>Cases: %{y:,}<extra></extra>'
        ))
        fig.update_layout(
            title=dict(text=f"<b>{crime_type} Cases Over Time</b>",
                       font=dict(size=22, color='#ffffff', family='Arial Black')),
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=14, family='Arial Black'),
            xaxis=dict(title="Year", gridcolor='rgba(220,20,60,0.3)', color='#ffffff'),
            yaxis=dict(title="Cases", gridcolor='rgba(220,20,60,0.3)', color='#ffffff'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        colors_n = ['#ff0000','#ff6600','#ffcc00','#00ff00','#0099ff']
        for idx, col in enumerate(df_national.columns[1:]):
            fig2.add_trace(go.Scatter(
                x=df_national["Year"], y=df_national[col], mode='lines+markers', name=col,
                line=dict(width=4, color=colors_n[idx]), marker=dict(size=10)
            ))
        fig2.update_layout(
            title=dict(text="<b>All Crime Categories — National</b>",
                       font=dict(size=22, color='#ffffff', family='Arial Black')),
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=14, family='Arial Black'),
            legend=dict(bgcolor='rgba(0,0,0,0.9)', font=dict(color='#ffffff', size=13),
                        bordercolor='#dc143c', borderwidth=2),
            xaxis=dict(color='#ffffff'), yaxis=dict(color='#ffffff'),
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        cs1, cs2 = st.columns(2)
        with cs1:
            st.markdown("<h4>📊 STATISTICS</h4>", unsafe_allow_html=True)
            stats = pd.DataFrame({
                "Metric":["Mean","Median","Std Dev","Min","Max","Range","Growth Rate"],
                "Value":[f"{df_national[crime_type].mean():.2f}",
                         f"{df_national[crime_type].median():.2f}",
                         f"{df_national[crime_type].std():.2f}",
                         f"{df_national[crime_type].min():.0f}",
                         f"{df_national[crime_type].max():.0f}",
                         f"{df_national[crime_type].max()-df_national[crime_type].min():.0f}",
                         f"{growth_rate:+.2f}%"]
            })
            st.dataframe(stats, use_container_width=True, hide_index=True)
        with cs2:
            st.markdown("<h4>📈 YEAR-ON-YEAR CHANGES</h4>", unsafe_allow_html=True)
            yoy = df_national[crime_type].pct_change() * 100
            yoy_df = pd.DataFrame({"Year": df_national["Year"][1:], "YoY %": yoy[1:].round(2)})
            bar_c = ['#ff4444' if v > 0 else '#44ff88' for v in yoy_df["YoY %"]]
            fig_yoy2 = go.Figure(go.Bar(x=yoy_df["Year"], y=yoy_df["YoY %"],
                marker=dict(color=bar_c), text=[f"{v:+.1f}%" for v in yoy_df["YoY %"]],
                textposition='outside', textfont=dict(color='#ffffff', size=11)))
            fig_yoy2.add_hline(y=0, line_color='white', line_dash='dash')
            fig_yoy2.update_layout(plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', family='Arial Black'),
                xaxis=dict(color='#ffffff', dtick=1), yaxis=dict(color='#ffffff'), height=340)
            st.plotly_chart(fig_yoy2, use_container_width=True)

    with tab4:
        styled_n = df_national.style.background_gradient(cmap='Reds', subset=df_national.columns[1:]) \
            .set_properties(**{'color':'white','font-weight':'bold'})
        st.dataframe(styled_n, use_container_width=True, height=400)
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        csv_n = df_national.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download National Dataset", data=csv_n,
                           file_name=f"national_crime_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# ═══════════════════════════════════════════════════════
# MODULE 5 — AI PREDICTION ENGINE
# ═══════════════════════════════════════════════════════
elif "AI Prediction" in menu:
    st.markdown("<h2>🤖 AI-POWERED CRIME PREDICTION ENGINE</h2>", unsafe_allow_html=True)

    pred_mode = st.radio("🎯 Prediction Mode", ["🇮🇳 National Prediction","📍 State-wise Prediction"], horizontal=True)

    st.markdown("---")

    if "National" in pred_mode:
        crime_type = st.selectbox("🎯 Select Crime Category", df_national.columns[1:], key="pred_nat")
        X = df_national[["Year"]]; y = df_national[crime_type]

    else:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            pred_state = st.selectbox("🏛️ Select State/UT", STATE_LIST, key="pred_state")
        with col_p2:
            crime_type = st.selectbox("🔍 Select Crime Type", ["Total_Crimes"] + CRIME_TYPES, key="pred_crime_st")
        sdf_p = df_state[df_state["State"] == pred_state].sort_values("Year")
        X = sdf_p[["Year"]]; y = sdf_p[crime_type]

    # Train
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
    model.fit(X_tr, y_tr)
    r2_train = r2_score(y_tr, model.predict(X_tr))
    r2_test  = r2_score(y_te, model.predict(X_te))
    mae      = mean_absolute_error(y_te, model.predict(X_te))
    rmse     = np.sqrt(mean_squared_error(y_te, model.predict(X_te)))

    st.markdown("### 🎯 MODEL PERFORMANCE METRICS")
    pm1, pm2, pm3, pm4 = st.columns(4)
    with pm1: st.metric("🎯 R² Score", f"{r2_test:.3f}", f"Train: {r2_train:.3f}")
    with pm2: st.metric("📊 MAE", f"{mae:.2f}", "cases")
    with pm3: st.metric("📈 RMSE", f"{rmse:.2f}")
    with pm4:
        q = "Excellent" if r2_test > 0.9 else "Good" if r2_test > 0.7 else "Fair"
        st.metric("✅ Quality", q, f"{r2_test*100:.1f}%")

    st.markdown("---")
    st.markdown("### 🔮 MAKE PREDICTION")

    pi1, pi2, pi3 = st.columns([2,1,1])
    with pi1:
        future_year = st.number_input("📅 Future Year", min_value=2024, max_value=2040, value=2025, step=1)
    with pi2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 PREDICT NOW", use_container_width=True)
    with pi3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"📊 {future_year - 2023} year(s) ahead")

    if predict_btn:
        predicted = int(model.predict([[future_year]])[0])
        lower     = int(predicted * 0.9)
        upper     = int(predicted * 1.1)

        st.success("✅ PREDICTION ANALYSIS COMPLETE")

        label_line = f"State: {pred_state.title()} | " if "State" in pred_mode else ""
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #dc143c, #8b0000); padding: 40px;
                    border-radius: 20px; text-align: center; border: 4px solid #ff0000;
                    box-shadow: 0 15px 40px rgba(220,20,60,0.8); max-width:700px; margin: 0 auto;'>
            <h3 style='color:#ffffff; font-size:1.2rem; margin-bottom:16px; letter-spacing:2px;'>
                PREDICTED {crime_type.upper()} CASES</h3>
            <p style='color:rgba(255,255,255,0.8); font-size:0.95rem; margin-bottom:8px;'>{label_line}Year: {future_year}</p>
            <h1 style='color:#ffffff; font-size:4rem; margin:16px 0;
                       text-shadow:4px 4px 10px rgba(0,0,0,0.9);'>{predicted:,}</h1>
            <p style='color:#ffff00; font-size:1rem; font-weight:800; background:rgba(0,0,0,0.3);
                      display:inline-block; padding:8px 24px; border-radius:20px;'>
                CONFIDENCE RANGE: {lower:,} – {upper:,} CASES</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.markdown("---")

        vt1, vt2, vt3 = st.tabs(["📈 Forecast Chart","📊 Analysis","🎯 Model Info"])

        with vt1:
            hist_y = X.values.flatten()
            hist_v = y.values
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=hist_y, y=hist_v, mode='lines+markers', name='Historical',
                line=dict(color='#00ccff', width=5), marker=dict(size=12, color='#0099ff')
            ))
            fig_p.add_trace(go.Scatter(
                x=[future_year], y=[predicted], mode='markers', name='Prediction',
                marker=dict(size=36, color='#ffff00', symbol='star', line=dict(color='#ff0000', width=4))
            ))
            fig_p.add_trace(go.Scatter(
                x=[hist_y[-1], future_year], y=[hist_v[-1], predicted],
                mode='lines', name='Forecast', line=dict(color='#ff0000', width=4, dash='dash')
            ))
            fig_p.add_trace(go.Scatter(
                x=[future_year]*3, y=[lower, predicted, upper],
                fill='toself', fillcolor='rgba(255,255,0,0.3)',
                line=dict(color='#ffff00', width=3), name='Confidence'
            ))
            fig_p.update_layout(
                title=dict(text=f"<b>{crime_type} Forecast</b>",
                           font=dict(size=22, color='#ffffff', family='Arial Black')),
                plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', size=14, family='Arial Black'),
                xaxis=dict(title="Year", color='#ffffff', gridcolor='rgba(220,20,60,0.3)'),
                yaxis=dict(title="Cases", color='#ffffff', gridcolor='rgba(220,20,60,0.3)'),
                legend=dict(bgcolor='rgba(0,0,0,0.8)', font=dict(color='#ffffff', size=13),
                            bordercolor='#dc143c', borderwidth=2),
                height=520
            )
            st.plotly_chart(fig_p, use_container_width=True)

        with vt2:
            pa1, pa2 = st.columns(2)
            hist_avg   = float(y.mean())
            diff_avg   = predicted - hist_avg
            pct_diff   = (diff_avg / hist_avg) * 100
            with pa1:
                st.markdown("<h4>📊 METRICS</h4>", unsafe_allow_html=True)
                cmp_df = pd.DataFrame({
                    "Metric":["Historical Avg","Predicted","Difference","% Change","Latest Actual"],
                    "Value":[f"{hist_avg:.0f}",f"{predicted:,}",f"{diff_avg:+.0f}",
                             f"{pct_diff:+.2f}%",f"{int(y.iloc[-1]):,}"]
                })
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)
            with pa2:
                st.markdown("<h4>🚨 RISK ASSESSMENT</h4>", unsafe_allow_html=True)
                if pct_diff > 20:   st.error("🔴 HIGH RISK — Significant increase predicted")
                elif pct_diff > 5:  st.warning("🟡 MODERATE RISK — Notable increase expected")
                else:               st.success("🟢 LOW RISK — Stable or decreasing trend")
                st.markdown(f"""
                **Projected Change:** {diff_avg:+,.0f} cases  
                **Relative to Avg:** {pct_diff:+.2f}%  
                **Confidence Band:** ±10%  
                **Model Accuracy:** {r2_test*100:.1f}%
                """)

        with vt3:
            mi1, mi2 = st.columns(2)
            with mi1:
                st.markdown("<h4>📊 Feature Importance</h4>", unsafe_allow_html=True)
                fig_imp = go.Figure(go.Bar(
                    x=["Year"], y=model.feature_importances_,
                    marker=dict(color='#ff0000'),
                    text=[f"{v:.2%}" for v in model.feature_importances_],
                    textposition='outside', textfont=dict(color='#ffffff', size=14)
                ))
                fig_imp.update_layout(plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff', family='Arial Black'),
                    yaxis=dict(color='#ffffff'), height=340)
                st.plotly_chart(fig_imp, use_container_width=True)
            with mi2:
                st.markdown("<h4>🧠 Model Configuration</h4>", unsafe_allow_html=True)
                st.markdown(f"""
                - **Algorithm:** Random Forest Regressor  
                - **Trees:** 200  
                - **Max Depth:** 10  
                - **Training Samples:** {len(X_tr)}  
                - **Test Samples:** {len(X_te)}  
                - **R² Train:** {r2_train:.4f}  
                - **R² Test:** {r2_test:.4f}  
                - **MAE:** {mae:.2f}  
                - **RMSE:** {rmse:.2f}  
                """)

        st.markdown("---")
        st.markdown("### 📥 EXPORT PREDICTION REPORTS")
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            rpt = pd.DataFrame({"Metric":["Crime","Year","Predicted","Lower","Upper","Change%"],
                                "Value":[crime_type,future_year,predicted,lower,upper,f"{pct_diff:+.2f}%"]})
            st.download_button("📄 Detailed Report (CSV)", data=rpt.to_csv(index=False).encode(),
                               file_name=f"prediction_{crime_type}_{future_year}.csv", mime="text/csv", use_container_width=True)
        with dl2:
            summary = f"PREDICTION REPORT\nCrime: {crime_type}\nYear: {future_year}\nPredicted: {predicted:,}\nRange: {lower:,}–{upper:,}\nChange: {pct_diff:+.2f}%\nR²: {r2_test:.4f}"
            st.download_button("📋 Summary (TXT)", data=summary,
                               file_name=f"summary_{future_year}.txt", mime="text/plain", use_container_width=True)
        with dl3:
            fc_df = pd.concat([X.assign(**{crime_type: y.values}),
                               pd.DataFrame({"Year":[future_year], crime_type:[predicted]})], ignore_index=True)
            st.download_button("📊 Forecast Data (CSV)", data=fc_df.to_csv(index=False).encode(),
                               file_name=f"forecast_{crime_type}.csv", mime="text/csv", use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding:28px 36px;
            background:linear-gradient(135deg, rgba(220,20,60,0.4), rgba(139,0,0,0.4));
            border-radius:20px; border:3px solid #dc143c;'>
    <h3 style='color:#ff0000; margin-bottom:12px; font-size:1.5rem;'>🚨 CRIME INTELLIGENCE PRO 🚨</h3>
    <p style='color:#ffffff; font-size:1rem; font-weight:800; margin-bottom:6px;'>
        Professional Crime Analytics & Prediction Platform — India</p>
    <p style='color:#cccccc; font-size:0.9rem;'>
        AI/ML · Data Science · Strategic Intelligence · Hackathon Edition 🏆</p>
    <p style='color:#aaaaaa; font-size:0.8rem; margin-top:8px;'>
        Covering 34 States/UTs · 7 Crime Categories · 2015–2023</p>
</div>
""", unsafe_allow_html=True)