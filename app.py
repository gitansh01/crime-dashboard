import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

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
label, .stMarkdown, .stText, p, span, div { color: #ffffff !important; text-shadow: 1px 1px 3px rgba(0,0,0,0.8); }
.stSelectbox label { color: #ffffff !important; font-weight: 900 !important; font-size: 1.1rem !important; }
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #1a0a0a !important; color: #ffffff !important;
    border: 4px solid #dc143c !important; font-weight: 900 !important;
    font-size: 1.2rem !important; min-height: 52px !important;
}
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] div[class*="singleValue"],
.stSelectbox [class*="ValueContainer"] *, .stSelectbox [class*="singleValue"],
.stSelectbox [class*="placeholder"], .stSelectbox input {
    color: #ffffff !important; font-weight: 900 !important; -webkit-text-fill-color: #ffffff !important;
}
.stSelectbox svg { fill: #dc143c !important; }
[data-baseweb="popover"] { background-color: #1a0a0a !important; border: 4px solid #dc143c !important; }
.stSelectbox li, [data-baseweb="menu"] li, [role="option"] {
    background-color: #1a0a0a !important; color: #ffffff !important;
    font-weight: 900 !important; font-size: 1.1rem !important;
    padding: 16px 24px !important; border-bottom: 2px solid #333 !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stSelectbox li:hover, [role="option"]:hover { background: linear-gradient(135deg, #dc143c, #ff0000) !important; }
h1 { color:#ffffff !important; text-shadow:5px 5px 10px rgba(0,0,0,0.9); font-size:2.8rem !important; font-weight:900 !important; text-align:center; }
h2, h3, h4 { color:#ffffff !important; font-weight:900 !important; text-shadow:3px 3px 8px rgba(0,0,0,0.9); }
h2 { font-size:1.8rem !important; } h3 { font-size:1.5rem !important; } h4 { font-size:1.3rem !important; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(220,20,60,0.4), rgba(139,0,0,0.3));
    border: 4px solid #dc143c; border-radius: 15px; padding: 25px; box-shadow: 0 15px 40px rgba(220,20,60,0.6);
}
[data-testid="stMetricValue"] { font-size:2.5rem !important; font-weight:900 !important; color:#ff0000 !important; text-shadow:4px 4px 8px rgba(0,0,0,0.9); }
[data-testid="stMetricLabel"] { color:#ffffff !important; font-weight:800 !important; }
.stButton>button {
    background: linear-gradient(135deg, #dc143c, #8b0000) !important; color:white !important;
    font-weight:900 !important; font-size:1.1rem !important; border:4px solid #ff0000 !important;
    border-radius:12px !important; padding:16px 40px !important; box-shadow:0 8px 20px rgba(220,20,60,0.6);
}
.stButton>button:hover { transform:translateY(-3px); box-shadow:0 12px 30px rgba(220,20,60,0.8); }
.stDownloadButton>button {
    background: linear-gradient(135deg, #0066cc, #003d82) !important; color:#ffffff !important;
    font-weight:900 !important; font-size:1.1rem !important; border:4px solid #ffffff !important;
    border-radius:12px !important; padding:16px 36px !important; box-shadow:0 10px 25px rgba(0,102,204,0.7);
}
.stDownloadButton>button:hover { transform:translateY(-3px); box-shadow:0 15px 35px rgba(0,102,204,0.9); background: linear-gradient(135deg, #0080ff, #0066cc) !important; }
.stDownloadButton>button span { color:#ffffff !important; font-weight:900 !important; }
[data-testid="stDataFrame"] {
    background:rgba(10,10,30,0.9); border-radius:12px; border:3px solid #dc143c;
    box-shadow:0 12px 30px rgba(0,0,0,0.7); padding-top:8px !important; margin-bottom:0 !important;
}
[data-testid="stDataFrame"] th { background-color:#1a0020 !important; color:#ffffff !important; font-weight:900 !important; padding:12px !important; }
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] [role="gridcell"] { color:#ffffff !important; font-weight:700 !important; -webkit-text-fill-color:#ffffff !important; }
.stTabs [data-baseweb="tab-list"] { gap:10px; background:rgba(10,10,30,0.9); border-radius:15px; padding:12px; border:3px solid #dc143c; }
.stTabs [data-baseweb="tab"] { background:linear-gradient(135deg,rgba(220,20,60,0.3),rgba(139,0,0,0.3)); border-radius:12px; color:#ffffff !important; font-weight:900 !important; font-size:1rem !important; padding:14px 22px; border:2px solid transparent; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#dc143c,#ff0000); color:white !important; border:2px solid #ffffff; box-shadow:0 8px 20px rgba(220,20,60,0.7); }
.stRadio > div { background:rgba(26,26,60,0.7); border-radius:12px; padding:20px; border:3px solid #dc143c; }
.stRadio label { color:#ffffff !important; font-weight:900 !important; font-size:1.1rem !important; }
.stAlert { background:rgba(10,10,30,0.95) !important; border-radius:12px; padding:20px; color:#ffffff !important; font-weight:700 !important; }
.stSuccess { background:rgba(0,100,0,0.95) !important; border-left:6px solid #00ff00 !important; }
.stInfo { background:rgba(0,50,150,0.95) !important; border-left:6px solid #0099ff !important; }
.stWarning { background:rgba(200,100,0,0.95) !important; border-left:6px solid #ffcc00 !important; }
.stError { background:rgba(150,0,0,0.95) !important; border-left:6px solid #ff0000 !important; }
.stNumberInput input { background-color:#000000 !important; color:#ffffff !important; border:3px solid #dc143c !important; font-weight:900 !important; font-size:1.1rem !important; }
hr { border:3px solid #dc143c; margin:30px 0; box-shadow:0 3px 15px rgba(220,20,60,0.7); }
.streamlit-expanderHeader { background-color:#1a1a3c !important; color:#ffffff !important; font-weight:900 !important; font-size:1.2rem !important; border:4px solid #dc143c !important; border-radius:12px !important; padding:14px !important; }
.kpi-card { background:linear-gradient(135deg,rgba(220,20,60,0.25),rgba(10,10,30,0.9)); border:2px solid #dc143c; border-radius:14px; padding:20px 14px; text-align:center; box-shadow:0 8px 24px rgba(220,20,60,0.4); transition:transform 0.2s ease, box-shadow 0.2s ease; }
.kpi-card:hover { transform:translateY(-4px); box-shadow:0 14px 36px rgba(220,20,60,0.65); }
.kpi-title { font-size:0.75rem; letter-spacing:2px; text-transform:uppercase; color:rgba(255,255,255,0.7) !important; margin-bottom:6px; font-weight:700; }
.kpi-value { font-size:1.8rem; font-weight:900; color:#ff4444 !important; text-shadow:2px 2px 6px rgba(0,0,0,0.7); line-height:1.1; }
.kpi-sub { font-size:0.75rem; color:rgba(255,255,255,0.6) !important; margin-top:4px; }
.rank-card-red { background:linear-gradient(135deg,rgba(220,20,60,0.2),rgba(10,10,30,0.95)); border-left:5px solid #dc143c; border-radius:10px; padding:12px 16px; margin-bottom:8px; }
.rank-card-green { background:linear-gradient(135deg,rgba(0,150,50,0.2),rgba(10,10,30,0.95)); border-left:5px solid #00cc44; border-radius:10px; padding:12px 16px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

STATE_COORDS = {
    "ANDHRA PRADESH":(15.9129,79.74),"ARUNACHAL PRADESH":(28.218,94.7278),
    "ASSAM":(26.2006,92.9376),"BIHAR":(25.0961,85.3131),
    "CHHATTISGARH":(21.2787,81.8661),"GOA":(15.2993,74.124),
    "GUJARAT":(22.2587,71.1924),"HARYANA":(29.0588,76.0856),
    "HIMACHAL PRADESH":(31.1048,77.1734),"JAMMU & KASHMIR":(33.7782,76.5762),
    "JHARKHAND":(23.6102,85.2799),"KARNATAKA":(15.3173,75.7139),
    "KERALA":(10.8505,76.2711),"MADHYA PRADESH":(22.9734,78.6569),
    "MAHARASHTRA":(19.7515,75.7139),"MANIPUR":(24.6637,93.9063),
    "MEGHALAYA":(25.467,91.3662),"MIZORAM":(23.1645,92.9376),
    "NAGALAND":(26.1584,94.5624),"ODISHA":(20.9517,85.0985),
    "PUNJAB":(31.1471,75.3412),"RAJASTHAN":(27.0238,74.2179),
    "SIKKIM":(27.533,88.5122),"TAMIL NADU":(11.1271,78.6569),
    "TRIPURA":(23.9408,91.9882),"UTTAR PRADESH":(26.8467,80.9462),
    "UTTARAKHAND":(30.0668,79.0193),"WEST BENGAL":(22.9868,87.855),
    "A & N ISLANDS":(11.7401,92.6586),"CHANDIGARH":(30.7333,76.7794),
    "D & N HAVELI":(20.1809,73.0169),"DAMAN & DIU":(20.4283,72.8397),
    "LAKSHADWEEP":(10.5667,72.6417),"PUDUCHERRY":(11.9416,79.8083)
}

PLOT_CRIMES = ["MURDER","RAPE","KIDNAPPING & ABDUCTION","ROBBERY",
               "BURGLARY","THEFT","DACOITY","RIOTS","CHEATING","DOWRY DEATHS"]
COLORS = ['#ff0000','#ff6600','#ffcc00','#00ccff','#cc00ff',
          '#00ff88','#ff66cc','#4488ff','#ff4488','#88ff44']

@st.cache_data
def load_data():
    df = pd.read_csv("clean_state_data.csv")
    df_trend = pd.read_csv("india_trend.csv")
    df_pred  = pd.read_csv("state_predictions.csv")
    df_total = df[df["DISTRICT"] == "TOTAL"].copy()
    df_total["Latitude"]  = df_total["STATE/UT"].map(lambda s: STATE_COORDS.get(s,(20,80))[0])
    df_total["Longitude"] = df_total["STATE/UT"].map(lambda s: STATE_COORDS.get(s,(20,80))[1])
    df_pred["Risk Level"] = df_pred["Predicted Risk Score"].apply(
        lambda x: "HIGH" if x > 60 else ("MEDIUM" if x > 25 else "LOW"))
    return df_total, df_trend, df_pred

@st.cache_resource
def load_models():
    crime_model = joblib.load("state_crime_model.pkl")
    label_enc   = joblib.load("state_label_encoder.pkl")
    risk_model  = joblib.load("state_risk_model.pkl")
    return crime_model, label_enc, risk_model

df_state, df_trend, df_pred = load_data()
crime_model, label_enc, risk_model = load_models()
STATE_LIST = sorted(df_state["STATE/UT"].unique().tolist())
YEARS      = sorted(df_state["YEAR"].unique().tolist())

with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>🎯 COMMAND CENTER</h2>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("📊 Select Module", [
        "🏠 National Overview",
        "🗺️ India Crime Map",
        "📊 State-wise Analysis",
        "🔍 National Trend Analysis",
        "🤖 AI Prediction Engine",
    ], index=0)
    st.markdown("---")
    st.markdown("### 📈 System Status")
    st.metric("Total Records",    f"{len(df_state):,}")
    st.metric("States / UTs",     df_state["STATE/UT"].nunique())
    st.metric("Years Covered",    f"{YEARS[0]}–{YEARS[-1]}")
    st.metric("Crime Categories", len(PLOT_CRIMES))
    st.metric("Status",           "🟢 ONLINE")
    st.markdown("---")
    st.markdown("<p style='text-align:center;font-size:0.8rem;color:#aaa;'>Crime Intelligence Pro v3.0<br>Hackathon Edition 🏆<br>Data: Kaggle NCRB Dataset</p>", unsafe_allow_html=True)

st.markdown("<h1>🚨 CRIME INTELLIGENCE PRO 🚨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:1.3rem;color:#ff6666;font-weight:800;'>AI-Powered National Crime Analytics & Prediction System | India (2001–2012)</p>", unsafe_allow_html=True)
st.markdown("---")

# ═══════════ MODULE 1 — NATIONAL OVERVIEW ═══════════
if "National Overview" in menu:
    st.markdown("<h2>🏠 NATIONAL CRIME OVERVIEW — INDIA</h2>", unsafe_allow_html=True)
    latest  = df_state[df_state["YEAR"] == YEARS[-1]]
    prev    = df_state[df_state["YEAR"] == YEARS[-2]]
    total_l = int(latest["TOTAL IPC CRIMES"].sum())
    total_p = int(prev["TOTAL IPC CRIMES"].sum())
    yoy_pct = ((total_l - total_p) / total_p) * 100
    worst   = latest.loc[latest["TOTAL IPC CRIMES"].idxmax(), "STATE/UT"]
    safest  = latest.loc[latest["TOTAL IPC CRIMES"].idxmin(), "STATE/UT"]
    top_cat = latest[PLOT_CRIMES].sum().idxmax()
    all_total = int(df_state["TOTAL IPC CRIMES"].sum())
    kpi_data = [
        ("🇮🇳 Total Crimes 2012",  f"{total_l:,}",     f"YoY {yoy_pct:+.1f}%"),
        ("📊 All Years Total",      f"{all_total:,}",   "2001–2012"),
        ("⚠️ Highest Crime State",  worst.title(),      f"{int(latest['TOTAL IPC CRIMES'].max()):,} cases"),
        ("✅ Safest State/UT",      safest.title(),     f"{int(latest['TOTAL IPC CRIMES'].min()):,} cases"),
        ("🔺 Top Crime Type",       top_cat.title(),    f"{int(latest[top_cat].sum()):,} cases"),
        ("📈 Avg YoY Growth",       f"{yoy_pct:.1f}%",  "2011→2012"),
    ]
    cols = st.columns(6)
    for col, (title, val, sub) in zip(cols, kpi_data):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">{title}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("---")
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("<h3>📊 Top 10 States by Total IPC Crimes (2012)</h3>", unsafe_allow_html=True)
        top10 = latest.nlargest(10, "TOTAL IPC CRIMES")[["STATE/UT","TOTAL IPC CRIMES"]].reset_index(drop=True)
        top10["STATE/UT"] = top10["STATE/UT"].str.title()
        fig_top = go.Figure(go.Bar(
            x=top10["TOTAL IPC CRIMES"], y=top10["STATE/UT"], orientation='h',
            marker=dict(color=top10["TOTAL IPC CRIMES"], colorscale=[[0,"#8b0000"],[0.5,"#dc143c"],[1,"#ff6666"]], line=dict(color="#ffffff",width=0.5)),
            text=[f"{v:,}" for v in top10["TOTAL IPC CRIMES"]], textposition="outside",
            textfont=dict(color="#ffffff", size=11, family="Arial Black")
        ))
        fig_top.update_layout(plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff',size=12,family='Arial Black'),
            yaxis=dict(color='#ffffff',autorange="reversed"), xaxis=dict(color='#ffffff',gridcolor='rgba(220,20,60,0.2)'),
            height=420, margin=dict(l=10,r=80,t=10,b=10))
        st.plotly_chart(fig_top, use_container_width=True)
    with col_r:
        st.markdown("<h3>🥧 Crime Category Distribution (2012)</h3>", unsafe_allow_html=True)
        cat_totals = latest[PLOT_CRIMES].sum()
        fig_pie = go.Figure(go.Pie(
            labels=[c.title() for c in cat_totals.index], values=cat_totals.values, hole=0.45,
            marker=dict(colors=COLORS, line=dict(color='#0a0a1e',width=2)),
            textfont=dict(color='#ffffff',size=11,family='Arial Black'),
            hovertemplate='<b>%{label}</b><br>Cases: %{value:,}<br>%{percent}<extra></extra>'
        ))
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff',family='Arial Black'),
            legend=dict(font=dict(color='#ffffff',size=10)), height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("---")
    st.markdown("<h3>📈 All-India Crime Trend (2001–2012)</h3>", unsafe_allow_html=True)
    nat_trend = df_state.groupby("YEAR")[PLOT_CRIMES + ["TOTAL IPC CRIMES"]].sum().reset_index()
    fig_trend = go.Figure()
    for i, ct in enumerate(PLOT_CRIMES):
        fig_trend.add_trace(go.Scatter(x=nat_trend["YEAR"], y=nat_trend[ct], mode='lines+markers',
            name=ct.title(), line=dict(width=3,color=COLORS[i]), marker=dict(size=7),
            hovertemplate=f'<b>{ct.title()}</b><br>Year: %{{x}}<br>Cases: %{{y:,}}<extra></extra>'))
    fig_trend.update_layout(plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff',size=13,family='Arial Black'),
        xaxis=dict(color='#ffffff',gridcolor='rgba(220,20,60,0.2)',title="Year",dtick=1),
        yaxis=dict(color='#ffffff',gridcolor='rgba(220,20,60,0.2)',title="Cases"),
        legend=dict(bgcolor='rgba(0,0,0,0.8)',font=dict(color='#ffffff',size=11),bordercolor='#dc143c',borderwidth=2),
        height=420, hovermode='x unified')
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown("---")
    st.markdown("<h3>🏆 State Rankings — All Crimes (2012)</h3>", unsafe_allow_html=True)
    rank_df = latest[["STATE/UT","TOTAL IPC CRIMES","MURDER","RAPE","KIDNAPPING & ABDUCTION","ROBBERY","THEFT","BURGLARY","RIOTS","DOWRY DEATHS"]].copy()
    rank_df = rank_df.sort_values("TOTAL IPC CRIMES", ascending=False).reset_index(drop=True)
    rank_df.index = rank_df.index + 1
    rank_df["STATE/UT"] = rank_df["STATE/UT"].str.title()
    styled_rank = rank_df.style.background_gradient(cmap='Reds', subset=["TOTAL IPC CRIMES","MURDER","RAPE","KIDNAPPING & ABDUCTION","ROBBERY","THEFT","BURGLARY","RIOTS","DOWRY DEATHS"]).set_properties(**{'color':'white','font-weight':'bold'})
    st.dataframe(styled_rank, use_container_width=True, height=420)

# ═══════════ MODULE 2 — INDIA MAP ═══════════
elif "India Crime Map" in menu:
    st.markdown("<h2>🗺️ INDIA CRIME HEATMAP</h2>", unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    with mc1: map_year  = st.selectbox("📅 Year", YEARS, index=len(YEARS)-1, key="map_yr")
    with mc2: map_crime = st.selectbox("🔍 Crime Type", ["TOTAL IPC CRIMES"] + PLOT_CRIMES, key="map_cr")
    with mc3: map_size  = st.selectbox("🔵 Bubble Size", ["Proportional","Equal"], key="map_sz")
    mdf = df_state[df_state["YEAR"] == map_year].copy()
    mdf["State_Title"] = mdf["STATE/UT"].str.title()
    max_val = mdf[map_crime].max()
    mdf["bubble"] = (mdf[map_crime] / max_val * 55 + 5) if map_size == "Proportional" else 22
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattergeo(
        lat=mdf["Latitude"], lon=mdf["Longitude"], mode='markers+text',
        marker=dict(size=mdf["bubble"], color=mdf[map_crime],
            colorscale=[[0,"#1a0020"],[0.3,"#8b0000"],[0.65,"#dc143c"],[1,"#ff6666"]],
            colorbar=dict(title=dict(text=f"<b>{map_crime.title()}</b>",font=dict(color="#ffffff",size=12)),
                tickfont=dict(color="#ffffff"), thickness=16, len=0.7,
                bgcolor="rgba(10,10,30,0.8)", bordercolor="#dc143c", borderwidth=2),
            cmin=0, cmax=max_val, line=dict(color='#ffffff',width=1), opacity=0.87),
        text=mdf["State_Title"], textfont=dict(color='#ffffff',size=8,family='Arial Black'), textposition="top center",
        customdata=np.stack([mdf["State_Title"],mdf[map_crime],mdf["TOTAL IPC CRIMES"],mdf["MURDER"]],axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>" + f"{map_crime.title()}: %{{customdata[1]:,}}<br>" + "Total IPC: %{customdata[2]:,}<br>Murder: %{customdata[3]:,}<extra></extra>"
    ))
    fig_map.update_layout(
        geo=dict(scope='asia', center=dict(lat=22.5,lon=82.5), projection_scale=4.5,
            showland=True, landcolor='rgba(20,20,50,0.95)', showocean=True, oceancolor='rgba(5,5,20,0.95)',
            showcountries=True, countrycolor='rgba(220,20,60,0.6)', showcoastlines=True, coastlinecolor='rgba(220,20,60,0.8)',
            bgcolor='rgba(0,0,0,0)', lataxis=dict(range=[6,38]), lonaxis=dict(range=[66,100])),
        paper_bgcolor='rgba(10,10,30,0.95)', height=640, margin=dict(l=0,r=0,t=40,b=10),
        title=dict(text=f"<b>India Crime Map — {map_crime.title()} ({map_year})</b>",
            font=dict(size=18,color='#ffffff',family='Arial Black'), x=0.5))
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("---")
    ml, mr = st.columns(2)
    with ml:
        st.markdown(f"<h3>🔴 Top 5 Most Affected ({map_year})</h3>", unsafe_allow_html=True)
        top5 = mdf.nlargest(5, map_crime).reset_index(drop=True)
        medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
        for i, row in top5.iterrows():
            st.markdown(f'<div class="rank-card-red"><div style="display:flex;justify-content:space-between;align-items:center;"><b>{medals[i]} {row["State_Title"]}</b><span style="color:#ff4444;font-weight:900;font-size:1.1rem;">{int(row[map_crime]):,}</span></div></div>', unsafe_allow_html=True)
    with mr:
        st.markdown(f"<h3>🟢 Top 5 Safest ({map_year})</h3>", unsafe_allow_html=True)
        bot5 = mdf.nsmallest(5, map_crime).reset_index(drop=True)
        for i, row in bot5.iterrows():
            st.markdown(f'<div class="rank-card-green"><div style="display:flex;justify-content:space-between;align-items:center;"><b>✅ {row["State_Title"]}</b><span style="color:#44ff88;font-weight:900;font-size:1.1rem;">{int(row[map_crime]):,}</span></div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h3>📊 State × Crime Category Heatmap</h3>", unsafe_allow_html=True)
    heat_df = mdf[["State_Title"] + PLOT_CRIMES].set_index("State_Title")
    fig_heat = go.Figure(go.Heatmap(
        z=heat_df.values, x=[c.title() for c in PLOT_CRIMES], y=heat_df.index,
        colorscale=[[0,"#0a0a1e"],[0.3,"#8b0000"],[0.7,"#dc143c"],[1,"#ff6666"]],
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:,}<extra></extra>',
        colorbar=dict(title=dict(text="Cases",font=dict(color="#ffffff")), tickfont=dict(color="#ffffff"),
            bgcolor="rgba(10,10,30,0.8)", bordercolor="#dc143c", borderwidth=2)))
    fig_heat.update_layout(plot_bgcolor='rgba(10,10,30,0.9)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff',size=11,family='Arial Black'),
        xaxis=dict(color='#ffffff'), yaxis=dict(color='#ffffff',autorange='reversed'),
        height=900, margin=dict(l=220,r=20,t=20,b=60))
    st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════ MODULE 3 — STATE-WISE ═══════════
elif "State-wise" in menu:
    st.markdown("<h2>📊 STATE-WISE DEEP ANALYSIS</h2>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1: sel_state = st.selectbox("🏛️ Select State/UT", STATE_LIST, key="st_sel")
    with sc2: sel_crime = st.selectbox("🔍 Crime Type", ["TOTAL IPC CRIMES"] + PLOT_CRIMES, key="st_cr")
    sdf = df_state[df_state["STATE/UT"] == sel_state].sort_values("YEAR")
    latest_s = sdf[sdf["YEAR"] == YEARS[-1]].iloc[0]
    prev_s   = sdf[sdf["YEAR"] == YEARS[-2]].iloc[0]
    growth_s = ((latest_s[sel_crime] - prev_s[sel_crime]) / prev_s[sel_crime]) * 100
    rank_2012 = df_state[df_state["YEAR"]==YEARS[-1]].sort_values(sel_crime, ascending=False)
    nat_rank  = rank_2012["STATE/UT"].tolist().index(sel_state) + 1
    pred_row  = df_pred[df_pred["STATE/UT"] == sel_state]
    pred_val  = pred_row["Predicted Crimes"].values[0] if len(pred_row) > 0 else 0
    risk_val  = pred_row["Predicted Risk Score"].values[0] if len(pred_row) > 0 else 0
    risk_lbl  = pred_row["Risk Level"].values[0] if len(pred_row) > 0 else "N/A"
    st.markdown(f"<h3>📍 {sel_state.title()} — {sel_crime.title()}</h3>", unsafe_allow_html=True)
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpis_s = [
        ("2012 Cases",      f"{int(latest_s[sel_crime]):,}", f"YoY {growth_s:+.1f}%"),
        ("National Rank",   f"#{nat_rank}",                  "out of 34"),
        ("12-Yr Total",     f"{int(sdf[sel_crime].sum()):,}","2001–2012"),
        ("Peak Year",       str(int(sdf.loc[sdf[sel_crime].idxmax(),'YEAR'])), f"{int(sdf[sel_crime].max()):,}"),
        ("AI Predicted",    f"{int(pred_val):,}",            "Future estimate"),
        ("Risk Score",      f"{risk_val:.1f}",               risk_lbl),
    ]
    for col, (t,v,s) in zip([k1,k2,k3,k4,k5,k6], kpis_s):
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-title">{t}</div><div class="kpi-value">{v}</div><div class="kpi-sub">{s}</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("---")
    ts1,ts2,ts3,ts4,ts5 = st.tabs(["📈 Trend","📊 Category Mix","📉 YoY Change","🆚 Compare States","🗂️ Raw Data"])
    with ts1:
        fig_st = go.Figure()
        fig_st.add_trace(go.Scatter(x=sdf["YEAR"], y=sdf[sel_crime], mode='lines+markers',
            line=dict(color='#ff0000',width=5), marker=dict(size=14,color='#dc143c',line=dict(color='white',width=3)),
            fill='tozeroy', fillcolor='rgba(220,20,60,0.12)',
            hovertemplate='<b>Year: %{x}</b><br>Cases: %{y:,}<extra></extra>'))
        z = np.polyfit(sdf["YEAR"], sdf[sel_crime], 1); p = np.poly1d(z)
        fig_st.add_trace(go.Scatter(x=sdf["YEAR"], y=p(sdf["YEAR"]), mode='lines', name='Trend Line',
            line=dict(color='#ffff00',width=2,dash='dash')))
        fig_st.update_layout(title=dict(text=f"<b>{sel_state.title()} — {sel_crime.title()}</b>",font=dict(size=18,color='#ffffff',family='Arial Black')),
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff',size=13,family='Arial Black'),
            xaxis=dict(title="Year",color='#ffffff',gridcolor='rgba(220,20,60,0.2)',dtick=1),
            yaxis=dict(title="Cases",color='#ffffff',gridcolor='rgba(220,20,60,0.2)'),
            legend=dict(bgcolor='rgba(0,0,0,0.7)',font=dict(color='#ffffff'),bordercolor='#dc143c',borderwidth=1), height=460)
        st.plotly_chart(fig_st, use_container_width=True)
    with ts2:
        cat_sums = sdf[sdf["YEAR"]==YEARS[-1]][PLOT_CRIMES].iloc[0]
        tc1, tc2 = st.columns(2)
        with tc1:
            fig_bar = go.Figure(go.Bar(x=[c.title() for c in PLOT_CRIMES], y=cat_sums.values,
                marker=dict(color=COLORS,line=dict(color='#ffffff',width=1)),
                text=[f"{int(v):,}" for v in cat_sums.values], textposition='outside',
                textfont=dict(color='#ffffff',size=10,family='Arial Black')))
            fig_bar.update_layout(title=dict(text="<b>Category Breakdown (2012)</b>",font=dict(size=16,color='#ffffff',family='Arial Black')),
                plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff',family='Arial Black'),
                xaxis=dict(color='#ffffff',tickangle=-35), yaxis=dict(color='#ffffff',gridcolor='rgba(220,20,60,0.2)'), height=420)
            st.plotly_chart(fig_bar, use_container_width=True)
        with tc2:
            fig_pie_s = go.Figure(go.Pie(labels=[c.title() for c in PLOT_CRIMES], values=cat_sums.values, hole=0.4,
                marker=dict(colors=COLORS,line=dict(color='#0a0a1e',width=2)),
                textfont=dict(color='#ffffff',size=11,family='Arial Black')))
            fig_pie_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#ffffff',family='Arial Black'),
                legend=dict(font=dict(color='#ffffff',size=10)), height=420)
            st.plotly_chart(fig_pie_s, use_container_width=True)
    with ts3:
        sdf2 = sdf.copy(); sdf2["YoY"] = sdf2[sel_crime].pct_change() * 100; sdf2 = sdf2.dropna(subset=["YoY"])
        bar_c = ['#ff4444' if v > 0 else '#44ff88' for v in sdf2["YoY"]]
        fig_yoy = go.Figure(go.Bar(x=sdf2["YEAR"], y=sdf2["YoY"], marker=dict(color=bar_c,line=dict(color='#ffffff',width=1)),
            text=[f"{v:+.1f}%" for v in sdf2["YoY"]], textposition='outside', textfont=dict(color='#ffffff',size=11,family='Arial Black')))
        fig_yoy.add_hline(y=0, line_color='#ffffff', line_width=2, line_dash='dash')
        fig_yoy.update_layout(title=dict(text=f"<b>YoY % Change — {sel_crime.title()}</b>",font=dict(size=18,color='#ffffff',family='Arial Black')),
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff',family='Arial Black'),
            xaxis=dict(color='#ffffff',dtick=1), yaxis=dict(color='#ffffff',gridcolor='rgba(220,20,60,0.2)',title="% Change"), height=420)
        st.plotly_chart(fig_yoy, use_container_width=True)
    with ts4:
        available_states = [s for s in STATE_LIST if s != sel_state]
        default_compare  = available_states[:3]
        compare_states   = st.multiselect("🆚 Select States to Compare", available_states, default=default_compare, max_selections=5, key="cmp_st")
        states_to_plot   = [sel_state] + compare_states
        fig_cmp = go.Figure()
        cmp_colors = ['#ff0000','#00ccff','#ffcc00','#00ff88','#cc00ff','#ff6600']
        for ci, st_name in enumerate(states_to_plot):
            cdf = df_state[df_state["STATE/UT"] == st_name].sort_values("YEAR")
            fig_cmp.add_trace(go.Scatter(x=cdf["YEAR"], y=cdf[sel_crime], mode='lines+markers',
                name=st_name.title(), line=dict(width=3,color=cmp_colors[ci%6]), marker=dict(size=8),
                hovertemplate=f'<b>{st_name.title()}</b><br>Year: %{{x}}<br>Cases: %{{y:,}}<extra></extra>'))
        fig_cmp.update_layout(title=dict(text=f"<b>Multi-State Comparison — {sel_crime.title()}</b>",font=dict(size=18,color='#ffffff',family='Arial Black')),
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff',family='Arial Black'),
            xaxis=dict(color='#ffffff',title="Year",dtick=1), yaxis=dict(color='#ffffff',title="Cases",gridcolor='rgba(220,20,60,0.2)'),
            legend=dict(bgcolor='rgba(0,0,0,0.8)',font=dict(color='#ffffff',size=12),bordercolor='#dc143c',borderwidth=2),
            height=480, hovermode='x unified')
        st.plotly_chart(fig_cmp, use_container_width=True)
    with ts5:
        disp = ["YEAR"] + PLOT_CRIMES + ["TOTAL IPC CRIMES"]
        styled_s = sdf[disp].style.background_gradient(cmap='Reds', subset=PLOT_CRIMES+["TOTAL IPC CRIMES"]).set_properties(**{'color':'white','font-weight':'bold'})
        st.dataframe(styled_s, use_container_width=True, height=400)
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        csv_s = sdf[disp].to_csv(index=False).encode('utf-8')
        st.download_button(f"📥 Download {sel_state.title()} Data", data=csv_s, file_name=f"{sel_state.replace(' ','_')}_crime.csv", mime="text/csv")

# ═══════════ MODULE 4 — NATIONAL TREND ═══════════
elif "National Trend" in menu:
    st.markdown("<h2>🔍 NATIONAL CRIME TREND ANALYSIS</h2>", unsafe_allow_html=True)
    crime_type = st.selectbox("🎯 Select Crime Category", ["TOTAL IPC CRIMES"] + PLOT_CRIMES, key="nat_cr")
    nat = df_state.groupby("YEAR")[PLOT_CRIMES + ["TOTAL IPC CRIMES"]].sum().reset_index()
    series = nat[crime_type]
    total_cases = int(series.sum()); max_yr = int(nat.loc[series.idxmax(),"YEAR"])
    min_yr = int(nat.loc[series.idxmin(),"YEAR"]); growth_rate = round((series.iloc[-1]-series.iloc[0])/series.iloc[0]*100,2)
    st.markdown("### 🎯 KEY PERFORMANCE INDICATORS")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("📊 12-Year Total", f"{total_cases:,}")
    with c2: st.metric("📅 Peak Year", max_yr, f"{int(series.max()):,} cases")
    with c3: st.metric("📉 Lowest Year", min_yr, f"{int(series.min()):,} cases")
    with c4: st.metric("📈 Overall Growth", f"{growth_rate:+.2f}%", "2001→2012")
    st.markdown("---")
    t1,t2,t3,t4 = st.tabs(["📈 Trend","📊 Compare","📉 Stats","🗂️ Data"])
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nat["YEAR"], y=nat[crime_type], mode='lines+markers',
            line=dict(color='#ff0000',width=5), marker=dict(size=14,color='#dc143c',line=dict(color='white',width=3)),
            fill='tozeroy', fillcolor='rgba(220,20,60,0.1)', hovertemplate='<b>Year: %{x}</b><br>Cases: %{y:,}<extra></extra>'))
        fig.update_layout(title=dict(text=f"<b>{crime_type.title()} — National Trend</b>",font=dict(size=22,color='#ffffff',family='Arial Black')),
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff',size=14,family='Arial Black'),
            xaxis=dict(title="Year",color='#ffffff',gridcolor='rgba(220,20,60,0.3)',dtick=1),
            yaxis=dict(title="Cases",color='#ffffff',gridcolor='rgba(220,20,60,0.3)'), height=480)
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        fig2 = go.Figure()
        for i, ct in enumerate(PLOT_CRIMES):
            fig2.add_trace(go.Scatter(x=nat["YEAR"], y=nat[ct], mode='lines+markers',
                name=ct.title(), line=dict(width=3,color=COLORS[i]), marker=dict(size=8)))
        fig2.update_layout(title=dict(text="<b>All Crime Categories — National (2001–2012)</b>",font=dict(size=20,color='#ffffff',family='Arial Black')),
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff',size=13,family='Arial Black'),
            xaxis=dict(color='#ffffff',dtick=1), yaxis=dict(color='#ffffff',gridcolor='rgba(220,20,60,0.2)'),
            legend=dict(bgcolor='rgba(0,0,0,0.9)',font=dict(color='#ffffff',size=11),bordercolor='#dc143c',borderwidth=2),
            height=480, hovermode='x unified')
        st.plotly_chart(fig2, use_container_width=True)
    with t3:
        cs1, cs2 = st.columns(2)
        with cs1:
            st.markdown("<h4>📊 STATISTICS</h4>", unsafe_allow_html=True)
            stats = pd.DataFrame({"Metric":["Mean","Median","Std Dev","Min","Max","Range","Growth Rate"],
                "Value":[f"{series.mean():.2f}",f"{series.median():.2f}",f"{series.std():.2f}",f"{series.min():,}",f"{series.max():,}",f"{series.max()-series.min():,}",f"{growth_rate:+.2f}%"]})
            st.dataframe(stats, use_container_width=True, hide_index=True)
        with cs2:
            yoy = series.pct_change() * 100
            yoy_df = pd.DataFrame({"YEAR":nat["YEAR"].iloc[1:],"YoY %":yoy.iloc[1:].round(2)})
            bar_c = ['#ff4444' if v > 0 else '#44ff88' for v in yoy_df["YoY %"]]
            fig_yoy = go.Figure(go.Bar(x=yoy_df["YEAR"], y=yoy_df["YoY %"], marker=dict(color=bar_c),
                text=[f"{v:+.1f}%" for v in yoy_df["YoY %"]], textposition='outside', textfont=dict(color='#ffffff',size=10)))
            fig_yoy.add_hline(y=0, line_color='white', line_dash='dash')
            fig_yoy.update_layout(plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff',family='Arial Black'),
                xaxis=dict(color='#ffffff',dtick=1), yaxis=dict(color='#ffffff',title="% Change"), height=340)
            st.plotly_chart(fig_yoy, use_container_width=True)
    with t4:
        disp_nat = nat[["YEAR"]+PLOT_CRIMES+["TOTAL IPC CRIMES"]]
        styled_n = disp_nat.style.background_gradient(cmap='Reds', subset=PLOT_CRIMES+["TOTAL IPC CRIMES"]).set_properties(**{'color':'white','font-weight':'bold'})
        st.dataframe(styled_n, use_container_width=True, height=420)
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.download_button("📥 Download National Trend Data", data=disp_nat.to_csv(index=False).encode('utf-8'), file_name="national_crime_trend.csv", mime="text/csv")

# ═══════════ MODULE 5 — AI PREDICTION ═══════════
elif "AI Prediction" in menu:
    st.markdown("<h2>🤖 AI-POWERED CRIME PREDICTION ENGINE</h2>", unsafe_allow_html=True)
    st.info("🧠 Models trained on real NCRB Kaggle data (2001–2012) using Random Forest Regressor")
    tab_p1, tab_p2 = st.tabs(["📍 State-wise Prediction","📊 All States Forecast"])

    with tab_p1:
        st.markdown("<h3>📍 Predict Crimes for a Specific State</h3>", unsafe_allow_html=True)
        pred_state = st.selectbox("🏛️ Select State/UT", STATE_LIST, key="pred_st")
        enc_val  = label_enc.transform([pred_state])[0]
        X_input  = pd.DataFrame({'State_Encoded': [enc_val]})
        crime_pred = crime_model.predict(X_input)[0]
        risk_score = risk_model.predict(X_input)[0]
        if risk_score > 60:   risk_label, risk_color = "🔴 HIGH RISK",   "#ff0000"
        elif risk_score > 25: risk_label, risk_color = "🟡 MEDIUM RISK", "#ffcc00"
        else:                 risk_label, risk_color = "🟢 LOW RISK",    "#00ff88"
        hist_data = df_state[df_state["STATE/UT"] == pred_state]["TOTAL IPC CRIMES"]
        hist_avg  = hist_data.mean(); diff_avg = crime_pred - hist_avg; pct_diff = (diff_avg/hist_avg)*100
        st.success("✅ PREDICTION ANALYSIS COMPLETE")
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#dc143c,#8b0000);padding:36px;border-radius:20px;
                    text-align:center;border:4px solid #ff0000;box-shadow:0 15px 40px rgba(220,20,60,0.8);
                    max-width:700px;margin:0 auto;'>
            <h3 style='color:#ffffff;font-size:1.1rem;margin-bottom:12px;letter-spacing:2px;'>PREDICTED TOTAL IPC CRIMES</h3>
            <p style='color:rgba(255,255,255,0.85);font-size:0.95rem;margin-bottom:8px;'>📍 {pred_state.title()}</p>
            <h1 style='color:#ffffff;font-size:3.8rem;margin:14px 0;text-shadow:4px 4px 10px rgba(0,0,0,0.9);'>{int(crime_pred):,}</h1>
            <p style='color:#ffff00;font-size:1rem;font-weight:800;background:rgba(0,0,0,0.3);
                      display:inline-block;padding:8px 24px;border-radius:20px;'>
                RISK SCORE: {risk_score:.2f} &nbsp;|&nbsp; {risk_label}</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.markdown("---")
        pv1,pv2,pv3 = st.tabs(["📈 Historical + Forecast","📊 Analysis","🎯 Model Info"])
        with pv1:
            hist_sdf = df_state[df_state["STATE/UT"]==pred_state].sort_values("YEAR")
            pred_yr  = YEARS[-1] + 1
            fig_pf = go.Figure()
            fig_pf.add_trace(go.Scatter(x=hist_sdf["YEAR"], y=hist_sdf["TOTAL IPC CRIMES"], mode='lines+markers',
                name='Historical (2001–2012)', line=dict(color='#00ccff',width=5), marker=dict(size=12,color='#0099ff')))
            fig_pf.add_trace(go.Scatter(x=[pred_yr], y=[crime_pred], mode='markers', name='AI Prediction',
                marker=dict(size=36,color='#ffff00',symbol='star',line=dict(color='#ff0000',width=4))))
            fig_pf.add_trace(go.Scatter(x=[YEARS[-1],pred_yr], y=[hist_sdf["TOTAL IPC CRIMES"].iloc[-1],crime_pred],
                mode='lines', name='Forecast', line=dict(color='#ff0000',width=4,dash='dash')))
            lower=int(crime_pred*0.9); upper=int(crime_pred*1.1)
            fig_pf.add_trace(go.Scatter(x=[pred_yr]*3, y=[lower,crime_pred,upper],
                fill='toself', fillcolor='rgba(255,255,0,0.25)', line=dict(color='#ffff00',width=2), name='Confidence ±10%'))
            fig_pf.update_layout(title=dict(text=f"<b>{pred_state.title()} — Historical + AI Forecast</b>",font=dict(size=18,color='#ffffff',family='Arial Black')),
                plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff',size=13,family='Arial Black'),
                xaxis=dict(title="Year",color='#ffffff',gridcolor='rgba(220,20,60,0.3)'),
                yaxis=dict(title="Total IPC Crimes",color='#ffffff',gridcolor='rgba(220,20,60,0.3)'),
                legend=dict(bgcolor='rgba(0,0,0,0.8)',font=dict(color='#ffffff',size=12),bordercolor='#dc143c',borderwidth=2), height=500)
            st.plotly_chart(fig_pf, use_container_width=True)
        with pv2:
            pa1, pa2 = st.columns(2)
            with pa1:
                st.markdown("<h4>📊 METRICS</h4>", unsafe_allow_html=True)
                cmp_df = pd.DataFrame({"Metric":["Historical Avg (2001–2012)","AI Predicted","Difference","% Change","Historical Max","Historical Min"],
                    "Value":[f"{hist_avg:,.0f}",f"{int(crime_pred):,}",f"{diff_avg:+,.0f}",f"{pct_diff:+.2f}%",f"{int(hist_data.max()):,}",f"{int(hist_data.min()):,}"]})
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)
            with pa2:
                st.markdown("<h4>🚨 RISK ASSESSMENT</h4>", unsafe_allow_html=True)
                if risk_score > 60:   st.error(f"🔴 HIGH RISK — Risk Score: {risk_score:.2f}/100")
                elif risk_score > 25: st.warning(f"🟡 MEDIUM RISK — Risk Score: {risk_score:.2f}/100")
                else:                 st.success(f"🟢 LOW RISK — Risk Score: {risk_score:.2f}/100")
                fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=risk_score,
                    title={'text':"Risk Score",'font':{'color':'#ffffff','size':16}},
                    number={'font':{'color':'#ffffff','size':28}},
                    gauge={'axis':{'range':[0,100],'tickcolor':'#ffffff','tickfont':{'color':'#ffffff'}},
                        'bar':{'color':risk_color},
                        'steps':[{'range':[0,25],'color':'rgba(0,150,50,0.3)'},{'range':[25,60],'color':'rgba(200,150,0,0.3)'},{'range':[60,100],'color':'rgba(180,0,0,0.3)'}],
                        'threshold':{'line':{'color':'white','width':3},'thickness':0.8,'value':risk_score}}))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=240,
                    font=dict(color='#ffffff',family='Arial Black'), margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
        with pv3:
            mi1, mi2 = st.columns(2)
            with mi1:
                st.markdown("<h4>📊 Feature Importance</h4>", unsafe_allow_html=True)
                fig_imp = go.Figure(go.Bar(x=["State_Encoded"], y=crime_model.feature_importances_,
                    marker=dict(color='#ff0000'), text=[f"{v:.2%}" for v in crime_model.feature_importances_],
                    textposition='outside', textfont=dict(color='#ffffff',size=14)))
                fig_imp.update_layout(plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#ffffff',family='Arial Black'), yaxis=dict(color='#ffffff'), height=300)
                st.plotly_chart(fig_imp, use_container_width=True)
            with mi2:
                st.markdown("<h4>🧠 Model Configuration</h4>", unsafe_allow_html=True)
                st.markdown(f"""
                - **Algorithm:** Random Forest Regressor
                - **Feature:** State_Encoded (Label Encoded)
                - **Training Data:** 2001–2012 NCRB Kaggle
                - **States Encoded:** {len(label_enc.classes_)}
                - **Predicted State:** {pred_state.title()}
                - **Encoded Value:** {enc_val}
                - **Predicted Crimes:** {int(crime_pred):,}
                - **Risk Score:** {risk_score:.2f} / 100
                - **Risk Level:** {risk_label}
                """)

    with tab_p2:
        st.markdown("<h3>📊 All States — AI Predictions & Risk Scores</h3>", unsafe_allow_html=True)
        sort_by = st.selectbox("Sort by", ["Predicted Crimes ↓","Risk Score ↓","State Name ↑"], key="sort_pred")
        df_p = df_pred.copy(); df_p["STATE_TITLE"] = df_p["STATE/UT"].str.title()
        if sort_by == "Predicted Crimes ↓": df_p = df_p.sort_values("Predicted Crimes", ascending=False)
        elif sort_by == "Risk Score ↓":     df_p = df_p.sort_values("Predicted Risk Score", ascending=False)
        else:                               df_p = df_p.sort_values("STATE/UT")
        fig_all = go.Figure(go.Bar(x=df_p["STATE_TITLE"], y=df_p["Predicted Crimes"],
            marker=dict(color=df_p["Predicted Risk Score"],
                colorscale=[[0,"#00cc44"],[0.4,"#ffcc00"],[1,"#ff0000"]],
                colorbar=dict(title=dict(text="Risk Score",font=dict(color="#ffffff")), tickfont=dict(color="#ffffff"),
                    bgcolor="rgba(10,10,30,0.8)", bordercolor="#dc143c"),
                line=dict(color='#ffffff',width=0.5)),
            text=[f"{int(v):,}" for v in df_p["Predicted Crimes"]], textposition='outside',
            textfont=dict(color='#ffffff',size=9,family='Arial Black'),
            hovertemplate='<b>%{x}</b><br>Predicted: %{y:,}<extra></extra>'))
        fig_all.update_layout(title=dict(text="<b>All States — Predicted Total IPC Crimes</b>",font=dict(size=18,color='#ffffff',family='Arial Black')),
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff',size=10,family='Arial Black'),
            xaxis=dict(color='#ffffff',tickangle=-45), yaxis=dict(color='#ffffff',gridcolor='rgba(220,20,60,0.2)',title="Predicted Crimes"), height=520)
        st.plotly_chart(fig_all, use_container_width=True)
        fig_risk = go.Figure(go.Scatter(x=df_p["Predicted Crimes"], y=df_p["Predicted Risk Score"],
            mode='markers+text',
            marker=dict(size=14,color=df_p["Predicted Risk Score"],colorscale=[[0,"#00cc44"],[0.4,"#ffcc00"],[1,"#ff0000"]],line=dict(color='#ffffff',width=1),opacity=0.9),
            text=df_p["STATE_TITLE"], textposition='top center', textfont=dict(color='#ffffff',size=8),
            hovertemplate='<b>%{text}</b><br>Predicted: %{x:,}<br>Risk: %{y:.2f}<extra></extra>'))
        fig_risk.add_hline(y=60, line_color='#ff4444', line_dash='dash', line_width=2, annotation_text="HIGH RISK threshold", annotation_font_color="#ff4444")
        fig_risk.add_hline(y=25, line_color='#ffcc00', line_dash='dash', line_width=2, annotation_text="MEDIUM RISK threshold", annotation_font_color="#ffcc00")
        fig_risk.update_layout(title=dict(text="<b>Crime Volume vs Risk Score — All States</b>",font=dict(size=18,color='#ffffff',family='Arial Black')),
            plot_bgcolor='rgba(10,10,30,0.7)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff',family='Arial Black'),
            xaxis=dict(color='#ffffff',title="Predicted Crimes",gridcolor='rgba(220,20,60,0.2)'),
            yaxis=dict(color='#ffffff',title="Risk Score",gridcolor='rgba(220,20,60,0.2)'), height=520)
        st.plotly_chart(fig_risk, use_container_width=True)
        st.markdown("<h3>📋 Complete Predictions Table</h3>", unsafe_allow_html=True)
        disp_pred = df_p[["STATE_TITLE","Predicted Crimes","Predicted Risk Score","Risk Level"]].copy()
        disp_pred.columns = ["State/UT","Predicted Crimes","Risk Score","Risk Level"]
        disp_pred = disp_pred.reset_index(drop=True); disp_pred.index = disp_pred.index + 1
        styled_pred = disp_pred.style.background_gradient(cmap='RdYlGn_r', subset=["Predicted Crimes","Risk Score"]).set_properties(**{'color':'white','font-weight':'bold'})
        st.dataframe(styled_pred, use_container_width=True, height=500)
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.download_button("📥 Download All Predictions", data=disp_pred.to_csv(index=False).encode(), file_name="state_predictions_all.csv", mime="text/csv")

st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:28px 36px;background:linear-gradient(135deg,rgba(220,20,60,0.4),rgba(139,0,0,0.4));border-radius:20px;border:3px solid #dc143c;'>
    <h3 style='color:#ff0000;margin-bottom:12px;font-size:1.5rem;'>🚨 CRIME INTELLIGENCE PRO 🚨</h3>
    <p style='color:#ffffff;font-size:1rem;font-weight:800;margin-bottom:6px;'>Professional Crime Analytics & Prediction Platform — India</p>
    <p style='color:#cccccc;font-size:0.9rem;'>AI/ML · Random Forest · Real NCRB Kaggle Data · Hackathon Edition 🏆</p>
    <p style='color:#aaaaaa;font-size:0.8rem;margin-top:8px;'>34 States/UTs · 10 Crime Categories · 2001–2012 · 3 Trained ML Models</p>
</div>""", unsafe_allow_html=True)