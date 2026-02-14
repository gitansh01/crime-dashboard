import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Crime Intelligence Dashboard",
    layout="wide"
)

# ---------------------- DARK PROFESSIONAL CSS ----------------------
st.markdown("""
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Titles */
h1, h2, h3, h4 {
    color: #00F5FF;
}

/* Buttons */
.stButton>button {
    background-color: #00F5FF;
    color: black;
    border-radius: 10px;
    font-weight: bold;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #1f2937;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- TITLE SECTION ----------------------
st.title("🚨 Crime Intelligence & Prediction Dashboard")
st.markdown("### Advanced Crime Analytics for Strategic Decision Making")

# ---------------------- SAMPLE DATA ----------------------
years = np.array([2015,2016,2017,2018,2019,2020,2021,2022,2023])
theft = np.array([400,450,480,500,550,600,650,700,720])
assault = np.array([300,320,350,370,400,420,450,470,500])
burglary = np.array([200,210,230,240,260,280,300,320,340])

df = pd.DataFrame({
    "Year": years,
    "Theft": theft,
    "Assault": assault,
    "Burglary": burglary
})

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("📊 Navigation")
menu = st.sidebar.radio("Select Module",
                        ["Crime Analysis", "Crime Prediction"])

# =====================================================
# =================== CRIME ANALYSIS ==================
# =====================================================
if menu == "Crime Analysis":

    st.subheader("📈 Crime Trend Analysis")

    col1, col2 = st.columns(2)

    with col1:
        crime_type = st.selectbox(
            "Select Crime Type",
            ["Theft", "Assault", "Burglary"]
        )

    with col2:
        st.metric(
            label=f"Total {crime_type} Cases (2015-2023)",
            value=int(df[crime_type].sum())
        )

    st.markdown("### Crime Data Table")
    st.dataframe(df, use_container_width=True)

    # Line Chart
    st.markdown("### Trend Visualization")

    fig, ax = plt.subplots()
    ax.plot(df["Year"], df[crime_type], marker='o', linewidth=3)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cases")
    ax.set_title(f"{crime_type} Crime Trend")
    ax.set_facecolor("#1f2937")
    fig.patch.set_facecolor("#1f2937")

    st.pyplot(fig)

    st.success("Crime Analysis Module Loaded Successfully")

# =====================================================
# =================== CRIME PREDICTION ================
# =====================================================
elif menu == "Crime Prediction":

    st.subheader("🤖 Crime Prediction using Machine Learning")

    crime_type = st.selectbox(
        "Select Crime Type for Prediction",
        ["Theft", "Assault", "Burglary"]
    )

    # Prepare Data
    X = df[["Year"]]
    y = df[crime_type]

    model = LinearRegression()
    model.fit(X, y)

    future_year = st.number_input(
        "Enter Future Year for Prediction",
        min_value=2024,
        max_value=2035,
        value=2025
    )

    if st.button("Predict Crime"):
        prediction = model.predict([[future_year]])
        predicted_value = int(prediction[0])

        st.markdown("## 🔮 Prediction Result")
        st.metric(
            label=f"Predicted {crime_type} Cases in {future_year}",
            value=predicted_value
        )

        # Plot Prediction
        fig2, ax2 = plt.subplots()
        ax2.plot(df["Year"], df[crime_type], marker='o', label="Historical")
        ax2.scatter(future_year, predicted_value, s=200)
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Cases")
        ax2.set_title(f"{crime_type} Prediction for {future_year}")
        ax2.set_facecolor("#1f2937")
        fig2.patch.set_facecolor("#1f2937")

        st.pyplot(fig2)

        st.success("Prediction Completed Successfully")

# ---------------------- FOOTER ----------------------
st.markdown("""
<hr style="border:1px solid #00F5FF;">
<center>Designed for Crime Data Intelligence | ML Powered Analytics</center>
""", unsafe_allow_html=True)
