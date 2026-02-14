import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Crime Dashboard", layout="wide")

# Title
st.title("🚔 Crime Dashboard")
st.write("A simple interactive crime data dashboard")

# Sample Data (Replace with your CSV later)
data = {
    "Year": [2019, 2020, 2021, 2022, 2023],
    "Theft": [500, 650, 700, 800, 750],
    "Assault": [300, 350, 400, 420, 390],
    "Burglary": [200, 180, 210, 230, 220]
}

df = pd.DataFrame(data)

# Sidebar
st.sidebar.header("Filter Data")
crime_type = st.sidebar.selectbox(
    "Select Crime Type",
    ["Theft", "Assault", "Burglary"]
)

# Display Data
st.subheader("Crime Data Table")
st.dataframe(df)

# Plot Graph
st.subheader(f"{crime_type} Over Years")

fig, ax = plt.subplots()
ax.plot(df["Year"], df[crime_type], marker='o')
ax.set_xlabel("Year")
ax.set_ylabel("Number of Cases")
ax.set_title(f"{crime_type} Trend")

st.pyplot(fig)

st.success("Dashboard Loaded Successfully ✅")
