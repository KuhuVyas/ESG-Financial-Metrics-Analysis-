import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ESG Analytics Dashboard", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed_esg_financial_data.csv")

df = load_data()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("🔎 Filters")

# Identify categorical columns safely
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Try common ESG dataset filters
region_col = [col for col in categorical_cols if "region" in col.lower()]
industry_col = [col for col in categorical_cols if "industry" in col.lower()]
year_col = [col for col in df.columns if "year" in col.lower()]

filtered_df = df.copy()

if region_col:
    region = st.sidebar.multiselect("Region", df[region_col[0]].unique())
    if region:
        filtered_df = filtered_df[filtered_df[region_col[0]].isin(region)]

if industry_col:
    industry = st.sidebar.multiselect("Industry", df[industry_col[0]].unique())
    if industry:
        filtered_df = filtered_df[filtered_df[industry_col[0]].isin(industry)]

if year_col:
    year = st.sidebar.multiselect("Year", df[year_col[0]].unique())
    if year:
        filtered_df = filtered_df[filtered_df[year_col[0]].isin(year)]

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 ESG Financial Analytics Dashboard")

# -----------------------------
# KPI SECTION
# -----------------------------
st.subheader("📌 Key Metrics")

numeric_df = filtered_df.select_dtypes(include=np.number)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Companies", len(filtered_df))

with col2:
    st.metric("Avg ESG Score", round(numeric_df.mean().mean(), 2))

with col3:
    st.metric("Avg Revenue", round(numeric_df.mean().max(), 2))

with col4:
    st.metric("Avg Profit", round(numeric_df.mean().min(), 2))

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("📄 Data Snapshot")
st.dataframe(filtered_df.head())

# -----------------------------
# VISUALIZATION SECTION
# -----------------------------
st.subheader("📈 Insights & Trends")

# Layout
col1, col2 = st.columns(2)

# ---- Correlation Heatmap
with col1:
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---- Distribution Plot
with col2:
    st.write("### Distribution")
    selected_col = st.selectbox("Select Metric", numeric_df.columns)
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_df[selected_col], kde=True, ax=ax2)
    ax2.set_title(selected_col)
    st.pyplot(fig2)

# -----------------------------
# COMPARISON SECTION
# -----------------------------
st.subheader("🔍 Metric Comparison")

col1, col2 = st.columns(2)

x_axis = col1.selectbox("X-axis", numeric_df.columns)
y_axis = col2.selectbox("Y-axis", numeric_df.columns)

fig3, ax3 = plt.subplots()
sns.scatterplot(x=filtered_df[x_axis], y=filtered_df[y_axis], ax=ax3)
ax3.set_xlabel(x_axis)
ax3.set_ylabel(y_axis)

st.pyplot(fig3)

# -----------------------------
# TOP PERFORMERS
# -----------------------------
st.subheader("🏆 Top Performers")

sort_col = st.selectbox("Sort by", numeric_df.columns)

top_df = filtered_df.sort_values(by=sort_col, ascending=False).head(10)

st.dataframe(top_df)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built for ESG Financial Metrics Analysis | Streamlit Dashboard")