import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -----------------------------
# PAGE CONFIG + DARK THEME
# -----------------------------
st.set_page_config(page_title="ESG Intelligence Dashboard", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: white;
    }
    .stMetric {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed_esg_financial_data.csv")

df = load_data()

# -----------------------------
# ESG SEGMENTATION
# -----------------------------
def esg_category(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

# auto-detect ESG column
esg_col = [col for col in df.columns if "esg" in col.lower()]
if esg_col:
    df["ESG_Category"] = df[esg_col[0]].apply(esg_category)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("🔎 Filters")

categorical_cols = df.select_dtypes(include='object').columns.tolist()

region_col = [c for c in categorical_cols if "region" in c.lower()]
industry_col = [c for c in categorical_cols if "industry" in c.lower()]
year_col = [c for c in df.columns if "year" in c.lower()]

filtered_df = df.copy()

if region_col:
    selected = st.sidebar.multiselect("Region", df[region_col[0]].unique())
    if selected:
        filtered_df = filtered_df[filtered_df[region_col[0]].isin(selected)]

if industry_col:
    selected = st.sidebar.multiselect("Industry", df[industry_col[0]].unique())
    if selected:
        filtered_df = filtered_df[filtered_df[industry_col[0]].isin(selected)]

if year_col:
    selected = st.sidebar.multiselect("Year", sorted(df[year_col[0]].unique()))
    if selected:
        filtered_df = filtered_df[filtered_df[year_col[0]].isin(selected)]

# -----------------------------
# TITLE
# -----------------------------
st.title("🌍 ESG Intelligence Dashboard")

# -----------------------------
# KPI SECTION
# -----------------------------
st.subheader("📊 Key Performance Indicators")

numeric_df = filtered_df.select_dtypes(include=np.number)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Companies", len(filtered_df))

with col2:
    if esg_col:
        st.metric("Avg ESG Score", round(filtered_df[esg_col[0]].mean(), 2))

with col3:
    revenue_col = [c for c in numeric_df.columns if "revenue" in c.lower()]
    if revenue_col:
        st.metric("Avg Revenue", round(filtered_df[revenue_col[0]].mean(), 2))

with col4:
    profit_col = [c for c in numeric_df.columns if "profit" in c.lower()]
    if profit_col:
        st.metric("Avg Profit", round(filtered_df[profit_col[0]].mean(), 2))

# -----------------------------
# ESG CATEGORY DISTRIBUTION
# -----------------------------
if "ESG_Category" in filtered_df.columns:
    st.subheader("📊 ESG Score Segmentation")

    fig, ax = plt.subplots()
    filtered_df["ESG_Category"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("ESG Category Distribution")
    st.pyplot(fig)

# -----------------------------
# TIME SERIES ANALYSIS
# -----------------------------
if year_col:
    st.subheader("📈 Time Series Trends")

    metric = st.selectbox("Select Metric", numeric_df.columns)

    trend = filtered_df.groupby(year_col[0])[metric].mean()

    fig, ax = plt.subplots()
    trend.plot(marker='o', ax=ax)
    ax.set_title(f"{metric} over Time")
    st.pyplot(fig)

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
st.subheader("🔗 Correlation Analysis")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -----------------------------
# SCATTER INSIGHT
# -----------------------------
st.subheader("🔍 Feature Relationship")

x = st.selectbox("X-axis", numeric_df.columns)
y = st.selectbox("Y-axis", numeric_df.columns)

fig, ax = plt.subplots()
sns.scatterplot(x=filtered_df[x], y=filtered_df[y], ax=ax)
st.pyplot(fig)

# -----------------------------
# TOP PERFORMERS
# -----------------------------
st.subheader("🏆 Top Companies")

sort_col = st.selectbox("Sort By", numeric_df.columns)
top_df = filtered_df.sort_values(by=sort_col, ascending=False).head(10)
st.dataframe(top_df)

# -----------------------------
# ML PREDICTION
# -----------------------------
st.subheader("🤖 ESG Prediction Engine")

try:
    model = joblib.load("model.pkl")
    st.success("Model Loaded Successfully")

    input_data = {}

    for col in numeric_df.columns:
        input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()))

    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        pred = model.predict(input_df)
        st.success(f"Prediction: {pred[0]}")

except:
    st.warning("Model not found. Add model.pkl to enable predictions")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built with Streamlit | ESG Intelligence System")