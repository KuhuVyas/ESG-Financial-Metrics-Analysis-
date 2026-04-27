import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="ESG Intelligence Dashboard", layout="wide")

# -----------------------------
# CLEAN DARK UI
# -----------------------------
st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.stMetric {background-color: #1c1f26; padding: 15px; border-radius: 10px;}
h1, h2, h3 {color: #EAEAEA;}
</style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")
sns.set_theme(style="darkgrid")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed_esg_financial_data.csv")

df = load_data()

# -----------------------------
# STRICT COLUMN MAPPING
# -----------------------------
NUMERIC_COLS = [
    'revenue', 'profit_margin', 'market_cap', 'growth_rate',
    'esg_overall', 'esg_environmental', 'esg_social', 'esg_governance',
    'carbon_emissions', 'water_usage', 'energy_consumption'
]

df = df.dropna(subset=NUMERIC_COLS)

# -----------------------------
# ESG CATEGORY (FIXED)
# -----------------------------
def esg_category(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

df["ESG_Category"] = df["esg_overall"].apply(esg_category)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("Filters")

filtered_df = df.copy()

region = st.sidebar.multiselect("Region", df["region"].unique())
industry = st.sidebar.multiselect("Industry", df["industry"].unique())
year = st.sidebar.multiselect("Year", sorted(df["year"].unique()))

if region:
    filtered_df = filtered_df[filtered_df["region"].isin(region)]

if industry:
    filtered_df = filtered_df[filtered_df["industry"].isin(industry)]

if year:
    filtered_df = filtered_df[filtered_df["year"].isin(year)]

numeric_df = filtered_df[NUMERIC_COLS]

# -----------------------------
# TITLE
# -----------------------------
st.title("ESG Financial Intelligence Dashboard")

# -----------------------------
# DATASET OVERVIEW
# -----------------------------
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(filtered_df))
col2.metric("Unique Companies", filtered_df["company_id"].nunique())
col3.metric("Industries Covered", filtered_df["industry"].nunique())

# -----------------------------
# KPI SECTION (CORRECTED)
# -----------------------------
st.subheader("Key Business KPIs")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Avg ESG Score", round(filtered_df["esg_overall"].mean(), 2))
k2.metric("Avg Profit Margin", round(filtered_df["profit_margin"].mean(), 2))
k3.metric("Avg Growth Rate", round(filtered_df["growth_rate"].mean(), 2))
k4.metric("Avg Market Cap", round(filtered_df["market_cap"].mean(), 2))

# -----------------------------
# ESG vs PROFITABILITY INSIGHT
# -----------------------------
st.subheader("ESG Impact on Profitability")

corr = filtered_df[['esg_overall', 'profit_margin']].corr().iloc[0,1]

fig, ax = plt.subplots(figsize=(5,3))
sns.regplot(
    data=filtered_df,
    x='esg_overall',
    y='profit_margin',
    scatter_kws={'alpha':0.5},
    ax=ax
)
ax.set_title("ESG vs Profit Margin")
st.pyplot(fig)

st.info(f"Correlation: {round(corr,2)} → {'Positive relationship' if corr>0 else 'Weak relationship'}")

# -----------------------------
# ESG COMPONENT IMPACT
# -----------------------------
st.subheader("ESG Pillar Impact")

impact = filtered_df[
    ['esg_environmental','esg_social','esg_governance','profit_margin']
].corr()['profit_margin'].drop('profit_margin')

fig, ax = plt.subplots(figsize=(5,3))
impact.sort_values().plot(kind='barh', ax=ax)
ax.set_title("Impact on Profitability")
st.pyplot(fig)

# -----------------------------
# MARKET VALUE RELATION
# -----------------------------
st.subheader("ESG vs Market Valuation")

fig, ax = plt.subplots(figsize=(5,3))
sns.regplot(
    data=filtered_df,
    x='esg_overall',
    y='market_cap',
    ax=ax
)
ax.set_yscale('log')
ax.set_title("ESG vs Market Cap (Log Scale)")
st.pyplot(fig)

# -----------------------------
# CORRELATION MATRIX (FIXED)
# -----------------------------
st.subheader("Correlation Matrix")

corr_df = numeric_df.select_dtypes(include=np.number)

fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(
    corr_df.corr(),
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    ax=ax
)
ax.set_title("Feature Correlation")
st.pyplot(fig)

# -----------------------------
# TOP PERFORMERS
# -----------------------------
st.subheader("Top Performers")

top_df = filtered_df.sort_values(by="profit_margin", ascending=False).head(10)
st.dataframe(top_df[['company_name','industry','profit_margin','esg_overall']], use_container_width=True)

# -----------------------------
# ML PREDICTION ENGINE (FIXED)
# -----------------------------
st.subheader("Growth Prediction Engine")

MODEL_PATH = "model.pkl"

try:
    if os.path.exists(MODEL_PATH):

        saved = joblib.load(MODEL_PATH)

        model = saved["model"]
        scaler = saved["scaler"]
        features = saved["features"]

        st.success("Model Loaded")

        input_data = {}

        for col in features:
            min_val = float(df[col].min())
            max_val = float(df[col].max())

            input_data[col] = st.number_input(
                col,
                min_value=min_val,
                max_value=max_val,
                value=min_val
            )

        input_df = pd.DataFrame([input_data])

        if st.button("Predict Growth"):
            input_scaled = scaler.transform(input_df[features])
            pred = model.predict(input_scaled)
            st.success(f"Predicted Growth Rate: {round(pred[0],3)}")

    else:
        st.error("Model file not found")

except Exception as e:
    st.error("Prediction engine failed")
    st.write(str(e))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("ESG Analytics Platform | Data-driven Sustainability Insights")