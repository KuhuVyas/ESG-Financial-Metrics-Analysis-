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
# STYLING
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
# DATA CLEANING
# -----------------------------
NUMERIC_COLS = [
    'revenue', 'profit_margin', 'market_cap', 'growth_rate',
    'esg_overall', 'esg_environmental', 'esg_social', 'esg_governance'
]

df = df.dropna(subset=NUMERIC_COLS)

# -----------------------------
# ESG CATEGORY
# -----------------------------
df['ESG_Category'] = pd.qcut(df['esg_overall'], 3, labels=['Low', 'Medium', 'High'])

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("Filters")

st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.metric("Companies", df['company_id'].nunique())
st.sidebar.metric("Years", df['year'].nunique())

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

# -----------------------------
# TITLE
# -----------------------------
st.markdown("""
<h1 style='text-align: center;'>📊 ESG Financial Intelligence Dashboard</h1>
<p style='text-align: center; color: gray;'>Sustainability-driven Financial Insights & Growth Prediction</p>
""", unsafe_allow_html=True)

# -----------------------------
# DATASET OVERVIEW
# -----------------------------
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(filtered_df))
col2.metric("Unique Companies", filtered_df["company_id"].nunique())
col3.metric("Industries Covered", filtered_df["industry"].nunique())

# -----------------------------
# KPIs
# -----------------------------
st.subheader("📌 Key Performance Indicators")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Avg ESG", f"{filtered_df['esg_overall'].mean():.2f}")
k2.metric("Avg Growth", f"{filtered_df['growth_rate'].mean():.2f}")
k3.metric("Avg Profit", f"{filtered_df['profit_margin'].mean():.2f}")
k4.metric("Top ESG", f"{filtered_df['esg_overall'].max():.2f}")
k5.metric("Top Growth", f"{filtered_df['growth_rate'].max():.2f}")

# -----------------------------
# TREND
# -----------------------------
st.subheader("📈 Growth & ESG Trends")

trend = filtered_df.groupby('year')[['growth_rate','esg_overall']].mean().reset_index()

fig, ax = plt.subplots(figsize=(6,3))
ax.plot(trend['year'], trend['growth_rate'], label='Growth')
ax.plot(trend['year'], trend['esg_overall'], label='ESG')
ax.legend()
st.pyplot(fig)

# -----------------------------
# ESG vs PROFIT
# -----------------------------
st.subheader("ESG Impact on Profitability")

sample_df = filtered_df.sample(min(800, len(filtered_df)), random_state=42)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.regplot(data=sample_df, x='esg_overall', y='profit_margin', ax=ax)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.boxplot(data=sample_df, x='ESG_Category', y='profit_margin', ax=ax)
    st.pyplot(fig)

# -----------------------------
# CORRELATION
# -----------------------------
st.subheader("📊 Correlation Insights")

fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(
    filtered_df[NUMERIC_COLS].corr(),
    annot=True, fmt=".2f", cmap="coolwarm", ax=ax
)
st.pyplot(fig)

# -----------------------------
# INSIGHT
# -----------------------------
corr = filtered_df[['esg_overall','growth_rate']].corr().iloc[0,1]

if corr > 0.2:
    insight = "Strong positive ESG-growth relationship"
elif corr > 0:
    insight = "Moderate ESG-growth relationship"
else:
    insight = "Weak ESG-growth relationship"

st.info(f"📌 {insight} (Correlation = {corr:.2f})")

# -----------------------------
# TOP PERFORMERS
# -----------------------------
st.subheader("Top Performers")

top_df = filtered_df.sort_values(by="profit_margin", ascending=False).head(10)

st.dataframe(
    top_df[['company_name','industry','profit_margin','growth_rate','esg_overall']],
    use_container_width=True
)

# -----------------------------
# ML MODEL
# -----------------------------
st.subheader("🤖 Growth Prediction Engine")

MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    scaler = saved["scaler"]
    features = saved["features"]

    st.success("Model Loaded Successfully")

    input_data = {}

    for col in features:
        min_val = float(df[col].min())
        max_val = float(df[col].max())

        input_data[col] = st.slider(
            f"{col}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val + max_val) / 2
        )

    input_df = pd.DataFrame([input_data])

    if st.button("Predict Growth"):
        input_scaled = scaler.transform(input_df[features])
        pred = model.predict(input_scaled)

        st.success(f"Predicted Growth Rate: {pred[0]:.3f}")

        if pred[0] > 0.6:
            st.info("🚀 High Growth Potential")
        elif pred[0] > 0.4:
            st.info("📈 Moderate Growth")
        else:
            st.warning("⚠️ Low Growth")

else:
    st.error("Model file not found")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("ESG Analytics Platform | Data-driven Sustainability Insights")