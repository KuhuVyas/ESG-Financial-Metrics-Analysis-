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
# DARK THEME + UI
# -----------------------------
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
h1, h2, h3 {
    color: #EAEAEA;
}
</style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")
sns.set_theme(style="darkgrid")

FIG_SIZE = (5, 3)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed_esg_financial_data.csv")

df = load_data()

# -----------------------------
# AUTO DETECT COLUMNS
# -----------------------------
numeric_df = df.select_dtypes(include=np.number)

esg_col = [c for c in df.columns if "esg" in c.lower()]
revenue_col = [c for c in df.columns if "revenue" in c.lower()]
profit_col = [c for c in df.columns if "profit" in c.lower()]
year_col = [c for c in df.columns if "year" in c.lower()]
region_col = [c for c in df.columns if "region" in c.lower()]
industry_col = [c for c in df.columns if "industry" in c.lower()]

# -----------------------------
# ESG CATEGORY
# -----------------------------
def esg_category(score):
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"

if esg_col:
    df["ESG_Category"] = df[esg_col[0]].apply(esg_category)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("🔎 Filters")

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

numeric_df = filtered_df.select_dtypes(include=np.number)

# -----------------------------
# TITLE
# -----------------------------
st.title("🌍 ESG Intelligence Dashboard")

# -----------------------------
# KPI SECTION (REAL BUSINESS KPIs)
# -----------------------------
st.subheader("📊 Key Insights")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Companies", len(filtered_df))

with col2:
    if esg_col:
        avg_esg = filtered_df[esg_col[0]].mean()
        st.metric("Avg ESG Score", round(avg_esg, 2))

with col3:
    if revenue_col:
        avg_rev = filtered_df[revenue_col[0]].mean()
        st.metric("Avg Revenue", f"{round(avg_rev,2)}")

with col4:
    if profit_col:
        avg_profit = filtered_df[profit_col[0]].mean()
        st.metric("Avg Profit", f"{round(avg_profit,2)}")

# -----------------------------
# BUSINESS INSIGHT SUMMARY
# -----------------------------
st.markdown("### 📌 Business Insights")

if esg_col and profit_col:
    corr = filtered_df[[esg_col[0], profit_col[0]]].corr().iloc[0, 1]

    st.info(f"""
- ESG Score vs Profit correlation: **{round(corr,2)}**
- {"High ESG companies tend to be more profitable" if corr > 0.4 else "Weak relationship between ESG and profitability"}
- Median ESG Score: **{round(filtered_df[esg_col[0]].median(),2)}**
""")

# -----------------------------
# ESG DISTRIBUTION
# -----------------------------
if "ESG_Category" in filtered_df.columns:
    st.subheader("📊 ESG Segmentation")

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    sns.countplot(
        data=filtered_df,
        x="ESG_Category",
        palette="viridis",
        ax=ax
    )

    ax.set_title("ESG Category Distribution", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# -----------------------------
# TIME SERIES
# -----------------------------
if year_col:
    st.subheader("📈 Trend Analysis")

    metric = st.selectbox("Select Metric", numeric_df.columns)

    trend = filtered_df.groupby(year_col[0])[metric].mean()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    trend.plot(marker='o', linewidth=2, ax=ax)

    ax.set_title(f"{metric} Trend", fontsize=12)
    ax.grid(alpha=0.3)

    st.pyplot(fig)

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
st.subheader("🔗 Correlation Overview")

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"shrink": 0.7},
    ax=ax
)

ax.set_title("Feature Correlation", fontsize=12)
st.pyplot(fig)

# -----------------------------
# SCATTER RELATIONSHIP
# -----------------------------
st.subheader("🔍 Relationship Analysis")

x = st.selectbox("X-axis", numeric_df.columns)
y = st.selectbox("Y-axis", numeric_df.columns)

sample_df = filtered_df.sample(min(1000, len(filtered_df)))

fig, ax = plt.subplots(figsize=FIG_SIZE)

sns.scatterplot(
    data=sample_df,
    x=x,
    y=y,
    alpha=0.5,
    s=30,
    ax=ax
)

sns.regplot(
    data=sample_df,
    x=x,
    y=y,
    scatter=False,
    ax=ax,
    line_kws={"color": "red"}
)

ax.set_title(f"{x} vs {y}", fontsize=12)
st.pyplot(fig)

# Insight
corr_xy = filtered_df[[x, y]].corr().iloc[0, 1]

st.info(f"""
Correlation: **{round(corr_xy,2)}**
→ {"Strong" if abs(corr_xy)>0.6 else "Moderate" if abs(corr_xy)>0.3 else "Weak"} relationship
""")

# -----------------------------
# TOP PERFORMERS
# -----------------------------
st.subheader("🏆 Top Performers")

sort_col = st.selectbox("Sort By", numeric_df.columns)
top_df = filtered_df.sort_values(by=sort_col, ascending=False).head(10)

st.dataframe(top_df, use_container_width=True)

# -----------------------------
# MODEL LOADING (FIXED)
# -----------------------------
st.subheader("🤖 ESG Prediction Engine")

MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    saved = joblib.load(MODEL_PATH)

if isinstance(saved, dict):
    model = saved["model"]
    features = saved["features"]
else:
    model = saved
    features = numeric_df.columns.tolist()

    st.success("✅ Model Loaded")

    input_data = {}

    for col in features:

        if col in df.columns:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
        else   :
        # fallback for missing columns
            min_val = 0.0
            max_val = 100.0

        input_data[col] = st.number_input(
        label=col,
        min_value=min_val,
        max_value=max_val,
        value=min_val
    )

    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        pred = model.predict(input_df)
        st.success(f"Predicted Value: {round(pred[0], 3)}")

else:
    st.error("❌ model.pkl not found. Please train and save the model.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("ESG Intelligence System | Production-grade Analytics Dashboard")