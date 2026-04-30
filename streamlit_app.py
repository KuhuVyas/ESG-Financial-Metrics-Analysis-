import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="ESG Intelligence Platform", layout="wide")

# -----------------------------
# DARK UI
# -----------------------------
st.markdown("""
<style>
body {background-color: #0b0f14; color: #EAEAEA;}
.stMetric {
    background: linear-gradient(145deg, #1a1f2b, #11151c);
    padding: 12px;
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

df = df.dropna()

df['ESG_Category'] = pd.qcut(df['esg_overall'], 3, labels=['Low', 'Medium', 'High'])

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Filters")

region = st.sidebar.multiselect("Region", df["region"].unique())
industry = st.sidebar.multiselect("Industry", df["industry"].unique())
year = st.sidebar.multiselect("Year", sorted(df["year"].unique()))

filtered_df = df.copy()

if region:
    filtered_df = filtered_df[filtered_df["region"].isin(region)]
if industry:
    filtered_df = filtered_df[filtered_df["industry"].isin(industry)]
if year:
    filtered_df = filtered_df[filtered_df["year"].isin(year)]

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align: center;'>📊 ESG Intelligence Platform</h1>
<p style='text-align: center; color: gray;'>Financial • Sustainability • Predictive Intelligence</p>
""", unsafe_allow_html=True)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 ML Engine", "📈 Insights"])

# =====================================================
# 📊 DASHBOARD
# =====================================================
with tab1:

    # KPIs
    st.subheader("Key Metrics")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Avg ESG", f"{filtered_df['esg_overall'].mean():.2f}")
    k2.metric("Avg Growth", f"{filtered_df['growth_rate'].mean():.2f}")
    k3.metric("Avg Profit", f"{filtered_df['profit_margin'].mean():.2f}")
    k4.metric("Companies", filtered_df["company_id"].nunique())

    # -----------------------------
    # PIE CHART (REGION)
    # -----------------------------
    st.subheader("🌍 Regional Distribution")

    region_df = filtered_df['region'].value_counts().reset_index()
    region_df.columns = ['Region', 'Count']

    fig = px.pie(region_df, names='Region', values='Count',
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # BAR CHART (INDUSTRY)
    # -----------------------------
    st.subheader("🏭 Industry Distribution")

    industry_df = filtered_df['industry'].value_counts().reset_index()
    industry_df.columns = ['Industry', 'Count']

    fig = px.bar(industry_df, x='Industry', y='Count',
                 color='Count', color_continuous_scale='Teal')
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # HISTOGRAM
    # -----------------------------
    st.subheader("📊 Growth Distribution")

    fig = px.histogram(filtered_df, x='growth_rate', nbins=40,
                       color_discrete_sequence=['#00ADB5'])
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # TREND LINE
    # -----------------------------
    st.subheader("📈 Trend Over Time")

    trend = filtered_df.groupby('year')[['growth_rate','esg_overall']].mean().reset_index()

    fig = px.line(trend, x='year', y=['growth_rate','esg_overall'],
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # SCATTER (KEY RELATION)
    # -----------------------------
    st.subheader("📉 ESG vs Profit")

    fig = px.scatter(filtered_df.sample(min(1000,len(filtered_df))),
                     x='esg_overall',
                     y='profit_margin',
                     color='ESG_Category',
                     opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 🤖 ML ENGINE
# =====================================================
with tab2:

    st.subheader("Growth Prediction")

    MODEL_PATH = "model.pkl"

    if os.path.exists(MODEL_PATH):

        saved = joblib.load(MODEL_PATH)
        model = saved["model"]
        scaler = saved["scaler"]

        # 🔥 TOP FEATURES ONLY
        features = [
            'growth_rate_lag1',
            'revenue_growth',
            'profit_trend',
            'profit_margin',
            'esg_overall'
        ]

        input_data = {}

        for col in features:
            input_data[col] = st.slider(
                col,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )

        input_df = pd.DataFrame([input_data])

        if st.button("Predict Growth"):
            pred = model.predict(scaler.transform(input_df))[0]

            st.success(f"Predicted Growth: {pred:.3f}")

            if pred > 0.6:
                st.info("🚀 High Growth Company")
            elif pred > 0.4:
                st.info("📈 Moderate Growth")
            else:
                st.warning("⚠️ Low Growth")

    else:
        st.error("Model not found")

# =====================================================
# 📈 INSIGHTS
# =====================================================
with tab3:

    st.subheader("📊 Correlation Heatmap")

    numeric_df = filtered_df.select_dtypes(include=['number'])

    fig = px.imshow(
    numeric_df.corr(),
    color_continuous_scale='RdBu_r',
    text_auto=True
)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Top Companies")

    top = filtered_df.sort_values(by='growth_rate', ascending=False).head(10)

    st.dataframe(top[['company_name','growth_rate','esg_overall']],
                 use_container_width=True)

    # Insight
    corr = filtered_df[['esg_overall','growth_rate']].corr().iloc[0,1]

    if corr > 0.2:
        msg = "Positive ESG-growth relationship"
    else:
        msg = "Weak ESG-growth relationship"

    st.info(f"{msg} (corr = {corr:.2f})")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("ESG Intelligence Platform | Corporate Analytics Dashboard")