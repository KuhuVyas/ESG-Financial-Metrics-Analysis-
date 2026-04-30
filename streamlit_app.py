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
st.set_page_config(page_title="ESG Intelligence Platform", layout="wide")

# -----------------------------
# BLOOMBERG-STYLE UI
# -----------------------------
st.markdown("""
<style>
body {background-color: #0b0f14; color: #EAEAEA;}
.stMetric {
    background: linear-gradient(145deg, #1a1f2b, #11151c);
    padding: 15px;
    border-radius: 10px;
}
h1 {color: #f5f5f5;}
h2, h3 {color: #d1d5db;}
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
# CLEAN DATA
# -----------------------------
NUMERIC_COLS = [
    'revenue', 'profit_margin', 'market_cap', 'growth_rate',
    'esg_overall', 'esg_environmental', 'esg_social', 'esg_governance'
]

df = df.dropna(subset=NUMERIC_COLS)

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
<p style='text-align: center; color: gray;'>Financial Performance • Sustainability • Predictive Intelligence</p>
""", unsafe_allow_html=True)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 ML Engine", "📈 Insights"])

# =====================================================
# 📊 TAB 1 — DASHBOARD
# =====================================================
with tab1:

    st.subheader("Key Metrics")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Avg ESG", f"{filtered_df['esg_overall'].mean():.2f}")
    k2.metric("Avg Growth", f"{filtered_df['growth_rate'].mean():.2f}")
    k3.metric("Avg Profit", f"{filtered_df['profit_margin'].mean():.2f}")
    k4.metric("Companies", filtered_df["company_id"].nunique())

    # -----------------------------
    # Trend
    # -----------------------------
    st.subheader("Trend Analysis")

    trend = filtered_df.groupby('year')[['growth_rate','esg_overall']].mean().reset_index()

    fig, ax = plt.subplots()
    ax.plot(trend['year'], trend['growth_rate'], label='Growth')
    ax.plot(trend['year'], trend['esg_overall'], label='ESG')
    ax.legend()
    st.pyplot(fig)

    # -----------------------------
    # Distribution
    # -----------------------------
    st.subheader("Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['growth_rate'], bins=30, kde=True, ax=ax)
        ax.set_title("Growth Rate Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_df, x='ESG_Category', y='profit_margin', ax=ax)
        st.pyplot(fig)

    # -----------------------------
    # ESG vs Market
    # -----------------------------
    st.subheader("Market Relationship")

    sample_df = filtered_df.sample(min(800, len(filtered_df)), random_state=42)

    fig, ax = plt.subplots()
    sns.regplot(data=sample_df, x='esg_overall', y='market_cap', ax=ax)
    ax.set_yscale('log')
    st.pyplot(fig)

# =====================================================
# 🤖 TAB 2 — ML ENGINE
# =====================================================
with tab2:

    st.subheader("Growth Prediction Engine")

    MODEL_PATH = "model.pkl"

    if os.path.exists(MODEL_PATH):

        saved = joblib.load(MODEL_PATH)
        model = saved["model"]
        scaler = saved["scaler"]

        st.success("Model Ready")

        # 🔥 Only important features
        selected_features = [
            'growth_rate_lag1',
            'revenue_growth',
            'profit_trend',
            'profit_margin',
            'esg_overall'
        ]

        input_data = {}

        for col in selected_features:
            min_val = float(df[col].min())
            max_val = float(df[col].max())

            input_data[col] = st.slider(
                col,
                min_value=min_val,
                max_value=max_val,
                value=(min_val + max_val)/2
            )

        input_df = pd.DataFrame([input_data])

        if st.button("Predict Growth"):
            input_scaled = scaler.transform(input_df[selected_features])
            pred = model.predict(input_scaled)

            st.success(f"Predicted Growth Rate: {pred[0]:.3f}")

    else:
        st.error("Model not found")

# =====================================================
# 📈 TAB 3 — INSIGHTS
# =====================================================
with tab3:

    st.subheader("Correlation Matrix")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(
        filtered_df[NUMERIC_COLS].corr(),
        annot=True, cmap="coolwarm", ax=ax
    )
    st.pyplot(fig)

    st.subheader("Top Companies")

    top_df = filtered_df.sort_values(by="growth_rate", ascending=False).head(10)

    st.dataframe(
        top_df[['company_name','industry','growth_rate','esg_overall']],
        use_container_width=True
    )

    # -----------------------------
    # ESG Impact Insight
    # -----------------------------
    corr = filtered_df[['esg_overall','growth_rate']].corr().iloc[0,1]

    if corr > 0.2:
        insight = "Positive ESG-growth relationship"
    else:
        insight = "Weak ESG-growth relationship"

    st.info(f"📌 {insight} (Correlation = {corr:.2f})")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Corporate ESG Analytics Platform | Built for Strategic Decision Making")