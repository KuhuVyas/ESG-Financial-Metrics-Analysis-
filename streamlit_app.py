import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ESG Dashboard", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed_esg_financial_data.csv")

df = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📊 ESG Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "Visualization"]
)

# -----------------------------
# OVERVIEW
# -----------------------------
if page == "Overview":
    st.title("📌 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])

    with col2:
        st.write("### Preview")
        st.dataframe(df.head())

# -----------------------------
# EDA
# -----------------------------
elif page == "EDA":
    st.title("📈 Exploratory Data Analysis")

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    numeric_cols = df.select_dtypes(include=np.number).columns

    selected_col = st.selectbox("Select feature", numeric_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[selected_col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {selected_col}")
    st.pyplot(fig)

# -----------------------------
# VISUALIZATION
# -----------------------------
elif page == "Visualization":
    st.title("📊 Data Visualization")

    numeric_df = df.select_dtypes(include=np.number)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Scatter plot
    st.subheader("Feature Comparison")

    col1 = st.selectbox("X-axis", numeric_df.columns)
    col2 = st.selectbox("Y-axis", numeric_df.columns)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=df[col1], y=df[col2], ax=ax2)
    ax2.set_xlabel(col1)
    ax2.set_ylabel(col2)

    st.pyplot(fig2)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("ESG Financial Metrics Analysis Dashboard")