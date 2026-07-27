import streamlit as st
import pandas as pd

from config import APP_TITLE, PROCESSED_DATA_PATH
from src.visualization import Visualizer

# PAGE CONFIGURATION
st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED_DATA_PATH)

df = load_data()

# TITLE
st.title("🌍 Country Clustering for Strategic Aid Allocation")
st.markdown(
    """
This application helps **HELP International** identify countries that should
receive humanitarian aid by clustering countries using
**Machine Learning (K-Means Clustering)**.
"""
)
st.divider()

# PROJECT OVERVIEW
st.header("📖 Project Overview")
st.markdown(
    """
HELP International has raised **$10 million** to support countries that
require humanitarian assistance.

Using socio-economic and health indicators, countries are grouped into
similar clusters. The clusters help identify which countries should receive
priority attention.

The deployed model follows the same workflow as the experimentation notebook.
"""
)

# WORKFLOW
st.header("⚙️ Machine Learning Workflow")
workflow = [
    "Load Raw Dataset",
    "Feature Engineering",
    "Outlier Treatment (IQR Clipping)",
    "Feature Scaling (StandardScaler)",
    "Dimensionality Reduction (PCA - 6 Components)",
    "K-Means Clustering (K = 3)",
    "Cluster Assignment",
    "Aid Priority Recommendation"
]

for step_no, step in enumerate(workflow, start=1):
    st.write(f"**{step_no}.** {step}")

st.divider()

# DATASET SUMMARY
st.header("📊 Dataset Summary")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Countries", len(df))
with col2:
    st.metric("Features", len(df.columns))
with col3:
    st.metric("Clusters", df["Cluster"].nunique())
with col4:
    st.metric("PCA Components", 6)
with col5:
    st.metric("Aid Budget", "$10M")

st.divider()

# CLUSTER SUMMARY
st.header("🎯 Cluster Summary")
cluster_counts = (df["Cluster"].value_counts().sort_index())
cluster_df = cluster_counts.reset_index()
cluster_df.columns = ["Cluster", "Countries"]
col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(cluster_df, use_container_width=True,hide_index=True)
with col2:
    fig = Visualizer.plot_cluster_distribution(df)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# CLUSTER INTERPRETATION
st.header("📌 Cluster Interpretation")
cluster_info = {
    0: (
        "🟢 Cluster 0",
        "Developed economies with higher income, better life expectancy "
        "and lower aid requirement."
    ),
    1: (
        "🔴 Cluster 1",
        "Countries with higher child mortality, lower income and GDP, "
        "requiring the highest priority for humanitarian aid."
    ),
    2: (
        "🟡 Cluster 2",
        "Developing countries with moderate socio-economic indicators "
        "requiring monitoring and selective support."
    )
}

for cluster, (title, description) in cluster_info.items():
    with st.expander(title):
        st.write(description)

st.divider()

# NAVIGATION
st.header("🚀 Explore the Application")
st.info(
    """
Use the navigation menu on the left to explore the project.

### Available Pages

- 📂 Data Explorer
- 📈 Visualizations
- 🎯 Predict Country
- ℹ️ About
"""
)

st.success(
    "The machine learning model has already been trained. "
    "You can immediately explore the dataset or predict the cluster "
    "for a new country."
)