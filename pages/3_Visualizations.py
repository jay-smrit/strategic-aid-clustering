import pandas as pd
import streamlit as st

from config import PROCESSED_DATA_PATH
from src.constants import RADAR_FEATURES, MODEL_FEATURES
from src.visualization import Visualizer

# PAGE CONFIGURATION
st.set_page_config(page_title="Visualizations", page_icon="📈", layout="wide")

# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED_DATA_PATH)
df = load_data()

# TITLE
st.title("📈 Visualizations Dashboard")
st.write(
    """
Explore interactive visualizations generated from the
processed dataset and K-Means clustering results.
"""
)
st.divider()

# CLUSTER OVERVIEW
st.header("🌍 Cluster Overview")
col1, col2 = st.columns(2)
with col1:
    fig = Visualizer.plot_cluster_distribution(df)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = Visualizer.plot_pca_clusters(df)
    st.plotly_chart(fig, use_container_width=True)
st.divider()

# DISTRIBUTION ANALYSIS
st.header("📊 Distribution Analysis")
feature = st.selectbox("Select Feature", MODEL_FEATURES, key="distribution_feature")
col1, col2 = st.columns(2)
with col1:
    fig = Visualizer.plot_histogram(df, feature)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = Visualizer.plot_boxplot(df, feature)
    st.plotly_chart(fig, use_container_width=True)
st.divider()

# CORRELATION ANALYSIS
st.header("📌 Correlation Analysis")
fig = Visualizer.plot_correlation_heatmap(df)
st.plotly_chart(fig, use_container_width=True)
st.divider()

# FEATURE RELATIONSHIPS
st.header("🔗 Feature Relationships")
col1, col2 = st.columns(2)
with col1:
    x_feature = st.selectbox("X-axis", MODEL_FEATURES, key="x_feature")
with col2:
    y_feature = st.selectbox("Y-axis", MODEL_FEATURES, index=1, key="y_feature")
fig = Visualizer.plot_scatter(df, x_feature, y_feature)
st.plotly_chart(fig, use_container_width=True)
st.divider()

# CLUSTER COMPARISON
st.header("🎯 Cluster Comparison")
col1, col2 = st.columns(2)
with col1:
    fig = Visualizer.plot_cluster_comparison(df, MODEL_FEATURES)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = Visualizer.plot_cluster_radar(df, RADAR_FEATURES)
    st.plotly_chart(fig, use_container_width=True)