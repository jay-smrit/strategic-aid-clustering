import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.constants import CLUSTER_COLORS

class Visualizer:

    # COMMON LAYOUT
    @staticmethod
    def update_layout(fig, title):
        fig.update_layout(
            title={"text": title, "x":0.5, "xanchor":"center"},
            template = "plotly_white",
            height = 550,
            margin = dict(l=30, r=30, t=70, b=30),
            legend_title = "",
            font = dict(family="Arial", size=14)
        )
        return fig

    # CLUSTER DISTRIBUTION
    @staticmethod
    def plot_cluster_distribution(df):
        cluster_counts = df['Cluster'].value_counts().sort_index().reset_index()
        cluster_counts.columns = ["Cluster","Countries"]

        fig = px.bar(cluster_counts, x="Cluster", y="Countries", color="Cluster", 
                     color_discrete_map=CLUSTER_COLORS, text="Countries")

        fig.update_traces(textposition="outside")

        return Visualizer.update_layout(fig, "Country Distribution Across Clusters")

    # MISSING VALUES
    @staticmethod
    def plot_missing_values(df):
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Feature","Missing"]

        fig = px.bar(missing, x="Feature", y="Missing", text="Missing")

        fig.update_traces(marker_color="#EF553B", textposition="outside")
        fig.update_xaxes(tickangle=45)

        return Visualizer.update_layout(fig, "Missing Values")

    # HISTOGRAM
    @staticmethod
    def plot_histogram(df, feature):
        fig = px.histogram(df, x=feature, nbins=30, color_discrete_sequence=["#1f77b4"])
        return Visualizer.update_layout(fig, f"{feature} Distribution")

    # BOXPLOT
    @staticmethod
    def plot_boxplot(df, feature):
        fig = px.box(df, y=feature, color="Cluster", color_discrete_map=CLUSTER_COLORS, points="outliers")
        return Visualizer.update_layout(fig, f"{feature} by Cluster")

    # CORRELATION HEATMAP
    @staticmethod
    def plot_correlation_heatmap(df):
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(title={"text": "Correlation Heatmap", "x": 0.5}, height=700)
        return fig

    # PCA SCATTER PLOT
    @staticmethod
    def plot_pca_clusters(df):
        required_columns = ["PC1", "PC2", "Cluster"]
        missing = [col for col in required_columns if col not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Missing PCA columns : {missing}")
        hover_cols = []
        if "country" in df.columns:
            hover_cols.append("country")
        fig = px.scatter(df, x="PC1", y="PC2", color="Cluster", 
                         hover_data=hover_cols, color_discrete_map=CLUSTER_COLORS, opacity=0.85)
        fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="black")))
        return Visualizer.update_layout(fig, "PCA Projection of COuntry Clusters")

    # FEATURE SCATTER PLOT
    @staticmethod
    def plot_scatter(df, x_feature, y_feature):
        hover_cols = []
        if "country" in df.columns:
            hover_cols.append("country")
        fig = px.scatter(df, x=x_feature, y=y_feature, color="Cluster", 
                            hover_data=hover_cols, color_discrete_map=CLUSTER_COLORS, opacity=0.80)
        fig.update_traces(marker=dict(size=9, line=dict(width=0.4, color="black")))
        return Visualizer.update_layout(fig, f"{x_feature} vs {y_feature}")

    # CLUSTER COMPARISON
    @staticmethod
    def plot_cluster_comparison(df, features):
        cluster_summary = df.groupby("Cluster")[features].mean().reset_index()
        melted = cluster_summary.melt(id_vars="Cluster", var_name="Feature", value_name="Value")
        fig = px.bar(melted, x="Feature", y="Value", color="Cluster", 
                     barmode="group",color_discrete_map=CLUSTER_COLORS, text_auto=".2f")       
        fig.update_xaxes(tickangle=45)

        return Visualizer.update_layout(fig, "Average Feature Values by Cluster")

    # CLUSTER RADAR CHART
    @staticmethod
    def plot_cluster_radar(df, features):
        summary = df.groupby("Cluster")[features].mean()       
        fig = go.Figure()
        for cluster in summary.index:
            values = summary.loc[cluster].tolist()
            values.append(values[0])
            theta = features.copy()
            theta.append(features[0])
            fig.add_trace(go.Scatterpolar(r=values, theta=theta, fill="toself", name=f"Cluster {cluster}"))
        fig.update_layout(
            title="Cluster Profile Comparison", polar=dict(radialaxis=dict(visible=True)),
            template="plotly_white",
            height=650
        )
        return fig

    # FEATURE SUMMARY TABLE
    @staticmethod
    def feature_summary(df):
        summary = df.describe().T
        summary["Missing"] = df.isna().sum()
        return summary.round(2)

    # CLUSTER SUMMARY TABLE
    @staticmethod
    def cluster_summary(df, features):
        summary = df.groupby("Cluster")[features].mean().round(2)
        return summary

    # Numeric Features

    @staticmethod
    def numeric_features(df):
        numeric = (df.select_dtypes(include="number").columns.tolist())
        ignore = ["Cluster"]
        return [col for col in numeric if col not in ignore]

    
    