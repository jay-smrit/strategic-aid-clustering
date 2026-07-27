from pathlib import Path
import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import PIPELINE_PATH

from src.constants import MODEL_FEATURES, N_CLUSTERS, RANDOM_STATE
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor

class CountryClusterModel:
    def __init__(self, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.scaler = None
        self.pca = None
        self.kmeans = None

        self.processed_df = None

        self.X_scaled = None
        self.X_pca = None

    def train(self, df: pd.DataFrame):
        
        df = df.copy()
        df = FeatureEngineer.create_features(df) # Feature Enginnering
        df = Preprocessor.clip_outliers(df) # Outlier Treatment
        self.scaler, self.X_scaled = (Preprocessor.fit_scaler(df)) # scaling
        self.pca, self.X_pca = (Preprocessor.fit_pca(self.X_scaled)) # PCA
        
        df["PC1"] = self.X_pca[:, 0]
        df["PC2"] = self.X_pca[:, 1]

        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            init='k-means++',
            random_state=self.random_state)

        self.kmeans.fit(self.X_pca)
        df["Cluster"] = self.kmeans.labels_
        self.processed_df = df

        return df

    def evaluate(self):
        if self.kmeans is None:
            raise ValueError("Train the model before evaluation")

        metrics = {
            "Inertia" : self.kmeans.inertia_,
            "Silhouette score" : silhouette_score(self.X_pca, self.kmeans.labels_)
        }

        return metrics

    # CLUSTER SUMMARY
    def cluster_summary(self):
        if self.processed_df is None:
            raise ValueError("Model has not been trained")

        summary = (self.processed_df.groupby("Cluster")[MODEL_FEATURES].mean().round(2))

        return summary

    # SAVE PIPELINE
    def save_pipeline(self):

        metrics = self.evaluate()
        pipeline = {
            "scaler": self.scaler,
            "pca": self.pca,
            "kmeans": self.kmeans,
            "feature_order": MODEL_FEATURES,
            "n_clusters": self.n_clusters,
            "pca_components": self.pca.n_components_,
            "metrics": metrics
        }

        joblib.dump(pipeline, PIPELINE_PATH)

    # LOAD PIPELINE
    @staticmethod
    def load_pipeline():
        return joblib.load(PIPELINE_PATH)

    # MODEL INFORMATION
    def model_info(self):
        if self.kmeans is None:
            raise ValueError("Model has not been trained.")

        info = {
            "Algorithm": "K-Means",
            "Clusters": self.n_clusters,
            "PCA Components": self.pca.n_components_,
            "Random State": self.random_state
        }
        return info

    # GET PROCESSED DATASET
    def get_processed_data(self):
        if self.processed_df is None:
            raise ValueError("Train the model first.")
        return self.processed_df.copy()