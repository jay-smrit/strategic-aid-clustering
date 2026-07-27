"""
predict.py
----------

Prediction module for the Country Clustering for Strategic Aid Allocation
application.

Responsibilities
----------------
1. Load trained pipeline
2. Perform feature engineering
3. Scale features
4. Apply PCA transformation
5. Predict cluster
6. Calculate confidence
7. Return prediction results

Author : Jayasmritha
"""

import joblib
import numpy as np
import pandas as pd

from config import PIPELINE_PATH

from src.schemas import SCHEMA

from src.feature_engineering import FeatureEngineer

from src.constants import (
    MODEL_FEATURES,
    CLUSTER_LABELS,
    CLUSTER_MESSAGES,
    CLUSTER_COLORS
)


class CountryPredictor:
    """
    Performs prediction using the trained clustering pipeline.
    """

    def __init__(self):
        self.pipeline = self.load_pipeline()
        self.scaler = self.pipeline["scaler"]
        self.pca = self.pipeline["pca"]
        self.kmeans = self.pipeline["kmeans"]
        self.feature_order = self.pipeline["feature_order"]

    # Load Pipeline
    @staticmethod
    def load_pipeline():
        return joblib.load(PIPELINE_PATH)

    # Prepare Input Data
    def prepare_input(self,df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = FeatureEngineer.create_features(df)
        missing = [col for col in self.feature_order if col not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Missing features : {missing}")
        return df[self.feature_order]

    # Scaling
    def scale_data(self, X: pd.DataFrame):
        return self.scaler.transform(X)

    # PCA Transformation
    def apply_pca(self,X_scaled):
        return self.pca.transform(X_scaled)

    # Cluster Prediction
    def predict_cluster(self, X_pca):
        return self.kmeans.predict(X_pca)

    # Predict One Country
    def predict(self,input_df: pd.DataFrame):
        X = self.prepare_input(input_df)
        X_scaled = self.scale_data(X)
        X_pca = self.apply_pca(X_scaled)
        prediction = self.predict_cluster(X_pca)
        cluster = int(prediction[0])

        result = {
            "Cluster": cluster,
            "Priority": CLUSTER_LABELS[cluster],
            "Message": CLUSTER_MESSAGES[cluster],
            "Colour": CLUSTER_COLORS[cluster],
        }

        return result

    # Batch Prediction
    def predict_batch(self,df: pd.DataFrame) -> pd.DataFrame:
        prediction_df = df.copy()

        X = self.prepare_input(prediction_df)
        X_scaled = self.scale_data(X)
        X_pca = self.apply_pca(X_scaled)
        clusters = self.predict_cluster(X_pca)

        prediction_df["Cluster"] = clusters
        prediction_df["Priority"] = prediction_df["Cluster"].map(CLUSTER_LABELS)
        prediction_df["Recommendation"] = prediction_df["Cluster"].map(CLUSTER_MESSAGES)

        return prediction_df

    # Prediction Summary
    def prediction_summary(self,prediction_df: pd.DataFrame) -> dict:
        cluster_counts = (prediction_df["Cluster"].value_counts().sort_index())
        priority_counts = (prediction_df["Priority"].value_counts().to_dict())

        return {
            "Total Countries": len(prediction_df),
            "Cluster Counts": cluster_counts.to_dict(),
            "Priority Counts": priority_counts
        }

    # Cluster Information
    def get_cluster_details(self,cluster: int) -> dict:
        return {
            "Cluster": cluster,
            "Priority": CLUSTER_LABELS[cluster],
            "Recommendation": CLUSTER_MESSAGES[cluster],
            "Colour": CLUSTER_COLORS[cluster]
        }

    # Cluster Centres
    def get_cluster_centers(self):
        return self.kmeans.cluster_centers_

    # Model Information
    def get_model_info(self):
        return {
            "Algorithm": "K-Means Clustering",
            "Number of Clusters": self.kmeans.n_clusters,
            "PCA Components": self.pca.n_components_,
            "Number of Features": len(self.feature_order)
        }

    # Pipeline Information
    def pipeline_information(self):
        return {
            "Features Used": self.feature_order,
            "Total Features": len(self.feature_order),
            "Pipeline Keys": list(self.pipeline.keys())
        }

    # Validate Input
    def validate_input(self,df: pd.DataFrame):

        required = required = SCHEMA.raw_features
        missing = [col for col in self.feature_order if col not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Missing required columns: {missing}")
        return True
    
    # Predict From Dictionary
    def predict_from_dict(self, input_data: dict):
        df = pd.DataFrame([input_data])
        return self.predict(df)