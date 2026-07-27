"""
Contains all preprocessing functions:
• IQR clipping
• Feature selection
• Scaling
• PCA
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import PIPELINE_PATH
from src.constants import MODEL_FEATURES, PCA_COMPONENTS
from src.feature_engineering import FeatureEngineer

class Preprocessor:

    @staticmethod
    def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clips outliers in the specified columns of the DataFrame using the IQR method.
        """
        df = df.copy()
        for col in MODEL_FEATURES:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper= Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower, upper)
        return df

#-----------------------------------------------------------------------------------
# STANDARD SCALING
    @staticmethod
    def fit_scaler(df: pd.DataFrame):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[MODEL_FEATURES])
        return scaler, X_scaled

    @staticmethod
    def transform_scaler(df, scaler):
        return scaler.transform(df[MODEL_FEATURES])

    # @staticmethod
    # def save_scaler(scaler):
    #     joblib.dump(scaler, SCALER_PATH)

    # @staticmethod
    # def load_scaler():
    #     return joblib.load(SCALER_PATH)

#-----------------------------------------------------------------------------------
# PCA
    @staticmethod
    def fit_pca(X):
        pca = PCA(n_components=6, random_state=42)
        X_pca = pca.fit_transform(X)
        return pca, X_pca

    @staticmethod
    def transform_pca(X, pca):
        return pca.transform(X)

    # @staticmethod
    # def save_pca(pca):
    #     joblib.dump(pca, PCA_PATH)

    # @staticmethod
    # def load_pca():
    #     return joblib.load(PCA_PATH)
#-----------------------------------------------------------------------------------
# COMPLETE TRAINING PIPELINE
    @staticmethod
    def prepare_training_data(df: pd.DataFrame):
        df = FeatureEngineer.create_features(df)
        df = Preprocessor.clip_outliers(df)
        scaler, X_scaled = Preprocessor.fit_scaler(df)
        pca, X_pca = Preprocessor.fit_pca(X_scaled)

        return {
            "processed_df": df,
            "scaled_data": X_scaled,
            "pca_data": X_pca,
            "scaler": scaler,
            "pca": pca
        }
