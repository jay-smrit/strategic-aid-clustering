"""
train_model.py
--------------

Training script for the Country Clustering for Strategic Aid Allocation
application.

Workflow
--------
1. Load raw dataset
2. Validate dataset
3. Train clustering model
4. Save pipeline
5. Save processed dataset
6. Display training summary

Run:
    python train_model.py
"""

import pandas as pd

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

from src.data_loader import DataLoader
from src.model import CountryClusterModel


def main():
    print("=" * 70)
    print("Country Clustering for Strategic Aid Allocation")
    print("=" * 70)

    # Load Dataset-
    print("\nLoading dataset...")
    loader = DataLoader()
    df = loader.load_raw_data()
    loader.validate_columns(df)
    df = loader.remove_duplicates(df)
    print(f"Dataset loaded successfully.")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # Train Model
    print("\nTraining K-Means model...")
    model = CountryClusterModel()
    processed_df = model.train(df)
    print("Model training completed.")

    # Evaluation
    metrics = model.evaluate()
    print("\nModel Evaluation")
    print("-" * 30)
    # print(metrics)
    print(f"Inertia           : {metrics['Inertia']:.2f}")
    print(f"Silhouette Score  : {metrics['Silhouette score']:.4f}")

    # Cluster Summary
    print("\nCluster Summary")
    print("-" * 30)
    print(model.cluster_summary())

    # Save Pipeline
    print("\nSaving pipeline...")
    model.save_pipeline()
    print("Pipeline saved successfully.")

    # Save Processed Dataset
    print("\nSaving processed dataset...")
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("processed_data.csv saved successfully.")

    # Model Information
    info = model.model_info()
    print("\nModel Information")
    print("-" * 30)
    for key, value in info.items():
        print(f"{key:<20}: {value}")
    print("\nTraining completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    main()