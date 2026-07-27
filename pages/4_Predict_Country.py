#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:21:47 2026

@author: jayasmritha
"""

import pandas as pd
import streamlit as st

from src.predict import CountryPredictor
from src.schemas import SCHEMA
from src.constants import CLUSTER_MESSAGES
from src.visualization import Visualizer

# PAGE CONFIGURATION
st.set_page_config(page_title="Predict Country", page_icon="🎯", layout="wide")

# LOAD PREDICTOR
@st.cache_resource
def load_predictor():
    return CountryPredictor()
predictor = load_predictor()

# TITLE
st.title("🎯 Predict Country Cluster")
st.write(
    """
Enter the country's socio-economic and health indicators
to predict its cluster and aid priority.
"""
)

st.divider()

# PREDICTION FORM
st.header("📝 Enter Country Details")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    inputs = {}
    with col1:
        for feature in SCHEMA.raw_features[:len(SCHEMA.raw_features)//2]:
            inputs[feature] = st.number_input(label=feature, value=0.0, format="%.2f")
    with col2:
        for feature in SCHEMA.raw_features[len(SCHEMA.raw_features)//2:]:
            inputs[feature] = st.number_input(label=feature, value=0.0, format="%.2f")
    submitted = st.form_submit_button("🚀 Predict Cluster", use_container_width=True)

# PREDICTION
if submitted:
    input_df = pd.DataFrame([inputs])
    try:
        result = predictor.predict(input_df)

        # st.write(type(result)) # Added temorarily 2 lines
        # st.write(result)

        if isinstance(result, pd.DataFrame):
            cluster = int(result.iloc[0]["Cluster"])
            priority = CLUSTER_MESSAGES[cluster]["priority"]
            recommendation = CLUSTER_MESSAGES[cluster]["message"]

        elif isinstance(result, dict):
            cluster = int(result["Cluster"])
            priority = result["Priority"]
            recommendation = result["Message"]

        else:
            cluster = int(result)
            priority = CLUSTER_MESSAGES[cluster]["priority"]
            recommendation = CLUSTER_MESSAGES[cluster]["message"]
       
        st.success("Prediction completed successfully.")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Cluster", cluster)

        with col2:
            st.metric("Aid Priority", priority)

        st.subheader("Recommendation")
        st.info(recommendation)

    except Exception as e:
        st.error(f"Prediction failed.\n\n{e}")

# BATCH PREDICTION
st.divider()
st.header("📂 Batch Prediction")
st.write(
    """
Upload a CSV file containing one or more countries.
The uploaded file must contain the same raw features
used to train the clustering model.
"""
)
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)
if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.dataframe(
            batch_df.head(),
            use_container_width=True,
            hide_index=True
        )

        # VALIDATE REQUIRED COLUMNS
        required_columns = SCHEMA.raw_features
        missing_columns = [col for col in required_columns if col not in batch_df.columns]
        if missing_columns:
            st.error("The uploaded CSV is missing the following columns:")
            st.write(missing_columns)

        else:
            if st.button("🚀 Predict All Countries", use_container_width=True):
                results = predictor.predict(batch_df)
                st.success("Batch prediction completed successfully.")
                st.subheader("Prediction Results")
                st.dataframe(results, use_container_width=True, hide_index=True)

                st.divider()
                st.header("📊 Prediction Dashboard")
                cluster_counts = (results["Cluster"].value_counts().sort_index())
                high_priority = int(cluster_counts.get(1, 0))
                medium_priority = int(cluster_counts.get(2, 0))
                low_priority = int(cluster_counts.get(0, 0))

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Countries Processed", len(results))
                with col2:
                    st.metric("🔴 High Priority", high_priority)
                with col3:
                    st.metric("🟡 Medium Priority", medium_priority)
                with col4:
                    st.metric("🟢 Low Priority", low_priority)
                st.divider()

                st.subheader("Cluster Distribution")
                fig = Visualizer.plot_cluster_distribution(results)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Aid Recommendations")
                recommendation_rows = []
                for cluster in sorted(cluster_counts.index):
                    recommendation_rows.append({
                        "Cluster": cluster,
                        "Priority": CLUSTER_MESSAGES[cluster]["priority"],
                        "Recommendation": CLUSTER_MESSAGES[cluster]["recommendation"],
                        "Countries": int(cluster_counts[cluster])
                    })
                recommendation_df = pd.DataFrame(recommendation_rows)
                st.dataframe(recommendation_df, use_container_width=True, hide_index=True)

                csv = results.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Predictions", 
                    data=csv, 
                    file_name="country_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(str(e))

st.success(
    """
Batch prediction completed successfully.

Review the dashboard above to identify countries
requiring immediate humanitarian assistance.
"""
)