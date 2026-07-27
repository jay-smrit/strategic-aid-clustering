# 🌍 Strategic Aid Clustering for HELP International

## Project Overview

This project was developed as part of a Machine Learning Business Case Study to help **HELP International**, a humanitarian NGO, identify countries that should be prioritised for aid allocation.

Using socio-economic and health indicators, countries are grouped into clusters using the **K-Means Clustering** algorithm. The notebook implementation was then converted into a modular **Streamlit application** to allow interactive data exploration and prediction for new countries.

---

# Business Problem

HELP International has raised **$10 million** for humanitarian aid. Since the available funds are limited, the organisation requires a data-driven approach to identify countries with the greatest need for assistance.

The objective of this project is to:

* Analyse socio-economic and health indicators.
* Cluster countries with similar characteristics.
* Identify High, Medium and Low priority countries.
* Support strategic aid allocation through machine learning.

---

# Notebook Workflow

The notebook follows the complete machine learning workflow from raw data to business recommendations.

### 1. Data Loading

* Load the country dataset.
* Inspect data types and feature information.
* Check for missing values and duplicates.

### 2. Exploratory Data Analysis (EDA)

Performed exploratory analysis to understand:

* Feature distributions
* Correlation between variables
* Outliers
* Country-wise socio-economic characteristics

Visualisations include:

* Histograms
* Box plots
* Correlation Heatmap
* Pairwise analysis

### 3. Outlier Treatment

Outliers were handled using the **Interquartile Range (IQR)** method by clipping values outside the lower and upper bounds.

### 4. Feature Engineering

Additional features were created to improve clustering performance.

Engineered features include:

* Export / Import Ratio
* Trade Balance
* Mortality / Income Ratio
* Fertility / Income Ratio
* Health / Life Expectancy Ratio
* Aid Need Score
* Wellbeing Score

### 5. Feature Scaling

Features were standardised using **StandardScaler** before dimensionality reduction and clustering.

### 6. Principal Component Analysis (PCA)

Principal Component Analysis (PCA) was applied to reduce dimensionality while retaining most of the information in the dataset.

The final implementation uses **6 principal components**, matching the notebook.

### 7. K-Means Clustering

Countries were grouped into **3 clusters** using the K-Means algorithm.

Each cluster represents a different aid priority level:

* High Priority
* Medium Priority
* Low Priority

### 8. Cluster Evaluation

Model performance was evaluated using:

* Inertia
* Silhouette Score

### 9. Cluster Interpretation

Each cluster was analysed using socio-economic indicators to understand:

* Development level
* Healthcare conditions
* Economic strength
* Humanitarian aid requirements

Business recommendations were then prepared for each cluster.

---

# Project Structure

```
strategic-aid-clustering/
│
├── app.py
├── config.py
├── train_model.py
├── requirements.txt
│
├── data/
│   ├── Country-data.csv
│   └── processed_country_data.csv
│
├── models/
│   └── cluster_pipeline.pkl
│
├── pages/
│   ├── 1_Home.py
│   ├── 2_Data_Explorer.py
│   ├── 3_Visualizations.py
│   ├── 4_Predict_Country.py
│   └── 5_About.py
│
└── src/
    ├── constants.py
    ├── config.py
    ├── data_loader.py
    ├── feature_engineering.py
    ├── preprocessing.py
    ├── model.py
    ├── predict.py
    ├── visualization.py
    ├── schemas.py
    └── utils.py
```

### Project Modules

| File                     | Purpose                                                         |
| ------------------------ | --------------------------------------------------------------- |
| `config.py`              | Stores project paths and configuration variables                |
| `data_loader.py`         | Loads raw and processed datasets                                |
| `feature_engineering.py` | Creates engineered features used during training and prediction |
| `preprocessing.py`       | Performs outlier treatment, scaling and PCA                     |
| `model.py`               | Trains the K-Means model and saves the complete pipeline        |
| `predict.py`             | Loads the trained pipeline and performs single/batch prediction |
| `visualization.py`       | Contains reusable Plotly visualisations                         |
| `schemas.py`             | Defines dataset schema and validation                           |
| `constants.py`           | Stores project constants and cluster messages                   |

---

# Streamlit Application

The notebook implementation was converted into a modular Streamlit application to provide an interactive interface for users.

The application consists of five pages.

### 🏠 Home

Provides an overview of the dataset including:

* Dataset summary
* Cluster summary
* Cluster distribution

### 📊 Data Explorer

Allows users to:

* Preview the dataset
* View descriptive statistics
* Explore feature information

### 📈 Visualizations

Interactive visualisations include:

* Histograms
* Box Plots
* Correlation Heatmap
* PCA Scatter Plot
* Cluster Comparison
* Radar Chart

### 🎯 Predict Country

Supports:

* Single country prediction using manual inputs.
* Batch prediction through CSV upload.
* Automatic cluster assignment with aid recommendations.

### ℹ️ About

Provides project background, methodology and application overview.

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

# Future Improvements

Possible extensions include:

* Automatic optimal cluster selection.
* Comparison with additional clustering algorithms.
* Interactive geographic visualisations.
* Cloud deployment.
* Explainable AI for cluster interpretation.

---

# Author

**Jayasmritha**

Machine Learning Business Case Study – Strategic Aid Clustering using K-Means and Streamlit.
