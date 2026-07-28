# 🌍 Strategic Aid Allocation Dashboard & API

An end-to-end Machine Learning and Data Science project designed to help **HELP International** allocate $10 Million in humanitarian aid. This repository uses **K-Means Clustering**, **Principal Component Analysis (PCA)**, and advanced feature engineering to identify and categorize countries in direst socio-economic and health need.

---

## 📌 Project Overview

HELP International is an international humanitarian NGO committed to fighting poverty and providing basic amenities to underdeveloped nations. This project leverages socio-economic and health indicators (such as child mortality, GDP per capita, income, life expectancy, and health spending) to group countries into distinct priority categories:

* 🔴 **High Priority (Underdeveloped):** High child mortality, low GDP per capita, low income.
* 🟡 **Medium Priority (Developing):** Moderate socio-economic indicators.
* 🟢 **Low Priority (Developed):** High life expectancy, high GDP per capita, robust infrastructure.

---

## 🛠️ Key Features & Pipeline

1. **Outlier Treatment:** Handled via $1.5 \times \text{IQR}$ clipping to retain country representation while minimizing extreme distortion.
2. **Feature Engineering:** Calculated 7 custom socio-economic and health ratios (e.g., `export_import_ratio`, `trade_balance`, `mortality_income_ratio`, `aid_need_score`, `wellbeing_score`).
3. **Clustering & PCA:** Standardized data via `StandardScaler`, applied `KMeans` ($K=3$ default), and visualized separation using 2D `PCA`.
4. **Dual Deployment:**
   * **Streamlit Web Dashboard:** Interactive frontend for exploratory data analysis, cluster profiling, and single-country prediction.
   * **Flask REST API:** Production backend exposing clean endpoints (`/api/analyze` and `/api/predict`) for scalable integration.

---

## 📁 Repository Structure

```text
├── Country-data.csv                # Raw dataset
├── Strategic_Aid_Allocation.ipynb  # Jupyter Notebook (EDA, Modeling & Evaluation)
├── app_streamlit.py                # Streamlit Web Application
├── app_flask.py                    # Flask Web API Application
├── requirements.txt                # Dependencies
└── README.md                       # Documentation