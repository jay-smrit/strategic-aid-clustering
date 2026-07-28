# 🌍 Clustering Countries for Strategic Aid Allocation

## 📌 Project Overview

This project applies **Unsupervised Machine Learning (K-Means Clustering)** to group countries based on socio-economic and health indicators. The objective is to assist **HELP International**, a humanitarian NGO, in identifying countries that require the highest priority for aid allocation.

Instead of predicting a predefined target, this project discovers natural patterns within the data to classify countries into meaningful clusters representing different levels of development and humanitarian need.

---

## 🎯 Business Objective

HELP International has limited resources and must distribute aid strategically.

Using clustering techniques, this project identifies countries that share similar economic and social characteristics, enabling data-driven prioritisation of humanitarian assistance.

---

## 📂 Dataset

The dataset contains socio-economic and health indicators for multiple countries.

### Original Features

* Child Mortality
* Exports
* Health Expenditure
* Imports
* Income
* Inflation
* Life Expectancy
* Total Fertility
* GDP per Capita

---

## 🛠 Project Workflow

### 1. Data Understanding

* Dataset exploration
* Feature identification
* Missing value analysis
* Duplicate check
* Descriptive statistics

---

### 2. Data Cleaning

* Missing value verification
* Duplicate removal
* Data type validation

---

### 3. Exploratory Data Analysis

Performed extensive EDA including:

* Distribution plots
* Boxplots
* Correlation heatmap
* Pairwise feature analysis
* Outlier detection

---

### 4. Outlier Treatment

Outliers were handled using the **Interquartile Range (IQR)** method to reduce the impact of extreme values while preserving overall data distribution.

---

### 5. Feature Engineering

Several domain-specific features were created to improve clustering quality, including:

* Export-Import Ratio
* Trade Balance
* Mortality-Income Ratio
* Fertility-Income Ratio
* Health-Life Ratio
* Aid Need Score
* Wellbeing Score

These engineered features provide a better representation of a country's socio-economic condition.

---

### 6. Feature Scaling

Standardisation was performed using **StandardScaler** to ensure all features contribute equally during clustering.

---

### 7. Dimensionality Reduction

Principal Component Analysis (PCA) was applied to reduce dimensionality while retaining most of the variance in the dataset.

Benefits include:

* Reduced feature space
* Faster clustering
* Better visualisation
* Reduced multicollinearity

---

### 8. K-Means Clustering

Countries were grouped using the K-Means clustering algorithm.

The optimal number of clusters was determined using:

* Elbow Method
* Silhouette Score

---

### 9. Cluster Analysis

Each cluster was analysed using its average socio-economic indicators.

The clusters were interpreted as:

| Cluster   | Interpretation            | Aid Priority    |
| --------- | ------------------------- | --------------- |
| Cluster 0 | Developing Countries      | Medium Priority |
| Cluster 1 | Least Developed Countries | High Priority   |
| Cluster 2 | Developed Countries       | Low Priority    |

---

### 10. Business Recommendations

Based on the cluster characteristics:

* **High Priority** countries require immediate humanitarian assistance.
* **Medium Priority** countries require continued development support and monitoring.
* **Low Priority** countries are economically stable and require minimal immediate intervention.

---

## 📊 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Plotly
* Scikit-learn
* Jupyter Notebook

---

## 📁 Repository Structure

```text
.
├── Clustering_Countries_Strategic_Aid.ipynb
├── Country-data.csv
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook
```

4. Open:

```
Clustering_Countries_Strategic_Aid.ipynb
```

5. Run all cells sequentially.

---

## 📈 Key Learning Outcomes

This project demonstrates:

* Exploratory Data Analysis (EDA)
* Feature Engineering
* Outlier Treatment
* Feature Scaling
* Principal Component Analysis (PCA)
* K-Means Clustering
* Cluster Interpretation
* Business Recommendation Development
* End-to-End Unsupervised Machine Learning Workflow

---
