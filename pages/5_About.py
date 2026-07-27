import streamlit as st

# PAGE CONFIGURATION

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

# TITLE
st.title("ℹ️ About the Project")
st.markdown("""
This application was developed to help **HELP International**
identify countries requiring humanitarian aid using
**Machine Learning (K-Means Clustering)**.
""")
st.divider()

# BUSINESS PROBLEM
st.header("🎯 Business Problem")
st.markdown("""
HELP International has raised **$10 million** to support
countries facing socio-economic and health challenges.

The objective is to identify countries that require
the highest priority for humanitarian assistance using
unsupervised machine learning techniques.
""")

st.divider()

# DATASET
st.header("📊 Dataset")
st.markdown("""
The dataset contains socio-economic and health indicators for
countries around the world.

Key variables include:

- Child Mortality
- Exports
- Health Expenditure
- Imports
- Net Income
- Inflation
- Life Expectancy
- Total Fertility
- GDP per Capita
""")

st.divider()

# MACHINE LEARNING PIPELINE
st.header("⚙️ Machine Learning Pipeline")
pipeline_steps = [
    "Load Raw Dataset",
    "Feature Engineering",
    "Outlier Treatment (IQR Clipping)",
    "Feature Scaling (StandardScaler)",
    "Principal Component Analysis (6 Components)",
    "K-Means Clustering (K = 3)",
    "Cluster Assignment",
    "Aid Priority Recommendation"
]

for i, step in enumerate(pipeline_steps, start=1):
    st.write(f"**{i}.** {step}")

st.divider()

# TECHNOLOGIES
st.header("🛠️ Technologies Used")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
### Programming
- Python
### Data Analysis
- Pandas
- NumPy
### Machine Learning
- Scikit-learn
""")
with col2:
    st.markdown("""
### Visualization
- Plotly
### Web Application
- Streamlit
### Model Persistence
- Joblib
""")
st.divider()

# PROJECT STRUCTURE
st.header("📁 Project Structure")
st.code(
"""
strategic-aid-clustering/
│
├── app.py
├── config.py
├── train_model.py
│
├── pages/
├── src/
├── models/
├── data/
├── output/
└── requirements.txt
""",
language="text"
)

st.divider()

# MODEL SUMMARY
st.header("🤖 Model Summary")
st.markdown("""
- Algorithm          : K-Means Clustering
- Optimal Clusters   : 3
- Dimensionality     : PCA (6 Components)
- Scaling            : StandardScaler
- Evaluation Metric  : Silhouette Score
""")

st.divider()


# FUTURE  ENHANCEMENTS
st.header("🚀 Future Enhancements")
st.markdown("""
- Automatic hyperparameter tuning
- Additional clustering algorithms
- Interactive geographic visualizations
- Real-time data integration
- Cloud deployment
- Model monitoring dashboard
""")
st.divider()

# AUTHOR
st.header("👩‍💻 Author")
st.success("""
Developed by **Jayasmritha**
Business Case Study
Country Clustering for Strategic Aid Allocation
""")