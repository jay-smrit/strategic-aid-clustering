import pandas as pd
import streamlit as st

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.visualization import Visualizer

# PAGE CONFIGURATION
st.set_page_config(page_title="Data Explorer", page_icon="📂", layout="wide")

# LOAD DATA
@st.cache_data
def load_data():
    # return pd.read_csv(PROCESSED_DATA_PATH)
    return pd.read_csv(RAW_DATA_PATH)
df = load_data()

# TITLE
st.title("📂 Data Explorer")
st.write("Explore the processed dataset used for training the K-Means clustering model.")
st.divider()

# DATASET OVERVIEW
st.header("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows",df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    st.metric("Numeric Features", len(df.select_dtypes(include="number").columns))
with col4:
    st.metric("Missing Values", int(df.isna().sum().sum()))
st.divider()

# DATASET PREVIEW
st.header("🔍 Dataset Preview")
rows = st.slider("Number of rows", min_value=5, max_value=50, value=10)
view_option = st.radio("Preview Type", ["First Rows", "Last Rows", "Random Sample"],horizontal=True)
if view_option == "First Rows":
    preview = df.head(rows)
elif view_option == "Last Rows":
    preview = df.tail(rows)
else:
    preview = df.sample(rows, random_state=42)
st.dataframe(preview, use_container_width=True, hide_index=True)
st.divider()

# FATURE INFORMATION
st.header("📑 Feature Information")
feature_info = pd.DataFrame({
    "Feature": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing": df.isna().sum().values
})
st.dataframe(feature_info, use_container_width=True, hide_index=True)
st.divider()

# SUMMARY STATISTICS
st.header("📈 Summary Statistics")
summary = Visualizer.feature_summary(df)
st.dataframe(summary, use_container_width=True)
st.divider()


# # MISSING VALUES
# st.header("❗ Missing Values")
# fig = Visualizer.plot_missing_values(df)
# st.plotly_chart(fig, use_container_width=True)
# st.divider()

# DOWNLOAD DATASET
# st.header("⬇️ Download Processed Dataset")
# csv = df.to_csv(index=False)
# st.download_button(
#     label="Download Processed Dataset", 
#     data=csv,
#     file_name="processed_country_data.csv",
#     mime="text/csv"
# )