import streamlit as st
from config import APP_TITLE, APP_ICON, LAYOUT

# PAGRE CONFIGURATION
st.set_page_config(
    page_title = APP_TITLE,
    page_icon = APP_ICON,
    layout = LAYOUT,
    initial_sidebar_state = "expanded"
)
# ---------------------------------------------------------------------------------------------------------------------
# CUSTOM CSS
st.markdown(
    """
    <style>

    .main-title{
        font-size:40px;
        font-weight:700;
        color:#1f77b4;
        text-align:center;
    }

    .sub-title{
        font-size:20px;
        color:#555555;
        text-align:center;
    }

    .section-header{
        font-size:28px;
        font-weight:600;
        color:#1f77b4;
        margin-top:15px;
    }

    .info-box{
        background-color:#F7F9FC;
        padding:18px;
        border-radius:10px;
        border-left:5px solid #1f77b4;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------------------------------------------------
# SIDEBAR
with st.sidebar:
    st.title("🌍 Strategic Aid")
    st.markdown("---")
    st.success("Navigation")
    st.markdown(
        """
       Select a page from the sidebar.

        **Available Pages**
        - Home
        - Data Explorer
        - Visualizations
        - Predict Country
        - About
        """
    )

    st.markdown("---")

    st.info(
        """
        **Model**

        - KMeans Clustering
        - K = 3
        - PCA Components = 6
        """
    )

# ---------------------------------------------------------------------------------------------------------------------
# MAIN PAGE
st.markdown(
    "<div class='main-title'>Country Clustering for Strategic Aid Allocation</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Helping HELP International prioritize countries for humanitarian aid using Machine Learning.</div>",
    unsafe_allow_html=True
)

st.write("")

# ---------------------------------------------------------------------------------------------------------------------
# PROJECT OVERVIEW
st.markdown(
    "<div class='section-header'>Project Overview</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class='info-box'>

HELP International has raised **$10 million** to support countries in need.

This application uses **Machine Learning (K-Means Clustering)** 
to categorize countries based on their socio-economic and health indicators.

The resulting clusters help identify countries requiring:

- 🔴 High Priority Aid
- 🟡 Medium Priority Monitoring
- 🟢 Low Priority Support

</div>
""",
    unsafe_allow_html=True
)

st.write("")

# ---------------------------------------------------------------------------------------------------------------------
# WORKFLOW
st.markdown(
    "<div class='section-header'>Workflow</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
1. Load Country Dataset
2. Perform Feature Engineering
3. Apply Outlier Treatment
4. Standardize Features
5. Apply PCA (6 Components)
6. Cluster Countries using KMeans
7. Recommend Aid Priority
"""
)

st.write("")

# ---------------------------------------------------------------------------------------------------------------------
# QUICK STATISTICS
st.markdown(
    "<div class='section-header'>Project Highlights</div>",
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Countries", "167")

with col2:
    st.metric("Clusters", "3")

with col3:
    st.metric("PCA Components", "6")

st.write("")

# ---------------------------------------------------------------------------------------------------------------------
# INSTRUCTIONS
st.markdown(
    "<div class='section-header'>Getting Started</div>",
    unsafe_allow_html=True
)

st.info(
    """
1. Open **Data Explorer** to understand the dataset.
2. Visit **Visualizations** to analyse country clusters.
3. Use **Predict Country** to estimate the aid priority for a new country.
4. Read **About** for project details and methodology.
    """
)

st.success(
    "The trained KMeans model and preprocessing pipeline are already loaded. "
    "Use the pages in the sidebar to explore the project."
)
