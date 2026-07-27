from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from config import APP_TITLE, APP_ICON, LAYOUT
from config import SCALER_PATH, PCA_PATH, KMEANS_PATH

# 1. PAGE CONFIGURATION
def set_page_config():
    
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout=LAYOUT,
        initial_sidebar_state="expanded"
    )

# 2. PAGE TITLE
def page_title(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)

    st.divider()

# 3. KPI CARDS
def display_kpis(summary: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Countries", summary["Rows"])
    col2.metric("Features", summary["Columns"])
    col3.metric("Missing Values", summary["Missing Values"])
    col4.metric("Duplicates", summary["Duplicate Rows"])

# 4. DATAFRAME DISPLAY
def dataframe_view(df):
    st.dataframe(df, use_container_width=True, hide_index=True)

# 5. DOWNLOAD BUTTON
def download_csv(df, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, filename, "text/csv")

# 6. SUCCESS MESSAGE
def success(msg):
    st.success(msg)

# 7. WARNING MESSAGE
def warning(msg):
    st.warning(msg)

# 8. ERROR MESSAGE
def error(msg):
    st.error(msg)

# 9. INFO MESSAGE
def info(msg):
    st.info(msg)

# 10. HORIZONTAL LINE
def divider():
    st.divider()

# 11. FOOTER
def footer():
    st.divider()
    st.caption(
        f"""
Country Clustering for Strategic Aid Allocation

Developed using streamlit • {datetime.now().year}
"""
    )

# 12. PERCENTAGE FORMATTER
def percentage(x):
    return f"{x:.2f}%"

# 13. CURRENCY FORMATTER
def currency(x):
    return "${:,.0f}".format(x)

# 14. NUMBER FORMATTER
def number(x):
    return "{:,.2f}".format(x)

# 15. CHECK IF MODEL EXISTS
def model_exist():
    return (
        Path(SCALER_PATH).exists() and
        Path(PCA_PATH).exists() and
        Path(KMEANS_PATH).exists()
    )
