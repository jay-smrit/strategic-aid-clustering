import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Strategic Aid Allocation Dashboard",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("Country-data.csv")
        except Exception:
            return None
    return df

def preprocessing(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)

    return df

def feature_engineering(df):
    df['export_import_ratio'] = df['exports'] / (df['imports'] + 1e-5)
    df['trade_balance'] = df['exports'] - df['imports']
    df['mortality_income_ratio'] = df['child_mort'] / (df['income'] + 1e-5)
    df['fertility_income_ratio'] = df['total_fer'] / (df['income'] + 1e-5)
    df['health_life_ratio'] = df['health'] / (df['life_expec'] + 1e-5)
    df['aid_need_score'] = (df['child_mort'] * df['total_fer']) / (df['income'] + 1e-5)
    df['wellbeing_score'] = (df['life_expec'] * df['health']) / (df['child_mort'] + 1e-5)

    return df

def run_clustering(df, n_clusters=3):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[num_cols]
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Create result dataframe
    df_result = df.copy()
    df_result['Cluster'] = cluster_labels
    
    cluster_summary = df_result.groupby('Cluster')[num_cols].mean()
    
    # Sort cluster IDs by child mortality (descending) to consistently map priority
    sorted_clusters = cluster_summary.sort_values(by='child_mort', ascending=False).index.tolist()
    
    # Dynamic priority labeling based on n_clusters
    priority_mapping = {}
    priority_mapping[sorted_clusters[0]] = "High Priority (Underdeveloped)"
    if n_clusters > 1:
        priority_mapping[sorted_clusters[-1]] = "Low Priority (Developed)"
    for cid in sorted_clusters[1:-1]:
        priority_mapping[cid] = "Medium Priority (Developing)"
    
    df_result['Priority_Category'] = df_result['Cluster'].map(priority_mapping)
    return df_result, num_cols, scaler, kmeans, X_scaled

# -----------------------------------------------------------------------------
# Sidebar Navigation & Upload
# -----------------------------------------------------------------------------
st.sidebar.title("🌍 HELP International")
st.sidebar.markdown("### Strategic Resource Allocation")

uploaded_file = st.sidebar.file_uploader("Upload 'Country-data.csv'", type=["csv"])
df_raw = load_data(uploaded_file)

if df_raw is None:
    st.title("🌍 Strategic Aid Allocation Dashboard")
    st.info("Please upload `Country-data.csv` using the sidebar to proceed.")
    st.stop()

# Apply preprocessing & engineering on raw data
df_processed_base = preprocessing(df_raw.copy())
df_processed_base = feature_engineering(df_processed_base)

# Interactive controls
n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=6, value=3)

# Filter options
priority_filter = st.sidebar.multiselect(
    "Filter Priority Category",
    options=["High Priority (Underdeveloped)", "Medium Priority (Developing)", "Low Priority (Developed)"],
    default=["High Priority (Underdeveloped)", "Medium Priority (Developing)", "Low Priority (Developed)"]
)

# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
df_processed, num_cols, scaler, kmeans, X_scaled = run_clustering(df_processed_base, n_clusters=n_clusters)
df_filtered = df_processed[df_processed['Priority_Category'].isin(priority_filter)]

st.title("🌍 Country Clustering for Strategic Aid Allocation")
st.markdown("""
This application utilizes **K-Means Clustering** on key socio-economic and health indicators 
to assist HELP International in identifying countries in direst need of humanitarian aid.
""")

st.markdown("---")

# Key Metrics Overview
col1, col2, col3, col4 = st.columns(4)
high_priority_count = (df_processed['Priority_Category'] == 'High Priority (Underdeveloped)').sum()
med_priority_count = (df_processed['Priority_Category'] == 'Medium Priority (Developing)').sum()
low_priority_count = (df_processed['Priority_Category'] == 'Low Priority (Developed)').sum()

col1.metric("Total Countries Analyzed", len(df_processed))
col2.metric("High Priority (Need Aid)", high_priority_count, delta_color="inverse")
col3.metric("Medium Priority", med_priority_count)
col4.metric("Low Priority", low_priority_count)

st.markdown("---")

# Main Tabs Structure
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Priority Country Focus", 
    "📊 Cluster Analysis & Profile", 
    "🗺️ PCA Visualizations", 
    "🔍 Country Search & Predictor"
])

# -----------------------------------------------------------------------------
# Tab 1: Priority Country Focus
# -----------------------------------------------------------------------------
with tab1:
    st.header("Countries Requiring Immediate Assistance")
    
    high_priority_df = df_processed[df_processed['Priority_Category'] == 'High Priority (Underdeveloped)']
    
    st.warning(f"**{len(high_priority_df)} Countries** have been identified as High Priority for the $10M Aid Budget allocation.")
    
    # Sort options
    sort_by = st.selectbox("Sort Priority List By:", ['child_mort', 'income', 'gdpp'], index=0)
    ascending = True if sort_by in ['income', 'gdpp'] else False
    
    sorted_high_prio = high_priority_df.sort_values(by=sort_by, ascending=ascending)
    
    st.dataframe(
        sorted_high_prio[['country', 'child_mort', 'income', 'gdpp', 'health', 'life_expec', 'total_fer']],
        use_container_width=True
    )
    
    st.download_button(
        label="📥 Download High-Priority Countries CSV",
        data=sorted_high_prio.to_csv(index=False),
        file_name="high_priority_countries.csv",
        mime="text/csv"
    )

# -----------------------------------------------------------------------------
# Tab 2: Cluster Analysis & Profile
# -----------------------------------------------------------------------------
with tab2:
    st.header("Cluster Profile Averages")
    
    profile_df = df_processed.groupby('Priority_Category')[num_cols].mean().round(2)
    st.dataframe(profile_df, use_container_width=True)
    
    st.subheader("Feature Comparisons Across Clusters")
    feature_to_plot = st.selectbox("Select Feature to Compare Across Clusters:", num_cols, index=0)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df_processed, x='Priority_Category', y=feature_to_plot, palette='Set2', ax=ax)
    plt.xticks(rotation=15)
    plt.title(f"Distribution of {feature_to_plot} by Priority Category")
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# Tab 3: PCA Visualizations
# -----------------------------------------------------------------------------
with tab3:
    st.header("2D Projection via Principal Component Analysis (PCA)")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(X_pca, columns=['PCA Component 1', 'PCA Component 2'])
    pca_df['Priority_Category'] = df_processed['Priority_Category']
    pca_df['Country'] = df_processed['country']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=pca_df,
        x='PCA Component 1',
        y='PCA Component 2',
        hue='Priority_Category',
        style='Priority_Category',
        palette='bright',
        s=80,
        ax=ax
    )
    
    # Annotate top extreme high priority countries
    for idx, row in pca_df.iterrows():
        if row['Priority_Category'] == 'High Priority (Underdeveloped)':
            ax.annotate(row['Country'], (row['PCA Component 1'], row['PCA Component 2']),
                        fontsize=8, alpha=0.7)
            
    plt.title("Cluster Separation via 2D PCA")
    st.pyplot(fig)
    
    st.caption(f"Explained Variance Ratio: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# -----------------------------------------------------------------------------
# Tab 4: Interactive Country Search & Predictor
# -----------------------------------------------------------------------------
with tab4:
    st.header("Interactive Country Lookup & Manual Predictor")
    
    selected_country = st.selectbox("Select a country to view details:", df_processed['country'].unique())
    country_data = df_processed[df_processed['country'] == selected_country].iloc[0]
    
    st.subheader(f"Status for **{selected_country}**: {country_data['Priority_Category']}")
    
    # Show metric cards for selected country
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Child Mortality", country_data['child_mort'])
    c2.metric("Income per Person", f"${country_data['income']:,}")
    c3.metric("GDP per Capita", f"${country_data['gdpp']:,}")
    c4.metric("Life Expectancy", country_data['life_expec'])
    
    st.markdown("---")
    st.subheader("Predict Cluster for Custom Country Metrics")
    
    with st.form("custom_predict_form"):
        col_a, col_b, col_c = st.columns(3)
        cm = col_a.number_input("Child Mortality", min_value=0.0, max_value=300.0, value=50.0)
        exp = col_b.number_input("Exports (% GDP)", min_value=0.0, max_value=200.0, value=25.0)
        hlth = col_c.number_input("Health Spending (% GDP)", min_value=0.0, max_value=50.0, value=5.0)
        
        imp = col_a.number_input("Imports (% GDP)", min_value=0.0, max_value=200.0, value=30.0)
        inc = col_b.number_input("Income", min_value=0, max_value=150000, value=2000)
        inf = col_c.number_input("Inflation", min_value=-10.0, max_value=200.0, value=5.0)
        
        le = col_a.number_input("Life Expectancy", min_value=20.0, max_value=100.0, value=60.0)
        tf = col_b.number_input("Total Fertility", min_value=0.0, max_value=10.0, value=4.0)
        gdp = col_c.number_input("GDP per Capita", min_value=0, max_value=150000, value=1000)
        
        submit_btn = st.form_submit_button("Predict Priority Category")
        
    if submit_btn:
        # 1. Construct single-row DataFrame with base inputs
        input_df = pd.DataFrame([{
            'child_mort': cm,
            'exports': exp,
            'health': hlth,
            'imports': imp,
            'income': inc,
            'inflation': inf,
            'life_expec': le,
            'total_fer': tf,
            'gdpp': gdp
        }])
        
        # 2. Dynamically compute the 7 engineered features
        input_df = feature_engineering(input_df)
        
        # 3. Align column order to match training feature set
        input_df = input_df[num_cols]
        
        # 4. Scale and Predict
        input_scaled = scaler.transform(input_df)
        predicted_cluster = kmeans.predict(input_scaled)[0]
        
        # Map cluster to Priority Category
        pred_priority = df_processed[df_processed['Cluster'] == predicted_cluster]['Priority_Category'].iloc[0]
        
        st.success(f"The input metrics classify this region as: **{pred_priority}**")