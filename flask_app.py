import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Data Preprocessing & Feature Engineering
# -----------------------------------------------------------------------------
def preprocessing(df):
    """Clips outliers using 1.5 * IQR bounds on numeric columns."""
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean[col] = np.clip(df_clean[col], lower, upper)
    return df_clean

def feature_engineering(df):
    """Generates 7 socio-economic and health ratio indicators."""
    df_eng = df.copy()
    eps = 1e-5  # Prevents division by zero
    
    df_eng['export_import_ratio'] = df_eng['exports'] / (df_eng['imports'] + eps)
    df_eng['trade_balance'] = df_eng['exports'] - df_eng['imports']
    df_eng['mortality_income_ratio'] = df_eng['child_mort'] / (df_eng['income'] + eps)
    df_eng['fertility_income_ratio'] = df_eng['total_fer'] / (df_eng['income'] + eps)
    df_eng['health_life_ratio'] = df_eng['health'] / (df_eng['life_expec'] + eps)
    df_eng['aid_need_score'] = (df_eng['child_mort'] * df_eng['total_fer']) / (df_eng['income'] + eps)
    df_eng['wellbeing_score'] = (df_eng['life_expec'] * df_eng['health']) / (df_eng['child_mort'] + eps)
    return df_eng

def process_clustering(df, n_clusters=3):
    """Executes full preprocessing, feature engineering, scaling, and K-Means."""
    df_prep = preprocessing(df)
    df_feat = feature_engineering(df_prep)
    
    num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    X = df_feat[num_cols]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_result = df_feat.copy()
    df_result['Cluster'] = clusters
    
    # Priority mapping based on child mortality average
    cluster_summary = df_result.groupby('Cluster')[num_cols].mean()
    sorted_clusters = cluster_summary.sort_values(by='child_mort', ascending=False).index.tolist()
    
    priority_mapping = {}
    priority_mapping[sorted_clusters[0]] = "High Priority (Underdeveloped)"
    if n_clusters > 1:
        priority_mapping[sorted_clusters[-1]] = "Low Priority (Developed)"
    for cid in sorted_clusters[1:-1]:
        priority_mapping[cid] = "Medium Priority (Developing)"

    df_result['Priority_Category'] = df_result['Cluster'].map(priority_mapping)
    return df_result, num_cols, scaler, kmeans, X_scaled

def get_data():
    if os.path.exists('Country-data.csv'):
        return pd.read_csv('Country-data.csv')
    return None

# -----------------------------------------------------------------------------
# Web Routes & REST API Endpoints
# -----------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main API endpoint returning cluster metrics and PCA visualization data."""
    data = request.json or {}
    n_clusters = int(data.get('n_clusters', 3))
    
    df = get_data()
    if df is None:
        return jsonify({'error': 'Country-data.csv file not found on server.'}), 400

    df_res, num_cols, scaler, kmeans, X_scaled = process_clustering(df, n_clusters)
    
    # Generate 2D PCA Coordinates
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_scaled)
    df_res['pca_x'] = pca_res[:, 0]
    df_res['pca_y'] = pca_res[:, 1]
    
    category_counts = df_res['Priority_Category'].value_counts().to_dict()
    
    high_prio_list = df_res[df_res['Priority_Category'] == 'High Priority (Underdeveloped)'][
        ['country', 'child_mort', 'income', 'gdpp', 'life_expec', 'total_fer', 'aid_need_score']
    ].sort_values(by='child_mort', ascending=False).to_dict(orient='records')

    all_countries = df_res[
        ['country', 'child_mort', 'income', 'gdpp', 'life_expec', 'Priority_Category', 'pca_x', 'pca_y']
    ].to_dict(orient='records')

    return jsonify({
        'total_countries': len(df_res),
        'category_counts': category_counts,
        'high_priority_countries': high_prio_list,
        'all_countries': all_countries,
        'explained_variance': float(np.sum(pca.explained_variance_ratio_) * 100)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict API endpoint that dynamically calculates feature engineering for custom inputs."""
    data = request.json
    try:
        n_clusters = int(data.get('n_clusters', 3))
        df_raw = get_data()
        if df_raw is None:
            return jsonify({'error': 'Country-data.csv not found on server.'}), 400
            
        df_res, num_cols, scaler, kmeans, _ = process_clustering(df_raw, n_clusters)
        
        # 1. Build DataFrame for the single input row
        input_df = pd.DataFrame([{
            'child_mort': float(data['child_mort']),
            'exports': float(data['exports']),
            'health': float(data['health']),
            'imports': float(data['imports']),
            'income': float(data['income']),
            'inflation': float(data['inflation']),
            'life_expec': float(data['life_expec']),
            'total_fer': float(data['total_fer']),
            'gdpp': float(data['gdpp'])
        }])
        
        # 2. Run feature engineering on input row (computes all 7 ratio features)
        input_df = feature_engineering(input_df)
        
        # 3. Align features to exact shape (16 numerical columns)
        input_df = input_df[num_cols]
        
        # 4. Scale and Predict
        input_scaled = scaler.transform(input_df)
        cluster_pred = kmeans.predict(input_scaled)[0]
        
        priority = df_res[df_res['Cluster'] == cluster_pred]['Priority_Category'].iloc[0]
        
        return jsonify({
            'cluster': int(cluster_pred),
            'priority_category': priority
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)