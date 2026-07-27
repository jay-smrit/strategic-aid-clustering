#MODEL CONFIGURATION
N_CLUSTERS = 3
RANDOM_STATE = 42
PCA_COMPONENTS = 6

# ORIGINAL DATASET FEATURES
ORIGINAL_FEATURES = [
    "child_mort",
    "exports",
    "health",
    "imports",
    "income",
    "inflation",
    "life_expec",
    "total_fer",
    "gdpp"
]

# ENGINEERED FEATURES
ENGINEERED_FEATURES = [
    'export_import_ratio',
    'trade_balance',
    'mortality_income_ratio',
    'fertility_income_ratio',
    'health_life_ratio',
    'aid_need_score',
    'wellbeing_score'
]

# RADAR FEATURES
RADAR_FEATURES = [
    "child_mort",
    "income",
    "gdpp",
    "life_expec",
    "total_fer"
]

#FEATURES USED BY MODEL
MODEL_FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES

# FEATURE LABELS
FEATURE_LABELS = {
    "child_mort": "Child Mortality (per 1,000 births)",
    "exports": "Exports (% of GDP)",
    "health": "Health Expenditure (% of GDP)",
    "imports": "Imports (% of GDP)",
    "income": "Net Income per Person",
    "inflation": "Inflation (%)",
    "life_expec": "Life Expectancy (Years)",
    "total_fer": "Total Fertility Rate",
    "gdpp": "GDP per Capita",
    "export_import_ratio": "Export / Import Ratio",
    "trade_balance": "Trade Balance",
    "mortality_income_ratio": "Mortality / Income Ratio",
    "fertility_income_ratio": "Fertility / Income Ratio",
    "health_life_ratio": "Health / Life Expectancy Ratio",
    "aid_need_score": "Aid Need Score",
    "wellbeing_score": "Wellbeing Score"
}

# CLUSTER LABELS
CLUSTER_LABELS = {
    0: "Medium Priority", 
    1: "High Priority", 
    2: "Low Priority"
}

# CLUSTER_MESSAGES

CLUSTER_MESSAGES = {
        0: """
🟡 **Medium Priority**

-  Moderate child mortality
-  Moderate GDP
-  Moderate income
-  Moderate socio-economic indicators
-  Moderate development level
-  Continue monitoring

Development assistance recommended.

""",
        1: """
🔴 **High Priority**

- Very high child mortality
- Low GDP
- Low income
- Low life expectancy
- Poor healthcare indicators

*Immediate humanitarian assistance is recommended.**
""",
        2: """
🟢 **Low Priority**

- Strong economy
- High GDP
- High life expectancy
- Good healthcare indicators

**No immediate aid required.**
"""
}

#PRIORITY CLUSTER COLORS
CLUSTER_COLORS = {
     0: "#F4B400",
     1: "#DB4437",
     2: "#0F9D58"
}

# PRIORITY RANKING
PRIORITY_ORDER = {
    "High Priority": 1,
    "Medium Priority": 2,
    "Low Priority": 3
}

# DASHBOARD COLORS
SUCCESS_COLOR = "#16A34A"
WARNING_COLOR = "#F59E0B"
DANGER_COLOR = "#DC2626"
INFO_COLOR = "#2563EB"

# PLOTLY COLOR SEQUENCE
PLOTLY_COLORS = ["#DB4437","#F4B400","#0F9D58"]