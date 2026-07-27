from pathlib import Path

# PROJECT DIRECTORIES
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
PAGES_DIR = BASE_DIR / "pages"
SRC_DIR = BASE_DIR / "src"
LOG_DIR = BASE_DIR / "logs"

# DATASET
RAW_DATA_PATH = DATA_DIR / "Country-data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"

# MODEL FILES
PIPELINE_PATH = MODEL_DIR / "cluster_pipeline.pkl"

# APPLICATION
APP_TITLE = "🌍 Country Clustering for Strategic Aid Allocation"
APP_ICON = "🌍"
LAYOUT = "wide"
SIDEBAR_TITLE = "Navigation"

# PLOT CONFIGURATION
PLOT_TEMPLATE = "plotly_white"
PLOT_HEIGHT = 600
PLOT_WIDTH = 1000

# Create required directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
PAGES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
