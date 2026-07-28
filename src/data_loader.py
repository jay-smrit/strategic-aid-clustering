import pandas as pd
import streamlit as st

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.schemas import REQUIRED_COLUMNS, NUMERIC_COLUMNS

class DataLoader:
    def __init__(self):
        self.raw_data_path = RAW_DATA_PATH
        self.processed_data_path = PROCESSED_DATA_PATH

    @st.cache_data(show_spinner=False)
    def load_raw_data(_self) -> pd.DataFrame:
        df = pd.read_csv(_self.raw_data_path)
        return df

    @st.cache_data(show_spinner=False)
    def load_processed_data(_self) -> pd.DataFrame:
            df = pd.read_csv(_self.processed_data_path)
            return df

    @staticmethod
    def validate_columns(df: pd.DataFrame) -> bool:
         missing = set(REQUIRED_COLUMNS) - set(df.columns)
         if missing:
              raise ValueError(f"Missing columns : {missing}")
         return True

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
         return df.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def get_numeric_columns(df: pd.DataFrame):
         return [col for col in NUMERIC_COLUMNS if col in df.columns]

    @staticmethod
    def dataset_summary(df: pd.DataFrame):
         return {
              "Rows": df.shape[0],
              "Columns": df.shape[1],
              "Missing Values": int(df.isnull().sum().sum()),
              "Duplicate Rows": int(df.duplicated().sum())
         }