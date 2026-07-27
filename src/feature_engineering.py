import numpy as np
import pandas as pd

class FeatureEngineer:

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['export_import_ratio'] = df['exports'] / df['imports']
        df['trade_balance'] = df['exports'] - df['imports']
        df['mortality_income_ratio'] = df['child_mort'] / df['income']
        df['fertility_income_ratio'] = df['total_fer'] / df['income']
        df['health_life_ratio'] = df['health'] / df['life_expec']
        df['aid_need_score'] = (df['child_mort'] * df['total_fer']) / df['income']
        df['wellbeing_score'] = (df['life_expec'] * df['health']) / df['child_mort']

        return df