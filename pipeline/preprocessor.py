"""Data Preprocessor

Feature engineering and data preprocessing for credit risk models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Production-grade data preprocessing pipeline.
    Handles missing values, outliers, feature engineering, and scaling.
    """
    
    def __init__(self, scaler_type: str = 'robust'):
        self.scaler_type = scaler_type
        self.scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        logger.info(f"Initialized DataPreprocessor with {scaler_type} scaler")
    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        df = self._handle_missing(df)
        df = self._engineer_features(df)
        df = self._handle_outliers(df)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if target_col and target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(target_col)
        
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        self.feature_names = df.columns.tolist()
        self.is_fitted = True
        
        logger.info(f"Fitted and transformed {len(df)} samples")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Must fit preprocessor before transform")
        
        df = self._handle_missing(df)
        df = self._engineer_features(df)
        df = self._handle_outliers(df)
        
        numeric_cols = [c for c in self.feature_names if c in df.columns and df[c].dtype in [np.float64, np.int64]]
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', inplace=True)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'debt' in df.columns and 'income' in df.columns:
            df['debt_to_income'] = df['debt'] / (df['income'] + 1e-6)
        
        if 'balance' in df.columns and 'credit_limit' in df.columns:
            df['credit_utilization'] = df['balance'] / (df['credit_limit'] + 1e-6)
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
