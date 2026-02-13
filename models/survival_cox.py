"""Cox Proportional Hazards Model for Time-to-Default

Survival analysis for estimating time until borrower default.
Uses Cox regression for LGD and recovery timing estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
except ImportError:
    raise ImportError("lifelines required: pip install lifelines")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SurvivalConfig:
    """Configuration for Cox Survival Model"""
    penalizer: float = 0.1
    l1_ratio: float = 0.0
    alpha: float = 0.05
    

class CoxSurvivalModel:
    """Cox Proportional Hazards for Default Timing"""
    
    def __init__(self, config: SurvivalConfig):
        self.config = config
        self.model = CoxPHFitter(
            penalizer=config.penalizer,
            l1_ratio=config.l1_ratio,
            alpha=config.alpha
        )
        self.is_fitted = False
        
    def train(self, df: pd.DataFrame, duration_col: str, event_col: str) -> Dict:
        """
        Train Cox model
        
        Args:
            df: DataFrame with features, duration, and event indicator
            duration_col: name of time-to-event column
            event_col: name of binary event indicator (1=default, 0=censored)
        """
        logger.info("Training Cox Proportional Hazards model...")
        self.model.fit(df, duration_col=duration_col, event_col=event_col)
        self.is_fitted = True
        
        # Calculate concordance index
        c_index = concordance_index(
            df[duration_col],
            -self.model.predict_partial_hazard(df),
            df[event_col]
        )
        
        logger.info(f"Cox model fitted. C-index: {c_index:.4f}")
        return {'c_index': c_index}
    
    def predict_survival(self, X: pd.DataFrame, times: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Predict survival probabilities at given times
        
        Args:
            X: Feature matrix
            times: Time points for survival prediction
        
        Returns:
            DataFrame of survival probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        return self.model.predict_survival_function(X, times=times)
    
    def predict_hazard(self, X: pd.DataFrame) -> np.ndarray:
        """Predict partial hazard for borrowers"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_partial_hazard(X).values
    
    def predict_median_survival(self, X: pd.DataFrame) -> np.ndarray:
        """Predict median time to default"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_median(X).values
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients and confidence intervals"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.summary
    
    def save(self, filepath: str):
        """Save model to pickle"""
        import joblib
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, config: SurvivalConfig):
        """Load model from pickle"""
        import joblib
        instance = cls(config)
        instance.model = joblib.load(filepath)
        instance.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
        return instance
