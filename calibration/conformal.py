"""Conformal Prediction

Implements split conformal and adaptive conformal prediction intervals.
Provides distribution-free uncertainty quantification for credit risk models.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ConformalPredictor:
    """
    Conformal Prediction for Uncertainty Quantification
    
    Provides distribution-free prediction intervals using split conformal
    and adaptive conformal methods. Ensures valid coverage guarantees.
    
    Mathematical Foundation:
    For miscoverage rate α, constructs prediction set C(x) such that:
    P(Y ∈ C(X)) ≥ 1 - α
    
    Attributes:
        alpha: Miscoverage rate (significance level)
        method: 'split' or 'adaptive'
        calibration_scores: Cached nonconformity scores
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        method: str = 'split'
    ):
        """
        Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
            method: Conformal method ('split', 'adaptive')
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        if method not in ['split', 'adaptive']:
            raise ValueError(f"Method must be 'split' or 'adaptive', got {method}")
            
        self.alpha = alpha
        self.method = method
        self.calibration_scores = None
        self.quantile = None
        self.is_calibrated = False
        
        logger.info(f"Initialized ConformalPredictor with alpha={alpha}, method={method}")
    
    def calibrate(
        self,
        cal_predictions: np.ndarray,
        cal_targets: np.ndarray
    ) -> 'ConformalPredictor':
        """
        Calibrate conformal predictor on calibration set.
        
        Computes nonconformity scores and quantile for prediction intervals.
        
        Args:
            cal_predictions: Calibration predictions [n_cal,]
            cal_targets: True calibration targets [n_cal,]
            
        Returns:
            Self for method chaining
        """
        cal_predictions = np.asarray(cal_predictions)
        cal_targets = np.asarray(cal_targets)
        
        if cal_predictions.shape != cal_targets.shape:
            raise ValueError("Predictions and targets must have same shape")
        if len(cal_predictions) < 10:
            logger.warning("Small calibration set may lead to poor coverage")
        
        # Compute nonconformity scores (absolute residuals)
        self.calibration_scores = np.abs(cal_predictions - cal_targets)
        
        # Compute quantile for prediction intervals
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(self.calibration_scores, q_level)
        
        self.is_calibrated = True
        logger.info(
            f"Calibrated on {n} samples, quantile={self.quantile:.4f} "
            f"for {(1-self.alpha)*100:.1f}% coverage"
        )
        
        return self
    
    def predict_interval(
        self,
        predictions: np.ndarray,
        return_dict: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Dict]:
        """
        Construct prediction intervals for new predictions.
        
        Args:
            predictions: Model predictions [n_samples,]
            return_dict: If True, return dict with additional info
            
        Returns:
            (lower, upper) bounds or dict with bounds and metadata
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before prediction")
        
        predictions = np.asarray(predictions)
        
        # Construct symmetric prediction intervals
        lower = predictions - self.quantile
        upper = predictions + self.quantile
        
        # Ensure non-negative for PD predictions
        lower = np.maximum(lower, 0.0)
        upper = np.minimum(upper, 1.0)
        
        if return_dict:
            return {
                'predictions': predictions,
                'lower': lower,
                'upper': upper,
                'interval_width': upper - lower,
                'coverage_level': 1 - self.alpha,
                'quantile': self.quantile
            }
        
        return lower, upper
    
    def validate_coverage(
        self,
        test_predictions: np.ndarray,
        test_targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Validate empirical coverage on test set.
        
        Args:
            test_predictions: Test set predictions
            test_targets: True test targets
            
        Returns:
            Dict with coverage metrics
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before validation")
        
        test_predictions = np.asarray(test_predictions)
        test_targets = np.asarray(test_targets)
        
        lower, upper = self.predict_interval(test_predictions)
        
        # Check coverage
        covered = (test_targets >= lower) & (test_targets <= upper)
        empirical_coverage = np.mean(covered)
        
        # Interval widths
        widths = upper - lower
        
        metrics = {
            'empirical_coverage': empirical_coverage,
            'target_coverage': 1 - self.alpha,
            'coverage_gap': empirical_coverage - (1 - self.alpha),
            'mean_width': np.mean(widths),
            'median_width': np.median(widths),
            'std_width': np.std(widths)
        }
        
        logger.info(
            f"Validation coverage: {empirical_coverage:.3f} "
            f"(target: {1-self.alpha:.3f}), "
            f"mean width: {metrics['mean_width']:.4f}"
        )
        
        return metrics
    
    def adaptive_recalibrate(
        self,
        new_predictions: np.ndarray,
        new_targets: np.ndarray,
        window_size: int = 100
    ) -> 'ConformalPredictor':
        """
        Adaptive recalibration for non-stationary distributions.
        
        Updates conformal quantile using sliding window of recent observations.
        
        Args:
            new_predictions: Recent predictions
            new_targets: Recent true values
            window_size: Size of sliding window
            
        Returns:
            Self for method chaining
        """
        if self.method != 'adaptive':
            logger.warning("Adaptive recalibration called on non-adaptive method")
        
        new_predictions = np.asarray(new_predictions)
        new_targets = np.asarray(new_targets)
        
        # Compute new nonconformity scores
        new_scores = np.abs(new_predictions - new_targets)
        
        # Update calibration scores with sliding window
        if self.calibration_scores is None:
            self.calibration_scores = new_scores
        else:
            self.calibration_scores = np.concatenate([
                self.calibration_scores,
                new_scores
            ])[-window_size:]
        
        # Recompute quantile
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(self.calibration_scores, q_level)
        
        self.is_calibrated = True
        logger.info(f"Adaptively recalibrated with {len(new_scores)} new samples")
        
        return self
