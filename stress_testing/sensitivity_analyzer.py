"""Sensitivity Analyzer for Model Parameter Testing

Performs sensitivity analysis on risk model parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable
import logging

logger = logging.getLogger(__name__)


class SensitivityAnalyzer:
    """Analyzes model sensitivity to parameter changes"""
    
    def __init__(self):
        self.results = {}
        
    def analyze_parameter_sensitivity(self, model_func: Callable, 
                                      base_params: Dict, 
                                      param_ranges: Dict[str, tuple]) -> pd.DataFrame:
        """
        Analyze sensitivity of model output to parameter variations
        
        Args:
            model_func: Function that takes parameters and returns risk metric
            base_params: Baseline parameter values
            param_ranges: Dict of (param_name, (min, max)) tuples
        
        Returns:
            DataFrame with sensitivity results
        """
        results = []
        
        for param_name, (min_val, max_val) in param_ranges.items():
            test_values = np.linspace(min_val, max_val, 10)
            
            for test_val in test_values:
                params = base_params.copy()
                params[param_name] = test_val
                
                output = model_func(**params)
                
                results.append({
                    'parameter': param_name,
                    'value': test_val,
                    'output': output
                })
        
        df = pd.DataFrame(results)
        self.results = df
        logger.info(f"Sensitivity analysis complete for {len(param_ranges)} parameters")
        return df
    
    def get_elasticity(self, parameter: str) -> float:
        """
        Calculate output elasticity with respect to parameter
        
        Returns:
            Elasticity measure
        """
        if self.results.empty:
            raise ValueError("No results available")
        
        param_data = self.results[self.results['parameter'] == parameter]
        if len(param_data) < 2:
            return 0.0
        
        # Simple finite difference approximation
        d_output = param_data['output'].diff().mean()
        d_param = param_data['value'].diff().mean()
        
        if d_param == 0:
            return 0.0
        
        elasticity = d_output / d_param
        return elasticity
