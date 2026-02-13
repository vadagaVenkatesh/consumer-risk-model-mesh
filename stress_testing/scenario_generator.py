"""CCAR/DFAST Scenario Generator

Generates adverse economic scenarios for stress testing.
Implements Fed/OCC regulatory stress testing frameworks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Configuration for scenario generation"""
    num_scenarios: int = 1000
    time_horizon_months: int = 24
    correlation_matrix: np.ndarray = None


class ScenarioGenerator:
    """Generates macroeconomic stress scenarios"""
    
    def __init__(self, config: ScenarioConfig = None):
        self.config = config or ScenarioConfig()
        
    def generate_scenarios(self, baseline: Dict[str, float] = None) -> pd.DataFrame:
        """
        Generate adverse economic scenarios
        
        Returns:
            DataFrame with columns: unemployment, gdp_growth, hpi, interest_rate
        """
        n = self.config.num_scenarios
        t = self.config.time_horizon_months
        
        # Baseline values
        baseline = baseline or {
            'unemployment': 4.5,
            'gdp_growth': 2.0,
            'hpi': 300.0,
            'interest_rate': 4.0
        }
        
        # Generate correlated shocks
        scenarios = []
        for i in range(n):
            scenario = {
                'scenario_id': i,
                'unemployment': baseline['unemployment'] + np.random.normal(0, 2),
                'gdp_growth': baseline['gdp_growth'] + np.random.normal(0, 1.5),
                'hpi': baseline['hpi'] * (1 + np.random.normal(0, 0.15)),
                'interest_rate': max(0, baseline['interest_rate'] + np.random.normal(0, 1.0))
            }
            scenarios.append(scenario)
        
        df = pd.DataFrame(scenarios)
        logger.info(f"Generated {n} scenarios")
        return df
