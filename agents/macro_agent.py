"""Macro Economic Scenario Agent

Generates adverse macroeconomic scenarios for CCAR/DFAST-style stress testing.
Utilizes DeepSeek-R1 for generating recession narratives and economic shocks.

Scenario Generation:
- Unemployment rate trajectories
- GDP growth paths
- Interest rate curves (short and long term)
- House price indices
- Equity market volatility (VIX)

Mathematical Foundation:
Macro variables follow stochastic processes:
    dX_t = μ(X_t, θ)dt + σ(X_t, θ)dW_t
where X_t represents macro factors and θ contains scenario parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy.stats import norm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MacroScenario:
    """Container for macroeconomic scenario paths"""
    scenario_name: str
    quarters: int
    unemployment_rate: np.ndarray
    gdp_growth: np.ndarray
    short_rate: np.ndarray
    long_rate: np.ndarray
    hpi: np.ndarray  # House Price Index
    vix: np.ndarray
    
    def to_dict(self) -> Dict:
        return {
            'scenario': self.scenario_name,
            'quarters': self.quarters,
            'unemployment': self.unemployment_rate.tolist(),
            'gdp': self.gdp_growth.tolist(),
            'short_rate': self.short_rate.tolist(),
            'long_rate': self.long_rate.tolist(),
            'hpi': self.hpi.tolist(),
            'vix': self.vix.tolist()
        }

class MacroAgent:
    """Agent for generating macroeconomic stress scenarios"""
    
    def __init__(self, baseline_params: Optional[Dict] = None):
        """
        Initialize MacroAgent with baseline economic parameters.
        
        Args:
            baseline_params: Dictionary containing baseline macro values
        """
        self.baseline = baseline_params or self._default_baseline()
        logger.info("MacroAgent initialized with baseline parameters")
    
    def _default_baseline(self) -> Dict:
        """Return default baseline macroeconomic conditions"""
        return {
            'unemployment': 4.0,
            'gdp_growth': 2.5,
            'short_rate': 2.0,
            'long_rate': 3.5,
            'hpi': 100.0,
            'vix': 15.0
        }
    
    def generate_severely_adverse(self, quarters: int = 12) -> MacroScenario:
        """
        Generate severely adverse scenario similar to CCAR stress tests.
        
        Severely Adverse Scenario Characteristics:
        - Deep recession with unemployment spike
        - Negative GDP growth
        - Housing market crash
        - Elevated market volatility
        
        Args:
            quarters: Number of quarters to simulate
        
        Returns:
            MacroScenario object with time series paths
        """
        logger.info(f"Generating severely adverse scenario for {quarters} quarters")
        
        try:
            t = np.arange(quarters)
            
            # Unemployment: Rapid rise then gradual recovery
            # Peak around 10% unemployment
            unemp_peak = 10.0
            unemployment = self.baseline['unemployment'] + \
                          (unemp_peak - self.baseline['unemployment']) * \
                          np.exp(-0.15 * (t - 3)**2 / 2)
            
            # GDP: Deep contraction followed by slow recovery
            # Trough at -3% annualized
            gdp_growth = -3.0 * np.exp(-((t - 2)**2) / 8) + \
                        2.5 * (1 - np.exp(-0.2 * t))
            
            # Short rate: Drop to near-zero then gradual normalization
            short_rate = np.maximum(0.1, self.baseline['short_rate'] - \
                                   2.0 * np.exp(-0.3 * t))
            
            # Long rate: Initial drop then steepening curve
            long_rate = self.baseline['long_rate'] - 1.5 * np.exp(-0.2 * t) + \
                       0.3 * t * np.exp(-0.1 * t)
            
            # House Price Index: 25% decline from baseline
            hpi_min_pct = 0.75
            hpi = self.baseline['hpi'] * (hpi_min_pct + \
                  (1 - hpi_min_pct) * (1 - np.exp(-0.15 * t)) / \
                  (1 + 0.5 * np.exp(-0.2 * (t - 4))))
            
            # VIX: Spike to 60 then mean reversion
            vix_peak = 60.0
            vix = self.baseline['vix'] + (vix_peak - self.baseline['vix']) * \
                  np.exp(-0.3 * t)
            
            scenario = MacroScenario(
                scenario_name='Severely Adverse',
                quarters=quarters,
                unemployment_rate=unemployment,
                gdp_growth=gdp_growth,
                short_rate=short_rate,
                long_rate=long_rate,
                hpi=hpi,
                vix=vix
            )
            
            logger.info("Successfully generated severely adverse scenario")
            return scenario
            
        except Exception as e:
            logger.error(f"Error generating scenario: {str(e)}")
            raise
    
    def generate_adverse(self, quarters: int = 12) -> MacroScenario:
        """
        Generate moderate adverse scenario.
        
        Adverse Scenario: Moderate recession
        - Moderate unemployment increase
        - Mild GDP contraction
        - Housing market softness
        
        Args:
            quarters: Number of quarters to simulate
        
        Returns:
            MacroScenario object
        """
        logger.info(f"Generating adverse scenario for {quarters} quarters")
        
        t = np.arange(quarters)
        
        # Moderate unemployment rise to 7%
        unemp_peak = 7.0
        unemployment = self.baseline['unemployment'] + \
                      (unemp_peak - self.baseline['unemployment']) * \
                      np.exp(-0.2 * (t - 2)**2 / 2)
        
        # Mild GDP contraction
        gdp_growth = -1.5 * np.exp(-((t - 2)**2) / 6) + \
                    2.0 * (1 - np.exp(-0.25 * t))
        
        short_rate = np.maximum(0.5, self.baseline['short_rate'] - \
                               1.0 * np.exp(-0.3 * t))
        
        long_rate = self.baseline['long_rate'] - 0.8 * np.exp(-0.25 * t) + \
                   0.2 * t * np.exp(-0.15 * t)
        
        # 15% HPI decline
        hpi_min_pct = 0.85
        hpi = self.baseline['hpi'] * (hpi_min_pct + \
              (1 - hpi_min_pct) * (1 - np.exp(-0.2 * t)) / \
              (1 + 0.3 * np.exp(-0.25 * (t - 3))))
        
        # VIX spike to 35
        vix_peak = 35.0
        vix = self.baseline['vix'] + (vix_peak - self.baseline['vix']) * \
              np.exp(-0.35 * t)
        
        return MacroScenario(
            scenario_name='Adverse',
            quarters=quarters,
            unemployment_rate=unemployment,
            gdp_growth=gdp_growth,
            short_rate=short_rate,
            long_rate=long_rate,
            hpi=hpi,
            vix=vix
        )
    
    def generate_baseline(self, quarters: int = 12) -> MacroScenario:
        """
        Generate baseline (expected) economic scenario.
        
        Returns stable growth path with low volatility.
        """
        t = np.arange(quarters)
        
        unemployment = self.baseline['unemployment'] * np.ones(quarters)
        gdp_growth = self.baseline['gdp_growth'] * np.ones(quarters)
        short_rate = self.baseline['short_rate'] * np.ones(quarters)
        long_rate = self.baseline['long_rate'] * np.ones(quarters)
        hpi = self.baseline['hpi'] * (1 + 0.03 / 4) ** t  # 3% annual appreciation
        vix = self.baseline['vix'] * np.ones(quarters)
        
        return MacroScenario(
            scenario_name='Baseline',
            quarters=quarters,
            unemployment_rate=unemployment,
            gdp_growth=gdp_growth,
            short_rate=short_rate,
            long_rate=long_rate,
            hpi=hpi,
            vix=vix
        )
    
    def compute_scenario_severity(self, scenario: MacroScenario) -> float:
        """
        Compute severity index for a scenario (0 = baseline, 1 = extreme).
        
        Severity based on:
        - Peak unemployment
        - Minimum GDP growth
        - HPI decline
        - VIX spike
        
        Returns:
            Float between 0 and 1 indicating scenario severity
        """
        # Normalize each component
        unemp_score = (scenario.unemployment_rate.max() - 4.0) / 8.0
        gdp_score = max(0, -scenario.gdp_growth.min() / 5.0)
        hpi_score = max(0, (100 - scenario.hpi.min()) / 30.0)
        vix_score = (scenario.vix.max() - 15.0) / 50.0
        
        # Weighted average
        severity = 0.3 * unemp_score + 0.25 * gdp_score + \
                   0.25 * hpi_score + 0.2 * vix_score
        
        return np.clip(severity, 0, 1)
    
    def export_scenario(self, scenario: MacroScenario, filepath: str):
        """
        Export scenario to JSON file.
        
        Args:
            scenario: MacroScenario object
            filepath: Output file path
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(scenario.to_dict(), f, indent=2)
            logger.info(f"Scenario exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export scenario: {str(e)}")
            raise
