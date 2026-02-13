"""Model Tracker

Tracks model performance metrics, versions, and experiments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class ModelTracker:
    """
    Production model tracking system.
    Logs experiments, metrics, and model metadata.
    """
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiments = []
        self.current_run = None
        logger.info(f"Initialized ModelTracker for '{experiment_name}'")
    
    def start_run(self, run_name: str, params: Optional[Dict] = None) -> None:
        """Start a new experiment run."""
        self.current_run = {
            'run_name': run_name,
            'start_time': datetime.now().isoformat(),
            'params': params or {},
            'metrics': {},
            'artifacts': []
        }
        logger.info(f"Started run: {run_name}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self.current_run is None:
            raise RuntimeError("No active run")
        self.current_run['params'].update(params)
        logger.debug(f"Logged params: {params}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a performance metric."""
        if self.current_run is None:
            raise RuntimeError("No active run")
        
        if name not in self.current_run['metrics']:
            self.current_run['metrics'][name] = []
        
        metric_entry = {'value': value, 'timestamp': datetime.now().isoformat()}
        if step is not None:
            metric_entry['step'] = step
        
        self.current_run['metrics'][name].append(metric_entry)
        logger.debug(f"Logged metric {name}: {value}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_artifact(self, artifact_name: str, artifact_path: str) -> None:
        """Log artifact location."""
        if self.current_run is None:
            raise RuntimeError("No active run")
        
        self.current_run['artifacts'].append({
            'name': artifact_name,
            'path': artifact_path,
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"Logged artifact: {artifact_name}")
    
    def end_run(self, status: str = 'completed') -> None:
        """End current run."""
        if self.current_run is None:
            raise RuntimeError("No active run")
        
        self.current_run['end_time'] = datetime.now().isoformat()
        self.current_run['status'] = status
        self.experiments.append(self.current_run)
        
        logger.info(f"Ended run: {self.current_run['run_name']} ({status})")
        self.current_run = None
    
    def get_best_run(self, metric_name: str, mode: str = 'max') -> Optional[Dict]:
        """Get best run based on metric."""
        if not self.experiments:
            return None
        
        valid_runs = [
            run for run in self.experiments
            if metric_name in run['metrics'] and run['metrics'][metric_name]
        ]
        
        if not valid_runs:
            return None
        
        best_run = max(
            valid_runs,
            key=lambda r: r['metrics'][metric_name][-1]['value']
            if mode == 'max' else
            -r['metrics'][metric_name][-1]['value']
        )
        
        return best_run
    
    def get_runs_dataframe(self) -> pd.DataFrame:
        """Convert experiments to DataFrame."""
        if not self.experiments:
            return pd.DataFrame()
        
        rows = []
        for run in self.experiments:
            row = {
                'run_name': run['run_name'],
                'status': run['status'],
                'start_time': run['start_time'],
                'end_time': run.get('end_time', None)
            }
            
            # Add params
            row.update({f"param_{k}": v for k, v in run['params'].items()})
            
            # Add final metrics
            for metric_name, metric_values in run['metrics'].items():
                if metric_values:
                    row[f"metric_{metric_name}"] = metric_values[-1]['value']
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def export_experiments(self, filepath: str) -> None:
        """Export all experiments to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        logger.info(f"Exported {len(self.experiments)} experiments to {filepath}")
