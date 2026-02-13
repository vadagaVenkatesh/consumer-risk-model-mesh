"""Risk Model Mesh Orchestrator

Main entry point that coordinates agents and models for end-to-end
credit risk assessment pipeline.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import argparse
import json

# Import agents
from agents.macro_agent import MacroAgent, MacroConfig
from agents.sentiment_agent import SentimentAgent, SentimentConfig
from agents.structure_agent import StructureAgent

# Import models
from models.attention_lstm import AttentionLSTM, LSTMConfig
from models.contagion_gnn import ContagionGNN, GNNConfig
from models.survival_cox import CoxSurvivalModel, SurvivalConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RiskMeshConfig:
    """Configuration for Risk Model Mesh"""
    macro_config: MacroConfig
    sentiment_config: SentimentConfig
    lstm_config: LSTMConfig
    gnn_config: GNNConfig
    survival_config: SurvivalConfig
    output_dir: str = './outputs'


class RiskModelMesh:
    """Orchestrates agentic data processing and risk modeling pipeline"""
    
    def __init__(self, config: RiskMeshConfig):
        self.config = config
        logger.info("Initializing Risk Model Mesh...")
        
        # Initialize agents (Layer 1: Data Processing)
        self.macro_agent = MacroAgent(config.macro_config)
        self.sentiment_agent = SentimentAgent(config.sentiment_config)
        self.structure_agent = StructureAgent()
        
        # Initialize models (Layer 2: Risk Quantification)
        self.lstm_model = AttentionLSTM(config.lstm_config)
        self.gnn_model = ContagionGNN(config.gnn_config)
        self.survival_model = CoxSurvivalModel(config.survival_config)
        
        logger.info("Risk Model Mesh initialized successfully")
    
    def process_borrower_data(self, borrower_id: str, documents: List[str], 
                             transaction_history: pd.DataFrame) -> Dict:
        """
        Process borrower data through agentic layer
        
        Args:
            borrower_id: Unique borrower identifier
            documents: List of document paths (PDFs, images)
            transaction_history: Transaction data
        
        Returns:
            Processed features dictionary
        """
        logger.info(f"Processing borrower {borrower_id}...")
        
        # Agent 1: Generate macro scenarios
        scenarios = self.macro_agent.generate_scenarios(
            num_scenarios=1000,
            time_horizon_months=24
        )
        logger.info(f"Generated {len(scenarios)} economic scenarios")
        
        # Agent 2: Analyze sentiment from communications
        # (Simulated for demonstration)
        sentiment_features = {
            'distress_score': 0.3,
            'urgency': 0.2,
            'sentiment': 'neutral'
        }
        logger.info("Sentiment analysis completed")
        
        # Agent 3: Extract structured data from documents  
        structured_data = {}
        for doc in documents:
            extracted = self.structure_agent.extract_from_pdf(doc)
            if extracted.confidence_scores:
                avg_confidence = np.mean(list(extracted.confidence_scores.values()))
                if avg_confidence > 0.7:
                    structured_data.update(extracted.extracted_fields)
        logger.info(f"Extracted data from {len(documents)} documents")
        
        return {
            'borrower_id': borrower_id,
            'macro_scenarios': scenarios,
            'sentiment_features': sentiment_features,
            'structured_data': structured_data,
            'transaction_history': transaction_history
        }
    
    def estimate_risk_metrics(self, processed_data: Dict) -> Dict:
        """
        Generate risk estimates using model mesh
        
        Args:
            processed_data: Output from process_borrower_data
        
        Returns:
            Risk metrics dictionary
        """
        logger.info("Estimating risk metrics...")
        
        # PD estimation via LSTM (for thin-file borrowers)
        transaction_seq = self._prepare_sequence_data(
            processed_data['transaction_history']
        )
        pd_prob = self.lstm_model.predict(transaction_seq)[0]
        
        # Systemic risk via GNN (network effects)
        # (Requires borrower network - simplified here)
        systemic_risk = 0.15  # Placeholder
        
        # Time-to-default via Cox model
        # (Requires prepared survival data)
        median_ttd = 18.0  # months (placeholder)
        
        # LGD estimation (simplified Basel approach)
        lgd = 0.45  # Standard unsecured consumer LGD
        
        # EAD calculation
        ead = processed_data.get('outstanding_balance', 10000.0)
        
        # Expected Loss = PD * LGD * EAD
        expected_loss = pd_prob * lgd * ead
        
        risk_metrics = {
            'borrower_id': processed_data['borrower_id'],
            'pd_12m': float(pd_prob),
            'lgd': lgd,
            'ead': ead,
            'expected_loss': expected_loss,
            'systemic_risk': systemic_risk,
            'median_time_to_default_months': median_ttd,
            'risk_rating': self._assign_risk_rating(pd_prob)
        }
        
        logger.info(f"Risk assessment complete: PD={pd_prob:.4f}, EL=${expected_loss:.2f}")
        return risk_metrics
    
    def _prepare_sequence_data(self, transaction_df: pd.DataFrame) -> np.ndarray:
        """Convert transactions to LSTM input format"""
        # Simplified: Create 12-month sequences with 15 features
        seq_length = self.config.lstm_config.sequence_length
        n_features = self.config.lstm_config.input_features
        
        # Generate synthetic sequence for demonstration
        sequence = np.random.randn(1, seq_length, n_features)
        return sequence
    
    def _assign_risk_rating(self, pd: float) -> str:
        """Map PD to internal risk rating"""
        if pd < 0.01:
            return 'AAA'
        elif pd < 0.05:
            return 'AA'
        elif pd < 0.10:
            return 'A'
        elif pd < 0.20:
            return 'BBB'
        elif pd < 0.30:
            return 'BB'
        elif pd < 0.50:
            return 'B'
        else:
            return 'CCC'
    
    def run_portfolio_assessment(self, portfolio_data: pd.DataFrame) -> pd.DataFrame:
        """
        Run risk assessment on entire portfolio
        
        Args:
            portfolio_data: DataFrame with borrower information
        
        Returns:
            DataFrame with risk metrics for all borrowers
        """
        logger.info(f"Running portfolio assessment on {len(portfolio_data)} borrowers...")
        
        results = []
        for idx, row in portfolio_data.iterrows():
            # Process each borrower
            processed = self.process_borrower_data(
                borrower_id=row['borrower_id'],
                documents=row.get('documents', []),
                transaction_history=row.get('transactions', pd.DataFrame())
            )
            
            # Estimate risk
            risk_metrics = self.estimate_risk_metrics(processed)
            results.append(risk_metrics)
        
        results_df = pd.DataFrame(results)
        logger.info(f"Portfolio assessment complete. Mean PD: {results_df['pd_12m'].mean():.4f}")
        
        return results_df


def main():
    """Main entry point for Risk Model Mesh"""
    parser = argparse.ArgumentParser(description='Consumer Risk Model Mesh')
    parser.add_argument('--mode', choices=['single', 'portfolio'], default='single',
                       help='Assessment mode')
    parser.add_argument('--borrower-id', type=str, help='Borrower ID for single assessment')
    parser.add_argument('--portfolio-file', type=str, help='Portfolio CSV file')
    parser.add_argument('--output', type=str, default='./outputs/risk_results.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = RiskMeshConfig(
        macro_config=MacroConfig(),
        sentiment_config=SentimentConfig(),
        lstm_config=LSTMConfig(),
        gnn_config=GNNConfig(),
        survival_config=SurvivalConfig()
    )
    
    # Initialize mesh
    mesh = RiskModelMesh(config)
    
    if args.mode == 'single':
        # Single borrower assessment
        logger.info(f"Running single borrower assessment for {args.borrower_id}")
        
        # Demo data
        documents = ['./data/paystub.pdf', './data/bank_statement.pdf']
        transactions = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=12, freq='M'),
            'amount': np.random.randn(12) * 1000 + 5000
        })
        
        processed = mesh.process_borrower_data(
            borrower_id=args.borrower_id or 'DEMO_001',
            documents=documents,
            transaction_history=transactions
        )
        
        risk_metrics = mesh.estimate_risk_metrics(processed)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(risk_metrics, f, indent=2)
        
        logger.info(f"Results saved to {args.output}")
        print(json.dumps(risk_metrics, indent=2))
        
    elif args.mode == 'portfolio':
        # Portfolio assessment
        logger.info(f"Running portfolio assessment from {args.portfolio_file}")
        
        # Load portfolio (demo with synthetic data if file not provided)
        if args.portfolio_file:
            portfolio = pd.read_csv(args.portfolio_file)
        else:
            # Generate demo portfolio
            portfolio = pd.DataFrame({
                'borrower_id': [f'BORR_{i:04d}' for i in range(100)],
                'outstanding_balance': np.random.uniform(5000, 50000, 100)
            })
        
        results = mesh.run_portfolio_assessment(portfolio)
        
        # Save results
        output_path = args.output.replace('.json', '.csv')
        results.to_csv(output_path, index=False)
        
        logger.info(f"Portfolio results saved to {output_path}")
        print(results.describe())


if __name__ == '__main__':
    main()
