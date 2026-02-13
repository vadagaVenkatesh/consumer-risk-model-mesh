"""Sentiment Analysis Agent

Analyzes borrower communications (emails, chat logs, customer service calls)
to extract early warning signals of financial distress.

Utilizes Mistral LLM for sentiment classification and risk signal detection.

Sentiment Features:
- Financial distress indicators (job loss, medical issues, divorce)
- Payment intention signals
- Emotional state (anxiety, frustration, desperation)
- Negotiation willingness

Output: Risk adjustment factor applied to base PD
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentSignal:
    """Container for sentiment analysis results"""
    borrower_id: str
    timestamp: datetime
    communication_type: str  # email, chat, call_transcript
    distress_score: float  # 0-1, higher = more distress
    payment_intention: float  # 0-1, higher = stronger intent to pay
    risk_keywords: List[str]
    sentiment_category: str  # cooperative, distressed, hostile, neutral
    
    def to_dict(self) -> Dict:
        return {
            'borrower_id': self.borrower_id,
            'timestamp': self.timestamp.isoformat(),
            'type': self.communication_type,
            'distress': self.distress_score,
            'intention': self.payment_intention,
            'keywords': self.risk_keywords,
            'category': self.sentiment_category
        }

class SentimentAgent:
    """Agent for analyzing borrower communication sentiment"""
    
    def __init__(self):
        """
        Initialize SentimentAgent with risk keyword dictionaries.
        
        In production, this would integrate with Mistral API.
        For demo, uses rule-based classification.
        """
        self.distress_keywords = [
            'unemployed', 'laid off', 'fired', 'lost job',
            'medical bills', 'hospital', 'emergency',
            'divorce', 'separated', 'eviction',
            'bankruptcy', 'garnishment', 'lawsuit',
            'can\'t afford', 'struggling', 'desperate'
        ]
        
        self.positive_intention_keywords = [
            'will pay', 'payment plan', 'working on it',
            'need extension', 'temporary hardship',
            'want to resolve', 'committed to paying'
        ]
        
        self.hostile_keywords = [
            'never paying', 'sue me', 'harassment',
            'leave me alone', 'stop calling', 'lawyer'
        ]
        
        logger.info("SentimentAgent initialized with keyword dictionaries")
    
    def analyze_text(self, 
                    text: str, 
                    borrower_id: str,
                    communication_type: str = 'email') -> SentimentSignal:
        """
        Analyze text communication for financial distress signals.
        
        Args:
            text: Communication text to analyze
            borrower_id: Unique borrower identifier
            communication_type: Type of communication
        
        Returns:
            SentimentSignal object with analysis results
        """
        try:
            text_lower = text.lower()
            
            # Detect distress keywords
            distress_matches = [kw for kw in self.distress_keywords 
                              if kw in text_lower]
            distress_score = min(len(distress_matches) * 0.2, 1.0)
            
            # Detect payment intention
            intention_matches = [kw for kw in self.positive_intention_keywords
                               if kw in text_lower]
            intention_score = min(len(intention_matches) * 0.25, 1.0)
            
            # Detect hostile sentiment
            hostile_matches = [kw for kw in self.hostile_keywords
                             if kw in text_lower]
            is_hostile = len(hostile_matches) > 0
            
            # Classify sentiment category
            if is_hostile:
                category = 'hostile'
            elif distress_score > 0.5 and intention_score > 0.3:
                category = 'distressed_cooperative'
            elif distress_score > 0.5:
                category = 'distressed'
            elif intention_score > 0.3:
                category = 'cooperative'
            else:
                category = 'neutral'
            
            # Combine all risk keywords found
            risk_keywords = list(set(distress_matches + hostile_matches))
            
            signal = SentimentSignal(
                borrower_id=borrower_id,
                timestamp=datetime.now(),
                communication_type=communication_type,
                distress_score=distress_score,
                payment_intention=intention_score,
                risk_keywords=risk_keywords,
                sentiment_category=category
            )
            
            logger.info(f"Analyzed sentiment for {borrower_id}: {category}")
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            raise
    
    def compute_risk_adjustment(self, signal: SentimentSignal) -> float:
        """
        Compute risk adjustment multiplier based on sentiment.
        
        Risk Adjustment Logic:
        - Distressed + Cooperative: 1.2x PD (temporary hardship)
        - Distressed + Low Cooperation: 1.8x PD (high risk)
        - Hostile: 2.5x PD (litigation risk)
        - Cooperative: 0.8x PD (willingness to pay)
        - Neutral: 1.0x PD (no adjustment)
        
        Returns:
            Float multiplier for base PD (0.5 to 3.0)
        """
        if signal.sentiment_category == 'hostile':
            adjustment = 2.5
        elif signal.sentiment_category == 'distressed_cooperative':
            adjustment = 1.2
        elif signal.sentiment_category == 'distressed':
            # High distress, low intention
            adjustment = 1.5 + (signal.distress_score * 0.5)
        elif signal.sentiment_category == 'cooperative':
            adjustment = 0.8
        else:  # neutral
            adjustment = 1.0
        
        # Clamp to reasonable range
        adjustment = np.clip(adjustment, 0.5, 3.0)
        
        logger.debug(f"Risk adjustment for {signal.borrower_id}: {adjustment}x")
        return adjustment
    
    def analyze_communication_history(self, 
                                     texts: List[str],
                                     borrower_id: str) -> Tuple[float, List[SentimentSignal]]:
        """
        Analyze multiple communications and aggregate risk signal.
        
        Args:
            texts: List of communication texts
            borrower_id: Borrower identifier
        
        Returns:
            Tuple of (aggregated_risk_adjustment, list_of_signals)
        """
        signals = []
        
        for text in texts:
            signal = self.analyze_text(text, borrower_id)
            signals.append(signal)
        
        if not signals:
            return 1.0, []
        
        # Aggregate using weighted average (more recent = higher weight)
        weights = np.exp(np.linspace(-1, 0, len(signals)))
        weights = weights / weights.sum()
        
        adjustments = [self.compute_risk_adjustment(s) for s in signals]
        aggregated_adjustment = np.average(adjustments, weights=weights)
        
        logger.info(f"Aggregated risk adjustment for {borrower_id}: {aggregated_adjustment:.2f}x")
        return aggregated_adjustment, signals
    
    def extract_hardship_reasons(self, text: str) -> List[str]:
        """
        Extract specific hardship reasons from text.
        
        Returns:
            List of hardship categories detected
        """
        hardship_patterns = {
            'unemployment': r'(unemployed|laid off|lost.*job|fired)',
            'medical': r'(medical|hospital|surgery|illness|injured)',
            'divorce': r'(divorce|separated|custody)',
            'other_debt': r'(other.*debt|credit.*card|student.*loan)',
            'reduced_income': r'(reduced.*hours|pay.*cut|lower.*income)'
        }
        
        text_lower = text.lower()
        reasons = []
        
        for reason, pattern in hardship_patterns.items():
            if re.search(pattern, text_lower):
                reasons.append(reason)
        
        return reasons
    
    def generate_summary_report(self, 
                               signals: List[SentimentSignal]) -> pd.DataFrame:
        """
        Generate summary DataFrame from sentiment signals.
        
        Args:
            signals: List of SentimentSignal objects
        
        Returns:
            DataFrame with summary statistics
        """
        if not signals:
            return pd.DataFrame()
        
        data = [s.to_dict() for s in signals]
        df = pd.DataFrame(data)
        
        # Add aggregated metrics
        summary = pd.DataFrame({
            'borrower_id': df['borrower_id'].unique(),
            'avg_distress': df.groupby('borrower_id')['distress'].mean(),
            'avg_intention': df.groupby('borrower_id')['intention'].mean(),
            'num_communications': df.groupby('borrower_id').size(),
            'most_recent_category': df.groupby('borrower_id')['category'].last()
        })
        
        return summary
