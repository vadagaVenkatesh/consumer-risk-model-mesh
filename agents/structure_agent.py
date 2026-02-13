"""Structure Extraction Agent

Extracts structured features from unstructured documents (PDFs, images, scanned documents).
Utilizes Qwen 2.5 for document understanding and JSON extraction.

Extraction Tasks:
- Income verification from pay stubs
- Employment history from resumes
- Asset documentation from bank statements
- Loan application data from PDFs

Output: Structured JSON with verified borrower attributes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import json
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedData:
    """Container for structured data extracted from documents"""
    borrower_id: str
    document_type: str
    extracted_fields: Dict
    confidence_scores: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'borrower_id': self.borrower_id,
            'doc_type': self.document_type,
            'fields': self.extracted_fields,
            'confidence': self.confidence_scores,
            'timestamp': self.timestamp.isoformat()
        }

class StructureAgent:
    """Agent for extracting structured data from unstructured documents"""
    
    def __init__(self):
        """
        Initialize StructureAgent.
        
        In production, integrates with Qwen 2.5 or similar document AI.
        For demo, uses regex-based extraction.
        """
        logger.info("StructureAgent initialized")
    
    def extract_income(self, text: str, borrower_id: str) -> ExtractedData:
        """
        Extract income information from pay stub text.
        
        Fields extracted:
        - Gross monthly income
        - Net monthly income
        - YTD earnings
        - Employer name
        - Pay period
        
        Args:
            text: Document text
            borrower_id: Borrower identifier
        
        Returns:
            ExtractedData object
        """
        try:
            fields = {}
            confidence = {}
            
            # Extract gross income (various formats)
            gross_patterns = [
                r'gross[:\s]+\$?([\d,]+\.?\d*)',
                r'total[\s]+earnings[:\s]+\$?([\d,]+\.?\d*)',
                r'gross[\s]+pay[:\s]+\$?([\d,]+\.?\d*)'
            ]
            
            for pattern in gross_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    fields['gross_income'] = float(match.group(1).replace(',', ''))
                    confidence['gross_income'] = 0.9
                    break
            
            # Extract net income
            net_patterns = [
                r'net[\s]+pay[:\s]+\$?([\d,]+\.?\d*)',
                r'take[\s]+home[:\s]+\$?([\d,]+\.?\d*)'
            ]
            
            for pattern in net_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    fields['net_income'] = float(match.group(1).replace(',', ''))
                    confidence['net_income'] = 0.85
                    break
            
            # Extract YTD earnings
            ytd_match = re.search(r'ytd[:\s]+\$?([\d,]+\.?\d*)', text, re.IGNORECASE)
            if ytd_match:
                fields['ytd_earnings'] = float(ytd_match.group(1).replace(',', ''))
                confidence['ytd_earnings'] = 0.8
            
            # Extract employer
            employer_match = re.search(r'employer[:\s]+([\w\s]+)', text, re.IGNORECASE)
            if employer_match:
                fields['employer'] = employer_match.group(1).strip()
                confidence['employer'] = 0.7
            
            extracted = ExtractedData(
                borrower_id=borrower_id,
                document_type='pay_stub',
                extracted_fields=fields,
                confidence_scores=confidence,
                timestamp=datetime.now()
            )
            
            logger.info(f"Extracted {len(fields)} fields from pay stub for {borrower_id}")
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting income: {str(e)}")
            raise
    
    def extract_employment(self, text: str, borrower_id: str) -> ExtractedData:
        """
        Extract employment history from resume/application.
        
        Fields extracted:
        - Current employer
        - Years of employment
        - Job title
        - Employment gaps
        
        Args:
            text: Document text
            borrower_id: Borrower identifier
        
        Returns:
            ExtractedData object
        """
        fields = {}
        confidence = {}
        
        # Extract job title
        title_patterns = [
            r'title[:\s]+([\w\s]+)',
            r'position[:\s]+([\w\s]+)',
            r'role[:\s]+([\w\s]+)'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['job_title'] = match.group(1).strip()
                confidence['job_title'] = 0.75
                break
        
        # Extract years of experience
        years_match = re.search(r'(\d+)[\s]*years?[\s]*(?:of)?[\s]*(?:experience)?', text, re.IGNORECASE)
        if years_match:
            fields['years_experience'] = int(years_match.group(1))
            confidence['years_experience'] = 0.8
        
        # Extract current employer
        employer_patterns = [
            r'current[\s]+employer[:\s]+([\w\s&,\.]+)',
            r'company[:\s]+([\w\s&,\.]+)'
        ]
        
        for pattern in employer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['current_employer'] = match.group(1).strip()
                confidence['current_employer'] = 0.7
                break
        
        extracted = ExtractedData(
            borrower_id=borrower_id,
            document_type='employment_verification',
            extracted_fields=fields,
            confidence_scores=confidence,
            timestamp=datetime.now()
        )
        
        logger.info(f"Extracted employment data for {borrower_id}")
        return extracted
    
    def extract_assets(self, text: str, borrower_id: str) -> ExtractedData:
        """
        Extract asset information from bank statements.
        
        Fields extracted:
        - Account balance
        - Average monthly balance
        - Large deposits (>$1000)
        - NSF occurrences
        
        Args:
            text: Document text
            borrower_id: Borrower identifier
        
        Returns:
            ExtractedData object
        """
        fields = {}
        confidence = {}
        
        # Extract account balance
        balance_patterns = [
            r'balance[:\s]+\$?([\d,]+\.?\d*)',
            r'ending[\s]+balance[:\s]+\$?([\d,]+\.?\d*)',
            r'current[\s]+balance[:\s]+\$?([\d,]+\.?\d*)'
        ]
        
        for pattern in balance_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields['account_balance'] = float(match.group(1).replace(',', ''))
                confidence['account_balance'] = 0.9
                break
        
        # Detect NSF (Non-Sufficient Funds)
        nsf_count = len(re.findall(r'nsf|insufficient|overdraft', text, re.IGNORECASE))
        if nsf_count > 0:
            fields['nsf_occurrences'] = nsf_count
            confidence['nsf_occurrences'] = 0.95
        
        # Extract all transactions to compute average
        transactions = re.findall(r'\$([\d,]+\.\d{2})', text)
        if transactions:
            amounts = [float(t.replace(',', '')) for t in transactions]
            fields['num_transactions'] = len(amounts)
            fields['avg_transaction'] = np.mean(amounts)
            confidence['avg_transaction'] = 0.7
        
        extracted = ExtractedData(
            borrower_id=borrower_id,
            document_type='bank_statement',
            extracted_fields=fields,
            confidence_scores=confidence,
            timestamp=datetime.now()
        )
        
        logger.info(f"Extracted asset data for {borrower_id}")
        return extracted
    
    def compute_dti_ratio(self, income_data: ExtractedData, 
                         debt_payments: float) -> float:
        """
        Compute Debt-to-Income (DTI) ratio.
        
        DTI = Total Monthly Debt Payments / Gross Monthly Income
        
        Key metric for credit underwriting:
        - DTI < 36%: Excellent
        - DTI 36-43%: Acceptable
        - DTI > 43%: High risk (Qualified Mortgage threshold)
        
        Args:
            income_data: ExtractedData with income fields
            debt_payments: Total monthly debt payments
        
        Returns:
            DTI ratio as percentage
        """
        gross_income = income_data.extracted_fields.get('gross_income', 0)
        
        if gross_income == 0:
            logger.warning("Cannot compute DTI: gross income is zero")
            return 999.0  # Flag as invalid
        
        dti = (debt_payments / gross_income) * 100
        
        logger.info(f"Computed DTI: {dti:.2f}%")
        return dti
    
    def validate_extraction(self, extracted: ExtractedData, 
                          min_confidence: float = 0.7) -> bool:
        """
        Validate that extracted data meets confidence thresholds.
        
        Args:
            extracted: ExtractedData object
            min_confidence: Minimum confidence threshold
        
        Returns:
            True if all fields meet confidence threshold
        """
        for field, confidence in extracted.confidence_scores.items():
            if confidence < min_confidence:
                logger.warning(f"Field '{field}' below confidence threshold: {confidence:.2f}")
                return False
        
        return True
    
    def aggregate_extractions(self, 
                             extractions: List[ExtractedData]) -> pd.DataFrame:
        """
        Aggregate multiple extractions into summary DataFrame.
        
        Args:
            extractions: List of ExtractedData objects
        
        Returns:
            DataFrame with aggregated data
        """
        if not extractions:
            return pd.DataFrame()
        
        data = [e.to_dict() for e in extractions]
        df = pd.DataFrame(data)
        
        return df
