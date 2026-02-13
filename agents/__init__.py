"""Agents Layer - Local LLM Integration for Data Preprocessing

This module contains specialized agents that process different types of input data:
- MacroAgent: Generates adverse economic scenarios (DeepSeek-R1)
- SentimentAgent: Analyzes borrower communications (Mistral)
- StructureAgent: Extracts JSON from PDFs (Qwen 2.5)
"""

from .macro_agent import MacroAgent
from .sentiment_agent import SentimentAgent
from .structure_agent import StructureAgent

__all__ = ['MacroAgent', 'SentimentAgent', 'StructureAgent']
