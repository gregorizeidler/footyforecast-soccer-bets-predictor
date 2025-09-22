"""
FootyForecast AI Insights Module

This module provides AI-powered analysis and insights for soccer match prediction
using OpenAI's GPT models. It includes intelligent data analysis, prediction explanations,
educational content, market analysis, executive reports, and conversational interface.

Author: Gregori Zeidler
Year: 2025
"""

from .openai_client import OpenAIClient
from .data_analyzer import DataAnalyzer
from .prediction_explainer import PredictionExplainer
from .report_generator import ReportGenerator
from .chat_assistant import ChatAssistant
from .market_analyzer import MarketAnalyzer

__version__ = "1.0.0"
__all__ = [
    "OpenAIClient",
    "DataAnalyzer", 
    "PredictionExplainer",
    "ReportGenerator",
    "ChatAssistant",
    "MarketAnalyzer"
]
