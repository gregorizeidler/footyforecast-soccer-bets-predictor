"""
FootyForecast Betting Intelligence Module

This module provides comprehensive betting analysis tools including:
- Portfolio tracking and ROI analysis
- Advanced risk management
- Arbitrage opportunity detection
- Kelly Criterion calculator
- Advanced analytics and streak analysis
- Auto-betting system with queue management
- Advanced ML model with 50+ features

Author: Gregori Zeidler
Year: 2025
"""

from .portfolio_tracker import PortfolioTracker
from .risk_manager import RiskManager
from .arbitrage_detector import ArbitrageDetector
from .kelly_calculator import KellyCalculator
from .betting_analyzer import BettingAnalyzer
from .advanced_analytics import AdvancedAnalytics
from .auto_betting import AutoBettingSystem, AutoBettingMode, BetStatus
from .advanced_model import AdvancedFootballModel

__version__ = "2.0.0"
__all__ = [
    "PortfolioTracker",
    "RiskManager", 
    "ArbitrageDetector",
    "KellyCalculator",
    "BettingAnalyzer",
    "AdvancedAnalytics",
    "AutoBettingSystem",
    "AutoBettingMode",
    "BetStatus",
    "AdvancedFootballModel"
]
