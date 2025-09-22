"""
AI-Powered Market Analysis for FootyForecast

This module provides intelligent analysis of betting markets,
value betting opportunities, and odds comparison.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from .openai_client import OpenAIClient

class MarketAnalyzer:
    """
    AI-powered betting market analyzer
    """
    
    def __init__(self, openai_client: OpenAIClient):
        """
        Initialize market analyzer
        
        Args:
            openai_client: OpenAI client instance
        """
        self.ai_client = openai_client
    
    def analyze_value_betting_opportunity(self, 
                                        model_probabilities: Dict[str, float],
                                        bookmaker_odds: Dict[str, float],
                                        team_names: Tuple[str, str]) -> Dict[str, Any]:
        """
        Analyze value betting opportunities by comparing model predictions with bookmaker odds
        
        Args:
            model_probabilities: Model predicted probabilities
            bookmaker_odds: Bookmaker odds
            team_names: (home_team, away_team) names
            
        Returns:
            Value betting analysis
        """
        # Convert odds to implied probabilities
        implied_probabilities = {}
        for outcome, odds in bookmaker_odds.items():
            implied_probabilities[outcome] = 1 / odds if odds > 0 else 0
        
        # Calculate value
        value_analysis = {}
        for outcome in model_probabilities:
            if outcome in implied_probabilities:
                model_prob = model_probabilities[outcome]
                implied_prob = implied_probabilities[outcome]
                
                # Calculate expected value
                if implied_prob > 0:
                    expected_value = (model_prob * bookmaker_odds[outcome]) - 1
                    value_analysis[outcome] = {
                        'model_probability': model_prob,
                        'implied_probability': implied_prob,
                        'odds': bookmaker_odds[outcome],
                        'expected_value': expected_value,
                        'has_value': expected_value > 0.05  # 5% threshold
                    }
        
        # Generate AI analysis
        analysis_data = {
            'home_team': team_names[0],
            'away_team': team_names[1],
            'value_analysis': value_analysis,
            'model_probabilities': model_probabilities,
            'bookmaker_odds': bookmaker_odds
        }
        
        ai_analysis = self.ai_client.analyze_with_context(
            data=analysis_data,
            analysis_type="market",
            context="Analyze value betting opportunities, explain which bets offer good value and why, and assess the risk-reward profile"
        )
        
        return {
            'value_calculations': value_analysis,
            'ai_analysis': ai_analysis,
            'recommendations': self._generate_betting_recommendations(value_analysis)
        }
    
    def analyze_market_efficiency(self, 
                                 historical_predictions: List[Dict[str, Any]],
                                 historical_odds: List[Dict[str, float]],
                                 actual_results: List[str]) -> str:
        """
        Analyze how efficient the betting market is compared to model predictions
        
        Args:
            historical_predictions: Historical model predictions
            historical_odds: Historical bookmaker odds
            actual_results: Actual match results
            
        Returns:
            Market efficiency analysis
        """
        efficiency_data = {
            'predictions': historical_predictions,
            'odds': historical_odds,
            'results': actual_results,
            'sample_size': len(actual_results)
        }
        
        return self.ai_client.analyze_with_context(
            data=efficiency_data,
            analysis_type="market",
            context="Analyze market efficiency by comparing how well bookmaker odds vs model predictions performed against actual results. Identify patterns where the market consistently over/undervalues certain types of matches"
        )
    
    def generate_bankroll_management_advice(self, 
                                          value_bets: List[Dict[str, Any]],
                                          bankroll_size: float,
                                          risk_tolerance: str = "medium") -> str:
        """
        Generate AI-powered bankroll management advice
        
        Args:
            value_bets: List of identified value betting opportunities
            bankroll_size: Available bankroll
            risk_tolerance: Risk tolerance level (low, medium, high)
            
        Returns:
            Bankroll management recommendations
        """
        management_data = {
            'value_opportunities': value_bets,
            'bankroll': bankroll_size,
            'risk_level': risk_tolerance
        }
        
        return self.ai_client.analyze_with_context(
            data=management_data,
            analysis_type="market",
            context=f"Provide bankroll management advice for these value betting opportunities, considering {risk_tolerance} risk tolerance and Kelly Criterion principles"
        )
    
    def analyze_odds_movement(self, 
                            odds_history: List[Dict[str, Any]],
                            team_names: Tuple[str, str],
                            match_info: Dict[str, Any]) -> str:
        """
        Analyze odds movement patterns and their implications
        
        Args:
            odds_history: Historical odds data with timestamps
            team_names: Team names
            match_info: Match information (date, importance, etc.)
            
        Returns:
            Odds movement analysis
        """
        movement_data = {
            'home_team': team_names[0],
            'away_team': team_names[1],
            'odds_history': odds_history,
            'match_info': match_info
        }
        
        return self.ai_client.analyze_with_context(
            data=movement_data,
            analysis_type="market",
            context="Analyze odds movement patterns and explain what they might indicate about market sentiment, insider information, or public betting patterns"
        )
    
    def compare_bookmaker_margins(self, 
                                 multiple_bookmaker_odds: Dict[str, Dict[str, float]],
                                 team_names: Tuple[str, str]) -> str:
        """
        Compare margins across different bookmakers
        
        Args:
            multiple_bookmaker_odds: Odds from different bookmakers
            team_names: Team names
            
        Returns:
            Bookmaker comparison analysis
        """
        comparison_data = {
            'home_team': team_names[0],
            'away_team': team_names[1],
            'bookmaker_odds': multiple_bookmaker_odds
        }
        
        # Calculate margins for each bookmaker
        margins = {}
        for bookmaker, odds in multiple_bookmaker_odds.items():
            total_implied_prob = sum(1/odd for odd in odds.values() if odd > 0)
            margins[bookmaker] = (total_implied_prob - 1) * 100  # Margin as percentage
        
        comparison_data['margins'] = margins
        
        return self.ai_client.analyze_with_context(
            data=comparison_data,
            analysis_type="market",
            context="Compare bookmaker margins and odds, recommend which bookmaker offers the best value for different outcomes"
        )
    
    def analyze_public_betting_sentiment(self, 
                                       betting_percentages: Dict[str, float],
                                       model_prediction: Dict[str, float],
                                       team_names: Tuple[str, str]) -> str:
        """
        Analyze public betting sentiment vs model predictions
        
        Args:
            betting_percentages: Percentage of bets on each outcome
            model_prediction: Model predicted probabilities
            team_names: Team names
            
        Returns:
            Public sentiment analysis
        """
        sentiment_data = {
            'home_team': team_names[0],
            'away_team': team_names[1],
            'public_betting': betting_percentages,
            'model_prediction': model_prediction
        }
        
        return self.ai_client.analyze_with_context(
            data=sentiment_data,
            analysis_type="market",
            context="Analyze public betting sentiment vs model predictions. Identify potential contrarian opportunities where the public might be biased"
        )
    
    def _generate_betting_recommendations(self, 
                                        value_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate specific betting recommendations based on value analysis
        
        Args:
            value_analysis: Value betting analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        for outcome, analysis in value_analysis.items():
            if analysis['has_value']:
                ev = analysis['expected_value']
                recommendations.append(
                    f"{outcome}: {ev:.2%} expected value at odds {analysis['odds']:.2f}"
                )
        
        if not recommendations:
            recommendations.append("No significant value opportunities identified")
        
        return recommendations
    
    def generate_market_report(self, 
                             match_data: Dict[str, Any],
                             odds_data: Dict[str, Any],
                             model_predictions: Dict[str, Any]) -> str:
        """
        Generate comprehensive market analysis report
        
        Args:
            match_data: Match information
            odds_data: Odds from various sources
            model_predictions: Model prediction results
            
        Returns:
            Comprehensive market report
        """
        report_data = {
            'match': match_data,
            'odds': odds_data,
            'predictions': model_predictions
        }
        
        return self.ai_client.analyze_with_context(
            data=report_data,
            analysis_type="market",
            context="Generate a comprehensive market analysis report covering value opportunities, market efficiency, risk assessment, and betting recommendations"
        )
