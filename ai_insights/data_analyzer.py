"""
Intelligent Data Analyzer for FootyForecast

This module provides AI-powered analysis of team performance data,
pattern detection, and automated insights generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from .openai_client import OpenAIClient
from .prompts.analysis_prompts import AnalysisPrompts

class DataAnalyzer:
    """
    AI-powered data analyzer for soccer team performance
    """
    
    def __init__(self, openai_client: OpenAIClient):
        """
        Initialize data analyzer
        
        Args:
            openai_client: OpenAI client instance
        """
        self.ai_client = openai_client
        self.prompts = AnalysisPrompts()
    
    def analyze_team_performance(self, 
                                team_data: Dict[str, Any], 
                                league_context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate comprehensive team performance analysis
        
        Args:
            team_data: Team statistics and performance data
            league_context: League-wide statistics for comparison
            
        Returns:
            Dictionary with different types of analysis
        """
        # Prepare data summary
        data_summary = self._prepare_team_summary(team_data)
        
        # Generate different types of analysis
        analyses = {}
        
        # Performance trends analysis
        analyses['performance_trends'] = self.ai_client.analyze_with_context(
            data=data_summary,
            analysis_type="team_performance",
            context="Focus on performance trends and patterns over recent matches"
        )
        
        # Strengths and weaknesses
        analyses['strengths_weaknesses'] = self.ai_client.analyze_with_context(
            data=data_summary,
            analysis_type="team_performance", 
            context="Identify key strengths and weaknesses based on statistics"
        )
        
        # Home vs Away analysis
        analyses['home_away_comparison'] = self.ai_client.analyze_with_context(
            data=data_summary,
            analysis_type="team_performance",
            context="Compare home and away performance, identify significant differences"
        )
        
        # League comparison (if context provided)
        if league_context:
            analyses['league_comparison'] = self.ai_client.analyze_with_context(
                data={**data_summary, 'league_averages': league_context},
                analysis_type="team_performance",
                context="Compare team performance against league averages"
            )
        
        return analyses
    
    def detect_performance_anomalies(self, 
                                   team_data: Dict[str, Any],
                                   threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect unusual patterns or anomalies in team performance
        
        Args:
            team_data: Team performance data
            threshold: Standard deviation threshold for anomaly detection
            
        Returns:
            Detected anomalies with AI explanations
        """
        anomalies = {}
        
        # Statistical anomaly detection
        numerical_stats = self._extract_numerical_stats(team_data)
        
        for stat_name, values in numerical_stats.items():
            if len(values) > 5:  # Need sufficient data
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Find outliers
                outliers = []
                for i, val in enumerate(values):
                    if abs(val - mean_val) > threshold * std_val:
                        outliers.append({'match': i+1, 'value': val, 'deviation': abs(val - mean_val)})
                
                if outliers:
                    anomalies[stat_name] = {
                        'outliers': outliers,
                        'explanation': self.ai_client.analyze_with_context(
                            data={
                                'statistic': stat_name,
                                'normal_range': f"{mean_val:.2f} ± {std_val:.2f}",
                                'outliers': outliers
                            },
                            analysis_type="team_performance",
                            context="Explain why these statistical outliers occurred and their significance"
                        )
                    }
        
        return anomalies
    
    def generate_team_insights(self, 
                             team_name: str,
                             team_data: Dict[str, Any],
                             recent_matches: List[Dict] = None) -> str:
        """
        Generate comprehensive insights about a team
        
        Args:
            team_name: Name of the team
            team_data: Team statistics
            recent_matches: Recent match results
            
        Returns:
            Comprehensive team insights
        """
        insight_data = {
            'team_name': team_name,
            'statistics': team_data,
            'recent_form': recent_matches[-5:] if recent_matches else None
        }
        
        prompt = self.prompts.get_team_insights_prompt(insight_data)
        
        return self.ai_client.generate_completion(
            prompt=prompt,
            system_message="You are an expert soccer analyst providing detailed team insights.",
            temperature=0.6
        )
    
    def compare_teams(self, 
                     team1_data: Dict[str, Any],
                     team2_data: Dict[str, Any],
                     team1_name: str,
                     team2_name: str) -> str:
        """
        Generate AI-powered comparison between two teams
        
        Args:
            team1_data: First team's data
            team2_data: Second team's data
            team1_name: First team's name
            team2_name: Second team's name
            
        Returns:
            Detailed comparison analysis
        """
        comparison_data = {
            'team1': {'name': team1_name, 'stats': team1_data},
            'team2': {'name': team2_name, 'stats': team2_data}
        }
        
        return self.ai_client.analyze_with_context(
            data=comparison_data,
            analysis_type="team_performance",
            context=f"Provide detailed comparison between {team1_name} and {team2_name}, highlighting key differences and tactical insights"
        )
    
    def _prepare_team_summary(self, team_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare team data summary for AI analysis
        
        Args:
            team_data: Raw team data
            
        Returns:
            Summarized data suitable for AI analysis
        """
        summary = {}
        
        # Extract key statistics
        if 'home_stats' in team_data:
            summary['home_performance'] = team_data['home_stats']
        
        if 'away_stats' in team_data:
            summary['away_performance'] = team_data['away_stats']
        
        if 'recent_form' in team_data:
            summary['recent_form'] = team_data['recent_form']
        
        if 'season_stats' in team_data:
            summary['season_overview'] = team_data['season_stats']
        
        return summary
    
    def _extract_numerical_stats(self, team_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Extract numerical statistics for anomaly detection
        
        Args:
            team_data: Team data dictionary
            
        Returns:
            Dictionary of numerical statistics
        """
        numerical_stats = {}
        
        # Extract from different data sections
        for section_name, section_data in team_data.items():
            if isinstance(section_data, dict):
                for stat_name, stat_value in section_data.items():
                    if isinstance(stat_value, (list, tuple)):
                        # Convert to float if possible
                        try:
                            float_values = [float(x) for x in stat_value if x is not None]
                            if float_values:
                                numerical_stats[f"{section_name}_{stat_name}"] = float_values
                        except (ValueError, TypeError):
                            continue
        
        return numerical_stats
