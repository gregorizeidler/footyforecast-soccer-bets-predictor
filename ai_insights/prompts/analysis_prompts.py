"""
Analysis prompt templates for AI-powered data analysis
"""

from typing import Dict, Any

class AnalysisPrompts:
    """
    Collection of prompt templates for data analysis
    """
    
    @staticmethod
    def get_team_insights_prompt(team_data: Dict[str, Any]) -> str:
        """
        Generate prompt for team insights analysis
        
        Args:
            team_data: Team data dictionary
            
        Returns:
            Formatted prompt
        """
        return f"""
        Analyze the following soccer team data and provide comprehensive insights:
        
        Team: {team_data.get('team_name', 'Unknown')}
        
        Statistics: {team_data.get('statistics', {})}
        
        Recent Form: {team_data.get('recent_form', 'Not available')}
        
        Please provide:
        1. Overall team assessment
        2. Key strengths and weaknesses
        3. Performance trends
        4. Tactical insights
        5. Predictions for future performance
        
        Make the analysis detailed but accessible to soccer fans.
        """
    
    @staticmethod
    def get_performance_comparison_prompt(team1_data: Dict, team2_data: Dict) -> str:
        """
        Generate prompt for team comparison analysis
        """
        return f"""
        Compare these two soccer teams based on their performance data:
        
        Team 1: {team1_data}
        
        Team 2: {team2_data}
        
        Provide a detailed comparison covering:
        1. Offensive capabilities
        2. Defensive strength
        3. Home vs Away performance
        4. Recent form
        5. Head-to-head advantages
        6. Prediction for direct matchup
        """
    
    @staticmethod
    def get_league_analysis_prompt(league_data: Dict[str, Any]) -> str:
        """
        Generate prompt for league-wide analysis
        """
        return f"""
        Analyze this soccer league data:
        
        {league_data}
        
        Provide insights on:
        1. League competitiveness
        2. Top performing teams
        3. Surprising performances
        4. Trends and patterns
        5. Predictability factors
        """
