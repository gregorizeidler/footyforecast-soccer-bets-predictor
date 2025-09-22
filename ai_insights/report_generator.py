"""
AI-Powered Report Generator for FootyForecast

This module generates comprehensive executive reports and automated
insights summaries using AI analysis.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from .openai_client import OpenAIClient

class ReportGenerator:
    """
    AI-powered report generator for executive summaries and insights
    """
    
    def __init__(self, openai_client: OpenAIClient):
        """
        Initialize report generator
        
        Args:
            openai_client: OpenAI client instance
        """
        self.ai_client = openai_client
    
    def generate_weekly_performance_report(self, 
                                         weekly_data: Dict[str, Any],
                                         model_performance: Dict[str, float],
                                         predictions_made: List[Dict[str, Any]]) -> str:
        """
        Generate weekly performance report
        
        Args:
            weekly_data: Week's match data and results
            model_performance: Model accuracy and performance metrics
            predictions_made: List of predictions made during the week
            
        Returns:
            Comprehensive weekly report
        """
        report_data = {
            'period': 'weekly',
            'data': weekly_data,
            'model_performance': model_performance,
            'predictions': predictions_made,
            'total_predictions': len(predictions_made)
        }
        
        return self.ai_client.analyze_with_context(
            data=report_data,
            analysis_type="report",
            context="Generate a comprehensive weekly performance report including model accuracy, key insights, notable predictions, and recommendations for improvement"
        )
    
    def generate_league_analysis_report(self, 
                                      league_name: str,
                                      league_data: Dict[str, Any],
                                      team_performances: Dict[str, Any],
                                      prediction_accuracy: Dict[str, float]) -> str:
        """
        Generate comprehensive league analysis report
        
        Args:
            league_name: Name of the league
            league_data: League statistics and information
            team_performances: Individual team performance data
            prediction_accuracy: Accuracy metrics for this league
            
        Returns:
            League analysis report
        """
        report_data = {
            'league': league_name,
            'league_stats': league_data,
            'team_data': team_performances,
            'accuracy_metrics': prediction_accuracy
        }
        
        return self.ai_client.analyze_with_context(
            data=report_data,
            analysis_type="report",
            context=f"Generate a comprehensive analysis report for {league_name}, including league trends, top performing teams, prediction accuracy, and strategic insights"
        )
    
    def generate_model_comparison_report(self, 
                                       model_performances: Dict[str, Dict[str, float]],
                                       test_period: str,
                                       league_context: Optional[str] = None) -> str:
        """
        Generate model comparison and performance report
        
        Args:
            model_performances: Performance metrics for different models
            test_period: Time period of the comparison
            league_context: Specific league context if applicable
            
        Returns:
            Model comparison report
        """
        report_data = {
            'models': model_performances,
            'period': test_period,
            'context': league_context or 'All leagues'
        }
        
        return self.ai_client.analyze_with_context(
            data=report_data,
            analysis_type="report",
            context="Compare different machine learning models' performance, identify strengths and weaknesses of each model, and recommend optimal model selection strategies"
        )
    
    def generate_market_insights_report(self, 
                                      market_data: Dict[str, Any],
                                      value_opportunities: List[Dict[str, Any]],
                                      market_efficiency_analysis: Dict[str, Any]) -> str:
        """
        Generate market insights and betting analysis report
        
        Args:
            market_data: Betting market data
            value_opportunities: Identified value betting opportunities
            market_efficiency_analysis: Market efficiency metrics
            
        Returns:
            Market insights report
        """
        report_data = {
            'market_overview': market_data,
            'value_bets': value_opportunities,
            'efficiency_analysis': market_efficiency_analysis
        }
        
        return self.ai_client.analyze_with_context(
            data=report_data,
            analysis_type="report",
            context="Generate a comprehensive market analysis report covering betting opportunities, market efficiency, risk assessment, and strategic recommendations"
        )
    
    def generate_executive_summary(self, 
                                 overall_performance: Dict[str, Any],
                                 key_insights: List[str],
                                 recommendations: List[str],
                                 time_period: str) -> str:
        """
        Generate executive summary for stakeholders
        
        Args:
            overall_performance: Overall system performance metrics
            key_insights: Key insights discovered
            recommendations: Strategic recommendations
            time_period: Reporting period
            
        Returns:
            Executive summary
        """
        summary_data = {
            'performance': overall_performance,
            'insights': key_insights,
            'recommendations': recommendations,
            'period': time_period
        }
        
        return self.ai_client.analyze_with_context(
            data=summary_data,
            analysis_type="report",
            context="Generate an executive summary suitable for stakeholders, focusing on high-level insights, business impact, and strategic recommendations"
        )
    
    def generate_automated_insights(self, 
                                  recent_data: Dict[str, Any],
                                  historical_comparison: Dict[str, Any],
                                  anomalies_detected: List[Dict[str, Any]]) -> str:
        """
        Generate automated insights from recent data analysis
        
        Args:
            recent_data: Recent performance and prediction data
            historical_comparison: Comparison with historical performance
            anomalies_detected: Any anomalies or unusual patterns detected
            
        Returns:
            Automated insights report
        """
        insights_data = {
            'recent_performance': recent_data,
            'historical_context': historical_comparison,
            'anomalies': anomalies_detected
        }
        
        return self.ai_client.analyze_with_context(
            data=insights_data,
            analysis_type="report",
            context="Generate automated insights highlighting significant changes, trends, anomalies, and their potential implications for future predictions"
        )
    
    def generate_prediction_accuracy_report(self, 
                                          accuracy_metrics: Dict[str, float],
                                          prediction_breakdown: Dict[str, Any],
                                          improvement_areas: List[str]) -> str:
        """
        Generate detailed prediction accuracy analysis
        
        Args:
            accuracy_metrics: Various accuracy measurements
            prediction_breakdown: Breakdown by different categories
            improvement_areas: Identified areas for improvement
            
        Returns:
            Accuracy analysis report
        """
        accuracy_data = {
            'metrics': accuracy_metrics,
            'breakdown': prediction_breakdown,
            'improvement_opportunities': improvement_areas
        }
        
        return self.ai_client.analyze_with_context(
            data=accuracy_data,
            analysis_type="report",
            context="Analyze prediction accuracy in detail, identify patterns in correct/incorrect predictions, and suggest specific improvements"
        )
    
    def generate_seasonal_trends_report(self, 
                                      seasonal_data: Dict[str, Any],
                                      year_over_year_comparison: Dict[str, Any],
                                      league_name: Optional[str] = None) -> str:
        """
        Generate seasonal trends and patterns report
        
        Args:
            seasonal_data: Data organized by season/time periods
            year_over_year_comparison: Comparison across different years
            league_name: Specific league if applicable
            
        Returns:
            Seasonal trends report
        """
        trends_data = {
            'seasonal_patterns': seasonal_data,
            'yearly_comparison': year_over_year_comparison,
            'league': league_name or 'All leagues'
        }
        
        return self.ai_client.analyze_with_context(
            data=trends_data,
            analysis_type="report",
            context="Analyze seasonal trends and patterns in soccer performance and predictions, identify cyclical patterns and their implications"
        )
    
    def generate_custom_report(self, 
                             report_title: str,
                             data: Dict[str, Any],
                             focus_areas: List[str],
                             target_audience: str = "technical") -> str:
        """
        Generate custom report based on specific requirements
        
        Args:
            report_title: Title/topic of the report
            data: Relevant data for the report
            focus_areas: Specific areas to focus on
            target_audience: Target audience (technical, executive, general)
            
        Returns:
            Custom report
        """
        custom_data = {
            'title': report_title,
            'data': data,
            'focus_areas': focus_areas,
            'audience': target_audience
        }
        
        context = f"Generate a {target_audience}-level report on '{report_title}' focusing on: {', '.join(focus_areas)}"
        
        return self.ai_client.analyze_with_context(
            data=custom_data,
            analysis_type="report",
            context=context
        )
