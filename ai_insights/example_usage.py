"""
Example usage of FootyForecast AI Insights

This file demonstrates how to use the AI-powered features
of FootyForecast for intelligent analysis and predictions.
"""

import os
from ai_insights import (
    OpenAIClient, 
    DataAnalyzer, 
    PredictionExplainer,
    ReportGenerator,
    ChatAssistant,
    MarketAnalyzer
)

def main():
    """
    Example usage of AI insights features
    """
    # Initialize OpenAI client (requires API key)
    # Set your OpenAI API key as environment variable: OPENAI_API_KEY
    try:
        ai_client = OpenAIClient()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OpenAI API key as environment variable OPENAI_API_KEY")
        return
    
    # Initialize AI modules
    data_analyzer = DataAnalyzer(ai_client)
    prediction_explainer = PredictionExplainer(ai_client)
    report_generator = ReportGenerator(ai_client)
    chat_assistant = ChatAssistant(ai_client)
    market_analyzer = MarketAnalyzer(ai_client)
    
    # Example 1: Team Performance Analysis
    print("=== Team Performance Analysis ===")
    team_data = {
        'team_name': 'Manchester City',
        'home_stats': {
            'wins': 8, 'losses': 1, 'goals_for': 24, 'goals_against': 6
        },
        'away_stats': {
            'wins': 6, 'losses': 2, 'goals_for': 18, 'goals_against': 10
        },
        'recent_form': ['W', 'W', 'W', 'D', 'W']
    }
    
    analysis = data_analyzer.analyze_team_performance(team_data)
    print("Performance Trends:", analysis.get('performance_trends', 'N/A'))
    
    # Example 2: Prediction Explanation
    print("\n=== Prediction Explanation ===")
    prediction_data = {
        'home_team': 'Arsenal',
        'away_team': 'Chelsea',
        'predicted_outcome': 'Home Win',
        'confidence': 0.68,
        'probabilities': {'Home': 0.68, 'Draw': 0.22, 'Away': 0.10}
    }
    
    model_features = {
        'home_wins_last_5': 4,
        'away_losses_last_5': 3,
        'home_goals_avg': 2.1,
        'away_goals_avg': 1.3
    }
    
    explanations = prediction_explainer.explain_prediction(
        prediction_data, model_features, team_names=('Arsenal', 'Chelsea')
    )
    print("Basic Explanation:", explanations.get('basic_explanation', 'N/A'))
    
    # Example 3: Market Analysis
    print("\n=== Market Analysis ===")
    model_probs = {'Home': 0.65, 'Draw': 0.25, 'Away': 0.10}
    bookmaker_odds = {'Home': 1.8, 'Draw': 3.5, 'Away': 8.0}
    
    value_analysis = market_analyzer.analyze_value_betting_opportunity(
        model_probs, bookmaker_odds, ('Arsenal', 'Chelsea')
    )
    print("Value Analysis:", value_analysis.get('recommendations', []))
    
    # Example 4: Chat Assistant
    print("\n=== Chat Assistant ===")
    response = chat_assistant.chat(
        "What does a Random Forest model do in soccer prediction?",
        current_context={'model_type': 'Random Forest', 'accuracy': 0.78}
    )
    print("AI Response:", response)
    
    # Example 5: Report Generation
    print("\n=== Report Generation ===")
    weekly_data = {
        'matches_analyzed': 25,
        'predictions_made': 25,
        'correct_predictions': 19
    }
    
    model_performance = {
        'accuracy': 0.76,
        'precision': 0.78,
        'recall': 0.74
    }
    
    report = report_generator.generate_weekly_performance_report(
        weekly_data, model_performance, []
    )
    print("Weekly Report:", report[:200] + "..." if len(report) > 200 else report)

if __name__ == "__main__":
    main()
