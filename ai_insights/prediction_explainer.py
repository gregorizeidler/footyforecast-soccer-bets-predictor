"""
AI-Powered Prediction Explainer for FootyForecast

This module provides intelligent explanations for model predictions,
helping users understand why certain outcomes were predicted.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from .openai_client import OpenAIClient
from .prompts.explanation_prompts import ExplanationPrompts

class PredictionExplainer:
    """
    AI-powered explainer for soccer match predictions
    """
    
    def __init__(self, openai_client: OpenAIClient):
        """
        Initialize prediction explainer
        
        Args:
            openai_client: OpenAI client instance
        """
        self.ai_client = openai_client
        self.prompts = ExplanationPrompts()
    
    def explain_prediction(self, 
                          prediction_result: Dict[str, Any],
                          model_features: Dict[str, float],
                          feature_importance: Dict[str, float] = None,
                          team_names: Tuple[str, str] = None) -> Dict[str, str]:
        """
        Generate comprehensive explanation for a prediction
        
        Args:
            prediction_result: Model prediction output
            model_features: Features used in prediction
            feature_importance: Importance scores for features
            team_names: (home_team, away_team) names
            
        Returns:
            Dictionary with different explanation types
        """
        explanations = {}
        
        # Basic prediction explanation
        explanations['basic_explanation'] = self._generate_basic_explanation(
            prediction_result, team_names
        )
        
        # Feature-based explanation
        explanations['feature_analysis'] = self._explain_key_features(
            model_features, feature_importance, team_names
        )
        
        # Confidence analysis
        explanations['confidence_analysis'] = self._analyze_prediction_confidence(
            prediction_result, model_features
        )
        
        # Risk factors
        explanations['risk_factors'] = self._identify_risk_factors(
            prediction_result, model_features, team_names
        )
        
        return explanations
    
    def explain_model_decision(self, 
                              model_name: str,
                              prediction_probabilities: Dict[str, float],
                              input_features: Dict[str, Any],
                              team_context: Dict[str, Any] = None) -> str:
        """
        Explain why a specific model made its decision
        
        Args:
            model_name: Name of the ML model
            prediction_probabilities: Probability distribution
            input_features: Input features used
            team_context: Additional team context
            
        Returns:
            Detailed model decision explanation
        """
        explanation_data = {
            'model': model_name,
            'probabilities': prediction_probabilities,
            'features': input_features,
            'context': team_context or {}
        }
        
        return self.ai_client.analyze_with_context(
            data=explanation_data,
            analysis_type="prediction",
            context=f"Explain why the {model_name} model made this specific prediction, focusing on the most influential factors"
        )
    
    def compare_model_predictions(self, 
                                 model_predictions: Dict[str, Dict[str, float]],
                                 team_names: Tuple[str, str]) -> str:
        """
        Compare and explain differences between multiple model predictions
        
        Args:
            model_predictions: Dictionary of model predictions
            team_names: (home_team, away_team) names
            
        Returns:
            Comparison explanation
        """
        comparison_data = {
            'home_team': team_names[0],
            'away_team': team_names[1],
            'model_predictions': model_predictions
        }
        
        return self.ai_client.analyze_with_context(
            data=comparison_data,
            analysis_type="prediction",
            context="Compare different model predictions and explain why they might disagree, highlighting which model might be more reliable for this specific match"
        )
    
    def explain_prediction_uncertainty(self, 
                                     prediction_result: Dict[str, Any],
                                     model_confidence: float,
                                     historical_accuracy: Dict[str, float] = None) -> str:
        """
        Explain uncertainty and reliability of predictions
        
        Args:
            prediction_result: Prediction output
            model_confidence: Model confidence score
            historical_accuracy: Historical accuracy data
            
        Returns:
            Uncertainty explanation
        """
        uncertainty_data = {
            'prediction': prediction_result,
            'confidence': model_confidence,
            'historical_performance': historical_accuracy or {}
        }
        
        return self.ai_client.analyze_with_context(
            data=uncertainty_data,
            analysis_type="prediction",
            context="Explain the uncertainty in this prediction, what factors contribute to lower/higher confidence, and how reliable this prediction is likely to be"
        )
    
    def generate_prediction_scenarios(self, 
                                    base_prediction: Dict[str, Any],
                                    variable_features: List[str],
                                    team_names: Tuple[str, str]) -> str:
        """
        Generate 'what-if' scenarios for predictions
        
        Args:
            base_prediction: Base prediction result
            variable_features: Features that could change
            team_names: Team names
            
        Returns:
            Scenario analysis
        """
        scenario_data = {
            'base_prediction': base_prediction,
            'variable_factors': variable_features,
            'home_team': team_names[0],
            'away_team': team_names[1]
        }
        
        return self.ai_client.analyze_with_context(
            data=scenario_data,
            analysis_type="prediction",
            context="Generate 'what-if' scenarios showing how the prediction might change if key factors were different (e.g., if home team had better recent form, if key players were injured, etc.)"
        )
    
    def _generate_basic_explanation(self, 
                                  prediction_result: Dict[str, Any],
                                  team_names: Optional[Tuple[str, str]]) -> str:
        """
        Generate basic prediction explanation
        """
        home_team = team_names[0] if team_names else "Home Team"
        away_team = team_names[1] if team_names else "Away Team"
        
        explanation_data = {
            'home_team': home_team,
            'away_team': away_team,
            'prediction': prediction_result
        }
        
        return self.ai_client.analyze_with_context(
            data=explanation_data,
            analysis_type="prediction",
            context="Provide a clear, simple explanation of this prediction that anyone can understand"
        )
    
    def _explain_key_features(self, 
                            model_features: Dict[str, float],
                            feature_importance: Optional[Dict[str, float]],
                            team_names: Optional[Tuple[str, str]]) -> str:
        """
        Explain the most important features in the prediction
        """
        # Sort features by importance if available
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: abs(x[1]), 
                                   reverse=True)[:5]  # Top 5 features
        else:
            sorted_features = list(model_features.items())[:5]
        
        feature_data = {
            'key_features': sorted_features,
            'all_features': model_features,
            'team_names': team_names
        }
        
        return self.ai_client.analyze_with_context(
            data=feature_data,
            analysis_type="prediction",
            context="Explain how the most important features influenced this prediction, making it clear why each feature matters"
        )
    
    def _analyze_prediction_confidence(self, 
                                     prediction_result: Dict[str, Any],
                                     model_features: Dict[str, float]) -> str:
        """
        Analyze and explain prediction confidence
        """
        confidence_data = {
            'prediction': prediction_result,
            'features': model_features
        }
        
        return self.ai_client.analyze_with_context(
            data=confidence_data,
            analysis_type="prediction",
            context="Analyze the confidence level of this prediction and explain what makes it more or less certain"
        )
    
    def _identify_risk_factors(self, 
                             prediction_result: Dict[str, Any],
                             model_features: Dict[str, float],
                             team_names: Optional[Tuple[str, str]]) -> str:
        """
        Identify potential risk factors that could affect the prediction
        """
        risk_data = {
            'prediction': prediction_result,
            'features': model_features,
            'teams': team_names
        }
        
        return self.ai_client.analyze_with_context(
            data=risk_data,
            analysis_type="prediction",
            context="Identify potential risk factors or scenarios that could make this prediction less accurate, such as unexpected events or factors not captured in the data"
        )
