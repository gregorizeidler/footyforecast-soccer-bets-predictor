"""
Explanation prompt templates for AI-powered prediction explanations
"""

from typing import Dict, Any

class ExplanationPrompts:
    """
    Collection of prompt templates for prediction explanations
    """
    
    @staticmethod
    def get_prediction_explanation_prompt(prediction_data: Dict[str, Any]) -> str:
        """
        Generate prompt for prediction explanation
        
        Args:
            prediction_data: Prediction data and context
            
        Returns:
            Formatted prompt
        """
        return f"""
        Explain this soccer match prediction in clear, understandable terms:
        
        Prediction Data: {prediction_data}
        
        Please explain:
        1. What the prediction means
        2. Why this outcome was predicted
        3. Key factors that influenced the decision
        4. Confidence level and what it means
        5. Potential risks or uncertainties
        
        Make the explanation accessible to both beginners and experts.
        """
    
    @staticmethod
    def get_model_decision_prompt(model_info: Dict[str, Any]) -> str:
        """
        Generate prompt for model decision explanation
        """
        return f"""
        Explain why this machine learning model made this specific decision:
        
        Model Information: {model_info}
        
        Focus on:
        1. How this type of model works
        2. Which features were most important
        3. Why these features led to this prediction
        4. Model limitations and assumptions
        5. When this model performs best/worst
        """
    
    @staticmethod
    def get_feature_importance_prompt(features_data: Dict[str, Any]) -> str:
        """
        Generate prompt for feature importance explanation
        """
        return f"""
        Explain the importance of these features in soccer prediction:
        
        Features: {features_data}
        
        For each important feature, explain:
        1. What it measures
        2. Why it's predictive
        3. How it influenced this specific prediction
        4. Typical ranges and what they mean
        5. Limitations or caveats
        """
