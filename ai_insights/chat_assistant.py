"""
AI-Powered Chat Assistant for FootyForecast

This module provides conversational AI interface for users to interact
with their data, get explanations, and receive guidance.
"""

from typing import Dict, List, Any, Optional, Tuple
import json
from .openai_client import OpenAIClient

class ChatAssistant:
    """
    Conversational AI assistant for FootyForecast users
    """
    
    def __init__(self, openai_client: OpenAIClient):
        """
        Initialize chat assistant
        
        Args:
            openai_client: OpenAI client instance
        """
        self.ai_client = openai_client
        self.conversation_history = []
        self.context_data = {}
    
    def chat(self, 
             user_message: str,
             current_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process user message and generate response
        
        Args:
            user_message: User's message/question
            current_context: Current application context (loaded data, etc.)
            
        Returns:
            AI assistant response
        """
        # Update context if provided
        if current_context:
            self.context_data.update(current_context)
        
        # Generate response
        response = self.ai_client.chat_response(
            user_message=user_message,
            conversation_history=self.conversation_history[-10:],  # Keep last 10 exchanges
            context_data=self.context_data
        )
        
        # Update conversation history
        self.conversation_history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ])
        
        return response
    
    def ask_about_data(self, 
                      question: str,
                      data_context: Dict[str, Any]) -> str:
        """
        Ask specific questions about loaded data
        
        Args:
            question: Question about the data
            data_context: Relevant data context
            
        Returns:
            Data-specific answer
        """
        enhanced_question = f"""
        Based on the current data context, please answer this question: {question}
        
        Available data: {json.dumps(data_context, indent=2)}
        """
        
        return self.ai_client.generate_completion(
            prompt=enhanced_question,
            system_message="You are a helpful data analyst assistant. Answer questions about soccer data clearly and accurately.",
            temperature=0.3
        )
    
    def explain_concept(self, 
                       concept: str,
                       user_level: str = "beginner") -> str:
        """
        Explain soccer analytics or machine learning concepts
        
        Args:
            concept: Concept to explain
            user_level: User's knowledge level (beginner, intermediate, advanced)
            
        Returns:
            Concept explanation
        """
        explanation_prompt = f"""
        Please explain the concept of "{concept}" in soccer analytics/machine learning.
        
        Target audience: {user_level} level
        
        Please provide:
        1. Clear definition
        2. Why it's important in soccer analysis
        3. How it's used in FootyForecast
        4. Practical example if applicable
        """
        
        return self.ai_client.generate_completion(
            prompt=explanation_prompt,
            system_message="You are an expert teacher in soccer analytics and machine learning. Explain concepts clearly for the specified audience level.",
            temperature=0.4
        )
    
    def get_prediction_help(self, 
                           prediction_data: Dict[str, Any],
                           user_question: str) -> str:
        """
        Help user understand predictions
        
        Args:
            prediction_data: Current prediction information
            user_question: Specific question about the prediction
            
        Returns:
            Prediction-specific help
        """
        help_prompt = f"""
        User question about prediction: {user_question}
        
        Prediction data: {json.dumps(prediction_data, indent=2)}
        
        Please provide helpful guidance about this prediction.
        """
        
        return self.ai_client.generate_completion(
            prompt=help_prompt,
            system_message="You are a helpful prediction analyst. Help users understand their predictions and make informed decisions.",
            temperature=0.5
        )
    
    def suggest_analysis(self, 
                        available_data: Dict[str, Any],
                        user_goal: str) -> str:
        """
        Suggest analysis approaches based on user goals
        
        Args:
            available_data: Data available for analysis
            user_goal: What the user wants to achieve
            
        Returns:
            Analysis suggestions
        """
        suggestion_prompt = f"""
        User goal: {user_goal}
        
        Available data: {json.dumps(available_data, indent=2)}
        
        Please suggest specific analysis approaches, features to use, and steps to take to achieve this goal.
        """
        
        return self.ai_client.generate_completion(
            prompt=suggestion_prompt,
            system_message="You are an expert data analyst. Suggest practical analysis approaches based on available data and user goals.",
            temperature=0.6
        )
    
    def troubleshoot_issue(self, 
                          issue_description: str,
                          error_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Help troubleshoot technical issues
        
        Args:
            issue_description: Description of the problem
            error_context: Any error messages or context
            
        Returns:
            Troubleshooting guidance
        """
        troubleshoot_prompt = f"""
        Issue: {issue_description}
        
        {f'Error context: {json.dumps(error_context, indent=2)}' if error_context else ''}
        
        Please provide step-by-step troubleshooting guidance.
        """
        
        return self.ai_client.generate_completion(
            prompt=troubleshoot_prompt,
            system_message="You are a helpful technical support assistant for FootyForecast. Provide clear troubleshooting steps.",
            temperature=0.3
        )
    
    def recommend_models(self, 
                        data_characteristics: Dict[str, Any],
                        prediction_goals: List[str]) -> str:
        """
        Recommend appropriate ML models based on data and goals
        
        Args:
            data_characteristics: Characteristics of the available data
            prediction_goals: What user wants to predict
            
        Returns:
            Model recommendations
        """
        recommendation_prompt = f"""
        Data characteristics: {json.dumps(data_characteristics, indent=2)}
        
        Prediction goals: {prediction_goals}
        
        Please recommend the most appropriate machine learning models and explain why they would work well for this scenario.
        """
        
        return self.ai_client.generate_completion(
            prompt=recommendation_prompt,
            system_message="You are an ML expert specializing in soccer prediction. Recommend appropriate models based on data and goals.",
            temperature=0.4
        )
    
    def generate_insights_summary(self, 
                                 session_data: Dict[str, Any]) -> str:
        """
        Generate summary of insights from current session
        
        Args:
            session_data: Data from current analysis session
            
        Returns:
            Insights summary
        """
        summary_prompt = f"""
        Session data: {json.dumps(session_data, indent=2)}
        
        Please provide a summary of key insights and findings from this analysis session.
        """
        
        return self.ai_client.generate_completion(
            prompt=summary_prompt,
            system_message="You are an expert analyst. Summarize key insights and findings from analysis sessions.",
            temperature=0.5
        )
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def set_context(self, context: Dict[str, Any]):
        """Set current context data"""
        self.context_data = context
    
    def get_conversation_summary(self) -> str:
        """
        Get summary of current conversation
        
        Returns:
            Conversation summary
        """
        if not self.conversation_history:
            return "No conversation history available."
        
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history[-10:]
        ])
        
        summary_prompt = f"""
        Conversation history:
        {conversation_text}
        
        Please provide a brief summary of this conversation, highlighting key topics discussed and any important conclusions.
        """
        
        return self.ai_client.generate_completion(
            prompt=summary_prompt,
            system_message="You are a helpful assistant. Summarize conversations concisely.",
            temperature=0.3
        )
