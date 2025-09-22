"""
OpenAI Client for FootyForecast AI Integration

This module handles all communication with OpenAI's API for generating
intelligent insights, explanations, and analysis.
"""

import openai
import os
from typing import Dict, List, Optional, Any
import json
import logging

class OpenAIClient:
    """
    Client for interacting with OpenAI's GPT models for soccer analysis
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
            model: GPT model to use (default: gpt-4)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        self.logger = logging.getLogger(__name__)
    
    def generate_completion(self, 
                          prompt: str, 
                          max_tokens: int = 1000,
                          temperature: float = 0.7,
                          system_message: Optional[str] = None) -> str:
        """
        Generate completion using OpenAI API
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens in response
            temperature: Creativity level (0-1)
            system_message: System instruction for the AI
            
        Returns:
            Generated text response
        """
        try:
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            messages.append({"role": "user", "content": prompt})
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return f"Error generating AI response: {str(e)}"
    
    def analyze_with_context(self, 
                           data: Dict[str, Any], 
                           analysis_type: str,
                           context: Optional[str] = None) -> str:
        """
        Analyze data with specific context and type
        
        Args:
            data: Data to analyze
            analysis_type: Type of analysis (team_performance, prediction, market, etc.)
            context: Additional context for analysis
            
        Returns:
            AI-generated analysis
        """
        system_messages = {
            "team_performance": "You are an expert soccer analyst specializing in team performance analysis.",
            "prediction": "You are a soccer prediction expert who explains model decisions clearly.",
            "market": "You are a sports betting analyst who understands market dynamics.",
            "education": "You are a patient teacher explaining soccer analytics concepts.",
            "report": "You are an executive analyst creating comprehensive reports."
        }
        
        system_msg = system_messages.get(analysis_type, 
                                       "You are a helpful soccer analytics assistant.")
        
        prompt = f"""
        Analysis Type: {analysis_type}
        
        Data: {json.dumps(data, indent=2)}
        
        {f'Additional Context: {context}' if context else ''}
        
        Please provide a comprehensive analysis based on this data.
        """
        
        return self.generate_completion(
            prompt=prompt,
            system_message=system_msg,
            temperature=0.6
        )
    
    def chat_response(self, 
                     user_message: str, 
                     conversation_history: List[Dict[str, str]] = None,
                     context_data: Optional[Dict] = None) -> str:
        """
        Generate conversational response with history
        
        Args:
            user_message: User's message
            conversation_history: Previous conversation messages
            context_data: Current data context
            
        Returns:
            AI response
        """
        system_msg = """
        You are FootyForecast AI Assistant, an expert in soccer analytics and machine learning.
        You help users understand their data, predictions, and provide insights about soccer matches.
        
        Be helpful, accurate, and explain complex concepts in simple terms.
        Always base your responses on the provided data when available.
        """
        
        messages = [{"role": "system", "content": system_msg}]
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add context data if available
        if context_data:
            context_prompt = f"Current data context: {json.dumps(context_data, indent=2)}\n\n{user_message}"
        else:
            context_prompt = user_message
            
        messages.append({"role": "user", "content": context_prompt})
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Chat response error: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."
