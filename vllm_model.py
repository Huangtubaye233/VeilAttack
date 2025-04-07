#!/usr/bin/env python3

# Standard library imports
import logging
import json
import os
from typing import Dict, Tuple, Optional
import re

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Third-party imports
from vllm import LLM, SamplingParams

class VLLMModel:
    """Class representing a model using vllm for inference"""
    
    def __init__(self, model_name: str, model_dir: Optional[str] = None):
        """Initialize the vllm model
        
        Args:
            model_name: Name of the model to load
            model_dir: Directory to cache the model (not used for vllm)
        """
        # Configure model loading options
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        # Initialize the model
        self.model = LLM(
            model=model_name,
            **model_kwargs
        )
        
        # Set default sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=200,
            repetition_penalty=1.2,
        )
    
    def query(self, user_input: str) -> Tuple[str, str]:
        """Query the model with a user input
        
        Args:
            user_input: The user's query
            
        Returns:
            Tuple[str, str]: The extracted answer and the full raw response
        """
        # Build the prompt
        prompt = self._build_prompt(user_input)
        
        try:
            # Generate response
            outputs = self.model.generate(prompt, self.sampling_params)
            
            # Get the first output
            output = outputs[0]
            full_response = output.outputs[0].text
            
            # Extract the answer
            answer = self._extract_answer(full_response)
            
            # Clean the answer
            answer = self._clean_response(answer)
            
            return answer, full_response
            
        except Exception as e:
            logger.error(f"Error querying model: {str(e)}")
            return f"Error: {str(e)}", f"Error: {str(e)}"
    
    def _build_prompt(self, user_input: str) -> str:
        """Build a prompt for the query
        
        Args:
            user_input: The current user query
            
        Returns:
            str: The complete prompt
        """
        # Start with very explicit system instruction
        prompt = """You are a helpful AI assistant. Follow these rules strictly:
1. Provide ONLY factual information related to the query
2. Keep responses under 50 words
3. Do not include hashtags, emojis, or unrelated content
4. Do not ask questions back to the user
5. Do not include any content about human resources, quantum mechanics, or other unrelated topics
6. Focus only on answering the specific question asked
7. Do not include any examples or placeholders in your response
8. Do not include any text like "Your response here" or similar placeholders

"""
        
        prompt += f"User: {user_input}\n\nAssistant: "
        
        return prompt
    
    def _extract_answer(self, response: str) -> str:
        """Extract the answer from the model's response
        
        Args:
            response: The full response from the model
            
        Returns:
            str: The extracted answer
        """
        # Try to extract the assistant's response
        if "Assistant:" in response:
            # Split by "Assistant:" and take the last part
            parts = response.split("Assistant:")
            return parts[-1].strip()
        
        # If no "Assistant:" prefix found, return the raw response
        return response.strip()
    
    def _clean_response(self, response: str) -> str:
        """Clean the response to remove unrelated content
        
        Args:
            response: The raw response to clean
            
        Returns:
            str: The cleaned response
        """
        # Remove any content after detecting certain patterns
        cutoff_patterns = [
            "Human:", "User:", "Question:", 
            "Human resources", "Quantum mechanics",
            "http", "www.", ".com", 
            "A)", "B)", "C)", "D)",
            "Your response here"
        ]
        
        for pattern in cutoff_patterns:
            if pattern in response:
                # Cut off at the pattern
                response = response.split(pattern)[0].strip()
        
        # Remove emojis and special characters
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
        response = emoji_pattern.sub(r'', response)
        
        # Remove hashtags
        response = re.sub(r'#\w+', '', response)
        
        # Remove excessive newlines
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Remove any trailing punctuation or spaces
        response = response.rstrip('.,;: \t\n')
        
        return response 