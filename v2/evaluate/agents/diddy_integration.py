"""
Diddy Agent Integration for Leaderboard v2
Wrapper to match LLMAgent interface
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from diddy_agent import DiddyAgent as _DiddyAgent
from typing import List, Dict, Any


class DiddyAgent:
    """Leaderboard-compatible Diddy agent wrapper"""
    
    def __init__(
        self,
        model_name: str = "diddy",
        domain: str = "",
        category: str = "",
        galaxy_logger=None,
        verbose: bool = False,
        history_manager=None,
        **kwargs
    ):
        """Initialize Diddy agent for leaderboard"""
        self.model_name = model_name
        self.domain = domain
        self.category = category
        self.verbose = verbose
        self.history_manager = history_manager
        
        # Initialize core Diddy agent
        self.agent = _DiddyAgent(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.0,
            max_tokens=4000,
            verbose=verbose
        )
        
        # Track metrics
        self.num_input_tokens = 0
        self.num_output_tokens = 0
        self.total_tokens = 0
        self.total_duration = 0
    
    def process_turn(
        self,
        conversation_history: List[Dict[str, str]],
        available_tools: List[Dict[str, Any]],
        user_message: str,
    ) -> Dict[str, Any]:
        """
        Process one turn - matches leaderboard interface
        
        Returns:
            Dict with agent_response, tool_calls, and metadata
        """
        response_text, tool_calls, metadata = self.agent.process_turn(
            conversation_history,
            available_tools,
            user_message,
            domain=self.domain,
            category=self.category,
        )
        
        # Update token metrics
        self.num_input_tokens += metadata.get("input_tokens", 0)
        self.num_output_tokens += metadata.get("output_tokens", 0)
        self.total_tokens += metadata.get("input_tokens", 0) + metadata.get("output_tokens", 0)
        
        return {
            "agent_response": response_text,
            "tool_calls": tool_calls,
            "metadata": metadata,
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics for leaderboard tracking"""
        return {
            "input_tokens": self.num_input_tokens,
            "output_tokens": self.num_output_tokens,
            "total_tokens": self.total_tokens,
        }


# Factory for leaderboard
def create_agent(**kwargs) -> DiddyAgent:
    return DiddyAgent(**kwargs)
