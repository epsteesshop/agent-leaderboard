#!/usr/bin/env python3
"""
Diddy Agent for Agent Leaderboard v2 Submission

A high-performance AI agent that:
- Selects tools intelligently (high TSQ)
- Completes tasks end-to-end (high AC)
- Reason about tool dependencies
- Handles multi-turn conversations
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
import anthropic

class DiddyAgent:
    """Diddy Agent - Optimized for task completion and tool selection"""
    
    def __init__(
        self,
        api_key: str = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        verbose: bool = False,
    ):
        import os
        # Use provided key or fall back to environment
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.model = "claude-3-5-sonnet-20241022"  # High performance
        
        # Metrics tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.start_time = time.time()
    
    def process_turn(
        self,
        conversation_history: List[Dict[str, str]],
        available_tools: List[Dict[str, Any]],
        current_user_message: str,
        domain: str = "",
        category: str = "",
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a single turn of conversation.
        
        Args:
            conversation_history: List of previous messages
            available_tools: Tool definitions
            current_user_message: Current user input
            domain: Domain context (banking, healthcare, etc.)
            category: Task category
        
        Returns:
            (response_text, tool_calls, metadata)
        """
        # Build tool definitions for Claude
        tools_def = self._format_tools_for_claude(available_tools)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(domain, category)
        
        # Convert conversation history to API format
        messages = []
        for msg in conversation_history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": current_user_message
        })
        
        # Call Claude with tools
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            tools=tools_def,
            messages=messages,
        )
        
        # Track tokens
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        self.total_calls += 1
        
        # Parse response
        response_text = ""
        tool_calls = []
        
        for block in response.content:
            if hasattr(block, 'text'):
                response_text = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "tool_name": block.name,
                    "parameters": block.input,
                    "tool_use_id": block.id,
                })
        
        metadata = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "stop_reason": response.stop_reason,
            "tool_calls_count": len(tool_calls),
        }
        
        if self.verbose:
            print(f"[Diddy] TSQ: {len(tool_calls)} tools selected | AC: Responding...")
            print(f"  Response: {response_text[:100]}...")
            if tool_calls:
                print(f"  Tools: {[t['tool_name'] for t in tool_calls]}")
        
        return response_text, tool_calls, metadata
    
    def _format_tools_for_claude(self, tools: List[Dict]) -> List[Dict]:
        """Convert tool definitions to Claude format"""
        claude_tools = []
        
        for tool in tools:
            claude_tool = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": {
                    "type": "object",
                    "properties": tool.get("parameters", {}).get("properties", {}),
                    "required": tool.get("parameters", {}).get("required", []),
                }
            }
            claude_tools.append(claude_tool)
        
        return claude_tools
    
    def _build_system_prompt(self, domain: str, category: str) -> str:
        """Build context-specific system prompt"""
        
        domain_prompts = {
            "banking": """You are a Banking Assistant. Your job is to:
1. Understand customer banking needs (transfers, balance checks, account management)
2. Select the RIGHT tools to complete their request
3. Provide accurate, helpful responses
4. Confirm actions before executing
Use available tools directly. Don't just provide guidance.""",
            
            "healthcare": """You are a Healthcare Assistant. Your job is to:
1. Manage patient records and appointments
2. Access health information accurately
3. Select appropriate healthcare tools
4. Maintain patient confidentiality
Use tools to complete healthcare tasks directly.""",
            
            "investment": """You are an Investment Assistant. Your job is to:
1. Help with portfolio management
2. Execute investment operations
3. Research investment options
4. Track performance
Use tools to complete investment operations directly.""",
            
            "telecom": """You are a Telecommunications Assistant. Your job is to:
1. Troubleshoot service issues
2. Manage account and services
3. Process service changes
4. Diagnose technical problems
Use tools directly to resolve customer issues.""",
        }
        
        base_prompt = domain_prompts.get(domain.lower(), 
            "You are a helpful assistant. Use available tools to complete tasks effectively.")
        
        task_guidance = f"\nTask Category: {category}" if category else ""
        
        return base_prompt + task_guidance + """

IMPORTANT:
- Be direct and action-oriented
- Select tools based on actual need, not guessing
- If uncertain about tool selection, ask clarifying questions
- Always prefer completing the task over explaining how you would do it
- Multiple tool calls in one turn are OK if they're needed
- Explain your tool selections briefly"""
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        elapsed = time.time() - self.start_time
        
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_tokens_per_call": (
                (self.total_input_tokens + self.total_output_tokens) / self.total_calls
                if self.total_calls > 0 else 0
            ),
            "elapsed_seconds": elapsed,
            "calls_per_second": self.total_calls / elapsed if elapsed > 0 else 0,
        }


# Integration function for leaderboard eval framework
def create_diddy_agent(**kwargs) -> DiddyAgent:
    """Factory function for leaderboard integration"""
    return DiddyAgent(**kwargs)


if __name__ == "__main__":
    # Quick test
    import os
    
    agent = DiddyAgent(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        verbose=True
    )
    
    # Test tools
    test_tools = [
        {
            "name": "check_balance",
            "description": "Check account balance",
            "parameters": {
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"]
            }
        },
        {
            "name": "transfer_funds",
            "description": "Transfer money between accounts",
            "parameters": {
                "properties": {
                    "from_account": {"type": "string"},
                    "to_account": {"type": "string"},
                    "amount": {"type": "number"}
                },
                "required": ["from_account", "to_account", "amount"]
            }
        }
    ]
    
    # Test conversation
    history = []
    response, tools, meta = agent.process_turn(
        history,
        test_tools,
        "What's my balance?",
        domain="banking"
    )
    
    print(f"\n✓ Agent response: {response}")
    print(f"✓ Tool calls: {len(tools)}")
    print(f"✓ Metrics: {agent.get_metrics()}")
