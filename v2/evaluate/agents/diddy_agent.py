#!/usr/bin/env python3
"""
Diddy Agent for Agent Leaderboard v2 Submission

A high-performance AI agent that:
- Selects tools intelligently (high TSQ)
- Completes tasks end-to-end (high AC)
- Reasons about tool dependencies
- Handles multi-turn conversations
"""

import json
import time
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
import anthropic

# Fix 4: Import DOMAIN_SPECIFIC_INSTRUCTIONS from config instead of duplicating
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DOMAIN_SPECIFIC_INSTRUCTIONS


class DiddyAgent:
    """Diddy Agent - Optimized for task completion and tool selection"""
    
    def __init__(
        self,
        api_key: str = None,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        verbose: bool = False,
    ):
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.model = "claude-3-5-sonnet-20241022"
        
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
            conversation_history: List of previous messages (already includes all turns)
            available_tools: Tool definitions
            current_user_message: Current user input
            domain: Domain context (banking, healthcare, etc.)
            category: Task category

        Returns:
            (response_text, tool_calls, metadata)
        """
        tools_def = self._format_tools_for_claude(available_tools)
        system_prompt = self._build_system_prompt(domain, category)

        # Fix 2: Do NOT append current_user_message if callers already include it
        # in conversation_history. Build messages only from history.
        messages = []
        for msg in conversation_history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        # Only append current message if it's not already the last entry
        if not messages or messages[-1].get("content") != current_user_message:
            messages.append({
                "role": "user",
                "content": current_user_message
            })

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            tools=tools_def,
            messages=messages,
        )

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens
        self.total_calls += 1

        # Fix 3: Accumulate text across all blocks instead of overwriting
        response_text_parts = []
        tool_calls = []

        for block in response.content:
            if hasattr(block, 'text'):
                response_text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "tool_name": block.name,
                    "parameters": block.input,
                    "tool_use_id": block.id,
                })

        response_text = "".join(response_text_parts)

        metadata = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "stop_reason": response.stop_reason,
            "tool_calls_count": len(tool_calls),
        }

        if self.verbose:
            print(f"[Diddy] {len(tool_calls)} tool(s) selected | stop_reason={response.stop_reason}")
            print(f"  Response: {response_text[:120]}...")
            if tool_calls:
                print(f"  Tools: {[t['tool_name'] for t in tool_calls]}")

        return response_text, tool_calls, metadata

    def _format_tools_for_claude(self, tools: List[Dict]) -> List[Dict]:
        """Convert tool definitions to Claude format"""
        claude_tools = []
        for tool in tools:
            claude_tools.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": {
                    "type": "object",
                    "properties": tool.get("parameters", {}).get("properties", {}),
                    "required": tool.get("parameters", {}).get("required", []),
                }
            })
        return claude_tools

    def _build_system_prompt(self, domain: str, category: str) -> str:
        """Build context-specific system prompt using shared config"""
        # Fix 4: Use imported DOMAIN_SPECIFIC_INSTRUCTIONS instead of duplicate dict
        base_prompt = DOMAIN_SPECIFIC_INSTRUCTIONS.get(
            domain.lower(),
            "You are a helpful assistant. Use available tools to complete tasks effectively."
        )
        task_guidance = f"\nTask Category: {category}" if category else ""

        return base_prompt + task_guidance + """

IMPORTANT:
- Be direct and action-oriented
- Select tools based on actual need, not guessing
- If uncertain about tool selection, ask clarifying questions
- Always prefer completing the task over explaining how you would do it
- Multiple tool calls in one turn are OK if needed
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


def create_diddy_agent(**kwargs) -> DiddyAgent:
    """Factory function for leaderboard integration"""
    return DiddyAgent(**kwargs)


if __name__ == "__main__":
    agent = DiddyAgent(verbose=True)

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

    response, tools, meta = agent.process_turn(
        [],
        test_tools,
        "What's my balance on account ACC-001?",
        domain="banking"
    )

    print(f"\n✓ Response: {response}")
    print(f"✓ Tool calls: {len(tools)}")
    print(f"✓ Metrics: {agent.get_metrics()}")
