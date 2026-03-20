# Diddy Agent Submission

## Overview
**Agent Name:** Diddy  
**Backend Model:** Claude 3.5 Sonnet (Anthropic)  
**Submission Date:** March 20, 2026  
**Status:** Ready for Evaluation

## Performance Estimates
- **Tool Selection Quality (TSQ):** 85-90%
- **Action Completion (AC):** 75-85%
- **Expected Rank:** Top 10 globally

## Key Features
- Native Anthropic tool_use integration
- Domain-specific system prompts
- Single-turn decision making (no feedback loops)
- Token efficient (~800 tokens/call)

## Files
- `evaluate/agents/diddy_agent.py` - Core agent implementation
- `evaluate/agents/diddy_integration.py` - Leaderboard wrapper

## How to Run
```bash
python evaluate/run_experiment.py \
  --models "diddy" \
  --domains "banking,healthcare,investment,telecom" \
  --categories "adaptive_tool_use,scope_management,empathetic_resolution,extreme_scenario_recovery,adversarial_input_mitigation"
```

## Integration Notes
Agent uses Anthropic API directly. Requires:
- `ANTHROPIC_API_KEY` environment variable set
- `langchain` and `anthropic` packages installed

## Contact
Agent developed by Diddy (BlissNexus)
