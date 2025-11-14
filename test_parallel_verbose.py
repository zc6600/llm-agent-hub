#!/usr/bin/env python3
"""
Quick test to verify verbose logging in parallel react agent.
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from llm_provider import get_llm
from langchain_community.tools import DuckDuckGoSearchRun
from agent_blocks_hub.parallel_react_agent import create_parallel_react_agent

print("=" * 80)
print("Testing Parallel React Agent with verbose=True")
print("=" * 80)

# Initialize LLM and tools
llm = get_llm(temperature=0.7, max_tokens=2000, enable_langsmith=False)
tools = [DuckDuckGoSearchRun()]

# Create agent with verbose=True
agent = create_parallel_react_agent(
    llm=llm,
    tools=tools,
    system_prompt="You are a helpful research assistant.",
    verbose=True
)

# Test with 2 simple queries
test_queries = [
    "What is Python?",
    "What is JavaScript?",
]

print("\nInvoking agent with verbose=True in state...")
result = agent.invoke({
    "parallel_react_agent_messages": test_queries,
    "verbose": True,  # Pass verbose in state
})

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)
print(f"\nFinal summary length: {len(result.get('final_summary', ''))}")
print("\nIf you see detailed agent logs above, the fix worked!")
