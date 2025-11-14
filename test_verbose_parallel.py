#!/usr/bin/env python3
"""
Test script to verify verbose logging in parallel react agent.
"""

import sys
sys.path.insert(0, '/Users/frank/Desktop/Towards\ AGI/llm-tool-hub/src')

from llm_provider import get_llm
from agent_blocks_hub.parallel_react_agent.agent import create_parallel_react_agent

# Try to import DuckDuckGo search
try:
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError:
    try:
        from langchain.tools import DuckDuckGoSearchRun
    except ImportError:
        # Fallback: create a dummy tool for testing
        from langchain_core.tools import BaseTool
        class DummySearchTool(BaseTool):
            name = "search"
            description = "Dummy search tool for testing"
            def _run(self, query: str) -> str:
                return f"Search results for: {query}"
        DuckDuckGoSearchRun = DummySearchTool

# Initialize components
llm = get_llm(enable_langsmith=False)
tools = [DuckDuckGoSearchRun()]

print("=" * 60)
print("Testing Parallel React Agent with verbose=True")
print("=" * 60)

# Test 1: Create agent with verbose=True
agent = create_parallel_react_agent(
    llm=llm,
    tools=tools,
    system_prompt="You are a helpful research assistant.",
    verbose=True  # VERBOSE MODE ENABLED
)

# Test 2: Invoke with some queries
test_queries = [
    "What is machine learning?",
    "What is deep learning?",
]

result = agent.invoke({
    "parallel_react_agent_messages": test_queries,
    "verbose": True,  # Also pass verbose in state
})

print("\n" + "=" * 60)
print("Test completed. Check output above for detailed verbose logs.")
print("=" * 60)
print(f"\nFinal summary length: {len(result.get('final_summary', ''))}")
