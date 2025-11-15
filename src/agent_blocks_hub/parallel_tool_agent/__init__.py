"""
Parallel Tool Agent - A lightweight system for parallel tool execution with optional summarization.

This module implements a fast parallel execution framework where:
1. Multiple tool calls are executed in parallel (no ReAct reasoning)
2. Each tool processes a different query with the same configuration
3. Optional LLM-based summarization can synthesize results

This is a lightweight alternative to parallel_react_agent, optimized for
simple information gathering tasks like paper searches.

Key Advantages:
- Much faster than parallel_react_agent (no reasoning loops)
- Optional summarization (can be disabled for maximum speed)
- Direct tool invocation
- Simpler state management

Basic Usage (Fast Mode - No Summarization):
    >>> from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
    >>> from agent_blocks_hub.parallel_tool_agent import create_parallel_tool_agent
    >>> 
    >>> # Create agent without LLM for maximum speed
    >>> agent = create_parallel_tool_agent(
    ...     tools=[SearchSemanticScholar()],
    ...     enable_summarization=False,  # Fastest mode
    ...     verbose=True
    ... )
    >>> 
    >>> result = agent.invoke({
    ...     "parallel_tool_agent_messages": [
    ...         "transformer neural networks",
    ...         "attention mechanism",
    ...     ]
    ... })
    >>> 
    >>> # Access individual results
    >>> for idx, res in result["tool_results"].items():
    ...     print(f"Query {idx}: {res['result']}")

Usage with Summarization:
    >>> from langchain_openai import ChatOpenAI
    >>> from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
    >>> from agent_blocks_hub.parallel_tool_agent import create_parallel_tool_agent
    >>> 
    >>> llm = ChatOpenAI(model="gpt-4")
    >>> agent = create_parallel_tool_agent(
    ...     llm=llm,
    ...     tools=[SearchSemanticScholar()],
    ...     enable_summarization=True,  # Enable intelligent summarization
    ...     system_prompt="Focus on recent breakthroughs"
    ... )
    >>> 
    >>> result = agent.invoke({
    ...     "parallel_tool_agent_messages": [
    ...         "transformer neural networks",
    ...         "attention mechanism",
    ...     ]
    ... })
    >>> 
    >>> print(result["final_summary"])  # Coherent, synthesized summary
"""

from .agent import create_parallel_tool_agent, get_compiled_graph
from .state import ParallelToolAgentState, ToolResult

__all__ = [
    "create_parallel_tool_agent",
    "get_compiled_graph",
    "ParallelToolAgentState",
    "ToolResult",
]
