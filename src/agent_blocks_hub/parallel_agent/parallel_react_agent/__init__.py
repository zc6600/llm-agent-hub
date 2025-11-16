"""
Parallel React Agent - A system for parallel multi-query research with result synthesis.

This module implements a parallel execution framework where:
1. Multiple ReAct agents process different queries in parallel
2. Each agent uses the same tools and system prompt
3. A Summarizing Agent integrates results from all queries into a coherent summary

The system follows the LangGraph architecture pattern used in deep_diver.

Basic Usage:
    >>> from langchain_openai import ChatOpenAI
    >>> from langchain_community.tools import DuckDuckGoSearchRun
    >>> from agent_blocks_hub.parallel_react_agent import create_parallel_react_agent
    >>> 
    >>> llm = ChatOpenAI(model="gpt-4")
    >>> tools = [DuckDuckGoSearchRun()]
    >>> 
    >>> agent = create_parallel_react_agent(
    ...     llm=llm,
    ...     tools=tools,
    ...     system_prompt="Be thorough and evidence-based"
    ... )
    >>> 
    >>> result = agent.invoke({
    ...     "parallel_react_agent_messages": [
    ...         "What is LangGraph?",
    ...         "What is ReAct?",
    ...     ]
    ... })
    >>> 
    >>> print(result["final_summary"])
"""

from .agent import create_parallel_react_agent, get_compiled_graph
from .state import ParallelReactAgentState, AgentResult
from .prompts import get_combined_system_prompt

__all__ = [
    "create_parallel_react_agent",
    "get_compiled_graph",
    "ParallelReactAgentState",
    "AgentResult",
    "get_combined_system_prompt",
]
