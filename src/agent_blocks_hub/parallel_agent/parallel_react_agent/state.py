"""
State definition for the Parallel React Agent.

Defines the state structure for parallel execution of multiple ReAct agents
with a summarizing agent to consolidate results.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.tools import BaseTool


class AgentResult(TypedDict, total=False):
    """Result from a single ReAct agent."""
    query_index: int
    query: str
    result: str
    intermediate_steps: List[Dict[str, Any]]  # (action, observation) pairs from ReAct loop
    success: bool
    error: Optional[str]


class ParallelReactAgentState(TypedDict, total=False):
    """
    State for the Parallel React Agent system.
    
    Manages parallel execution of multiple ReAct agents with centralized
    result summarization.
    """
    # Input queries
    parallel_react_agent_messages: List[str]  # List of queries to process in parallel
    
    # Configuration
    llm: Any  # Language model instance
    tools: List[BaseTool]  # Tools available to all agents
    system_prompt: str  # User-provided system prompt
    verbose: bool  # Whether to print detailed execution logs
    enable_summarization: bool  # Whether to run LLM-based summarization (default: True)
    
    # Parallel execution results
    agent_results: Dict[int, AgentResult]  # {query_index: result}
    
    # Final summary
    final_summary: str
