"""
State definition for the Parallel Tool Agent.

Defines the state structure for parallel execution of tool calls
with optional summarization.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.tools import BaseTool


class ToolResult(TypedDict, total=False):
    """Result from a single tool call."""
    query_index: int
    query: str
    result: str
    tool_name: str
    success: bool
    error: Optional[str]


class ParallelToolAgentState(TypedDict, total=False):
    """
    State for the Parallel Tool Agent system.
    
    Manages parallel execution of tool calls with optional result summarization.
    This is a lightweight alternative to parallel_react_agent that directly
    calls tools without ReAct reasoning loops.
    """
    # Input queries
    parallel_tool_agent_messages: List[str]  # List of queries to process in parallel
    
    # Configuration
    llm: Any  # Language model instance (only used for summarization if enabled)
    tools: List[BaseTool]  # Tools available for execution
    system_prompt: str  # User-provided system prompt (used in summarization)
    verbose: bool  # Whether to print detailed execution logs
    enable_summarization: bool  # Whether to run summarization step (default: False)
    
    # Tool selection (optional)
    tool_name: Optional[str]  # Specific tool to use (if None, uses first available tool)
    
    # Parallel execution results
    tool_results: Dict[int, ToolResult]  # {query_index: result}
    
    # Final summary (only populated if enable_summarization=True)
    final_summary: str
