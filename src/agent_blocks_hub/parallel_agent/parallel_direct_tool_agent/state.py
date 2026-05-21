"""
State definition for the Parallel Direct Tool Agent.

Defines the state structure for parallel execution of direct tool calls
with optional summarization. This agent bypasses LLM tool calling and
directly invokes tool.run() methods.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.tools import BaseTool


class ToolResult(TypedDict, total=False):
    """Result from a single direct tool call."""
    query_index: int
    query: str
    result: str
    tool_name: str
    success: bool
    error: Optional[str]
    remark: Optional[str]  # Optional LLM-generated remark for this result


class ParallelDirectToolAgentState(TypedDict, total=False):
    """
    State for the Parallel Direct Tool Agent system.
    
    Manages parallel execution of direct tool calls with optional result summarization.
    Unlike parallel_tool_agent, this bypasses LLM tool calling and directly invokes
    tool.run() methods, making it suitable for models with poor tool calling support.
    
    The interface is identical to ParallelToolAgentState for drop-in compatibility.
    """
    # Input queries
    parallel_agent_message: List[str]  # List of queries to process in parallel
    
    # Configuration
    llm: Any  # Language model instance (only used for summarization if enabled)
    tools: List[BaseTool]  # Tools available for execution
    system_prompt: str  # User-provided system prompt (used in summarization)
    verbose: bool  # Whether to print detailed execution logs
    enable_summarization: bool  # Whether to run summarization step (default: False)
    enable_remark: bool  # Whether to generate LLM remarks for individual results (default: False)
    remark_prompt: Optional[str]  # Custom prompt for remark generation
    summarization_prompt: Optional[str]  # Custom prompt for summarization
    
    # Tool selection (optional)
    tool_name: Optional[str]  # Specific tool to use (if None, uses first available tool)
    
    # Parallel execution results
    tool_results: Dict[int, ToolResult]  # {query_index: result}
    
    # Final summary (only populated if enable_summarization=True)
    final_summary: str
