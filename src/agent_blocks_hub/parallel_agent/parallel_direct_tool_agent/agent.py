"""
Parallel Direct Tool Agent creation and configuration.

Creates a LangGraph-based agent that directly executes tool.run() calls in parallel
WITHOUT LLM tool calling. This is useful for models with poor tool calling support
(e.g., DeepSeek V3) while maintaining the same interface as parallel_tool_agent.
"""

from typing import List, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .state import ParallelDirectToolAgentState
from .nodes import (
    initialize_state,
    run_parallel_direct_tools,
    summarize_results,
)


def create_parallel_direct_tool_agent(
    llm: Optional[BaseChatModel] = None,
    tools: List[BaseTool] = None,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    enable_summarization: bool = False,
    enable_remark: bool = False,
    remark_prompt: Optional[str] = None,
    summarization_prompt: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> Any:
    """
    Create a Parallel Direct Tool Agent using LangGraph.
    
    This agent DIRECTLY calls tool.run() in parallel, bypassing LLM tool calling entirely.
    This is useful for models with poor tool calling support (e.g., DeepSeek V3).
    
    Key differences from parallel_tool_agent:
    - Direct tool.run() invocation (NO llm.bind_tools)
    - No LLM involvement in tool selection/execution
    - LLM only used for optional summarization
    - Same interface and output format for drop-in compatibility
    
    Args:
        llm: Language model (only required if enable_summarization=True)
        tools: List of tools available for execution
        system_prompt: Optional user-provided system prompt for summarization
        verbose: Whether to print detailed execution logs (default: False)
        enable_summarization: Whether to run LLM summarization (default: False)
        tool_name: Specific tool to use (if None, uses first tool)
    
    Returns:
        Compiled LangGraph agent ready to invoke
        
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
        >>> 
        >>> # Fast mode: No summarization or remarks, direct tool calls only
        >>> agent = create_parallel_direct_tool_agent(
        ...     tools=[SearchSemanticScholar()],
        ...     enable_summarization=False,
        ...     enable_remark=False,  # No remarks
        ...     verbose=True
        ... )
        >>> 
        >>> result = agent.invoke({
        ...     "parallel_agent_message": [
        ...         "transformer neural networks",
        ...         "attention mechanism deep learning",
        ...     ]
        ... })
        >>> 
        >>> # Access individual tool results
        >>> for idx, tool_result in result["tool_results"].items():
        ...     print(f"Query {idx}: {tool_result['result']}")
        >>>
        >>> # Mode with remarks: Add LLM-generated remarks to individual results
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent_with_remarks = create_parallel_direct_tool_agent(
        ...     llm=llm,
        ...     tools=[SearchSemanticScholar()],
        ...     enable_summarization=False,
        ...     enable_remark=True,  # Enable remarks for each result
        ...     remark_prompt="Custom remark prompt: {result}",  # Optional custom prompt
        ...     verbose=True
        ... )
        >>> 
        >>> result = agent_with_remarks.invoke({
        ...     "parallel_agent_message": [
        ...         "transformer neural networks",
        ...         "attention mechanism deep learning",
        ...     ]
        ... })
        >>> 
        >>> # Access results with remarks
        >>> for idx, tool_result in result["tool_results"].items():
        ...     print(f"Query {idx}: {tool_result['result']}\nRemark: {tool_result['remark']}")
        >>>
        >>> # Full mode: Summarization with optional remarks
        >>> agent_full = create_parallel_direct_tool_agent(
        ...     llm=llm,
        ...     tools=[SearchSemanticScholar()],
        ...     enable_summarization=True,
        ...     enable_remark=True,
        ...     summarization_prompt="Custom summary prompt: {combined_results}",  # Optional
        ...     verbose=True
        ... )
        >>> 
        >>> result = agent_full.invoke({
        ...     "parallel_agent_message": [
        ...         "transformer neural networks",
        ...         "attention mechanism deep learning",
        ...     ]
        ... })
        >>> 
        >>> print(result["final_summary"])  # Synthesized summary including remarks
    """
    if tools is None:
        tools = []
    
    if system_prompt is None:
        system_prompt = ""
    
    if enable_summarization and llm is None:
        raise ValueError("LLM is required when enable_summarization=True")
    if enable_remark and llm is None:
        raise ValueError("LLM is required when enable_remark=True")
    
    # Create and return compiled graph
    graph = _create_graph(llm, tools, system_prompt, verbose, enable_summarization, enable_remark, remark_prompt, summarization_prompt, tool_name)
    return graph.compile()


def _create_graph(
    llm: Optional[BaseChatModel],
    tools: List[BaseTool],
    system_prompt: str,
    verbose: bool = False,
    enable_summarization: bool = False,
    enable_remark: bool = False,
    remark_prompt: Optional[str] = None,
    summarization_prompt: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> StateGraph:
    """
    Create the LangGraph state graph for the Parallel Direct Tool Agent.
    
    Args:
        llm: Language model (only used for summarization)
        tools: Available tools for direct execution
        system_prompt: User-provided system prompt
        verbose: Whether to print intermediate progress
        enable_summarization: Whether to run summarization step
        tool_name: Specific tool to use
        
    Returns:
        Configured StateGraph
    """
    # Create state graph
    workflow = StateGraph(ParallelDirectToolAgentState)
    
    # Define initialization node wrapper
    def init_node(state: ParallelDirectToolAgentState) -> Dict[str, Any]:
        state_verbose = state.get("verbose", verbose)
        state_enable_summarization = state.get("enable_summarization", enable_summarization)
        state_enable_remark = state.get("enable_remark", enable_remark)
        base_state = initialize_state(state)
        result = {
            **base_state,
            "llm": llm,
            "tools": tools,
            "system_prompt": system_prompt,
            "verbose": state_verbose,
            "enable_summarization": state_enable_summarization,
            "enable_remark": state_enable_remark,
            "tool_name": tool_name,
        }
        rp = state.get("remark_prompt", remark_prompt)
        if rp is not None:
            result["remark_prompt"] = rp
        sp = state.get("summarization_prompt", summarization_prompt)
        if sp is not None:
            result["summarization_prompt"] = sp
        return result
    
    # Define parallel direct tools node wrapper
    def parallel_direct_tools_node(state: ParallelDirectToolAgentState) -> Dict[str, Any]:
        """Execute direct tool calls in parallel."""
        return run_parallel_direct_tools(state)
    
    # Define summarization node wrapper
    def summarize_node(state: ParallelDirectToolAgentState) -> Dict[str, Any]:
        """Synthesize results into a summary."""
        return summarize_results(state)
    
    # Add nodes
    workflow.add_node("initialize", init_node)
    workflow.add_node("parallel_direct_tools", parallel_direct_tools_node)
    workflow.add_node("summarize", summarize_node)
    
    # Define edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "parallel_direct_tools")
    
    # Conditional edge: summarize only if enabled
    def should_summarize(state: ParallelDirectToolAgentState) -> str:
        """Determine if summarization should run."""
        if state.get("enable_summarization", False):
            return "summarize"
        return END
    
    workflow.add_conditional_edges(
        "parallel_direct_tools",
        should_summarize,
        {
            "summarize": "summarize",
            END: END,
        }
    )
    
    workflow.add_edge("summarize", END)
    
    return workflow
