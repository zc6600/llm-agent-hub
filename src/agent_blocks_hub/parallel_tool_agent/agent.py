"""
Parallel Tool Agent creation and configuration.

Creates a LangGraph-based agent that executes tool calls in parallel
with optional result summarization. This is a lightweight alternative
to parallel_react_agent for simple information gathering tasks.
"""

from typing import List, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .state import ParallelToolAgentState
from .nodes import (
    initialize_state,
    run_parallel_tools,
    summarize_results,
)


def create_parallel_tool_agent(
    llm: Optional[BaseChatModel] = None,
    tools: List[BaseTool] = None,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    enable_summarization: bool = False,
    tool_name: Optional[str] = None,
) -> Any:
    """
    Create a Parallel Tool Agent using LangGraph.
    
    The agent executes tool calls in parallel without ReAct reasoning loops,
    making it much faster for simple information gathering tasks like paper searches.
    
    Key differences from parallel_react_agent:
    - Direct tool invocation (no reasoning loops)
    - Optional summarization (can be disabled for maximum speed)
    - Simpler state management
    - Faster execution
    
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
        >>> # Fast mode: No summarization, direct tool calls only
        >>> agent = create_parallel_tool_agent(
        ...     tools=[SearchSemanticScholar()],
        ...     enable_summarization=False,  # Maximum speed
        ...     verbose=True
        ... )
        >>> 
        >>> result = agent.invoke({
        ...     "parallel_tool_agent_messages": [
        ...         "transformer neural networks",
        ...         "attention mechanism deep learning",
        ...     ]
        ... })
        >>> 
        >>> # Access individual tool results
        >>> for idx, tool_result in result["tool_results"].items():
        ...     print(f"Query {idx}: {tool_result['result']}")
        >>>
        >>> # Summarization mode: LLM synthesizes results
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent_with_summary = create_parallel_tool_agent(
        ...     llm=llm,
        ...     tools=[SearchSemanticScholar()],
        ...     enable_summarization=True,  # Enable intelligent summarization
        ...     verbose=True
        ... )
        >>> 
        >>> result = agent_with_summary.invoke({
        ...     "parallel_tool_agent_messages": [
        ...         "transformer neural networks",
        ...         "attention mechanism deep learning",
        ...     ]
        ... })
        >>> 
        >>> print(result["final_summary"])  # Synthesized, coherent summary
    """
    if tools is None:
        tools = []
    
    if system_prompt is None:
        system_prompt = ""
    
    if enable_summarization and llm is None:
        raise ValueError("LLM is required when enable_summarization=True")
    
    # Create and return compiled graph
    graph = _create_graph(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        verbose=verbose,
        enable_summarization=enable_summarization,
        tool_name=tool_name,
    )
    return graph.compile()


def _create_graph(
    llm: Optional[BaseChatModel],
    tools: List[BaseTool],
    system_prompt: str,
    verbose: bool,
    enable_summarization: bool,
    tool_name: Optional[str],
) -> StateGraph:
    """
    Create the LangGraph state graph for the Parallel Tool Agent.
    
    Args:
        llm: Language model (optional)
        tools: Available tools
        system_prompt: User-provided system prompt
        verbose: Whether to print detailed logs
        enable_summarization: Whether to enable summarization
        tool_name: Specific tool to use
        
    Returns:
        Configured StateGraph
    """
    # Create state graph
    workflow = StateGraph(ParallelToolAgentState)
    
    # Define initialization node wrapper
    def init_node(state: ParallelToolAgentState) -> Dict[str, Any]:
        """Initialize state with configuration."""
        # Use verbose from state if provided, otherwise use the one from agent creation
        state_verbose = state.get("verbose", verbose)
        state_enable_summarization = state.get("enable_summarization", enable_summarization)
        
        return {
            **initialize_state(state),
            "llm": llm,
            "tools": tools,
            "system_prompt": system_prompt,
            "verbose": state_verbose,
            "enable_summarization": state_enable_summarization,
            "tool_name": tool_name,
        }
    
    # Add nodes
    workflow.add_node("initialize", init_node)
    workflow.add_node("run_parallel_tools", run_parallel_tools)
    workflow.add_node("summarize", summarize_results)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "run_parallel_tools")
    workflow.add_edge("run_parallel_tools", "summarize")
    workflow.add_edge("summarize", END)
    
    return workflow


def get_compiled_graph(
    llm: Optional[BaseChatModel] = None,
    tools: List[BaseTool] = None,
    system_prompt: Optional[str] = None,
    enable_summarization: bool = False,
    save_graph_image: bool = True,
    image_name: str = "parallel_tool_agent_graph.png",
) -> Any:
    """
    Create and compile the Parallel Tool Agent graph.
    
    Optionally saves a PNG visualization of the graph.
    
    Args:
        llm: Language model (optional)
        tools: Available tools
        system_prompt: User-provided system prompt
        enable_summarization: Whether to enable summarization
        save_graph_image: Whether to save PNG image of the graph
        image_name: Name/path for the saved image
        
    Returns:
        Compiled LangGraph agent
    """
    agent = create_parallel_tool_agent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        enable_summarization=enable_summarization,
    )
    
    if save_graph_image:
        try:
            # Save PNG visualization
            agent.get_graph().draw_png(image_name)
            print(f"✓ Graph visualization saved to: {image_name}")
        except Exception as e:
            print(f"⚠️  Could not save graph image: {e}")
            print("   (This typically requires pygraphviz to be installed)")
            
            # Try saving as Mermaid instead
            try:
                mermaid_path = image_name.replace(".png", ".md")
                with open(mermaid_path, "w") as f:
                    f.write("```mermaid\n")
                    f.write(agent.get_graph().to_mermaid())
                    f.write("\n```")
                print(f"✓ Graph visualization saved to: {mermaid_path} (Mermaid format)")
            except Exception as e2:
                print(f"⚠️  Could not save Mermaid graph either: {e2}")
    
    return agent
