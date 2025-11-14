"""
Parallel React Agent creation and configuration.

Creates a LangGraph-based agent that runs multiple ReAct agents in parallel
and summarizes their results.
"""

from typing import List, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .state import ParallelReactAgentState
from .nodes import (
    initialize_state,
    run_parallel_agents,
    summarizing_agent,
)
from .prompts import get_combined_system_prompt


def create_parallel_react_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    verbose: bool = False,
) -> Any:
    """
    Create a Parallel React Agent using LangGraph.
    
    The agent runs multiple ReAct agents in parallel, each processing one query
    with the same tools and system prompt, then uses a Summarizing Agent to
    integrate and synthesize all results into a coherent output.
    
    Args:
        llm: Language model to use for all agents (Claude, GPT, etc.)
        tools: List of tools available to all agents
        system_prompt: Optional user-provided system prompt to combine with
                      the architecture's built-in prompts
        verbose: Whether to print detailed execution logs (default: False)
    
    Returns:
        Compiled LangGraph agent ready to invoke
        
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_community.tools import DuckDuckGoSearchRun
        >>> 
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> tools = [DuckDuckGoSearchRun()]
        >>> 
        >>> agent = create_parallel_react_agent(
        ...     llm=llm,
        ...     tools=tools,
        ...     system_prompt="Focus on recent developments",
        ...     verbose=True  # Enable detailed logging
        ... )
        >>> 
        >>> result = agent.invoke({
        ...     "parallel_react_agent_messages": [
        ...         "What is LangGraph?",
        ...         "What is ReAct pattern?",
        ...         "How to use agents in LangChain?"
        ...     ]
        ... })
        >>> 
        >>> print(result["final_summary"])
    """
    if system_prompt is None:
        system_prompt = ""
    
    # Create and return compiled graph
    graph = _create_graph(llm, tools, system_prompt, verbose)
    return graph.compile()


def _create_graph(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str,
    verbose: bool = False,
) -> StateGraph:
    """
    Create the LangGraph state graph for the Parallel React Agent.
    
    Args:
        llm: Language model
        tools: Available tools
        system_prompt: User-provided system prompt
        verbose: Whether to print detailed logs
        
    Returns:
        Configured StateGraph
    """
    # Create state graph
    workflow = StateGraph(ParallelReactAgentState)
    
    # Define initialization node wrapper
    def init_node(state: ParallelReactAgentState) -> Dict[str, Any]:
        """Initialize state with LLM, tools, and configuration."""
        # Use verbose from state if provided, otherwise use the one from agent creation
        state_verbose = state.get("verbose", verbose)
        return {
            **initialize_state(state),
            "llm": llm,
            "tools": tools,
            "system_prompt": system_prompt,
            "verbose": state_verbose,  # Set this last to ensure it's not overwritten
        }
    
    # Add nodes
    workflow.add_node("initialize", init_node)
    workflow.add_node("run_parallel_agents", run_parallel_agents)
    workflow.add_node("summarize", summarizing_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "run_parallel_agents")
    workflow.add_edge("run_parallel_agents", "summarize")
    workflow.add_edge("summarize", END)
    
    return workflow


def get_compiled_graph(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    save_graph_image: bool = True,
    image_name: str = "parallel_react_agent_graph.png",
) -> Any:
    """
    Create and compile the Parallel React Agent graph.
    
    Optionally saves a PNG visualization of the graph.
    
    Args:
        llm: Language model
        tools: Available tools
        system_prompt: User-provided system prompt
        save_graph_image: Whether to save PNG image of the graph
        image_name: Name/path for the saved image
        
    Returns:
        Compiled LangGraph agent
    """
    agent = create_parallel_react_agent(llm, tools, system_prompt)
    
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
