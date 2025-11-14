"""
Related Paper Searcher Lite Agent using LangGraph

This agent searches for papers related to a research query using Semantic Scholar,
evaluates their relevance using an LLM, and returns only related papers.
"""

from typing import Optional, List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from .state import RelatedPaperSearcherLiteState
from .nodes import search_papers_node, evaluate_papers_node
from llm_tool_hub.scientific_research_tool.search_semantic_scholar import SearchSemanticScholar


def create_related_paper_searcher_lite_agent(
    llm: BaseChatModel,
    search_tool: Optional[SearchSemanticScholar] = None
):
    """
    Create a Related Paper Searcher Lite agent using LangGraph.
    
    The agent performs two main steps:
    1. Search for papers using Semantic Scholar
    2. Evaluate papers using LLM to filter out unrelated ones
    
    Args:
        llm: Language model for paper evaluation
        search_tool: Semantic Scholar search tool (if None, will create default)
        
    Returns:
        Compiled LangGraph agent ready to invoke
        
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent = create_related_paper_searcher_lite_agent(llm=llm)
        >>> result = agent.invoke({"query": "transformer neural networks"})
        >>> print(result["related_papers"])
    """
    
    # Initialize search tool if not provided
    if search_tool is None:
        search_tool = SearchSemanticScholar()
    
    # Create the graph builder
    builder = StateGraph(RelatedPaperSearcherLiteState)
    
    # Add nodes
    builder.add_node(
        "search_papers",
        lambda state: search_papers_node(state, search_tool)
    )
    builder.add_node(
        "evaluate_papers",
        lambda state: evaluate_papers_node(state, llm)
    )
    
    # Add edges
    builder.add_edge(START, "search_papers")
    builder.add_edge("search_papers", "evaluate_papers")
    builder.add_edge("evaluate_papers", END)
    
    # Compile the graph
    agent = builder.compile()
    
    return agent
