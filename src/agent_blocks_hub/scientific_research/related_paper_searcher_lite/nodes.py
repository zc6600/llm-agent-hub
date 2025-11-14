"""
Node functions for Related Paper Searcher Lite Agent
"""

import json
import logging
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .state import RelatedPaperSearcherLiteState
from .prompts import EVALUATION_SYSTEM_PROMPT, EVALUATE_PAPERS_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def search_papers_node(state: RelatedPaperSearcherLiteState, search_tool: object) -> RelatedPaperSearcherLiteState:
    """
    Node: Search for papers using Semantic Scholar
    
    Args:
        state: Current state
        search_tool: SearchSemanticScholar tool instance
        
    Returns:
        Updated state with search results
    """
    query = state.get("query", "")
    
    if not query:
        raise ValueError("Query is required for paper search")
    
    logger.info(f"Searching for papers with query: {query}")
    
    # Run search tool
    search_results = search_tool.run(query=query, limit=10)
    
    return {
        **state,
        "search_results": search_results,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"Searched for papers on query: {query}")
        ]
    }


def evaluate_papers_node(state: RelatedPaperSearcherLiteState, llm: BaseChatModel) -> RelatedPaperSearcherLiteState:
    """
    Node: Evaluate papers for relevance using LLM
    
    Args:
        state: Current state with search results
        llm: Language model for evaluation
        
    Returns:
        Updated state with evaluation results
    """
    query = state.get("query", "")
    search_results = state.get("search_results", "")
    
    if not search_results:
        logger.warning("No search results to evaluate")
        return {
            **state,
            "evaluation_results": [],
            "related_papers": []
        }
    
    logger.info(f"Evaluating papers for relevance...")
    
    # Prepare evaluation prompt
    evaluation_prompt = EVALUATE_PAPERS_PROMPT_TEMPLATE.format(
        query=query,
        papers=search_results
    )
    
    # Call LLM for evaluation
    messages = [
        SystemMessage(content=EVALUATION_SYSTEM_PROMPT),
        HumanMessage(content=evaluation_prompt)
    ]
    
    response = llm.invoke(messages)
    evaluation_text = response.content
    
    # Parse evaluation results
    evaluation_results = _parse_evaluation_results(evaluation_text)
    
    # Filter out "Not related" papers
    related_papers = [
        result for result in evaluation_results
        if result.get("relevance") != "Not related"
    ]
    
    logger.info(f"Evaluation complete: {len(related_papers)} related papers out of {len(evaluation_results)} total")
    
    return {
        **state,
        "evaluation_results": evaluation_results,
        "related_papers": related_papers,
        "messages": state.get("messages", []) + [
            HumanMessage(content=f"Evaluated papers: {len(evaluation_results)} total, {len(related_papers)} related")
        ]
    }


def _parse_evaluation_results(evaluation_text: str) -> list:
    """
    Parse LLM evaluation output into structured format
    
    Args:
        evaluation_text: Raw LLM response
        
    Returns:
        List of evaluation results with paper info, relevance, and comment
    """
    results = []
    
    # Split by paper entries (looking for "Paper:" prefix)
    entries = evaluation_text.split("Paper:")
    
    for entry in entries[1:]:  # Skip first empty split
        lines = entry.strip().split("\n")
        
        paper_title = lines[0].strip() if lines else ""
        relevance = ""
        comment = ""
        
        for line in lines[1:]:
            if line.startswith("Relevance:"):
                relevance = line.replace("Relevance:", "").strip()
            elif line.startswith("Comment:"):
                comment = line.replace("Comment:", "").strip()
        
        if paper_title and relevance:
            results.append({
                "paper_title": paper_title,
                "relevance": relevance,
                "comment": comment
            })
    
    return results
