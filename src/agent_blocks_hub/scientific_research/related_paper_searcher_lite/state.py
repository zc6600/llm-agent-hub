"""
State definition for Related Paper Searcher Lite Agent
"""

from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages


class RelatedPaperSearcherLiteState(TypedDict):
    """
    State for the Related Paper Searcher Lite workflow.
    
    Fields:
        messages: List of conversation messages (appended with add_messages)
        query: The search query for papers
        search_results: Raw search results from Semantic Scholar
        evaluation_results: List of evaluated papers with relevance scores
        related_papers: Final list of related papers (excluding "Not related")
    """
    messages: Annotated[list, add_messages]
    query: str
    search_results: Optional[str] = None
    evaluation_results: list = []  # List of {"paper": {...}, "relevance": "...", "comment": "..."}
    related_papers: list = []  # Final filtered results
