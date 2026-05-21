"""
Literature Review Agent - Systematic Literature Review for Research Topics.

A 5-stage literature review system that comprehensively analyzes academic papers
by combining search, analysis, and synthesis capabilities.

Public API:
    - create_literature_review_agent: Main function to create the literature review agent
    - get_compiled_graph: Alternative function with graph visualization options

Example:
    >>> from langchain_openai import ChatOpenAI
    >>> from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
    >>> from llm_tool_hub.integrations.langchain_adapter import LangchainToolAdapter
    >>> from multi_agent_hub.scientific_research.literature_review import create_literature_review_agent
    >>>
    >>> llm = ChatOpenAI(model="gpt-4")
    >>> search_tool = LangchainToolAdapter.to_langchain_structured_tool(
    ...     SearchSemanticScholar()
    ... )
    >>>
    >>> agent = create_literature_review_agent(
    ...     llm=llm,
    ...     tools=[search_tool],
    ...     system_prompt="Focus on peer-reviewed papers from top conferences"
    ... )
    >>>
    >>> result = agent.invoke({
    ...     "review_topic": "transformer architectures for NLP"
    ... })
    >>>
    >>> print(result["final_review_report"]["comprehensive_report"])
"""

from .agent import create_literature_review_agent, get_compiled_graph
from .state import LiteratureReviewState

__all__ = [
    "create_literature_review_agent",
    "get_compiled_graph",
    "LiteratureReviewState",
]
