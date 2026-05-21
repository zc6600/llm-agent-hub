# src/llm_tool_hub/scientific_research_tool/__init__.py

from .search_semantic_scholar import SearchSemanticScholar
from .search_unpaywall import SearchUnpaywall

def SearchSemanticScholarLC():
    try:
        from ..integrations.langchain_adapter import LangchainToolAdapter
        return LangchainToolAdapter.to_langchain_structured_tool(SearchSemanticScholar())
    except ImportError as e:
        raise ImportError("LangChain core components are not installed. Install 'langchain-core' or use agent factory auto-adaptation.") from e

def SearchUnpaywallLC():
    try:
        from ..integrations.langchain_adapter import LangchainToolAdapter
        return LangchainToolAdapter.to_langchain_structured_tool(SearchUnpaywall())
    except ImportError as e:
        raise ImportError("LangChain core components are not installed. Install 'langchain-core' or use agent factory auto-adaptation.") from e

__all__ = ['SearchSemanticScholar', 'SearchUnpaywall', 'SearchSemanticScholarLC', 'SearchUnpaywallLC']
