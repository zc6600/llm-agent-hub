"""
Related Paper Searcher Lite Agent

This module provides an agent that searches for related papers using Semantic Scholar
and uses an LLM to evaluate their relevance, filtering out unrelated papers.
"""

from .agent import create_related_paper_searcher_lite_agent

__all__ = ["create_related_paper_searcher_lite_agent"]
