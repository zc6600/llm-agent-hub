"""
Ideation Agent - Research Ideation for Novel Ideas.

A 6-stage research ideation system that generates novel research ideas
by combining information gathering, gap analysis, and creative thinking.

Public API:
    - create_ideation_agent: Main function to create the ideation agent
    - get_compiled_graph: Alternative function with graph visualization options

Example:
    >>> from langchain_openai import ChatOpenAI
    >>> from langchain_community.tools import DuckDuckGoSearchRun
    >>> from multi_agent_hub.scientific_research.ideation import create_ideation_agent
    >>>
    >>> llm = ChatOpenAI(model="gpt-4")
    >>> tools = [DuckDuckGoSearchRun()]
    >>>
    >>> agent = create_ideation_agent(
    ...     llm=llm,
    ...     tools=tools,
    ...     system_prompt="Focus on sustainable technologies"
    ... )
    >>>
    >>> result = agent.invoke({
    ...     "ideation_message": "How can we improve renewable energy storage?"
    ... })
    >>>
    >>> print(result["final_idea_report"]["comprehensive_report"])
"""

from .agent import create_ideation_agent, get_compiled_graph
from .state import IdeationAgentState

__all__ = [
    "create_ideation_agent",
    "get_compiled_graph",
    "IdeationAgentState",
]
