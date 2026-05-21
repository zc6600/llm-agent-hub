"""
State definition for the Ideation Agent.

Defines the state structure for a 6-stage ideation process:
1. Gathering Information (Parallel)
2. Identify Research Gap
3. Gap-driven Query & Information Gathering (Parallel)
4. Bit Flipping Idea Generation
5. Sub-hypothesis Generation
6. Final Report Generation
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.tools import BaseTool


class GatheredInformation(TypedDict, total=False):
    """Information gathered from a single parallel query."""
    query_index: int
    query: str
    information: str
    success: bool
    error: Optional[str]


class CreativeIdea(TypedDict, total=False):
    """Creative idea generated from bit flipping."""
    idea: str
    reasoning: str
    novelty_score: float


class SubHypothesis(TypedDict, total=False):
    """Verifiable sub-hypothesis with experiment design."""
    research_question: str
    verification_experiment: str
    expected_outcome: str
    testable_metrics: List[str]


class IdeationAgentState(TypedDict, total=False):
    """
    State for the Ideation Agent system.

    Manages a 6-stage ideation process for generating novel research ideas:
    1. Parallel information gathering on the initial question
    2. Analysis to identify research gaps
    3. Gap-driven parallel queries to find existing solutions
    4. Bit-flipping creative idea generation
    5. Sub-hypothesis generation with verification experiments
    6. Final idea report synthesis
    """

    # Initial input
    ideation_message: str  # The original research question/topic

    # Configuration
    llm: Any  # Language model instance
    tools: List[BaseTool]  # Tools available to parallel agents
    system_prompt: str  # User-provided system prompt
    verbose: bool  # Whether to print intermediate progress (default: False)

    # Stage 1: Initial Information Gathering (Parallel)
    initial_queries: List[str]  # Generated queries from ideation_message
    gathered_information: List[GatheredInformation]  # Results from parallel queries

    # Stage 2: Research Gap Identification
    research_gap: str  # Identified gap in current knowledge/approaches

    # Stage 3: Gap-driven Information Gathering (Parallel)
    gap_driven_queries: List[str]  # Generated queries targeting the gap
    gap_driven_information: List[GatheredInformation]  # Existing solutions/approaches

    # Stage 4: Bit Flipping Creative Ideas
    creative_ideas: List[CreativeIdea]  # Generated creative ideas

    # Stage 5: Sub-hypothesis Generation
    sub_hypotheses: List[SubHypothesis]  # Verifiable hypotheses with experiments

    # Stage 6: Final Report
    final_idea_report: Dict[str, Any]  # Complete ideation report
