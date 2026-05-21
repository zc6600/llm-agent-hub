"""
State definition for the Literature Review Agent.

Defines the state structure for a literature review process:
1. Query Generation
2. Parallel Paper Search
3. Paper Analysis (Parallel)
4. Synthesis & Connections
5. Final Report Generation
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.tools import BaseTool


class PaperSearchResult(TypedDict, total=False):
    """Result from a single paper search query."""
    query_index: int
    query: str
    papers: str  # JSON string or formatted text of papers
    success: bool
    error: Optional[str]


class AnalyzedPaper(TypedDict, total=False):
    """Analyzed paper with extracted information."""
    paper_id: str
    title: str
    summary: str
    key_findings: List[str]
    methodology: str
    limitations: str
    relevance_score: float


class ThematicCluster(TypedDict, total=False):
    """Cluster of related papers around a theme."""
    theme: str
    papers: List[str]  # Paper IDs
    key_insights: List[str]
    research_gaps: List[str]


class LiteratureReviewState(TypedDict, total=False):
    """
    State for the Literature Review Agent system.

    Manages a 5-stage literature review process:
    1. Generate search queries from the research topic
    2. Parallel paper search across multiple queries
    3. Analyze individual papers in parallel
    4. Synthesize findings and identify connections
    5. Generate comprehensive literature review report
    """

    # Initial input
    review_topic: str  # The research topic to review

    # Configuration
    llm: Any  # Language model instance
    tools: List[BaseTool]  # Tools available to parallel agents
    system_prompt: str  # User-provided system prompt
    verbose: bool  # Whether to print intermediate progress (default: False)
    mode: str  # Parallel agent mode - "lite" or "full" (default: "lite")

    # Stage 1: Query Generation
    search_queries: List[str]  # Generated search queries

    # Stage 2: Parallel Paper Search
    search_results: List[PaperSearchResult]  # Results from parallel searches

    # Stage 3: Paper Analysis (Parallel)
    paper_analysis_tasks: List[Dict[str, Any]]  # Tasks for parallel processing
    analyzed_papers: List[AnalyzedPaper]  # Analyzed paper information

    # Stage 4: Synthesis & Connections
    thematic_clusters: List[ThematicCluster]  # Grouped papers by theme
    synthesis: str  # Overall synthesis of findings
    research_trends: List[str]  # Identified trends
    research_gaps: List[str]  # Identified gaps

    # Stage 5: Final Report
    final_review_report: Dict[str, Any]  # Complete literature review
    
    # Storage for all references found
    all_references: List[Dict[str, Any]]  # List of {title, remark, reference}
