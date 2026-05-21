"""
Node functions for the Literature Review Agent graph.

Implements the 5-stage literature review process:
1. Search query generation
2. Parallel paper search
3. Paper analysis (parallel)
4. Synthesis and thematic clustering
5. Final review report generation
"""

import json
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage
from .state import LiteratureReviewState, PaperSearchResult, AnalyzedPaper, ThematicCluster
from .prompts import (
    QUERY_GENERATION_PROMPT,
    PAPER_ANALYSIS_PROMPT,
    THEMATIC_CLUSTERING_PROMPT,
    SYNTHESIS_PROMPT,
    FINAL_REVIEW_REPORT_PROMPT,
)


def _log(message: str, verbose: bool = False):
    """Helper function for conditional logging."""
    if verbose:
        print(message)


def initialize_state(state: LiteratureReviewState, verbose: bool = False) -> Dict[str, Any]:
    """
    Node: Initialize state with default values.

    Args:
        state: Current state
        verbose: Whether to print debug information

    Returns:
        Updated state with initialization data
    """
    if verbose:
        print("\n[INITIALIZE] Setting up Literature Review Agent system")
        print(f"[INITIALIZE] Topic: {state.get('review_topic', '')[:80]}...")
        print(f"[INITIALIZE] Tools available: {len(state.get('tools', []))}")

    return {
        "search_queries": [],
        "search_results": [],
        "analyzed_papers": [],
        "thematic_clusters": [],
        "synthesis": "",
        "research_trends": [],
        "research_gaps": [],
        "final_review_report": {},
    }


def generate_search_queries(state: LiteratureReviewState) -> Dict[str, Any]:
    """
    Stage 1: Generate search queries from the research topic.

    Args:
        state: Current state with review_topic

    Returns:
        Updated state with search_queries
    """
    llm = state.get("llm")
    review_topic = state.get("review_topic", "")
    verbose = state.get("verbose", False)

    if not llm or not review_topic:
        raise ValueError("LLM and review_topic required")

    _log("\n[STAGE 1] Generating search queries", verbose)

    try:
        prompt = f"{QUERY_GENERATION_PROMPT}\n\nResearch topic: {review_topic}"
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        _log(f"[STAGE 1] LLM Response:\n{response_text[:300]}...", verbose)

        # Parse JSON response
        try:
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                queries = json.loads(json_str)
            else:
                queries = [q.strip().strip('"') for q in response_text.split("\n") if q.strip()][:5]
        except json.JSONDecodeError:
            queries = [line.strip() for line in response_text.split("\n") if line.strip()][:5]

        _log(f"[STAGE 1] ✓ Generated {len(queries)} queries", verbose)
        for i, q in enumerate(queries, 1):
            _log(f"  {i}. {q}", verbose)

        return {"search_queries": queries}

    except Exception as e:
        _log(f"[STAGE 1] ✗ Error: {str(e)}", verbose)
        # Fallback queries
        queries = [
            f"fundamentals of {review_topic}",
            f"recent advances in {review_topic}",
            f"{review_topic} applications",
            f"{review_topic} methodologies",
        ]
        return {"search_queries": queries}


def analyze_papers(state: LiteratureReviewState) -> Dict[str, Any]:
    """
    Stage 3: Analyze individual papers from search results.

    This creates analysis tasks that will be processed in parallel.

    Args:
        state: Current state with search_results

    Returns:
        Updated state with analysis queries for parallel processing
    """
    search_results = state.get("search_results", [])
    verbose = state.get("verbose", False)

    _log("\n[STAGE 3] Preparing paper analysis tasks", verbose)

    if not search_results:
        _log("[STAGE 3] No search results to analyze", verbose)
        return {"analyzed_papers": []}

    # Collect all papers from search results
    all_papers = []
    for result in search_results:
        if result.get("success") and result.get("papers"):
            # The papers field contains formatted information about papers
            papers_text = result.get("papers", "")
            # Create analysis task for this set of papers
            all_papers.append({
                "query": result.get("query", ""),
                "papers_info": papers_text,
            })

    _log(f"[STAGE 3] Collected {len(all_papers)} paper groups for analysis", verbose)

    # Store for parallel processing (this will be handled by parallel agent)
    return {"paper_analysis_tasks": all_papers}


def synthesize_findings(state: LiteratureReviewState) -> Dict[str, Any]:
    """
    Stage 4: Synthesize findings, identify themes, trends, and gaps.

    Args:
        state: Current state with analyzed_papers

    Returns:
        Updated state with thematic_clusters, synthesis, trends, and gaps
    """
    llm = state.get("llm")
    analyzed_papers = state.get("analyzed_papers", [])
    verbose = state.get("verbose", False)

    if not llm:
        raise ValueError("LLM not configured")

    _log("\n[STAGE 4] Synthesizing findings and identifying themes", verbose)

    if not analyzed_papers:
        _log("[STAGE 4] No analyzed papers to synthesize", verbose)
        return {
            "thematic_clusters": [],
            "synthesis": "",
            "research_trends": [],
            "research_gaps": [],
        }

    # Step 1: Identify thematic clusters
    _log("[STAGE 4.1] Identifying thematic clusters", verbose)
    papers_summary = "\n\n".join([
        f"Paper: {paper.get('title', 'Unknown')}\n"
        f"Summary: {paper.get('summary', '')}\n"
        f"Key Findings: {', '.join(paper.get('key_findings', []))}\n"
        f"Methodology: {paper.get('methodology', '')}"
        for paper in analyzed_papers
    ])

    try:
        cluster_prompt = f"{THEMATIC_CLUSTERING_PROMPT}\n\n{papers_summary}"
        cluster_response = llm.invoke([HumanMessage(content=cluster_prompt)])
        cluster_text = cluster_response.content

        # Parse clusters
        try:
            json_start = cluster_text.find("[")
            json_end = cluster_text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                clusters_data = json.loads(cluster_text[json_start:json_end])
            else:
                clusters_data = []
        except json.JSONDecodeError:
            clusters_data = []

        _log(f"[STAGE 4.1] ✓ Identified {len(clusters_data)} themes", verbose)

    except Exception as e:
        _log(f"[STAGE 4.1] ✗ Error in clustering: {str(e)}", verbose)
        clusters_data = []

    # Step 2: Generate synthesis, trends, and gaps
    _log("[STAGE 4.2] Generating synthesis and identifying trends/gaps", verbose)

    clusters_summary = json.dumps(clusters_data, indent=2) if clusters_data else "No clusters identified"
    synthesis_input = f"Analyzed Papers:\n{papers_summary}\n\nThematic Clusters:\n{clusters_summary}"

    try:
        synthesis_prompt = f"{SYNTHESIS_PROMPT}\n\n{synthesis_input}"
        synthesis_response = llm.invoke([HumanMessage(content=synthesis_prompt)])
        synthesis_text = synthesis_response.content

        # Parse synthesis JSON
        try:
            json_start = synthesis_text.find("{")
            json_end = synthesis_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                synthesis_data = json.loads(synthesis_text[json_start:json_end])
            else:
                synthesis_data = {
                    "synthesis": synthesis_text,
                    "research_trends": [],
                    "research_gaps": [],
                }
        except json.JSONDecodeError:
            synthesis_data = {
                "synthesis": synthesis_text,
                "research_trends": [],
                "research_gaps": [],
            }

        _log("[STAGE 4.2] ✓ Synthesis complete", verbose)

    except Exception as e:
        _log(f"[STAGE 4.2] ✗ Error in synthesis: {str(e)}", verbose)
        synthesis_data = {
            "synthesis": "",
            "research_trends": [],
            "research_gaps": [],
        }

    return {
        "thematic_clusters": clusters_data,
        "synthesis": synthesis_data.get("synthesis", ""),
        "research_trends": synthesis_data.get("research_trends", []),
        "research_gaps": synthesis_data.get("research_gaps", []),
    }


def generate_final_report(state: LiteratureReviewState) -> Dict[str, Any]:
    """
    Stage 5: Generate comprehensive literature review report.

    Args:
        state: Current state with all analysis results

    Returns:
        Updated state with final_review_report
    """
    llm = state.get("llm")
    review_topic = state.get("review_topic", "")
    search_results = state.get("search_results", [])
    analyzed_papers = state.get("analyzed_papers", [])
    thematic_clusters = state.get("thematic_clusters", [])
    synthesis = state.get("synthesis", "")
    research_trends = state.get("research_trends", [])
    research_gaps = state.get("research_gaps", [])
    verbose = state.get("verbose", False)

    if not llm:
        raise ValueError("LLM not configured")

    _log("\n[STAGE 5] Generating final literature review report", verbose)

    # Prepare comprehensive input
    all_references = state.get("all_references", [])
    
    # Format references for LLM context (Index + Remark)
    references_context = []
    for i, ref in enumerate(all_references, 1):
        references_context.append(f"[{i}] {ref.get('title', 'Unknown')}\nRemark: {ref.get('remark', '')}")
    
    references_text = "\n\n".join(references_context)

    report_input = {
        "review_topic": review_topic,
        "search_queries": state.get("search_queries", []),
        "search_results_summary": "\n\n".join([
            f"Query: {r.get('query', '')}\nPapers Found: {r.get('papers', '')}"
            for r in search_results if r.get("success")
        ]),
        "analyzed_papers": analyzed_papers,
        "thematic_clusters": thematic_clusters,
        "synthesis": synthesis,
        "research_trends": research_trends,
        "research_gaps": research_gaps,
        "available_references": references_text,
    }

    report_input_text = json.dumps(report_input, indent=2, ensure_ascii=False)

    try:
        prompt = f"{FINAL_REVIEW_REPORT_PROMPT}\n\n{report_input_text}"
        response = llm.invoke([HumanMessage(content=prompt)])
        report = response.content
        
        # Append full reference list to the report
        report += "\n\n# References\n"
        for i, ref in enumerate(all_references, 1):
            report += f"[{i}] {ref.get('reference', '')}\n"

        _log("[STAGE 5] ✓ Final report generated", verbose)
        _log(f"[STAGE 5] Report length: {len(report)} characters", verbose)

        return {
            "final_review_report": {
                "comprehensive_report": report,
                "topic": review_topic,
                "total_papers": len(analyzed_papers),
                "themes_identified": len(thematic_clusters),
                "word_count": len(report.split()),
                "references": all_references,
            }
        }

    except Exception as e:
        _log(f"[STAGE 5] ✗ Error: {str(e)}", verbose)
        return {
            "final_review_report": {
                "comprehensive_report": f"Error generating report: {str(e)}",
                "topic": review_topic,
                "error": str(e),
            }
        }
