"""
Node functions for the Ideation Agent graph.

Implements the 6-stage ideation process:
1. Initial query generation and parallel information gathering
2. Research gap analysis
3. Gap-driven query generation and information gathering
4. Bit-flipping creative idea generation
5. Sub-hypothesis generation with verification experiments
6. Final report synthesis
"""

import json
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage
from .state import IdeationAgentState, GatheredInformation, CreativeIdea, SubHypothesis
from .prompts import (
    INITIAL_QUERY_GENERATION_PROMPT,
    GAP_ANALYSIS_PROMPT,
    GAP_DRIVEN_QUERY_GENERATION_PROMPT,
    BIT_FLIPPING_PROMPT,
    SUB_HYPOTHESIS_GENERATION_PROMPT,
    FINAL_REPORT_GENERATION_PROMPT,
)


def _log(message: str, verbose: bool = False):
    """Helper function for conditional logging."""
    if verbose:
        print(message)


def initialize_state(state: IdeationAgentState, verbose: bool = False) -> Dict[str, Any]:
    """
    Node: Initialize state with LLM, tools, and configuration.

    Args:
        state: Current state
        verbose: Whether to print debug information

    Returns:
        Updated state with initialization data
    """
    if verbose:
        print("\n[INITIALIZE] Setting up Ideation Agent system")
        print(f"[INITIALIZE] Question: {state.get('ideation_message', '')[:80]}...")
        print(f"[INITIALIZE] Tools available: {len(state.get('tools', []))}")

    return {
        "initial_queries": [],
        "gathered_information": [],
        "research_gap": "",
        "gap_driven_queries": [],
        "gap_driven_information": [],
        "creative_ideas": [],
        "sub_hypotheses": [],
        "final_idea_report": {},
    }


def generate_initial_queries(state: IdeationAgentState) -> Dict[str, Any]:
    """
    Stage 1: Generate initial queries from the ideation message.

    Breaks down the research question into multiple specific, searchable queries
    for parallel information gathering.

    Args:
        state: Current state with ideation_message

    Returns:
        Updated state with initial_queries
    """
    llm = state.get("llm")
    ideation_message = state.get("ideation_message", "")
    verbose = state.get("verbose", False)

    if not llm or not ideation_message:
        raise ValueError("LLM and ideation_message required")

    if verbose:
        print("\n[STAGE 1] Generating initial queries from research question")

    try:
        # Create prompt for query generation
        prompt = f"{INITIAL_QUERY_GENERATION_PROMPT}\n\nResearch topic: {ideation_message}"

        # Call LLM directly
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        if verbose:
            print(f"[STAGE 1] LLM Response:\n{response_text}")

        # Parse JSON response
        try:
            # Try to extract JSON from response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                queries = json.loads(json_str)
            else:
                # Fallback: split by newlines and clean
                queries = [q.strip() for q in response_text.split("\n") if q.strip() and q.strip().startswith("\"")]
                queries = [q.strip('"') for q in queries]
        except json.JSONDecodeError:
            # Fallback: use simple parsing
            queries = [line.strip() for line in response_text.split("\n") if line.strip()][:5]

        if verbose:
            print(f"[STAGE 1] ✓ Generated {len(queries)} queries")
            for i, q in enumerate(queries, 1):
                print(f"  {i}. {q}")

        return {"initial_queries": queries}

    except Exception as e:
        if verbose:
            print(f"[STAGE 1] ✗ Error: {str(e)}")
        # Fallback: simple decomposition
        queries = [
            f"What are the fundamentals of {ideation_message}?",
            f"What recent developments exist in {ideation_message}?",
            f"What are current challenges in {ideation_message}?",
        ]
        return {"initial_queries": queries}


def run_initial_parallel_information_gathering(state: IdeationAgentState) -> Dict[str, Any]:
    """
    Stage 1 (continued): Run parallel information gathering for initial queries.

    This stage calls create_parallel_react_agent to gather information in parallel.

    Args:
        state: Current state with initial_queries

    Returns:
        Updated state with gathered_information
    """
    # This is a placeholder - the actual parallel execution happens via create_parallel_react_agent
    # In the graph flow, this will be called by passing initial_queries to create_parallel_react_agent

    print("\n[STAGE 1.5] Parallel information gathering")
    print("[STAGE 1.5] Note: This is executed via create_parallel_react_agent")

    # The gathered_information will be set by the graph orchestrator
    # For now, we just return the queries unchanged
    return {}


def analyze_research_gap(state: IdeationAgentState) -> Dict[str, Any]:
    """
    Stage 2: Analyze gathered information to identify research gaps.

    Args:
        state: Current state with gathered_information

    Returns:
        Updated state with research_gap
    """
    llm = state.get("llm")
    ideation_message = state.get("ideation_message", "")
    gathered_info = state.get("gathered_information", [])
    verbose = state.get("verbose", False)

    if not llm:
        raise ValueError("LLM not configured")

    _log("\n[STAGE 2] Analyzing research gap", verbose)

    if not gathered_info:
        _log("[STAGE 2] No information gathered", verbose)
        return {"research_gap": ""}

    # Prepare information summary
    info_summary = "\n\n".join(
        [f"Query {i['query_index']}: {i['query']}\n"
         f"Original Question: {ideation_message}\n"
         f"Information: {i['information']}"
         for i in gathered_info if i.get("success")]
    )

    try:
        prompt = f"{GAP_ANALYSIS_PROMPT}\n\nGathered Information:\n{info_summary}"

        response = llm.invoke([HumanMessage(content=prompt)])
        gap = response.content

        _log(f"[STAGE 2] ✓ Research gap identified", verbose)
        _log(f"[STAGE 2] Gap:\n{gap}", verbose)

        return {"research_gap": gap}

    except Exception as e:
        _log(f"[STAGE 2] ✗ Error: {str(e)}", verbose)
        return {"research_gap": "Insufficient information to identify gap"}


def generate_gap_driven_queries(state: IdeationAgentState) -> Dict[str, Any]:
    """
    Stage 3: Generate gap-driven queries targeting the identified research gap.

    Args:
        state: Current state with research_gap

    Returns:
        Updated state with gap_driven_queries
    """
    llm = state.get("llm")

    research_gap = state.get("research_gap", "")
    verbose = state.get("verbose", False)

    if not llm or not research_gap:
        raise ValueError("LLM and research_gap required")

    _log("\n[STAGE 3] Generating gap-driven queries", verbose)

    try:
        prompt = f"{GAP_DRIVEN_QUERY_GENERATION_PROMPT}\n\n{research_gap}"

        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        # Parse JSON response
        try:
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                queries = json.loads(json_str)
            else:
                queries = [line.strip() for line in response_text.split("\n") if line.strip()][:5]
        except json.JSONDecodeError:
            queries = [line.strip() for line in response_text.split("\n") if line.strip()][:5]

        _log(f"[STAGE 3] ✓ Generated {len(queries)} gap-driven queries", verbose)
        for i, q in enumerate(queries, 1):
            _log(f"  {i}. {q}", verbose)

        return {"gap_driven_queries": queries}

    except Exception as e:
        _log(f"[STAGE 3] ✗ Error: {str(e)}", verbose)
        return {"gap_driven_queries": []}


def run_gap_driven_parallel_information_gathering(state: IdeationAgentState) -> Dict[str, Any]:
    """
    Stage 3 (continued): Run parallel information gathering for gap-driven queries.

    Args:
        state: Current state

    Returns:
        Updated state with gap_driven_information
    """
    verbose = state.get("verbose", False)
    _log("\n[STAGE 3.5] Parallel gap-driven information gathering", verbose)
    _log("[STAGE 3.5] Note: This is executed via create_parallel_react_agent", verbose)

    return {}


def generate_creative_ideas(state: IdeationAgentState) -> Dict[str, Any]:
    """
    Stage 4: Bit-flip existing solutions to generate creative ideas.
    
    Generates a single, deeply analyzed creative idea through systematic reasoning.

    Args:
        state: Current state with research_gap and gap_driven_information

    Returns:
        Updated state with creative_ideas (single idea with detailed analysis)
    """
    llm = state.get("llm")
    gathered_info = state.get("gathered_information", [])
    research_gap = state.get("research_gap", "")
    gap_info = state.get("gap_driven_information", [])
    verbose = state.get("verbose", False)

    if not llm:
        raise ValueError("LLM not configured")

    _log("\n[STAGE 4] Generating creative idea through deep bit-flipping analysis", verbose)

    # Prepare existing approaches summary with full information
    approaches_summary = "\n\n".join(
        [f"Approach {i} from query '{info['query']}':\n{info['information']}"
         for i, info in enumerate(gap_info, 1) if info.get("success")]
    )
    
    # Prepare field information summary with full information
    field_info_summary = "\n\n".join(
        [f"Field knowledge {i} from query '{info['query']}':\n{info['information']}"
         for i, info in enumerate(gathered_info, 1) if info.get("success")]
    )

    try:
        prompt = f"{BIT_FLIPPING_PROMPT}\n\n{research_gap}\n\nField information:\n{field_info_summary}\n\nExisting approaches:\n{approaches_summary}"

        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        # Treat the entire response as a single, deeply analyzed creative idea
        idea = {
            "analysis": response_text,  # Full multi-stage reasoning
        }

        _log(f"[STAGE 4] ✓ Creative idea generated with deep analysis", verbose)
        preview = response_text[:200].replace('\n', ' ')
        _log(f"[STAGE 4] Analysis preview: {preview}...", verbose)

        return {"creative_ideas": [idea]}

    except Exception as e:
        _log(f"[STAGE 4] ✗ Error: {str(e)}", verbose)
        import traceback
        traceback.print_exc()
        return {"creative_ideas": []}


def generate_sub_hypotheses(state: IdeationAgentState) -> Dict[str, Any]:
    """
    Stage 5: Convert creative idea into testable scientific hypotheses.
    
    Takes the deeply analyzed creative idea and transforms it into concrete,
    verifiable sub-hypotheses with verification experiments.

    Args:
        state: Current state with creative_ideas

    Returns:
        Updated state with sub_hypotheses
    """
    llm = state.get("llm")
    creative_ideas = state.get("creative_ideas", [])
    verbose = state.get("verbose", False)

    if not llm:
        raise ValueError("LLM not configured")

    _log("\n[STAGE 5] Converting creative idea into sub-hypotheses and experiments", verbose)

    if not creative_ideas:
        _log("[STAGE 5] No creative idea to develop", verbose)
        return {"sub_hypotheses": []}

    # Extract the analysis from the single creative idea
    idea_analysis = creative_ideas[0].get("analysis", "")

    try:
        prompt = f"{SUB_HYPOTHESIS_GENERATION_PROMPT}\n\n{idea_analysis}"

        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        # Try to parse JSON array of hypotheses
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        
        if json_start >= 0 and json_end > json_start:
            try:
                json_str = response_text[json_start:json_end]
                hypotheses_data = json.loads(json_str)
                sub_hypotheses = [
                    {
                        "research_question": h.get("research_question", ""),
                        "verification_experiment": h.get("verification_experiment", ""),
                        "expected_outcome": h.get("expected_outcome", ""),
                        "testable_metrics": h.get("testable_metrics", []),
                    }
                    for h in hypotheses_data
                ]
            except json.JSONDecodeError:
                # If JSON parsing fails, treat entire response as single hypothesis
                sub_hypotheses = [{
                    "research_question": response_text,
                    "verification_experiment": response_text,
                    "expected_outcome": response_text,
                    "testable_metrics": [],
                }]
        else:
            # No JSON found, treat entire response as hypothesis
            sub_hypotheses = [{
                "research_question": response_text,
                "verification_experiment": response_text,
                "expected_outcome": response_text,
                "testable_metrics": [],
            }]

        _log(f"[STAGE 5] ✓ Generated {len(sub_hypotheses)} sub-hypothesis/hypotheses", verbose)

        return {"sub_hypotheses": sub_hypotheses}

    except Exception as e:
        _log(f"[STAGE 5] ✗ Error: {str(e)}", verbose)
        import traceback
        traceback.print_exc()
        return {"sub_hypotheses": []}


def generate_final_report(state: IdeationAgentState) -> Dict[str, Any]:
    """
    Stage 6: Synthesize all information into a comprehensive ideation report.

    Args:
        state: Current state with all previous stage outputs

    Returns:
        Updated state with final_idea_report
    """
    llm = state.get("llm")
    ideation_message = state.get("ideation_message", "")
    research_gap = state.get("research_gap", "")
    creative_ideas = state.get("creative_ideas", [])
    sub_hypotheses = state.get("sub_hypotheses", [])
    gathered_information = state.get("gathered_information", [])
    gap_driven_information = state.get("gap_driven_information", [])
    verbose = state.get("verbose", False)

    if not llm:
        raise ValueError("LLM not configured")

    _log("\n[STAGE 6] Generating final ideation report", verbose)

    # Prepare comprehensive input from creative idea analysis
    idea_analysis = creative_ideas[0].get("analysis", "") if creative_ideas else ""
    
    # Format hypotheses for readability
    hypotheses_text = "\n".join([
        f"\n--- Hypothesis {i} ---\n"
        f"Research Question: {h.get('research_question', '')}\n"
        f"Verification Experiment: {h.get('verification_experiment', '')}\n"
        f"Expected Outcome: {h.get('expected_outcome', '')}\n"
        f"Testable Metrics: {', '.join(h.get('testable_metrics', []))}"
        for i, h in enumerate(sub_hypotheses, 1)
    ])
    
    # Format gathered information with all context
    initial_info_text = "\n".join([
        f"\n--- Query {i}: {info.get('query', '')} ---\n"
        f"Information gathered:\n{info.get('information', '')}"
        for i, info in enumerate(gathered_information, 1)
        if info.get('success', False)
    ])
    
    # Format gap-driven information with all context
    gap_info_text = "\n".join([
        f"\n--- Gap Query {i}: {info.get('query', '')} ---\n"
        f"Information gathered:\n{info.get('information', '')}"
        for i, info in enumerate(gap_driven_information, 1)
        if info.get('success', False)
    ])

    try:
        prompt = (
            f"{FINAL_REPORT_GENERATION_PROMPT}\n\n"
            f"Original question: {ideation_message}\n\n"
            f"Research gap: {research_gap}\n\n"
            f"Initial research findings (with citations):\n{initial_info_text}\n\n"
            f"Gap-driven research findings (with citations):\n{gap_info_text}\n\n"
            f"Creative idea analysis:\n{idea_analysis}\n\n"
            f"Sub-hypotheses and experiments:\n{hypotheses_text}"
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        final_report_text = response.content

        # Create structured report
        final_report = {
            "title": f"Ideation Report: {ideation_message[:50]}...",
            "original_question": ideation_message,
            "research_gap": research_gap,
            "creative_idea_analysis": idea_analysis,
            "sub_hypotheses": sub_hypotheses,
            "comprehensive_report": final_report_text,
        }

        _log("[STAGE 6] ✓ Final report generated successfully", verbose)

        return {"final_idea_report": final_report}

    except Exception as e:
        _log(f"[STAGE 6] ✗ Error: {str(e)}", verbose)
        import traceback
        traceback.print_exc()
        return {
            "final_idea_report": {
                "original_question": ideation_message,
                "research_gap": research_gap,
                "creative_idea_analysis": idea_analysis,
                "sub_hypotheses": sub_hypotheses,
                "error": str(e),
            }
        }
