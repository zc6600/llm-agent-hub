"""
Literature Review Agent creation and configuration.

Creates a LangGraph-based literature review agent that implements a 5-stage process:
1. Generate search queries from research topic
2. Parallel paper search across multiple queries
3. Analyze papers in parallel
4. Synthesize findings and identify themes
5. Generate comprehensive literature review report
"""

from typing import List, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from agent_blocks_hub.parallel_agent import create_parallel_agent

from .state import LiteratureReviewState
from .prompts import REMARK_PROMPT
from .nodes import (
    initialize_state,
    generate_search_queries,
    analyze_papers,
    synthesize_findings,
    generate_final_report,
)


def create_literature_review_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    mode: str = "lite",
) -> Any:
    """
    Create a Literature Review Agent using LangGraph.

    The agent implements a 5-stage literature review process:
    1. Generate search queries from the research topic
    2. Execute parallel paper searches across queries
    3. Analyze individual papers in parallel
    4. Synthesize findings, identify themes, trends, and gaps
    5. Generate a comprehensive literature review report

    Args:
        llm: Language model to use for all stages
        tools: List of tools available to parallel agents (e.g., paper search tools)
        system_prompt: Optional user-provided system prompt for review context
        verbose: Whether to print intermediate progress and debug information (default: False)
        mode: Execution mode - "lite" for faster execution with direct tool calling,
              "full" for comprehensive analysis with ReAct reasoning (default: "lite")

    Returns:
        Compiled LangGraph agent ready to invoke

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
        >>> from llm_tool_hub.integrations.langchain_adapter import LangchainToolAdapter
        >>>
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> search_tool = LangchainToolAdapter.to_langchain_structured_tool(
        ...     SearchSemanticScholar()
        ... )
        >>>
        >>> # Use lite mode (faster, default)
        >>> agent = create_literature_review_agent(
        ...     llm=llm,
        ...     tools=[search_tool],
        ...     mode="lite",
        ...     system_prompt="Focus on peer-reviewed papers from top conferences",
        ...     verbose=True
        ... )
        >>>
        >>> # Or use full mode for comprehensive analysis with ReAct reasoning
        >>> agent_full = create_literature_review_agent(
        ...     llm=llm,
        ...     tools=[search_tool],
        ...     mode="full",
        ...     verbose=True
        ... )
        >>>
        >>> result = agent.invoke({
        ...     "review_topic": "transformer architectures for natural language processing"
        ... })
        >>>
        >>> print(result["final_review_report"]["comprehensive_report"])
    """
    if system_prompt is None:
        system_prompt = ""

    if mode not in ("lite", "full"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'lite' or 'full'")

    # Create and return compiled graph
    graph = _create_graph(llm, tools, system_prompt, verbose, mode)
    return graph.compile()


def _get_parallel_mode(mode: str) -> str:
    """Map user-friendly mode to parallel agent mode.
    
    Args:
        mode: User-facing mode ("lite" or "full")
        
    Returns:
        Parallel agent mode ("direct" or "react")
    """
    return "direct" if mode == "lite" else "react"


def _create_graph(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str,
    verbose: bool = False,
    mode: str = "lite",
) -> StateGraph:
    """
    Create the LangGraph state graph for the Literature Review Agent.

    Args:
        llm: Language model
        tools: Available tools for parallel agents
        system_prompt: User-provided system prompt
        verbose: Whether to print intermediate progress
        mode: Execution mode - "lite" or "full"

    Returns:
        Configured StateGraph
    """
    # Create state graph
    workflow = StateGraph(LiteratureReviewState)

    # Define initialization node wrapper
    def init_node(state: LiteratureReviewState) -> Dict[str, Any]:
        """Initialize state with LLM, tools, and configuration."""
        return {
            "llm": llm,
            "tools": tools,
            "system_prompt": system_prompt,
            "verbose": verbose,
            "mode": mode,
            **initialize_state(state, verbose),
        }

    # Stage 1: Generate search queries
    def stage1_node(state: LiteratureReviewState) -> Dict[str, Any]:
        """Stage 1: Generate search queries."""
        return generate_search_queries(state)

    # Stage 2: Parallel paper search
    def stage2_parallel_node(state: LiteratureReviewState) -> Dict[str, Any]:
        """Stage 2 (parallel): Execute paper searches using parallel agent."""
        search_queries = state.get("search_queries", [])
        verbose = state.get("verbose", False)
        user_mode = state.get("mode", "lite")
        parallel_mode = _get_parallel_mode(user_mode)

        if not search_queries:
            return {"search_results": []}

        if verbose:
            print(f"\n[STAGE 2] Running parallel paper search (mode={user_mode} -> {parallel_mode})")

        # Prepare search instructions
        tool_required_prompt = (
            "You are a research assistant helping with academic paper search. "
            "For EVERY query you receive, you MUST use the available tools to search for relevant papers. "
            "Do NOT answer from your own knowledge - ALWAYS use the tools to find papers. "
            "\n\n"
            "CRITICAL: Use SIMPLE, BROAD keywords when calling search tools:\n"
            "❌ BAD: 'Transformer architectures with multi-head attention for sequence-to-sequence NLP tasks'\n"
            "✅ GOOD: 'transformer attention NLP'\n"
            "\n"
            "Keep search queries SHORT and focused on core concepts.\n\n"
            + state.get("system_prompt", "")
        )

        # Shared logic for processing results
        search_results = []
        all_references = []
        
        # Execute parallel agent
        if parallel_mode in ("direct", "tool_calling"):
            parallel_agent = create_parallel_agent(
                llm=llm,
                tools=tools,
                mode=parallel_mode,
                system_prompt=tool_required_prompt,
                verbose=verbose,
                enable_summarization=False,
                enable_remark=True,
                remark_prompt=REMARK_PROMPT,
            )

            result = parallel_agent.invoke({
                "parallel_agent_message": search_queries,
                "verbose": verbose,
            })
        else:
            parallel_agent = create_parallel_agent(
                llm=llm,
                tools=tools,
                mode="react",
                system_prompt=state.get("system_prompt", ""),
                verbose=verbose,
                enable_summarization=False,
                enable_remark=True,
                remark_prompt=REMARK_PROMPT,
            )

            result = parallel_agent.invoke({
                "parallel_react_agent_messages": search_queries,
                "verbose": verbose,
            })

        # Process results
        remarks = result.get("remark", {})
        agent_results = result.get("agent_results", {})
        
        import json
        
        for idx, query in enumerate(search_queries):
            item = agent_results.get(idx, {})
            text = remarks.get(idx, "") or ""
            
            # Parse JSON remark
            papers_data = []
            try:
                # Find JSON content
                json_start = text.find("[")
                json_end = text.rfind("]") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = text[json_start:json_end]
                    raw_papers = json.loads(json_str)
                    
                    # Validate and transform structure
                    for p in raw_papers:
                        if isinstance(p, dict) and "title" in p:
                            # Reconstruct the markdown remark string
                            key_results_str = "\n".join([f"- {r}" for r in p.get("key_results", [])])
                            remark_md = (
                                f"**title**: {p.get('title')}\n"
                                f"**authors**: {p.get('authors')}\n"
                                f"**content**: {p.get('content')}\n"
                                f"**methodology**: {p.get('methodology')}\n"
                                f"**key results**: \n{key_results_str}\n"
                                f"**limitations**: {p.get('limitations')}\n"
                                f"**remark**: {p.get('relevance_assessment')}\n"
                                f"**take-home message**: {p.get('take_home_message')}\n"
                            )
                            
                            valid_paper = {
                                "title": p.get("title"),
                                "remark": remark_md,
                                "reference": p.get("citation_info")
                            }
                            papers_data.append(valid_paper)
                            all_references.append(valid_paper)
            except Exception as e:
                if verbose:
                    print(f"Error parsing remark JSON for query {idx}: {e}")
            
            search_results.append({
                "query_index": idx,
                "query": query,
                "papers": json.dumps(papers_data) if papers_data else text, # Store JSON string or raw text
                "success": item.get("success", bool(text)),
                "error": item.get("error"),
            })

        # Save references to JSON file
        try:
            with open("literature_review_references.json", "w") as f:
                json.dump(all_references, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"\n[STAGE 2] Saved {len(all_references)} references to literature_review_references.json")
        except Exception as e:
            if verbose:
                print(f"[STAGE 2] Error saving references file: {e}")

        return {"search_results": search_results, "all_references": all_references}

    # Stage 3: Paper analysis preparation
    def stage3_node(state: LiteratureReviewState) -> Dict[str, Any]:
        """Stage 3: Prepare paper analysis tasks."""
        return analyze_papers(state)

    # Stage 3 (parallel): Analyze papers
    def stage3_parallel_node(state: LiteratureReviewState) -> Dict[str, Any]:
        """Stage 3 (parallel): Analyze papers using parallel processing."""
        paper_tasks = state.get("paper_analysis_tasks", [])
        verbose = state.get("verbose", False)
        user_mode = state.get("mode", "lite")

        if not paper_tasks:
            return {"analyzed_papers": []}

        if verbose:
            print(f"\n[STAGE 3] Running parallel paper analysis (mode={user_mode})")

        # Create analysis queries for parallel processing
        analysis_queries = []
        for task in paper_tasks:
            query_text = (
                f"Analyze the following papers from the search query '{task['query']}':\n\n"
                f"{task['papers_info']}\n\n"
                f"For each paper, provide: summary, key findings, methodology, limitations, and relevance score (0-1)."
            )
            analysis_queries.append(query_text)

        # Use simpler direct LLM call instead of parallel agent for analysis
        # This is more suitable since we're not using tools, just analyzing text
        llm_instance = state.get("llm")
        analyzed_papers = []

        from langchain_core.messages import HumanMessage
        from .prompts import PAPER_ANALYSIS_PROMPT

        for i, query in enumerate(analysis_queries):
            if verbose:
                print(f"[STAGE 3] Analyzing paper group {i+1}/{len(analysis_queries)}")
            
            try:
                prompt = f"{PAPER_ANALYSIS_PROMPT}\n\n{query}"
                response = llm_instance.invoke([HumanMessage(content=prompt)])
                analysis_text = response.content

                # Try to parse JSON response
                import json
                try:
                    json_start = analysis_text.find("{")
                    json_end = analysis_text.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        paper_data = json.loads(analysis_text[json_start:json_end])
                        analyzed_papers.append(paper_data)
                    elif "[" in analysis_text:
                        # Multiple papers
                        json_start = analysis_text.find("[")
                        json_end = analysis_text.rfind("]") + 1
                        papers_list = json.loads(analysis_text[json_start:json_end])
                        analyzed_papers.extend(papers_list)
                except json.JSONDecodeError:
                    # Fallback: store as text summary
                    analyzed_papers.append({
                        "paper_id": f"paper_{i}",
                        "title": f"Papers from query {i+1}",
                        "summary": analysis_text[:500],
                        "key_findings": [],
                        "methodology": "",
                        "limitations": "",
                    })
            except Exception as e:
                if verbose:
                    print(f"[STAGE 3] Error analyzing paper group {i+1}: {e}")

        return {"analyzed_papers": analyzed_papers}

    # Stage 4: Synthesis
    def stage4_node(state: LiteratureReviewState) -> Dict[str, Any]:
        """Stage 4: Synthesize findings."""
        return synthesize_findings(state)

    # Stage 5: Final report
    def stage5_node(state: LiteratureReviewState) -> Dict[str, Any]:
        """Stage 5: Generate final literature review report."""
        return generate_final_report(state)

    # Add all nodes to workflow
    workflow.add_node("initialize", init_node)
    workflow.add_node("stage1_queries", stage1_node)
    workflow.add_node("stage2_parallel_search", stage2_parallel_node)
    workflow.add_node("stage3_prepare_analysis", stage3_node)
    workflow.add_node("stage3_parallel_analysis", stage3_parallel_node)
    workflow.add_node("stage4_synthesis", stage4_node)
    workflow.add_node("stage5_final_report", stage5_node)

    # Define edges (flow)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "stage1_queries")
    workflow.add_edge("stage1_queries", "stage2_parallel_search")
    workflow.add_edge("stage2_parallel_search", "stage3_prepare_analysis")
    workflow.add_edge("stage3_prepare_analysis", "stage3_parallel_analysis")
    workflow.add_edge("stage3_parallel_analysis", "stage4_synthesis")
    workflow.add_edge("stage4_synthesis", "stage5_final_report")
    workflow.add_edge("stage5_final_report", END)

    return workflow


def get_compiled_graph(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    save_graph_image: bool = True,
    image_name: str = "literature_review_agent_graph.png",
    mode: str = "lite",
) -> Any:
    """
    Create and compile the Literature Review Agent graph.

    Optionally saves a PNG visualization of the graph.

    Args:
        llm: Language model
        tools: Available tools
        system_prompt: User-provided system prompt
        save_graph_image: Whether to save PNG image of the graph
        image_name: Name/path for the saved image
        mode: Execution mode - "lite" or "full" (default: "lite")

    Returns:
        Compiled LangGraph agent
    """
    agent = create_literature_review_agent(llm, tools, system_prompt, verbose=False, mode=mode)

    if save_graph_image:
        try:
            agent.get_graph().draw_png(image_name)
            print(f"✓ Graph visualization saved to: {image_name}")
        except Exception as e:
            print(f"⚠️  Could not save graph image: {e}")
            print("   (This typically requires pygraphviz to be installed)")

            # Try saving as Mermaid instead
            try:
                mermaid_path = image_name.replace(".png", ".md")
                with open(mermaid_path, "w") as f:
                    f.write("```mermaid\n")
                    f.write(agent.get_graph().to_mermaid())
                    f.write("\n```")
                print(f"✓ Graph visualization saved to: {mermaid_path} (Mermaid format)")
            except Exception as e2:
                print(f"⚠️  Could not save Mermaid graph either: {e2}")

    return agent
