"""
Ideation Agent creation and configuration.

Creates a LangGraph-based ideation agent that implements a 6-stage research ideation process:
1. Initial parallel information gathering
2. Research gap identification
3. Gap-driven parallel information gathering
4. Creative idea generation through bit-flipping
5. Sub-hypothesis generation with verification experiments
6. Final comprehensive ideation report
"""

from typing import List, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from agent_blocks_hub.parallel_agent import create_parallel_agent


from .state import IdeationAgentState
from .prompts import REMARK_PROMPT
from .nodes import (
    initialize_state,
    generate_initial_queries,
    run_initial_parallel_information_gathering,
    analyze_research_gap,
    generate_gap_driven_queries,
    run_gap_driven_parallel_information_gathering,
    generate_creative_ideas,
    generate_sub_hypotheses,
    generate_final_report,
)


def create_ideation_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    mode: str = "lite",
) -> Any:
    """
    Create an Ideation Agent using LangGraph.

    The agent implements a 6-stage research ideation process:
    1. Generate queries and gather initial information (parallel)
    2. Analyze information to identify research gaps
    3. Generate gap-driven queries and gather targeted information (parallel)
    4. Apply bit-flipping to existing solutions to generate creative ideas
    5. Convert creative ideas into verifiable sub-hypotheses with experiments
    6. Synthesize everything into a comprehensive ideation report

    Args:
        llm: Language model to use for all stages
        tools: List of tools available to parallel agents
        system_prompt: Optional user-provided system prompt for ideation context
        verbose: Whether to print intermediate progress and debug information (default: False)
        mode: Parallel agent mode - "lite" uses parallel_tool_agent with summarization for faster execution,
              "full" uses parallel_react_agent for more detailed results with full reasoning (default: "lite")

    Returns:
        Compiled LangGraph agent ready to invoke

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_community.tools import DuckDuckGoSearchRun
        >>>
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> tools = [DuckDuckGoSearchRun()]
        >>>
        >>> # Use lite mode (faster with summarization, default)
        >>> agent = create_ideation_agent(
        ...     llm=llm,
        ...     tools=tools,
        ...     system_prompt="Focus on quantum computing applications",
        ...     verbose=True
        ... )
        >>>
        >>> # Or use full mode for more comprehensive results with detailed reasoning
        >>> agent_full = create_ideation_agent(
        ...     llm=llm,
        ...     tools=tools,
        ...     mode="full",
        ...     verbose=True
        ... )
        >>>
        >>> result = agent.invoke({
        ...     "ideation_message": "How can quantum computing improve drug discovery?"
        ... })
        >>>
        >>> print(result["final_idea_report"]["comprehensive_report"])
    """
    if system_prompt is None:
        system_prompt = ""

    if mode not in ("lite", "full"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'lite' or 'full'")

    # Create and return compiled graph
    graph = _create_graph(llm, tools, system_prompt, verbose, mode)
    return graph.compile()


def _create_graph(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str,
    verbose: bool = False,
    mode: str = "lite",
) -> StateGraph:
    """
    Create the LangGraph state graph for the Ideation Agent.

    Args:
        llm: Language model
        tools: Available tools for parallel agents
        system_prompt: User-provided system prompt
        verbose: Whether to print intermediate progress
        mode: Parallel agent mode - "lite" or "full"

    Returns:
        Configured StateGraph
    """
    # Create state graph
    workflow = StateGraph(IdeationAgentState)

    # Define initialization node wrapper
    def init_node(state: IdeationAgentState) -> Dict[str, Any]:
        """Initialize state with LLM, tools, and configuration."""
        return {
            "llm": llm,
            "tools": tools,
            "system_prompt": system_prompt,
            "verbose": verbose,
            "mode": mode,
            **initialize_state(state, verbose),
        }

    # Stage 1: Initial queries and parallel information gathering
    def stage1_node(state: IdeationAgentState) -> Dict[str, Any]:
        """Stage 1: Generate initial queries."""
        return generate_initial_queries(state)

    def stage1_parallel_node(state: IdeationAgentState) -> Dict[str, Any]:
        """Stage 1 (parallel): Gather information for initial queries using parallel agent."""
        initial_queries = state.get("initial_queries", [])
        verbose = state.get("verbose", False)
        parallel_mode = state.get("mode", "lite")

        if not initial_queries:
            return {"gathered_information": []}

        if verbose:
            print(f"\n[STAGE 1.5] Running parallel information gathering for initial queries (mode={parallel_mode})")

        if parallel_mode == "lite":
            # Use parallel_tool_agent with summarization for faster execution
            # Add explicit instruction to require tool usage with simple search keywords
            tool_required_prompt = (
                "You are a research assistant helping with scientific literature search. "
                "For EVERY query you receive, you MUST use the available tools to search for relevant academic papers and research. "
                "Do NOT answer from your own knowledge - ALWAYS use the tools to find current research papers. "
                "\n\n"
                "CRITICAL: When calling search tools, use SIMPLE, BROAD keywords (3-5 key terms max):\n"
                "❌ BAD: 'Foundational algorithms and limitations of machine learning in predicting 3D protein structure'\n"
                "✅ GOOD: 'machine learning protein structure prediction'\n"
                "\n"
                "Keep search queries SHORT and focused on core concepts to maximize results.\n\n"
                + state.get("system_prompt", "")
            )
            parallel_agent = create_parallel_agent(
                llm=llm,
                tools=tools,
                system_prompt=tool_required_prompt,
                verbose=verbose,
                enable_summarization=False,
                enable_remark=True,
                remark_prompt=REMARK_PROMPT,
                mode="direct"
            )

            result = parallel_agent.invoke({
                "parallel_agent_message": initial_queries,
                "verbose": verbose,
            })

            gathered_info = []
            remarks = result.get("remark", {})
            agent_results = result.get("agent_results", {})
            
            for idx, query in enumerate(initial_queries):
                item = agent_results.get(idx, {})
                text = remarks.get(idx, "") or ""
                gathered_info.append({
                    "query_index": idx,
                    "query": query,
                    "information": text,
                    "success": item.get("success", bool(text)),
                    "error": item.get("error"),
                })
        else:
            # Use parallel_react_agent for comprehensive results with full reasoning (full mode)
            parallel_agent = create_parallel_agent(
                llm=llm,
                tools=tools,
                system_prompt=state.get("system_prompt", ""),
                verbose=verbose,
                enable_summarization=False,
                enable_remark=True,
                remark_prompt=REMARK_PROMPT,
                mode="react"
            )

            result = parallel_agent.invoke({
                "parallel_react_agent_messages": initial_queries,
                "verbose": verbose,
            })

            gathered_info = []
            remarks = result.get("remark", {})
            agent_results = result.get("agent_results", {})
            
            if remarks or agent_results:
                for idx, query in enumerate(initial_queries):
                    item = agent_results.get(idx, {})
                    text = remarks.get(idx, "") or ""
                    gathered_info.append({
                        "query_index": idx,
                        "query": query,
                        "information": text,
                        "success": item.get("success", bool(text)),
                        "error": item.get("error"),
                    })
            else:
                for idx, query in enumerate(initial_queries):
                    gathered_info.append({
                        "query_index": idx,
                        "query": query,
                        "information": "",
                        "success": False,
                        "error": "No results returned from parallel agent",
                    })

        return {"gathered_information": gathered_info}

    # Stage 2: Research gap analysis
    def stage2_node(state: IdeationAgentState) -> Dict[str, Any]:
        """Stage 2: Analyze research gap."""
        return analyze_research_gap(state)

    # Stage 3: Gap-driven queries and parallel information gathering
    def stage3_node(state: IdeationAgentState) -> Dict[str, Any]:
        """Stage 3: Generate gap-driven queries."""
        return generate_gap_driven_queries(state)

    def stage3_parallel_node(state: IdeationAgentState) -> Dict[str, Any]:
        """Stage 3 (parallel): Gather information for gap-driven queries using parallel agent."""
        gap_driven_queries = state.get("gap_driven_queries", [])
        verbose = state.get("verbose", False)
        parallel_mode = state.get("mode", "lite")

        if not gap_driven_queries:
            return {"gap_driven_information": []}

        if verbose:
            print(f"\n[STAGE 3.5] Running parallel gap-driven information gathering (mode={parallel_mode})")

        if parallel_mode == "lite":
            # Use parallel_tool_agent with summarization for faster execution
            # Add explicit instruction to require tool usage with simple search keywords
            tool_required_prompt = (
                "You are a research assistant helping with scientific literature search. "
                "For EVERY query you receive, you MUST use the available tools to search for relevant academic papers and research. "
                "Do NOT answer from your own knowledge - ALWAYS use the tools to find current research papers. "
                "\n\n"
                "CRITICAL: When calling search tools, use SIMPLE, BROAD keywords (3-5 key terms max):\n"
                "❌ BAD: 'Emerging technologies for addressing challenges in real-time processing'\n"
                "✅ GOOD: 'real-time processing methods'\n"
                "\n"
                "Keep search queries SHORT and focused on core concepts to maximize results.\n\n"
                + state.get("system_prompt", "")
            )
            parallel_agent = create_parallel_agent(
                llm=llm,
                tools=tools,
                system_prompt=tool_required_prompt,
                verbose=verbose,
                enable_summarization=False,
                enable_remark=True,
                remark_prompt=REMARK_PROMPT,
                mode="tool_calling"
            )

            result = parallel_agent.invoke({
                "parallel_agent_message": gap_driven_queries,
                "verbose": verbose,
            })

            gap_info = []
            remarks = result.get("remark", {})
            agent_results = result.get("agent_results", {})
            
            for idx, query in enumerate(gap_driven_queries):
                item = agent_results.get(idx, {})
                text = remarks.get(idx, "") or ""
                gap_info.append({
                    "query_index": idx,
                    "query": query,
                    "information": text,
                    "success": item.get("success", bool(text)),
                    "error": item.get("error"),
                })
        else:
            # Use parallel_react_agent for comprehensive results with full reasoning (full mode)
            parallel_agent = create_parallel_agent(
                llm=llm,
                tools=tools,
                system_prompt=state.get("system_prompt", ""),
                verbose=verbose,
                enable_summarization=False,
                enable_remark=True,
                remark_prompt=REMARK_PROMPT,
                mode="react"
            )

            result = parallel_agent.invoke({
                "parallel_react_agent_messages": gap_driven_queries,
                "verbose": verbose,
            })

            gap_info = []
            remarks = result.get("remark", {})
            agent_results = result.get("agent_results", {})
            
            if remarks or agent_results:
                for idx, query in enumerate(gap_driven_queries):
                    item = agent_results.get(idx, {})
                    text = remarks.get(idx, "") or ""
                    gap_info.append({
                        "query_index": idx,
                        "query": query,
                        "information": text,
                        "success": item.get("success", bool(text)),
                        "error": item.get("error"),
                    })
            else:
                for idx, query in enumerate(gap_driven_queries):
                    gap_info.append({
                        "query_index": idx,
                        "query": query,
                        "information": "",
                        "success": False,
                        "error": "No results returned from parallel agent",
                    })

        return {"gap_driven_information": gap_info}

    # Stage 4: Bit-flipping creative ideas
    def stage4_node(state: IdeationAgentState) -> Dict[str, Any]:
        """Stage 4: Generate creative ideas."""
        return generate_creative_ideas(state)

    # Stage 5: Sub-hypothesis generation
    def stage5_node(state: IdeationAgentState) -> Dict[str, Any]:
        """Stage 5: Generate sub-hypotheses with verification experiments."""
        return generate_sub_hypotheses(state)

    # Stage 6: Final report synthesis
    def stage6_node(state: IdeationAgentState) -> Dict[str, Any]:
        """Stage 6: Generate final ideation report."""
        return generate_final_report(state)

    # Add all nodes to workflow
    workflow.add_node("initialize", init_node)
    workflow.add_node("stage1_queries", stage1_node)
    workflow.add_node("stage1_parallel", stage1_parallel_node)
    workflow.add_node("stage2_gap_analysis", stage2_node)
    workflow.add_node("stage3_queries", stage3_node)
    workflow.add_node("stage3_parallel", stage3_parallel_node)
    workflow.add_node("stage4_creative_ideas", stage4_node)
    workflow.add_node("stage5_sub_hypotheses", stage5_node)
    workflow.add_node("stage6_final_report", stage6_node)

    # Define edges (flow)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "stage1_queries")
    workflow.add_edge("stage1_queries", "stage1_parallel")
    workflow.add_edge("stage1_parallel", "stage2_gap_analysis")
    # For speed consideration
    #workflow.add_edge("stage2_gap_analysis", "stage3_queries")
    #workflow.add_edge("stage3_queries", "stage3_parallel")
    #workflow.add_edge("stage3_parallel", "stage4_creative_ideas")
    workflow.add_edge("stage2_gap_analysis", "stage4_creative_ideas")
    workflow.add_edge("stage4_creative_ideas", "stage5_sub_hypotheses")
    workflow.add_edge("stage5_sub_hypotheses", "stage6_final_report")
    workflow.add_edge("stage6_final_report", END)

    return workflow


def get_compiled_graph(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    save_graph_image: bool = True,
    image_name: str = "ideation_agent_graph.png",
    mode: str = "lite",
) -> Any:
    """
    Create and compile the Ideation Agent graph.

    Optionally saves a PNG visualization of the graph.

    Args:
        llm: Language model
        tools: Available tools
        system_prompt: User-provided system prompt
        save_graph_image: Whether to save PNG image of the graph
        image_name: Name/path for the saved image
        mode: Parallel agent mode - "lite" or "full" (default: "lite")

    Returns:
        Compiled LangGraph agent
    """
    agent = create_ideation_agent(llm, tools, system_prompt, mode=mode)

    if save_graph_image:
        try:
            # Save PNG visualization
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
