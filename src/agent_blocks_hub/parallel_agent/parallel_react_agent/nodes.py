"""
Node functions for the Parallel React Agent graph.

Implements parallel execution of ReAct agents and result summarization.
"""

import asyncio
from typing import Any, Dict, List
from langchain.agents import create_agent
from .state import ParallelReactAgentState, AgentResult
from .prompts import get_combined_system_prompt


def _log(message: str, verbose: bool = False):
    """Helper function for conditional logging."""
    if verbose:
        print(message)


def initialize_state(state: ParallelReactAgentState) -> Dict[str, Any]:
    """
    Node: Initialize state with LLM, tools, and configuration.
    
    Prepares the state for parallel agent execution.
    
    Args:
        state: Current state
        
    Returns:
        Updated state with initialization data
    """
    verbose = state.get("verbose", False)
    queries_count = len(state.get('parallel_react_agent_messages', []))
    tools_count = len(state.get('tools', []))
    
    print("\n[INITIALIZE] Setting up Parallel React Agent system")
    _log(f"[INITIALIZE] Number of queries: {queries_count}", verbose)
    _log(f"[INITIALIZE] Number of tools: {tools_count}", verbose)
    _log(f"[INITIALIZE] Verbose mode: {verbose}", verbose)
    
    return {
        "agent_results": {},
    }


def run_parallel_agents(state: ParallelReactAgentState) -> Dict[str, Any]:
    """
    Node: Run ReAct agents in parallel for each query.
    
    Creates and executes a ReAct agent for each query in parallel.
    Each agent uses the same tools and system prompt but processes
    different queries.
    
    Args:
        state: Current state with queries, LLM, and tools
        
    Returns:
        Updated state with agent results
    """
    queries = state.get("parallel_react_agent_messages", [])
    llm = state.get("llm")
    tools = state.get("tools", [])
    system_prompt = state.get("system_prompt", "")
    verbose = state.get("verbose", False)
    
    if not llm:
        raise ValueError("LLM not configured in state")
    
    if not queries:
        _log("[PARALLEL_AGENTS] No queries provided", verbose)
        return {"agent_results": {}}
    
    print(f"\n[PARALLEL_AGENTS] Starting parallel execution for {len(queries)} queries")
    print(f"[PARALLEL_AGENTS] Tools available: {[tool.name for tool in tools]}")
    _log(f"[PARALLEL_AGENTS] Queries: {queries}", verbose)
    
    # Combine system prompts
    combined_prompt = get_combined_system_prompt("react", system_prompt)
    
    # Use synchronous execution path for better logging visibility
    # This ensures all _log() calls with verbose=True are properly displayed
    _log(f"[PARALLEL_AGENTS] Using synchronous execution for verbose logging", verbose)
    
    results = []
    for idx, query in enumerate(queries):
        result = run_single_agent_sync(idx, query, llm, tools, combined_prompt, verbose)
        results.append(result)
    
    # Convert results list to dictionary
    agent_results = {result["query_index"]: result for result in results}
    
    print(f"\n[PARALLEL_AGENTS] Parallel execution completed")
    print(f"[PARALLEL_AGENTS] Successful: {sum(1 for r in results if r['success'])}/{len(results)}")
    _log(f"[PARALLEL_AGENTS] Detailed results available in state", verbose)
    
    return {
        "agent_results": agent_results,
    }


def run_single_agent_sync(
    query_index: int,
    query: str,
    llm: Any,
    tools: List[Any],
    system_prompt: str,
    verbose: bool = False
) -> AgentResult:
    """
    Run a single ReAct agent synchronously (fallback for when asyncio is not available).
    
    Args:
        query_index: Index of the query
        query: Query string
        llm: Language model
        tools: Tools available to agent
        system_prompt: System prompt for agent
        verbose: Whether to print detailed execution logs
        
    Returns:
        AgentResult with the agent's response
    """
    try:
        _log(f"\n  [Agent {query_index}] Processing query: {query}", verbose)
        _log(f"  [Agent {query_index}] Available tools: {[tool.name for tool in tools]}", verbose)
        
        # Create ReAct agent
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )
        
        _log(f"  [Agent {query_index}] Agent created, invoking...", verbose)
        
        # Invoke agent with new LangChain API format (messages instead of input)
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": query}
            ]
        })
        
        _log(f"  [Agent {query_index}] LLM Response received", verbose)
        
        # Extract output from messages (new LangChain API format)
        messages = result.get("messages", [])
        output = ""
        intermediate_steps = []
        
        # Extract content from the last AI message
        if messages:
            for msg in reversed(messages):
                # Look for AIMessage with content
                if hasattr(msg, 'content') and msg.content:
                    output = msg.content
                    break
        
        # Try to extract intermediate steps from message history
        # Tool calls would be in ToolMessage objects
        for msg in messages:
            msg_type = type(msg).__name__
            if msg_type == 'ToolMessage' or (hasattr(msg, 'type') and msg.type == 'tool'):
                intermediate_steps.append(msg)
        
        # Log intermediate steps if verbose
        if verbose:
            print(f"\n  [Agent {query_index}] === DETAILED EXECUTION LOG ===")
            print(f"  [Agent {query_index}] Intermediate steps: {len(intermediate_steps)} found")
            if intermediate_steps:
                for step_idx, step in enumerate(intermediate_steps):
                    print(f"\n  [Agent {query_index}] --- Step {step_idx + 1} ---")
                    # Handle both tuple format (action, observation) and dict format
                    if isinstance(step, tuple) and len(step) == 2:
                        action, observation = step
                        tool_name = getattr(action, 'tool', 'unknown')
                        tool_input = getattr(action, 'tool_input', '')
                        print(f"  [Agent {query_index}] Tool: '{tool_name}'")
                        print(f"  [Agent {query_index}] Input: {tool_input}")
                        print(f"  [Agent {query_index}] Output (FULL):")
                        print(f"  {observation}")
                        print(f"  [Agent {query_index}] (End of output for step {step_idx + 1})")
                    else:
                        # Generic handling for other formats
                        print(f"  [Agent {query_index}] Step data: {step}")
            
            print(f"\n  [Agent {query_index}] === FINAL AGENT OUTPUT ===")
            print(f"  [Agent {query_index}] Output length: {len(output)} chars")
            print(f"  [Agent {query_index}] Full output:")
            print(f"  {output}")
            print(f"  [Agent {query_index}] (End of final output)")
            print(f"  [Agent {query_index}] ✓ Completed")
        else:
            _log(f"  [Agent {query_index}] Output length: {len(output)} chars", verbose)
            _log(f"  [Agent {query_index}] ✓ Completed", verbose)
        
        return {
            "query_index": query_index,
            "query": query,
            "result": output,
            "intermediate_steps": intermediate_steps,
            "success": True,
            "error": None,
        }
    
    except Exception as e:
        _log(f"  [Agent {query_index}] ✗ Error: {str(e)}", verbose)
        import traceback
        _log(f"  [Agent {query_index}] Traceback: {traceback.format_exc()}", verbose)
        return {
            "query_index": query_index,
            "query": query,
            "result": "",
            "intermediate_steps": [],
            "success": False,
            "error": str(e),
        }


def summarizing_agent(state: ParallelReactAgentState) -> Dict[str, Any]:
    """
    Node: Optionally summarize results from all parallel agents.
    
    If enable_summarization=True, uses a Summarizing Agent to integrate
    and synthesize results. Otherwise, provides a simple concatenation.
    
    Args:
        state: Current state with agent results
        
    Returns:
        Updated state with final summary
    """
    agent_results = state.get("agent_results", {})
    llm = state.get("llm")
    tools = state.get("tools", [])
    system_prompt = state.get("system_prompt", "")
    queries = state.get("parallel_react_agent_messages", [])
    verbose = state.get("verbose", False)
    enable_summarization = state.get("enable_summarization", True)
    
    print("\n[SUMMARIZING_AGENT] Starting result synthesis")
    _log(f"[SUMMARIZING_AGENT] Summarization enabled: {enable_summarization}", verbose)
    
    if not agent_results:
        _log("[SUMMARIZING_AGENT] No results to summarize", verbose)
        return {"final_summary": "No results were generated from the parallel agents."}
    
    # If summarization is disabled, use simple concatenation
    if not enable_summarization:
        _log("[SUMMARIZING_AGENT] Summarization disabled, using simple concatenation", verbose)
        final_summary = _fallback_summary(queries, agent_results)
        print(f"[SUMMARIZING_AGENT] ✓ Summary prepared (no LLM synthesis)")
        return {"final_summary": final_summary}
    
    # Prepare summary input for LLM-based summarization
    summary_input = _prepare_summary_input(queries, agent_results, verbose)
    
    successful_results = sum(1 for r in agent_results.values() if r['success'])
    print(f"[SUMMARIZING_AGENT] Synthesizing {successful_results}/{len(agent_results)} results with LLM")
    
    try:
        # Get combined system prompt for summarizing agent
        combined_prompt = get_combined_system_prompt("summarizing", system_prompt)
        
        _log("[SUMMARIZING_AGENT] Creating summarizing agent...", verbose)
        
        # Create summarizing agent
        summarizing_agent_obj = create_agent(
            model=llm,
            tools=tools,  # Can use same tools for additional research if needed
            system_prompt=combined_prompt,
        )
        
        _log("[SUMMARIZING_AGENT] Invoking summarizing agent...", verbose)
        
        # Invoke summarizing agent with new LangChain API format (messages instead of input)
        result = summarizing_agent_obj.invoke({
            "messages": [
                {"role": "user", "content": summary_input}
            ]
        })
        
        # Extract output from messages (new LangChain API format)
        messages = result.get("messages", [])
        final_summary = ""
        
        # Extract content from the last AI message
        if messages:
            for msg in reversed(messages):
                # Look for AIMessage with content
                if hasattr(msg, 'content') and msg.content:
                    final_summary = msg.content
                    break

        
        if verbose:
            print(f"\n[SUMMARIZING_AGENT] === FINAL SUMMARY ===")
            print(f"[SUMMARIZING_AGENT] Summary length: {len(final_summary)} chars")
            print(f"[SUMMARIZING_AGENT] Full summary:")
            print(f"{final_summary}")
            print(f"[SUMMARIZING_AGENT] (End of summary)")
        else:
            _log(f"[SUMMARIZING_AGENT] Generated summary length: {len(final_summary)} chars", verbose)
        
        print("[SUMMARIZING_AGENT] ✓ Summary generated successfully")
        
    except Exception as e:
        print(f"[SUMMARIZING_AGENT] ✗ Error during summarization: {str(e)}")
        _log(f"[SUMMARIZING_AGENT] Using fallback summary", verbose)
        # Fallback: simple concatenation of results
        final_summary = _fallback_summary(queries, agent_results)
    
    return {
        "final_summary": final_summary,
    }


def _prepare_summary_input(queries: List[str], agent_results: Dict[int, AgentResult], verbose: bool = False) -> str:
    """
    Prepare input text for summarizing agent.
    
    Formats all query-result pairs for the summarizing agent to process.
    
    Args:
        queries: Original queries
        agent_results: Results from all agents
        verbose: Whether to print detailed logs
        
    Returns:
        Formatted string for summarizing agent
    """
    if verbose:
        print("\n[SUMMARY_INPUT] === PREPARING SUMMARY INPUT ===")
    
    summary_text = "Please analyze and synthesize the following research results:\n\n"
    
    for idx, query in enumerate(queries):
        result = agent_results.get(idx)
        if result:
            summary_text += f"Query {idx + 1}: {query}\n"
            if result["success"]:
                summary_text += f"Response: {result['result']}\n\n"
                if verbose:
                    print(f"\n[SUMMARY_INPUT] Query {idx + 1}: {query}")
                    print(f"[SUMMARY_INPUT] Response length: {len(result['result'])} chars")
                    print(f"[SUMMARY_INPUT] Full response:")
                    print(f"{result['result']}")
                    print(f"[SUMMARY_INPUT] (End of Query {idx + 1} response)")
            else:
                summary_text += f"Error: {result['error']}\n\n"
                if verbose:
                    print(f"[SUMMARY_INPUT] Query {idx + 1}: Failed with error: {result['error']}")
    
    summary_text += "\nPlease integrate these responses by analyzing their internal structures and logical connections, " \
                   "then provide a comprehensive and coherent synthesis."
    
    if verbose:
        print(f"\n[SUMMARY_INPUT] === SUMMARY INPUT PREPARED ===")
        print(f"[SUMMARY_INPUT] Total input length: {len(summary_text)} chars")
    
    return summary_text


def _fallback_summary(queries: List[str], agent_results: Dict[int, AgentResult]) -> str:
    """
    Fallback summary when agent fails.
    
    Simple concatenation of successful results.
    
    Args:
        queries: Original queries
        agent_results: Results from all agents
        
    Returns:
        Simple concatenated summary
    """
    summary = "## Research Results Summary\n\n"
    
    for idx, query in enumerate(queries):
        result = agent_results.get(idx)
        if result and result["success"]:
            summary += f"### Query {idx + 1}: {query}\n"
            summary += f"{result['result']}\n\n"
    
    return summary
