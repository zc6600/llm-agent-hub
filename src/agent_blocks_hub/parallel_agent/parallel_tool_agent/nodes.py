"""Node functions for the Parallel Tool Agent graph.

Implements parallel execution of LLM+tool calls with optional summarization.

Each query is first processed by an LLM bound with tools (``llm.bind_tools``),
allowing the model to decide *how* and *when* to call tools. The resulting
tool calls are then executed, and both the LLM's answer and tool outputs are
stored in the state. A second LLM pass can optionally synthesize a global
summary across all queries.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from .state import ParallelToolAgentState, ToolResult
from .prompts import get_combined_system_prompt


def _log(message: str, verbose: bool = False):
    """Helper function for conditional logging."""
    if verbose:
        print(message)


def initialize_state(state: ParallelToolAgentState) -> Dict[str, Any]:
    """
    Node: Initialize state with configuration.
    
    Prepares the state for parallel tool execution.
    
    Args:
        state: Current state
        
    Returns:
        Updated state with initialization data
    """
    verbose = state.get("verbose", False)
    queries_count = len(state.get('parallel_tool_agent_messages', []))
    tools_count = len(state.get('tools', []))
    enable_summarization = state.get("enable_summarization", False)
    
    print("\n[INITIALIZE] Setting up Parallel Tool Agent system")
    _log(f"[INITIALIZE] Number of queries: {queries_count}", verbose)
    _log(f"[INITIALIZE] Number of tools: {tools_count}", verbose)
    _log(f"[INITIALIZE] Summarization enabled: {enable_summarization}", verbose)
    _log(f"[INITIALIZE] Verbose mode: {verbose}", verbose)
    
    return {
        "tool_results": {},
        "final_summary": "",
    }


def run_parallel_tools(state: ParallelToolAgentState) -> Dict[str, Any]:
    """
     Node: Execute LLM+tool calls in parallel for each query.
    
     For each query, this node:
     1. Uses an LLM bound with tools (``llm.bind_tools(tools)``) to decide how to
         query tools.
     2. Executes any tool calls returned by the LLM.
     3. Stores a combined result containing the LLM's answer and a concise
         summary of tool executions.
    
    Args:
        state: Current state with queries and tools
        
    Returns:
        Updated state with tool results
    """
    queries = state.get("parallel_tool_agent_messages", [])
    tools = state.get("tools", [])
    verbose = state.get("verbose", False)
    system_prompt = state.get("system_prompt", "")
    llm = state.get("llm")
    
    if not queries:
        _log("[PARALLEL_TOOLS] No queries provided", verbose)
        return {"tool_results": {}}
    
    if not tools:
        raise ValueError("No tools configured in state")

    if llm is None:
        raise ValueError("LLM is required for LLM+tool querying in parallel_tool_agent")

    # Bind tools to LLM so it can decide how/when to call them
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    print(f"\n[PARALLEL_TOOLS] Starting parallel LLM+tool execution for {len(queries)} queries")
    print(f"[PARALLEL_TOOLS] Tools available for LLM binding: {[tool.name for tool in tools]}")
    _log(f"[PARALLEL_TOOLS] Queries: {queries}", verbose)
    
    # Execute LLM+tool pipeline in parallel for each query
    results: List[ToolResult] = []
    max_workers = min(len(queries), 5)  # Limit concurrent workers

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all query processing tasks
        future_to_idx = {
            executor.submit(
                _process_single_query,
                idx,
                query,
                llm_with_tools,
                tools,
                system_prompt,
                verbose,
            ): idx
            for idx, query in enumerate(queries)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
            except Exception as e:
                error_msg = f"Error in LLM+tool pipeline: {str(e)}"
                _log(f"  [Query {idx}] ✗ {error_msg}", verbose)
                result = {
                    "query_index": idx,
                    "query": queries[idx],
                    "result": "",
                    "tool_name": "llm_with_tools",
                    "success": False,
                    "error": error_msg,
                }
            results.append(result)
    
    # Convert results list to dictionary
    tool_results = {result["query_index"]: result for result in results}
    
    print(f"\n[PARALLEL_TOOLS] Parallel LLM+tool execution completed")
    print(f"[PARALLEL_TOOLS] Successful: {sum(1 for r in results if r['success'])}/{len(results)}")
    _log(f"[PARALLEL_TOOLS] Detailed results available in state", verbose)
    
    return {
        "tool_results": tool_results,
    }


def _process_single_query(
    query_index: int,
    query: str,
    llm_with_tools: Any,
    tools: List[Any],
    system_prompt: str,
    verbose: bool = False,
) -> ToolResult:
    """Process a single query using an LLM bound with tools.

    Flow:
    1. Build messages with optional system prompt and user query.
    2. Call ``llm_with_tools.invoke`` to let the LLM decide tool usage.
    3. Execute any returned tool calls and collect their outputs.
    4. Combine the LLM's answer and tool execution logs into ``ToolResult.result``.
    """
    _log(f"\n  [Query {query_index}] Processing query via LLM+tools: {query[:80]}...", verbose)

    # 1. Build messages
    messages: List[Any] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=query))

    # 2. Invoke LLM bound with tools
    response = llm_with_tools.invoke(messages)
    llm_answer = getattr(response, "content", "")

    tool_logs: List[str] = []

    # 3. Execute tool calls if present
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            selected_tool = next((t for t in tools if t.name == tool_name), None)
            if not selected_tool:
                msg = f"[Tool {tool_name}] not found for args={tool_args}"
                tool_logs.append(msg)
                _log(f"  [Query {query_index}] {msg}", verbose)
                continue

            try:
                result = selected_tool.invoke(tool_args)
                result_str = str(result)
                if len(result_str) > 400:
                    result_str = result_str[:400] + "..."
                msg = f"[Tool {tool_name}] args={tool_args} -> {result_str}"
                tool_logs.append(msg)
                _log(f"  [Query {query_index}] {msg}", verbose)
            except Exception as e:  # pragma: no cover - defensive logging
                err_msg = f"[Tool {tool_name}] ERROR: {str(e)} for args={tool_args}"
                tool_logs.append(err_msg)
                _log(f"  [Query {query_index}] {err_msg}", verbose)

    # 4. Combine into a single result string to keep state schema unchanged
    tool_section = "\n".join(tool_logs) if tool_logs else "No tool calls were executed."
    combined_result = f"""LLM Answer:\n{llm_answer}\n\nTool Executions:\n{tool_section}\n"""

    if verbose:
        print(f"\n  [Query {query_index}] === LLM ANSWER ===")
        print(llm_answer)
        print(f"  [Query {query_index}] === TOOL EXECUTIONS ===")
        print(tool_section)
        print(f"  [Query {query_index}] === END QUERY RESULT ===")

    return {
        "query_index": query_index,
        "query": query,
        "result": combined_result,
        "tool_name": "llm_with_tools",
        "success": True,
        "error": None,
    }


def summarize_results(state: ParallelToolAgentState) -> Dict[str, Any]:
    """
    Node: Optionally summarize all tool results into a coherent response.
    
    This node only runs if enable_summarization=True in the state.
    
    Args:
        state: Current state with tool results
        
    Returns:
        Updated state with final summary
    """
    enable_summarization = state.get("enable_summarization", False)
    verbose = state.get("verbose", False)
    
    # If summarization is disabled, just concatenate results
    if not enable_summarization:
        _log("[SUMMARIZE] Summarization disabled, concatenating results", verbose)
        tool_results = state.get("tool_results", {})
        
        concatenated = []
        for idx in sorted(tool_results.keys()):
            result = tool_results[idx]
            if result.get("success"):
                concatenated.append(f"Query: {result['query']}\n{result['result']}")
        
        return {"final_summary": "\n\n".join(concatenated)}
    
    # Run LLM-based summarization
    llm = state.get("llm")
    tool_results = state.get("tool_results", {})
    system_prompt = state.get("system_prompt", "")
    
    if not llm:
        raise ValueError("LLM required for summarization but not configured")
    
    print(f"\n[SUMMARIZE] Synthesizing results from {len(tool_results)} tool calls")
    _log(f"[SUMMARIZE] Using LLM for intelligent summarization", verbose)
    
    # Prepare input for summarization
    results_text = _format_results_for_summarization(tool_results, verbose)
    
    # Get combined system prompt
    combined_prompt = get_combined_system_prompt(system_prompt)
    
    # Create summarization prompt
    user_message = f"""Please synthesize the following tool results into a comprehensive summary:

{results_text}

Provide a well-structured, coherent summary that integrates all the information."""
    
    _log(f"[SUMMARIZE] Invoking LLM for summarization...", verbose)
    
    # Invoke LLM
    messages = [
        {"role": "system", "content": combined_prompt},
        {"role": "user", "content": user_message}
    ]
    
    response = llm.invoke([HumanMessage(content=f"{combined_prompt}\n\n{user_message}")])
    final_summary = response.content
    
    print(f"[SUMMARIZE] ✓ Summarization completed")
    print(f"[SUMMARIZE] Summary length: {len(final_summary)} chars")
    _log(f"[SUMMARIZE] Full summary:\n{final_summary}", verbose)
    
    return {
        "final_summary": final_summary,
    }


def _format_results_for_summarization(
    tool_results: Dict[int, ToolResult],
    verbose: bool = False
) -> str:
    """
    Format tool results into a text block for summarization.
    
    Args:
        tool_results: Dictionary of tool results
        verbose: Whether to log formatting details
        
    Returns:
        Formatted text string
    """
    formatted_parts = []
    
    for idx in sorted(tool_results.keys()):
        result = tool_results[idx]
        
        formatted_parts.append(f"\n{'='*80}")
        formatted_parts.append(f"Query {idx + 1}: {result['query']}")
        formatted_parts.append(f"Tool: {result['tool_name']}")
        formatted_parts.append(f"{'='*80}")
        
        if result.get("success"):
            formatted_parts.append(result['result'])
        else:
            formatted_parts.append(f"ERROR: {result.get('error', 'Unknown error')}")
    
    return "\n".join(formatted_parts)
