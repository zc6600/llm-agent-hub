"""
Node functions for the Parallel Direct Tool Agent graph.

Implements parallel execution of DIRECT tool calls with optional summarization.

Key difference from parallel_tool_agent:
- Directly calls tool.run(query) without LLM tool calling
- Bypasses llm.bind_tools() entirely
- Suitable for models with poor tool calling support (e.g., DeepSeek V3)
- Maintains same interface and behavior for summarization
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from .state import ParallelDirectToolAgentState, ToolResult
from .prompts import get_combined_system_prompt, SUMMARIZATION_PROMPT_TEMPLATE, DEFAULT_REMARK_PROMPT


def _log(message: str, verbose: bool = False):
    """Helper function for conditional logging."""
    if verbose:
        print(message)


def initialize_state(state: ParallelDirectToolAgentState) -> Dict[str, Any]:
    """
    Node: Initialize state with configuration.
    
    Prepares the state for parallel direct tool execution.
    
    Args:
        state: Current state
        
    Returns:
        Updated state with initialization data
    """
    verbose = state.get("verbose", False)
    queries_count = len(state.get('parallel_agent_message', []))
    tools = state.get('tools')
    tools_count = len(tools) if tools is not None else 0
    enable_summarization = state.get("enable_summarization", False)
    enable_remark = state.get("enable_remark", False)
    remark_prompt = state.get("remark_prompt")
    summarization_prompt = state.get("summarization_prompt")
    
    # Set default prompts if not provided
    if remark_prompt is None:
        remark_prompt = DEFAULT_REMARK_PROMPT
    if summarization_prompt is None:
        summarization_prompt = SUMMARIZATION_PROMPT_TEMPLATE
    
    print("\n[INITIALIZE] Setting up Parallel Direct Tool Agent system")
    _log(f"[INITIALIZE] Number of queries: {queries_count}", verbose)
    _log(f"[INITIALIZE] Number of tools: {tools_count}", verbose)
    _log(f"[INITIALIZE] Summarization enabled: {enable_summarization}", verbose)
    _log(f"[INITIALIZE] Remark enabled: {enable_remark}", verbose)
    _log(f"[INITIALIZE] Mode: DIRECT TOOL INVOCATION (no LLM tool calling)", verbose)
    
    return {
        "tool_results": {},
        "final_summary": "",
        "enable_remark": enable_remark,
        "remark_prompt": remark_prompt,
        "summarization_prompt": summarization_prompt,
    }


def run_parallel_direct_tools(state: ParallelDirectToolAgentState) -> Dict[str, Any]:
    """
    Node: Execute DIRECT tool calls in parallel for each query.
    
    Key difference: This directly calls tool.run(query) without LLM tool calling.
    
    For each query, this node:
    1. Directly invokes tool.run(query=query) (bypasses LLM)
    2. Captures the raw tool output
    3. Stores the result for optional summarization
    
    Args:
        state: Current state with queries and tools
        
    Returns:
        Updated state with tool results
    """
    queries = state.get("parallel_agent_message", [])
    tools = state.get("tools", [])
    verbose = state.get("verbose", False)
    tool_name = state.get("tool_name")
    enable_remark = state.get("enable_remark", False)
    remark_prompt = state.get("remark_prompt", "")
    llm = state.get("llm")
    if enable_remark and llm is None:
        raise ValueError("LLM is required when enable_remark=True")
    
    if not queries:
        _log("[PARALLEL_DIRECT_TOOLS] No queries provided", verbose)
        return {"tool_results": {}}
    
    if not tools:
        raise ValueError("No tools configured in state")

    # Select tool to use
    if tool_name:
        selected_tool = next((t for t in tools if t.name == tool_name), None)
        if not selected_tool:
            raise ValueError(f"Tool '{tool_name}' not found in available tools")
    else:
        selected_tool = tools[0]  # Use first tool by default

    print(f"\n[PARALLEL_DIRECT_TOOLS] Starting parallel DIRECT tool execution for {len(queries)} queries")
    print(f"[PARALLEL_DIRECT_TOOLS] Selected tool: {selected_tool.name}")
    print(f"[PARALLEL_DIRECT_TOOLS] Mode: Direct tool.run() invocation (NO LLM tool calling)")
    _log(f"[PARALLEL_DIRECT_TOOLS] Queries: {queries}", verbose)
    
    # Execute tool.run() in parallel for each query
    results: List[ToolResult] = []
    max_workers = min(len(queries), 5)  # Limit concurrent workers

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all query processing tasks
        future_to_idx = {
            executor.submit(
                _process_single_query_direct,
                idx,
                query,
                selected_tool,
                verbose,
                enable_remark,
                remark_prompt,
                llm,
            ): idx
            for idx, query in enumerate(queries)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
            except Exception as e:
                error_msg = f"Error in direct tool execution: {str(e)}"
                _log(f"  [Query {idx}] ✗ {error_msg}", verbose)
                result = {
                    "query_index": idx,
                    "query": queries[idx],
                    "result": "",
                    "tool_name": selected_tool.name,
                    "success": False,
                    "error": error_msg,
                    "remark": None,
                }
            results.append(result)
    
    # Convert results list to dictionary
    tool_results = {result["query_index"]: result for result in results}
    
    print(f"\n[PARALLEL_DIRECT_TOOLS] Parallel direct tool execution completed")
    print(f"[PARALLEL_DIRECT_TOOLS] Successful: {sum(1 for r in results if r['success'])}/{len(results)}")
    _log(f"[PARALLEL_DIRECT_TOOLS] Detailed results available in state", verbose)
    
    return {
        "tool_results": tool_results,
    }


def _process_single_query_direct(
    query_index: int,
    query: str,
    tool: Any,
    verbose: bool = False,
    enable_remark: bool = False,
    remark_prompt: str = "",
    llm: Any = None,
) -> ToolResult:
    """
    Process a single query by DIRECTLY calling tool.run().
    
    This bypasses LLM tool calling entirely and directly invokes the tool.
    
    Args:
        query_index: Index of the query
        query: The search query
        tool: The tool instance to call directly
        verbose: Whether to print detailed logs
        
    Returns:
        ToolResult with the direct tool output
    """
    _log(f"  [Query {query_index}] Processing query: '{query}'", verbose)
    if hasattr(tool, "invoke"):
        _log(f"  [Query {query_index}] Directly calling {tool.name}.invoke(args={{'query': '{query}'}})", verbose)
    else:
        _log(f"  [Query {query_index}] Directly calling {tool.name}.run(query='{query}')", verbose)
    
    try:
        # Direct call: prefer LangChain invoke if available, otherwise BaseTool.run
        if hasattr(tool, "invoke"):
            tool_output = tool.invoke({"query": query})
        else:
            tool_output = tool.run(query=query)
        
        _log(f"  [Query {query_index}] ✓ Tool returned {(str(tool_output))} characters", verbose)
        
        # Generate remark if enabled
        remark = None
        if enable_remark and llm is not None:
            try:
                _log(f"  [Query {query_index}] Generating remark with LLM", verbose)
                from langchain_core.messages import HumanMessage
                remark_message = HumanMessage(content=remark_prompt.format(query=query, result=str(tool_output), tool=tool.name))
                remark_response = llm.invoke([remark_message])

                remark = remark_response.content.strip()
                _log(f"  [Query {query_index}] ✓ Remark generated: {remark[:100]}...", verbose)
            except Exception as e:
                _log(f"  [Query {query_index}] ✗ Failed to generate remark: {str(e)}", verbose)
                remark = None
        
        return {
            "query_index": query_index,
            "query": query,
            "result": str(tool_output),
            "tool_name": tool.name,
            "success": True,
            "error": None,
            "remark": remark,
        }
        
    except Exception as e:
        error_msg = f"Tool execution failed: {str(e)}"
        _log(f"  [Query {query_index}] ✗ {error_msg}", verbose)
        
        return {
            "query_index": query_index,
            "query": query,
            "result": "",
            "tool_name": tool.name,
            "success": False,
            "error": error_msg,
            "remark": None,
        }


def summarize_results(state: ParallelDirectToolAgentState) -> Dict[str, Any]:
    """
    Node: Synthesize all tool results into a coherent summary using LLM.
    
    This step is identical to parallel_tool_agent - uses LLM to create
    an intelligent summary of all gathered information.
    
    Args:
        state: Current state with tool results
        
    Returns:
        Updated state with final summary
    """
    tool_results = state.get("tool_results", {})
    llm = state.get("llm")
    system_prompt = state.get("system_prompt", "")
    verbose = state.get("verbose", False)
    summarization_prompt = state.get("summarization_prompt", SUMMARIZATION_PROMPT_TEMPLATE)
    
    if not tool_results:
        _log("[SUMMARIZE] No results to summarize", verbose)
        return {"final_summary": "No results available for summarization."}
    
    if llm is None:
        raise ValueError("LLM is required for summarization")
    
    print(f"\n[SUMMARIZE] Synthesizing {len(tool_results)} tool results into summary")
    
    # Combine all results into a single text
    combined_results = []
    for idx in sorted(tool_results.keys()):
        result = tool_results[idx]
        query = result.get("query", "")
        tool_name = result.get("tool_name", "unknown")
        success = result.get("success", False)
        
        if success:
            result_text = result.get("result", "")
            remark = result.get("remark", None)
            result_entry = f"Query {idx}: {query}\nTool: {tool_name}\n\n{result_text}\n"
            if remark:
                result_entry += f"Remark: {remark}\n"
            combined_results.append(result_entry)
        else:
            error = result.get("error", "Unknown error")
            remark = result.get("remark", None)
            result_entry = f"Query {idx}: {query}\nTool: {tool_name}\nError: {error}\n"
            if remark:
                result_entry += f"Remark: {remark}\n"
            combined_results.append(result_entry)
    
    combined_text = "\n" + "="*80 + "\n".join(combined_results)
    
    # Generate summary using LLM
    _log("[SUMMARIZE] Generating summary with LLM", verbose)
    
    system_message = SystemMessage(content=get_combined_system_prompt(system_prompt))
    user_message = HumanMessage(content=summarization_prompt.format(
        combined_results=combined_text
    ))
    
    try:
        response = llm.invoke([system_message, user_message])
        summary = response.content
        
        print(f"[SUMMARIZE] ✓ Summary generated ({len(summary)} characters)")
        _log(f"[SUMMARIZE] Summary preview: {summary[:200]}...", verbose)
        
        return {"final_summary": summary}
        
    except Exception as e:
        error_msg = f"Error generating summary: {str(e)}"
        print(f"[SUMMARIZE] ✗ {error_msg}")
        return {"final_summary": f"Error: {error_msg}"}