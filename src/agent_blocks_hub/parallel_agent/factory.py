"""
Unified Parallel Agent Factory

A unified interface for creating parallel agents with different execution modes:
- "tool_calling": LLM decides when/how to call tools (original parallel_tool_agent)
- "direct": Directly call tool.run() without LLM tool calling (parallel_direct_tool_agent)
- "react": Full ReAct reasoning loops (parallel_react_agent)

This provides a single entry point with consistent interface across all modes.
"""

from typing import List, Optional, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool


def create_parallel_agent(
    llm: Optional[BaseChatModel] = None,
    tools: List[BaseTool] = None,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    mode: str = "direct",
    enable_summarization: bool = False,
    tool_name: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Unified factory function for creating parallel agents with different execution modes.
    
    This is the RECOMMENDED way to create parallel agents. It provides a single interface
    that automatically selects the appropriate implementation based on the mode parameter.
    
    Modes:
    ------
    1. "direct" (RECOMMENDED for poor tool-calling models like DeepSeek V3)
       - Directly calls tool.run() without LLM tool calling
       - Fastest and most reliable for simple information gathering
       - No LLM needed if enable_summarization=False
       - Best for: Semantic Scholar searches, simple API calls
       
    2. "tool_calling" (for models with good tool calling support)
       - LLM decides when/how to call tools using llm.bind_tools()
       - More intelligent tool usage decisions
       - Requires model with good tool calling support
       - Best for: GPT-4, Claude with complex multi-tool scenarios
       
    3. "react" (for complex reasoning tasks)
       - Full ReAct reasoning loops with thought chains
       - Most comprehensive but slowest
       - Best for: Complex research tasks requiring multi-step reasoning
    
    Args:
        llm: Language model instance
            - Required for mode="tool_calling" and mode="react"
            - Required if enable_summarization=True
            - Optional for mode="direct" with enable_summarization=False
        tools: List of tools available for execution
        system_prompt: Optional user-provided system prompt
        verbose: Whether to print detailed execution logs (default: False)
        mode: Execution mode - "direct", "tool_calling", or "react" (default: "direct")
        enable_summarization: Whether to run LLM summarization (default: False)
        tool_name: Specific tool to use (if None, uses first tool)
        **kwargs: Additional mode-specific parameters
    
    Returns:
        Compiled LangGraph agent ready to invoke
        
    Raises:
        ValueError: If mode is invalid or required parameters are missing
        
    Examples:
    ---------
    
    Example 1: Fast direct mode for DeepSeek V3 (RECOMMENDED)
    >>> from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
    >>> 
    >>> agent = create_parallel_agent(
    ...     tools=[SearchSemanticScholar()],
    ...     mode="direct",  # No LLM tool calling
    ...     enable_summarization=False,  # Maximum speed
    ...     verbose=True
    ... )
    >>> 
    >>> result = agent.invoke({
    ...     "parallel_tool_agent_messages": [
    ...         "transformer neural networks",
    ...         "attention mechanism",
    ...     ]
    ... })
    
    Example 2: Direct mode with LLM summarization
    >>> from langchain_openai import ChatOpenAI
    >>> 
    >>> llm = ChatOpenAI(model="deepseek-chat")
    >>> agent = create_parallel_agent(
    ...     llm=llm,
    ...     tools=[SearchSemanticScholar()],
    ...     mode="direct",
    ...     enable_summarization=True,  # LLM synthesizes results
    ...     verbose=True
    ... )
    
    Example 3: Tool calling mode for GPT-4
    >>> llm = ChatOpenAI(model="gpt-4")
    >>> agent = create_parallel_agent(
    ...     llm=llm,
    ...     tools=[SearchSemanticScholar()],
    ...     mode="tool_calling",  # LLM decides tool usage
    ...     enable_summarization=True,
    ...     verbose=True
    ... )
    
    Example 4: Full ReAct mode for complex reasoning
    >>> agent = create_parallel_agent(
    ...     llm=llm,
    ...     tools=[SearchSemanticScholar()],
    ...     mode="react",  # Full reasoning loops
    ...     verbose=True
    ... )
    
    Migration Guide:
    ---------------
    Old code:
        from agent_blocks_hub.parallel_tool_agent import create_parallel_tool_agent
        agent = create_parallel_tool_agent(llm=llm, tools=tools)
    
    New code (recommended):
        from agent_blocks_hub.parallel_agent import create_parallel_agent
        agent = create_parallel_agent(llm=llm, tools=tools, mode="direct")
    """
    if tools is None:
        tools = []
    
    if system_prompt is None:
        system_prompt = ""
    
    # Validate mode
    valid_modes = ["direct", "tool_calling", "react"]
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode: '{mode}'. Must be one of {valid_modes}.\n"
            f"Recommendation: Use 'direct' for models with poor tool calling support (e.g., DeepSeek V3)"
        )
    
    # Validate LLM requirements
    if mode in ["tool_calling", "react"] and llm is None:
        raise ValueError(f"LLM is required for mode='{mode}'")
    
    if enable_summarization and llm is None:
        raise ValueError("LLM is required when enable_summarization=True")
    
    # Normalize tools based on mode
    tools = _normalize_tools_for_mode(tools or [], mode)

    # Extract optional remark-related kwargs
    enable_remark = kwargs.get("enable_remark", False)
    remark_prompt = kwargs.get("remark_prompt")
    summarization_prompt = kwargs.get("summarization_prompt")

    # Route to appropriate implementation
    if mode == "direct":
        from .direct import create_parallel_direct_tool_agent as create_impl
        agent = create_impl(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            verbose=verbose,
            enable_summarization=enable_summarization,
            enable_remark=enable_remark,
            remark_prompt=remark_prompt,
            summarization_prompt=summarization_prompt,
            tool_name=tool_name,
        )
        # Wrap to provide unified output format
        return _wrap_agent_with_unified_output(agent, "tool_results")
    
    elif mode == "tool_calling":
        from .tool_calling import create_parallel_tool_agent as create_impl
        agent = create_impl(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            verbose=verbose,
            enable_summarization=enable_summarization,
            enable_remark=enable_remark,
            remark_prompt=remark_prompt,
            summarization_prompt=summarization_prompt,
            tool_name=tool_name,
        )
        # Wrap to provide unified output format
        return _wrap_agent_with_unified_output(agent, "tool_results")
    
    elif mode == "react":
        from .react import create_parallel_react_agent as create_impl
        agent = create_impl(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            verbose=verbose,
            enable_summarization=enable_summarization,
            enable_remark=enable_remark,
            remark_prompt=remark_prompt,
            summarization_prompt=summarization_prompt,
        )
        # Wrap to provide unified output format (react uses "agent_results")
        return _wrap_agent_with_unified_output(agent, "agent_results")


def _wrap_agent_with_unified_output(agent: Any, result_key: str) -> Any:
    """
    Wrap an agent to provide unified output format.
    
    Ensures all agents return results in a consistent format with both:
    - tool_results: For backward compatibility
    - agent_results: Unified key name
    
    Args:
        agent: The compiled agent
        result_key: The key the agent uses ("tool_results" or "agent_results")
        
    Returns:
        Wrapped agent with unified output
    """
    class UnifiedAgent:
        def __init__(self, inner_agent, result_key):
            self.inner_agent = inner_agent
            self.result_key = result_key
        
        def invoke(self, *args, **kwargs):
            result = self.inner_agent.invoke(*args, **kwargs)
            
            # Ensure both keys are present for compatibility
            if self.result_key == "tool_results" and "tool_results" in result:
                # Add agent_results alias
                result["agent_results"] = result["tool_results"]
            elif self.result_key == "agent_results" and "agent_results" in result:
                # Add tool_results alias
                result["tool_results"] = result["agent_results"]
            
            results_dict = result.get("agent_results") or result.get("tool_results") or {}
            tool_result = {}
            remark = {}
            tool_result_with_remark = {}
            query_map = {}
            for idx, item in results_dict.items():
                res_text = item.get("result", "")
                rm = item.get("remark")
                q = item.get("query")
                tool_result[idx] = res_text
                remark[idx] = rm
                tool_result_with_remark[idx] = res_text + ("\nRemark: " + rm if rm else "")
                if q is not None:
                    query_map[idx] = q
            if tool_result:
                result["tool_result"] = tool_result
            if tool_result_with_remark:
                result["tool_result_with_remark"] = tool_result_with_remark
            if remark:
                result["remark"] = remark
            if query_map:
                result["query"] = query_map
            if "final_summary" in result:
                result["summary"] = result.get("final_summary")

            return result
        
        def stream(self, *args, **kwargs):
            return self.inner_agent.stream(*args, **kwargs)
        
        def __getattr__(self, name):
            return getattr(self.inner_agent, name)
    
    return UnifiedAgent(agent, result_key)


def _normalize_tools_for_mode(tools: List[Any], mode: str) -> List[Any]:
    """
    Normalize provided tools to match the execution mode requirements.

    - For mode "tool_calling" and "react": ensure tools are LangChain tools with `.invoke`.
      If tools are llm-tool-hub BaseTool instances or lack `.invoke`, adapt them automatically.
    - For mode "direct": return tools unchanged.
    """
    if not tools:
        return []

    if mode in ["tool_calling", "react"]:
        try:
            from llm_tool_hub.base_tool import BaseTool as HubBaseTool
        except Exception:
            HubBaseTool = None  # Fallback if not available

        needs_adapter = False
        for t in tools:
            if hasattr(t, "invoke"):
                continue
            if HubBaseTool and isinstance(t, HubBaseTool):
                needs_adapter = True
                break
            # If no invoke and unknown type, still try adapting
            needs_adapter = True
            break

        if needs_adapter:
            try:
                from llm_tool_hub.integrations.langchain_adapter import LangchainToolAdapter
                return LangchainToolAdapter.to_langchain_structured_tool(tools)
            except ImportError as e:
                raise ImportError(
                    "Provided tools are not LangChain tools and automatic adaptation failed. "
                    "Please install 'langchain-core' or manually adapt tools using "
                    "LangchainToolAdapter.to_langchain_structured_tool(...)."
                ) from e

    return tools


# Backward compatibility aliases
def create_parallel_tool_agent(*args, **kwargs):
    """
    Backward compatibility wrapper for create_parallel_tool_agent.
    
    Deprecated: Use create_parallel_agent(mode="tool_calling") instead.
    """
    kwargs.setdefault("mode", "tool_calling")
    return create_parallel_agent(*args, **kwargs)


def create_parallel_direct_tool_agent(*args, **kwargs):
    """
    Backward compatibility wrapper for create_parallel_direct_tool_agent.
    
    Deprecated: Use create_parallel_agent(mode="direct") instead.
    """
    kwargs.setdefault("mode", "direct")
    return create_parallel_agent(*args, **kwargs)


def create_parallel_react_agent(*args, **kwargs):
    """
    Backward compatibility wrapper for create_parallel_react_agent.
    
    Deprecated: Use create_parallel_agent(mode="react") instead.
    """
    kwargs.setdefault("mode", "react")
    return create_parallel_agent(*args, **kwargs)
