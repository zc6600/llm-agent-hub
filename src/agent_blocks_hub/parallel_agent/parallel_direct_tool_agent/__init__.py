"""
Parallel Direct Tool Agent - Direct tool invocation without LLM tool calling.

A LangGraph-based agent that directly calls tools in parallel, bypassing LLM
tool calling entirely. This is useful for models with poor tool calling support
(e.g., DeepSeek V3) while still maintaining parallel execution and optional
LLM summarization.

Key differences from parallel_tool_agent:
- Direct tool.run() invocation (no LLM tool calling)
- Works with models that have poor tool calling support
- Same interface and output format as parallel_tool_agent
- Still supports parallel execution and optional summarization

Public API:
    - create_parallel_direct_tool_agent: Main function to create the agent
"""

from .agent import create_parallel_direct_tool_agent
from .state import ParallelDirectToolAgentState

__all__ = [
    "create_parallel_direct_tool_agent",
    "ParallelDirectToolAgentState",
]
