"""
Direct tool calling strategy - bypasses LLM tool calling.

This module re-exports the parallel_direct_tool_agent implementation.
"""

from .parallel_direct_tool_agent import create_parallel_direct_tool_agent

__all__ = ["create_parallel_direct_tool_agent"]
