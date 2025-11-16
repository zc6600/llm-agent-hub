"""
Tool calling strategy - LLM decides when/how to call tools.

This module re-exports the parallel_tool_agent implementation.
"""

from .parallel_tool_agent import create_parallel_tool_agent

__all__ = ["create_parallel_tool_agent"]
