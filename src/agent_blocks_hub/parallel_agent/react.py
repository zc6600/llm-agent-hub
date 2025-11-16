"""
ReAct strategy - full reasoning loops with observation-thought-action cycles.

This module re-exports the parallel_react_agent implementation.
"""

from .parallel_react_agent import create_parallel_react_agent

__all__ = ["create_parallel_react_agent"]
