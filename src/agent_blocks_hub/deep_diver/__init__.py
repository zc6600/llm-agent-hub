"""
Deep Diver Agent - A scientific method-based research agent using LangGraph.

This module implements a research agent that follows the scientific method
(inspired by Popper's philosophy and Lean Startup methodology).

Features:
- Automatic task classification (simple vs complex queries)
- Flexible workflow: factual queries skip hypothesis generation, complex tasks use full scientific method
- Supports manual task type override
- Configurable system prompts and iteration limits
"""

from . import compat  # noqa: F401  # Ensure compatibility patches are applied
from .agent import create_deepdiver_agent
from .state import DeepDiverState
from .nodes import (
    classify_task,
    formulate_problem,
    gather_information,
    generate_hypothesis,
    verify_hypothesis,
    final_answer,
    decide_hypothesis_needed
)
from .task_classifier import TaskClassifier

__all__ = [
    "create_deepdiver_agent",
    "DeepDiverState",
    "TaskClassifier",
    "MultiAgentOrchestrator",
    "SubTaskAgent",
    "classify_task",
    "formulate_problem",
    "gather_information",
    "generate_hypothesis",
    "verify_hypothesis",
    "final_answer",
    "decide_hypothesis_needed"
]
