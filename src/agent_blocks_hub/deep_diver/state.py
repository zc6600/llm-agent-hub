"""
State definition for the Deep Diver agent.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import MessagesState
from langchain_core.tools import BaseTool


class Hypothesis(TypedDict, total=False):
    """A hypothesis with its verification status."""
    content: str
    verification_result: Optional[str]
    confidence: Optional[float]
    evidence: Optional[List[str]]


class DeepDiverState(MessagesState, total=False):
    """
    State for the Deep Diver agent.
    
    Extends MessagesState to include scientific method components.
    All fields are optional and will be added as the workflow progresses.
    """
    # Problem formulation
    original_question: str
    decomposed_problems: List[str]
    
    # Task classification
    task_type: Optional[str]  # "simple" or "complex"
    task_classification_confidence: Optional[float]
    task_reasoning: Optional[str]
    use_hypothesis_generation: bool  # Controls whether to include hypothesis generation
    
    # Information gathering
    gathered_information: List[Dict[str, Any]]
    problem_research_results: List[Dict[str, Any]]  # Results from parallel problem research: [{problem_idx, problem, response, gathered_info}]
    
    # Hypothesis generation and verification
    hypotheses: List[Hypothesis]
    current_iteration: int
    max_iterations: int
    
    # Experience pool
    experience_pool: List[Dict[str, Any]]
    
    # Final answer
    final_answer: Optional[str]
    
    # Tools and LLM (passed through state)
    tools: List[BaseTool]
    llm: Any
    system_prompt: Optional[str]
    task_classifier: Optional[Any]  # TaskClassifier instance
