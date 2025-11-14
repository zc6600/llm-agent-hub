"""
Deep Diver Agent creation and configuration.

Creates a LangGraph-based agent that follows the scientific method.
"""

from typing import List, Optional, Dict, Any, Literal
from langgraph.graph import StateGraph, END, START
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .state import DeepDiverState
from .nodes import (
    classify_task,
    formulate_problem,
    gather_information,
    generate_hypothesis,
    verify_hypothesis,
    final_answer,
    should_continue_iteration,
    decide_hypothesis_needed,
    synthesize_results
)
from .prompts import DEFAULT_SYSTEM_PROMPT
from .task_classifier import TaskClassifier


def create_deepdiver_agent(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    max_iterations: int = 3,
    task_type: Literal["auto", "simple", "complex"] = "auto",
    enable_task_classification: bool = True
):
    """
    Create a Deep Diver agent using LangGraph with flexible workflow.
    
    The agent can follow two paths:
    1. SIMPLE PATH: Formulate → Gather → Final Answer (for factual queries)
    2. COMPLEX PATH: Formulate → Gather → Hypothesis → Verify → Final Answer (for research)
    
    The path is determined by task_type parameter or automatic classification.
    
    Args:
        llm: Language model to use for reasoning
        tools: List of tools available to the agent (e.g., internet_search)
        system_prompt: Optional system prompt to guide the agent's behavior
        max_iterations: Maximum number of hypothesis-verification iterations (default: 3)
        task_type: Task classification mode
            - "auto": LLM automatically classifies the task (recommended)
            - "simple": Always use simple path (no hypothesis generation)
            - "complex": Always use complex path (full scientific method)
        enable_task_classification: Whether to enable task classification node
        
    Returns:
        Compiled LangGraph agent ready to invoke
        
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> # Auto-classify tasks (recommended)
        >>> agent = create_deepdiver_agent(
        ...     llm=llm,
        ...     tools=[internet_search],
        ...     task_type="auto"  # Will use simple path for "What is X?" questions
        ... )
        >>> # Or force a specific workflow
        >>> agent_simple = create_deepdiver_agent(
        ...     llm=llm,
        ...     tools=[internet_search],
        ...     task_type="simple"  # Always skip hypothesis generation
        ... )
    """
    # Use default system prompt if none provided
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    # Create the graph
    graph = _create_graph(
        llm, tools, system_prompt, max_iterations,
        task_type, enable_task_classification
    )
    
    # Compile and return
    return graph.compile()


def _create_graph(
    llm: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: Optional[str],
    max_iterations: int,
    task_type: Literal["auto", "simple", "complex"] = "auto",
    enable_task_classification: bool = True
) -> StateGraph:
    """
    Create the LangGraph state graph for the Deep Diver agent.
    
    Args:
        llm: Language model
        tools: Available tools
        system_prompt: System prompt
        max_iterations: Maximum iterations
        task_type: Task classification mode ("auto", "simple", or "complex")
        enable_task_classification: Whether to use task classification
        
    Returns:
        Configured StateGraph
    """
    # Create state graph
    workflow = StateGraph(DeepDiverState)
    
    # Add initialization node to inject LLM, tools, and config into state
    def initialize_state(state: DeepDiverState) -> Dict[str, Any]:
        """Initialize state with LLM, tools, and configuration."""
        # Create task classifier with tools awareness
        classifier = None
        if enable_task_classification:
            classifier = TaskClassifier(llm=llm, tools=tools)
        
        return {
            "llm": llm,
            "tools": tools,
            "system_prompt": system_prompt,
            "max_iterations": max_iterations,
            "current_iteration": 0,
            "gathered_information": [],
            "hypotheses": [],
            "experience_pool": [],
            "task_classifier": classifier,
            "use_hypothesis_generation": task_type == "complex"  # Default value
        }
    
    # Add nodes
    workflow.add_node("initialize", initialize_state)
    
    if enable_task_classification:
        workflow.add_node("classify_task", classify_task)
    
    workflow.add_node("formulate_problem", formulate_problem)
    workflow.add_node("gather_information", gather_information)
    workflow.add_node("synthesize_results", synthesize_results)  # NEW: Aggregate parallel search results
    workflow.add_node("generate_hypothesis", generate_hypothesis)
    workflow.add_node("verify_hypothesis", verify_hypothesis)
    workflow.add_node("final_answer", final_answer)
    
    # Add edges - different flow based on task_type parameter
    # Start -> Initialize
    workflow.add_edge(START, "initialize")
    
    if enable_task_classification:
        # Initialize -> Classify Task
        workflow.add_edge("initialize", "classify_task")
        # Classify Task -> Formulate Problem
        workflow.add_edge("classify_task", "formulate_problem")
    else:
        # Initialize -> Formulate Problem directly
        workflow.add_edge("initialize", "formulate_problem")
    
    # Formulate Problem -> Gather Information (parallel search for each decomposed problem)
    workflow.add_edge("formulate_problem", "gather_information")
    
    # Gather Information -> Synthesize Results (aggregate parallel search results)
    workflow.add_edge("gather_information", "synthesize_results")
    
    # Synthesize Results -> Conditional (hypothesis generation or final answer)
    workflow.add_conditional_edges(
        "synthesize_results",
        decide_hypothesis_needed,
        {
            "hypothesis": "generate_hypothesis",
            "answer": "final_answer"
        }
    )
    
    # Generate Hypothesis -> Verify Hypothesis
    workflow.add_edge("generate_hypothesis", "verify_hypothesis")
    
    # Verify Hypothesis -> Decision (continue or finish)
    workflow.add_conditional_edges(
        "verify_hypothesis",
        should_continue_iteration,
        {
            "continue": "generate_hypothesis",  # Loop back to generate new hypotheses
            "finish": "final_answer"
        }
    )
    
    # Final Answer -> End
    workflow.add_edge("final_answer", END)
    
    return workflow
