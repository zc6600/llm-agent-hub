"""
Example: Using Deep Diver with flexible workflow (auto task classification).

This example demonstrates the new task classification feature that allows
Deep Diver to automatically choose between:
- SIMPLE PATH: For factual queries like "What is X?"
- COMPLEX PATH: For research queries like "How should I design X?"

This makes Deep Diver more efficient and flexible for different types of questions.
"""

import sys
import json
from pathlib import Path

# Ensure project src directory is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_community.tools import DuckDuckGoSearchRun

from llm_provider import get_llm
from agent_blocks_hub.deep_diver import create_deepdiver_agent


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def run_example(question: str, task_type: str = "auto"):
    """Run Deep Diver with a specific question and task type."""
    print_section(f"RUNNING: {question}\nTASK TYPE: {task_type}")
    
    # Initialize LLM and tools
    llm = get_llm(model="gpt-4o-mini")
    
    tools = [DuckDuckGoSearchRun()]
    
    # Create agent with specified task type
    agent = create_deepdiver_agent(
        llm=llm,
        tools=tools,
        task_type=task_type,
        enable_task_classification=True,
        max_iterations=2
    )
    
    # Prepare input
    from langchain_core.messages import HumanMessage
    
    input_state = {
        "messages": [HumanMessage(content=question)]
    }
    
    # Run the agent
    print("\n[EXECUTING] Starting workflow...\n")
    
    try:
        result = agent.invoke(input_state)
        
        print_section("FINAL RESULT")
        print(f"\nTask Type: {result.get('task_type', 'unknown')}")
        print(f"Classification Confidence: {result.get('task_classification_confidence', 0):.2f}")
        print(f"Reasoning: {result.get('task_reasoning', 'N/A')}")
        print(f"\nFinal Answer:\n{result.get('final_answer', 'No answer')}")

        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run examples with different task types."""
    
    # Example 1: Factual query (should use SIMPLE path)
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Factual Query (Auto-Classify)")
    print("=" * 80)
    print("""
This question is factual and doesn't require complex hypothesis testing.
The auto-classifier should detect this and use the SIMPLE path:
  formulate → gather information → final answer (skip hypothesis generation)
    """)
    
    run_example(
        question="What is LangGraph and what are its main components?",
        task_type="auto"
    )
    
    # Example 2: Complex research question (should use COMPLEX path)
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Complex Research Question (Auto-Classify)")
    print("=" * 80)
    print("""
This question requires deeper analysis and comparing options.
The auto-classifier should detect this and use the COMPLEX path:
  formulate → gather → hypothesis generation → verification → final answer
    """)
    
    run_example(
        question="How should I design a system that efficiently handles real-time data processing with high reliability?",
        task_type="auto"
    )
    
    # Example 3: Force simple path
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Force Simple Path (Bypass Classification)")
    print("=" * 80)
    print("""
Even for complex-looking questions, we can force the SIMPLE path by setting task_type="simple".
This skips hypothesis generation entirely.
    """)
    
    run_example(
        question="How should I design a system for real-time processing?",
        task_type="simple"
    )
    
    # Example 4: Force complex path
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Force Complex Path (Full Scientific Method)")
    print("=" * 80)
    print("""
For simple queries, we can force the COMPLEX path by setting task_type="complex".
This ensures full hypothesis generation and verification.
    """)
    
    run_example(
        question="What is LangGraph?",
        task_type="complex"
    )


if __name__ == "__main__":
    main()
