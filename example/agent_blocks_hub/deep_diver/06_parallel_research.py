"""
Example: Deep Diver with Parallel Problem Research

This example demonstrates the parallel research feature:
- Problems are decomposed into sub-tasks
- Each sub-task is researched IN PARALLEL for efficiency
- Results are synthesized before generating the final answer

Usage:
    python 06_parallel_research.py
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure project src directory is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_provider import get_llm
from src.llm_tool_hub.scientific_research_tool import internet_search
from src.agent_blocks_hub.deep_diver.agent import create_deepdiver_agent


def create_research_tools():
    """Create research tools for the agent."""
    return [internet_search]


def run_parallel_research_example():
    """
    Run a Deep Diver agent with parallel problem research.
    
    This example shows how:
    1. A complex question is decomposed into multiple sub-problems
    2. Each sub-problem is researched IN PARALLEL (for better performance)
    3. Results from all researches are synthesized
    4. A comprehensive final answer is generated
    """
    
    # Initialize LLM
    llm = get_llm(model="gpt-4", temperature=0.7)
    
    # Create research tools
    tools = create_research_tools()
    
    # Create the agent with parallel research enabled
    agent = create_deepdiver_agent(
        llm=llm,
        tools=tools,
        task_type="simple",  # Use simple path for factual research
        enable_task_classification=True
    )
    
    # Example research question
    question = """
    What are the key differences between machine learning, deep learning, and reinforcement learning?
    Provide a comprehensive comparison covering definitions, applications, and recent developments.
    """
    
    print("\n" + "="*80)
    print("PARALLEL RESEARCH EXAMPLE")
    print("="*80)
    print(f"\nResearch Question:\n{question}\n")
    print("="*80)
    print("\nWorkflow:")
    print("1. Question → Problem Decomposition")
    print("   ↓")
    print("2. Formulate Problem (break into sub-tasks)")
    print("   ↓")
    print("3. Gather Information (PARALLEL research for each sub-task)")
    print("   ├─ Sub-task 1 research")
    print("   ├─ Sub-task 2 research")
    print("   └─ Sub-task 3 research")
    print("   ↓")
    print("4. Synthesize Results (aggregate all findings)")
    print("   ↓")
    print("5. Generate Final Answer (comprehensive synthesis)")
    print("\n" + "="*80 + "\n")
    
    # Invoke the agent
    result = agent.invoke({"messages": [("human", question)]})
    
    # Print results
    print("\n" + "="*80)
    print("RESEARCH RESULTS")
    print("="*80)
    
    # Show decomposed problems
    if "decomposed_problems" in result:
        print("\n[DECOMPOSED PROBLEMS]")
        for i, problem in enumerate(result["decomposed_problems"], 1):
            print(f"  {i}. {problem}")
    
    # Show synthesis information
    if "synthesized_research" in result:
        print("\n[SYNTHESIZED RESEARCH]")
        for item in result["synthesized_research"]:
            print(f"\n  Problem {item['problem_idx']}: {item['problem']}")
            print(f"  Sources gathered: {item['num_sources']}")
            print(f"  Summary: {item['research_summary'][:150]}...")
    
    # Show final answer
    if "final_answer" in result:
        print("\n[FINAL ANSWER]")
        print("-" * 80)
        print(result["final_answer"])
        print("-" * 80)
    
    print("\n" + "="*80)
    print("✓ Research Complete!")
    print("="*80 + "\n")


def run_complex_research_example():
    """
    Run a Deep Diver agent with hypothesis generation AND parallel research.
    
    This shows a more complex workflow where:
    1. Problems are decomposed and researched in parallel
    2. Hypotheses are generated based on the research
    3. Hypotheses are verified through additional research
    4. A comprehensive answer is synthesized
    """
    
    # Initialize LLM
    llm = get_llm(model="gpt-4", temperature=0.7)
    
    # Create research tools
    tools = create_research_tools()
    
    # Create the agent with hypothesis generation
    agent = create_deepdiver_agent(
        llm=llm,
        tools=tools,
        task_type="complex",  # Use complex path with hypothesis generation
        enable_task_classification=True,
        max_iterations=2
    )
    
    # Complex research question
    question = """
    What is the current state of quantum computing technology and what are the
    major challenges to achieving practical quantum advantage?
    """
    
    print("\n" + "="*80)
    print("COMPLEX RESEARCH EXAMPLE (WITH HYPOTHESIS GENERATION)")
    print("="*80)
    print(f"\nResearch Question:\n{question}\n")
    print("="*80)
    print("\nWorkflow:")
    print("1. Question → Problem Decomposition")
    print("   ↓")
    print("2. Formulate Problem")
    print("   ↓")
    print("3. Gather Information (PARALLEL for each sub-task)")
    print("   ↓")
    print("4. Synthesize Results")
    print("   ↓")
    print("5. Generate Hypotheses (based on synthesized research)")
    print("   ↓")
    print("6. Verify Hypotheses (additional research to validate)")
    print("   ↓")
    print("7. Generate Final Answer")
    print("\n" + "="*80 + "\n")
    
    # Invoke the agent
    result = agent.invoke({"messages": [("human", question)]})
    
    # Print results
    print("\n" + "="*80)
    print("RESEARCH RESULTS")
    print("="*80)
    
    # Show decomposed problems
    if "decomposed_problems" in result:
        print("\n[DECOMPOSED PROBLEMS]")
        for i, problem in enumerate(result["decomposed_problems"], 1):
            print(f"  {i}. {problem}")
    
    # Show final answer
    if "final_answer" in result:
        print("\n[FINAL ANSWER]")
        print("-" * 80)
        print(result["final_answer"])
        print("-" * 80)
    
    print("\n" + "="*80)
    print("✓ Research Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    print("\nDeep Diver Parallel Research Examples")
    print("=====================================\n")
    print("Available examples:")
    print("  1. Simple Parallel Research (faster, no hypothesis generation)")
    print("  2. Complex Research (with hypothesis generation and verification)")
    print("  3. Exit\n")
    
    choice = input("Select example (1-3): ").strip()
    
    if choice == "1":
        run_parallel_research_example()
    elif choice == "2":
        run_complex_research_example()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice. Please select 1, 2, or 3.")
