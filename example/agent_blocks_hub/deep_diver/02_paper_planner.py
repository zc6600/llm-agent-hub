"""
Paper Planner usage example with detailed debugging.

This example demonstrates how to create and use a Paper Planner agent
to generate a detailed research plan based on a research idea.
"""

import sys
import json
from pathlib import Path

# Ensure project src directory is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_provider import get_llm
from multi_agent_hub.scientific_research.paper_planner import create_paper_planner_agent


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_debug(level: int, message: str, data=None):
    """Print debug message with indentation based on level."""
    indent = "  " * level
    print(f"{indent}[DEBUG {level}] {message}")
    if data is not None:
        if isinstance(data, (dict, list)):
            print(f"{indent}    {json.dumps(data, ensure_ascii=False, indent=2)}")
        else:
            print(f"{indent}    {data}")


def main():
    """Run a Paper Planner agent example."""
    print_section("INITIALIZING PAPER PLANNER AGENT")
    
    print_debug(0, "Configuring Language Model (LLM)")
    # Initialize LLM using llm_provider
    llm = get_llm(
        model="google/gemini-2.5-flash-preview-09-2025",
        temperature=0.7,
        max_tokens=100000
    )
    print_debug(1, "✓ LLM initialized", {
        "model": "google/gemini-2.5-flash-preview-09-2025",
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0.7,
        "max_tokens": 100000
    })
    
    # Research instructions (system prompt)
    research_instructions = """
    You are a thorough research planning assistant. Help create detailed, structured research plans:
    1. Analyze the research idea comprehensively
    2. Identify background and context
    3. Review related works and identify gaps
    4. Formulate clear hypotheses/research questions
    5. Design verification methods with benchmarks and metrics
    6. Suggest literature review directions
    
    Be critical, evidence-based, and thorough. Focus on feasibility and novelty.
    """
    
    print_debug(0, "Creating Paper Planner Agent")
    # Create the agent
    agent = create_paper_planner_agent(
        llm=llm,
        system_prompt=research_instructions,
        max_literature_iterations=3,  # Reduced for faster testing
        max_refinement_iterations=2
    )
    print_debug(1, "✓ Agent graph created and compiled")
    
    # Define the research request
    research_request = """
    I want to implement a tool for LLM that can ask questions, like ask_if_unclear(query: str) -> str.
    Unlike human-in-the-loop, an LLM (maybe with tools, or even a multiple agent system) hiding behind 
    the ask function will answer the query. I hypothesize an AI agent with this asking-tool can dive deeper 
    and avoid shortcut learning compared with llm_with_tools. I'm looking for benchmarks to verify this hypothesis.
    Please help me make a detailed research plan, including background, related works, hypotheses, 
    and verification methods (benchmark/dataset, model/algorithm, evaluation metrics).
    """
    
    print_section("STARTING AGENT EXECUTION")
    print_debug(0, f"Research Request: {research_request[:200]}...")
    
    print_debug(0, "Invoking agent...")
    # Invoke the agent
    result = agent.invoke({
        "messages": [{"role": "user", "content": research_request}],
        "original_request": research_request
    })
    print_debug(1, "✓ Agent execution completed")
    
    # Print results with debugging information
    print_section("EXECUTION SUMMARY")
    
    original_request = result.get('original_request')
    print_debug(1, "Original Request captured:", original_request[:100] + "...")
    
    current_plan = result.get('current_plan', [])
    print_debug(1, f"Number of plan iterations: {len(current_plan)}")
    for i, plan in enumerate(current_plan, 1):
        print_debug(2, f"Plan iteration {i}:", plan[:150] + "...")
    
    final_plan = result.get('final_plan', '')
    if final_plan:
        print_debug(1, "Final Plan generated:")
        lines = final_plan.split('\n')
        for line in lines[:15]:  # Show first 15 lines
            print_debug(2, line)
        if len(lines) > 15:
            print_debug(2, f"... and {len(lines) - 15} more lines")
    else:
        print_debug(1, "⚠ No final plan generated")
    
    literature_notes = result.get('literature_note', [])
    print_debug(1, f"Literature notes collected: {len(literature_notes)}")
    for i, note in enumerate(literature_notes[:2], 1):  # Show first 2 notes
        print_debug(2, f"Note {i} (first 100 chars):", note[:100] + "...")
    if len(literature_notes) > 2:
        print_debug(2, f"... and {len(literature_notes) - 2} more notes")
    
    searched_papers = result.get('searched_papers', [])
    print_debug(1, f"Papers searched: {len(searched_papers)}")
    for i, paper in enumerate(searched_papers[:5], 1):
        print_debug(2, f"Paper {i}: {paper}")
    if len(searched_papers) > 5:
        print_debug(2, f"... and {len(searched_papers) - 5} more papers")
    
    print_section("FINAL SUMMARY")
    print_debug(0, "Paper Planner Agent Execution Completed ✓", {
        "plan_iterations": len(current_plan),
        "literature_notes": len(literature_notes),
        "papers_searched": len(searched_papers),
        "final_plan_generated": bool(final_plan)
    })
    
    print("\n✅ Paper Planner workflow completed successfully!")
    print("📄 Check the output folder for generated research plan and notes.\n")


if __name__ == "__main__":
    main()
