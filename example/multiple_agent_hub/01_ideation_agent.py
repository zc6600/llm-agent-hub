"""
Example: Basic usage of the Ideation Agent.

This example demonstrates how to create and use the Ideation Agent
to generate novel research ideas through a 6-stage ideation process.

Run with:
    python example/deep_diver/01_ideation_agent.py
"""

import sys
import json
from pathlib import Path

# Ensure project src directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_provider import get_llm
from multi_agent_hub.scientific_research.ideation import create_ideation_agent
from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
from llm_tool_hub.integrations.langchain_adapter import LangchainToolAdapter


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
            print(f"{indent}    {json.dumps(data, ensure_ascii=False, indent=2)[:500]}...")
        else:
            print(f"{indent}    {str(data)[:500]}...")


def main():
    """Run a basic Ideation Agent example."""
    print_section("INITIALIZING IDEATION AGENT")

    print_debug(0, "Configuring Language Model (LLM)")
    # Initialize LLM from environment variables
    llm = get_llm(
        temperature=0.8,  # Slightly higher for creative thinking
        max_tokens=2000,
        enable_langsmith=False,
    )
    print_debug(1, "✓ LLM initialized")

    print_debug(0, "Initializing Tools")
    # Initialize Semantic Scholar search tool for academic research
    semantic_scholar_tool = SearchSemanticScholar()
    
    # Convert to LangChain tool using adapter
    search_tool = LangchainToolAdapter.to_langchain_structured_tool(semantic_scholar_tool)
    print_debug(1, "✓ Semantic Scholar search tool initialized and converted to LangChain format")

    # System prompt for ideation
    ideation_instructions = """
    You are a creative research strategist specializing in scientific ideation.
    Your goal is to help generate novel, impactful research ideas.
    
    When working through ideation:
    1. Search for peer-reviewed academic papers and research
    2. Identify genuine gaps in current knowledge
    3. Think creatively and challenge existing assumptions
    4. Focus on ideas that are both novel AND feasible
    5. Provide clear reasoning based on scientific evidence
    6. Cite relevant papers when making claims
    """

    print_debug(0, "Creating Ideation Agent")
    # Create the agent
    agent = create_ideation_agent(
        llm=llm,
        tools=[search_tool],
        system_prompt=ideation_instructions,
        verbose=True,  # Enable verbose mode to print summarize results
    )
    print_debug(1, "✓ Agent created and compiled")

    # Define the research question
    research_question = "How can machine learning improve protein structure prediction for drug discovery?"

    print_section("RUNNING IDEATION PROCESS")
    print_debug(0, f"Research Question: {research_question}")

    try:
        # Invoke the agent
        print_debug(0, "Invoking ideation agent...")
        result = agent.invoke({
            "ideation_message": research_question,
        })

        # Display results
        print_section("IDEATION RESULTS")

        if "final_idea_report" in result:
            report = result["final_idea_report"]

            print_debug(0, "Final Report Title")
            print(f"  {report.get('title', 'N/A')}")

            if "original_question" in report:
                print_debug(0, "Original Question")
                print(f"  {report['original_question']}")

            if "research_gap" in report:
                print_debug(0, "Identified Research Gap")
                print(f"  {report['research_gap']}")

            if "proposed_ideas" in report:
                print_debug(0, f"Proposed Ideas ({len(report['proposed_ideas'])} ideas)")
                for i, idea in enumerate(report["proposed_ideas"], 1):
                    print(f"\n  Idea {i}: {idea.get('idea', 'N/A')}")
                    print(f"    Reasoning: {idea.get('reasoning', 'N/A')}")

            if "hypotheses_and_experiments" in report:
                print_debug(0, f"Verifiable Sub-hypotheses ({len(report['hypotheses_and_experiments'])} hypotheses)")
                for i, hyp in enumerate(report["hypotheses_and_experiments"], 1):
                    print(f"\n  Hypothesis {i}:")
                    print(f"    Research Question: {hyp.get('research_question', 'N/A')}")
                    print(f"    Experiment: {hyp.get('verification_experiment', 'N/A')[:200]}...")
                    print(f"    Expected Outcome: {hyp.get('expected_outcome', 'N/A')}")

            if "comprehensive_report" in report:
                print_debug(0, "Comprehensive Ideation Report")
                print(f"\n{report['comprehensive_report']}")

        print_section("IDEATION PROCESS COMPLETE")
        print_debug(0, "✓ All stages completed successfully")

    except Exception as e:
        print_section("ERROR")
        print_debug(0, f"✗ Error during ideation process: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
