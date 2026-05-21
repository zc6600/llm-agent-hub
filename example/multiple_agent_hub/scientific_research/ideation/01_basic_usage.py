"""
Example 01: Basic usage of the Ideation Agent for scientific research.

This example demonstrates both 'lite' and 'full' modes of the Ideation Agent:

- Lite Mode (default): Uses parallel_react_agent for faster execution (~60s)
  - Best for: Quick iteration, cost-conscious workflows
  
- Full Mode: Uses parallel_tool_agent for detailed results (~120s)
  - Best for: Comprehensive analysis, detailed traceability

This example shows how to use LangChain-style tools with the Ideation Agent.
"""

import sys
import json
from pathlib import Path

# Ensure project root and src directory are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_provider import get_llm
from multi_agent_hub.scientific_research.ideation import create_ideation_agent
from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
from llm_tool_hub.integrations.langchain_adapter import LangchainToolAdapter


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_debug(level: int, message: str, data=None) -> None:
    """Print debug message with indentation based on level."""
    indent = "  " * level
    print(f"{indent}[DEBUG {level}] {message}")
    if data is not None:
        if isinstance(data, (dict, list)):
            print(f"{indent}    {json.dumps(data, ensure_ascii=False, indent=2)[:500]}...")
        else:
            print(f"{indent}    {str(data)[:500]}...")


def run_ideation(mode: str = "lite", verbose: bool = True) -> dict:
    """
    Run the Ideation Agent in specified mode.
    
    Args:
        mode: "lite" for fast execution or "full" for comprehensive analysis
        verbose: Whether to print progress information
    
    Returns:
        Agent result dictionary with ideation outputs
    """
    print_section(f"INITIALIZING IDEATION AGENT ({mode} mode)")

    # 1. Configure LLM
    print_debug(0, "Configuring Language Model (LLM)")
    llm = get_llm(
        temperature=0.8,  # Slightly higher for creative thinking
        max_tokens=20000,
        enable_langsmith=True,
    )
    print_debug(1, "✓ LLM initialized")

    # 2. Initialize tools (LangChain-adapted)
    print_debug(0, "Initializing Tools")
    semantic_scholar_tool = SearchSemanticScholar()
    search_tool = LangchainToolAdapter.to_langchain_structured_tool(semantic_scholar_tool)
    print_debug(1, "✓ Semantic Scholar search tool initialized and converted to LangChain format")

    # 3. System prompt for ideation
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

    # 4. Create the ideation agent with specified mode
    print_debug(0, f"Creating Ideation Agent ({mode} mode)")
    agent = create_ideation_agent(
        llm=llm,
        tools=[search_tool],
        system_prompt=ideation_instructions,
        verbose=verbose,
        mode=mode,  # Specify "lite" or "full"
    )
    print_debug(1, f"✓ Agent created and compiled (mode={mode})")

    # 5. Define research question
    research_question = "How can machine learning improve protein structure prediction for drug discovery?"

    print_section(f"RUNNING IDEATION PROCESS ({mode} mode)")
    print_debug(0, f"Research Question: {research_question}")

    try:
        print_debug(0, "Invoking ideation agent...")
        result = agent.invoke({
            "ideation_message": research_question,
        })

        return result

    except Exception as e:  # pragma: no cover
        print_section("ERROR")
        print_debug(0, f"✗ Error during ideation process: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def print_results(result: dict, mode: str = "lite") -> None:
    """
    Print ideation results in formatted output.
    
    Args:
        result: Agent result dictionary
        mode: Mode used (for logging purposes)
    """
    print_section(f"IDEATION RESULTS ({mode} mode)")

    if "final_idea_report" not in result:
        print_debug(0, "✗ No final report generated")
        return

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
        ideas = report["proposed_ideas"]
        print_debug(0, f"Proposed Ideas ({len(ideas)} ideas)")
        for i, idea in enumerate(ideas, 1):
            print(f"\n  Idea {i}: {idea.get('idea', 'N/A')}")
            print(f"    Reasoning: {idea.get('reasoning', 'N/A')[:400]}...")

    if "hypotheses_and_experiments" in report:
        hyps = report["hypotheses_and_experiments"]
        print_debug(0, f"Verifiable Sub-hypotheses ({len(hyps)} hypotheses)")
        for i, hyp in enumerate(hyps, 1):
            print(f"\n  Hypothesis {i}:")
            print(f"    Research Question: {hyp.get('research_question', 'N/A')}")
            print(f"    Experiment: {hyp.get('verification_experiment', 'N/A')[:300]}...")
            print(f"    Expected Outcome: {hyp.get('expected_outcome', 'N/A')}")

    if "comprehensive_report" in report:
        print_debug(0, "Comprehensive Ideation Report")
        print("\n" + "-" * 80)
        print(report["comprehensive_report"])
        print("-" * 80)

    print_section(f"IDEATION PROCESS COMPLETE ({mode} mode)")
    print_debug(0, "✓ All stages completed successfully")


def main() -> None:
    """Run both lite and full mode examples."""
    print_section("IDEATION AGENT - MODE COMPARISON")
    print_debug(0, "This example demonstrates both lite and full modes")
    print_debug(0, "")
    print_debug(0, "Lite Mode (default):")
    print_debug(1, "- Uses parallel_react_agent")
    print_debug(1, "- Faster execution (~60 seconds)")
    print_debug(1, "- Lower token usage (~9k-12k)")
    print_debug(1, "- Best for: Quick iteration, cost-aware workflows")
    print_debug(0, "")
    print_debug(0, "Full Mode:")
    print_debug(1, "- Uses parallel_tool_agent")
    print_debug(1, "- More detailed results (~120 seconds)")
    print_debug(1, "- Higher token usage (~15k-20k)")
    print_debug(1, "- Best for: Comprehensive analysis, detailed traceability")

    # Run in lite mode (default, recommended for most use cases)
    print_section("EXAMPLE 1: Lite Mode (Fast & Efficient)")
    print_debug(0, "Running ideation agent in lite mode (default)...")
    result_lite = run_ideation(mode="lite", verbose=True)
    
    if result_lite:
        print_results(result_lite, mode="lite")

    # Optionally run in full mode for comparison (uncomment to enable)
    # This will take longer but provide more detailed results
    # print_section("EXAMPLE 2: Full Mode (Comprehensive)")
    # print_debug(0, "Running ideation agent in full mode...")
    # result_full = run_ideation(mode="full", verbose=True)
    # 
    # if result_full:
    #     print_results(result_full, mode="full")


if __name__ == "__main__":
    main()
