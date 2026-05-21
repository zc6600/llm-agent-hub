"""
Example 01: Basic usage of the Literature Review Agent for scientific research.

This example demonstrates both 'lite' and 'full' modes of the Literature Review Agent:

- Lite Mode (default): Uses parallel_tool_agent for faster execution (~60s)
  - Best for: Quick literature surveys, cost-conscious workflows
  
- Full Mode: Uses parallel_react_agent for detailed results (~120s)
  - Best for: Comprehensive analysis, detailed traceability

This example shows how to use LangChain-style tools with the Literature Review Agent.
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
from multi_agent_hub.scientific_research.literature_review import create_literature_review_agent
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


def run_literature_review(mode: str = "lite", verbose: bool = True) -> dict:
    """
    Run the Literature Review Agent in specified mode.
    
    Args:
        mode: "lite" for fast execution or "full" for comprehensive analysis
        verbose: Whether to print progress information
    
    Returns:
        Agent result dictionary with literature review outputs
    """
    print_section(f"INITIALIZING LITERATURE REVIEW AGENT ({mode} mode)")

    # 1. Configure LLM
    print_debug(0, "Configuring Language Model (LLM)")
    llm = get_llm(
        temperature=0.7,  # Balanced for analysis and synthesis
        max_tokens=20000,
        enable_langsmith=True,
    )
    print_debug(1, "✓ LLM initialized")

    # 2. Initialize tools (LangChain-adapted)
    print_debug(0, "Initializing Tools")
    semantic_scholar_tool = SearchSemanticScholar()
    search_tool = LangchainToolAdapter.to_langchain_structured_tool(semantic_scholar_tool)
    print_debug(1, "✓ Semantic Scholar search tool initialized and converted to LangChain format")

    # 3. System prompt for literature review
    review_instructions = """
    You are a systematic literature reviewer specializing in academic research.
    Your goal is to comprehensively analyze and synthesize published research.

    When working through literature review:
    1. Search for peer-reviewed academic papers and research
    2. Focus on recent work (last 5 years) and foundational papers
    3. Analyze papers for key findings, methodologies, and limitations
    4. Identify themes, trends, and connections across papers
    5. Discover genuine gaps in current research
    6. Provide clear citations and references
    7. Maintain an objective, academic tone
    """

    # 4. Create the literature review agent with specified mode
    print_debug(0, f"Creating Literature Review Agent ({mode} mode)")
    agent = create_literature_review_agent(
        llm=llm,
        tools=[search_tool],
        system_prompt=review_instructions,
        verbose=verbose,
        mode=mode,  # Specify "lite" or "full"
    )
    print_debug(1, f"✓ Agent created and compiled (mode={mode})")

    # 5. Define research topic
    research_topic = "transformer architectures for natural language processing"

    print_section(f"RUNNING LITERATURE REVIEW PROCESS ({mode} mode)")
    print_debug(0, f"Research Topic: {research_topic}")

    try:
        print_debug(0, "Invoking literature review agent...")
        result = agent.invoke({
            "review_topic": research_topic,
        })

        return result

    except Exception as e:  # pragma: no cover
        print_section("ERROR")
        print_debug(0, f"✗ Error during literature review process: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def print_results(result: dict, mode: str = "lite") -> None:
    """
    Print literature review results in formatted output.
    
    Args:
        result: Agent result dictionary
        mode: Mode used (for logging purposes)
    """
    print_section(f"LITERATURE REVIEW RESULTS ({mode} mode)")

    if "final_review_report" not in result:
        print_debug(0, "✗ No final report generated")
        return

    report = result["final_review_report"]

    print_debug(0, "Review Topic")
    print(f"  {report.get('topic', 'N/A')}")

    print_debug(0, "Statistics")
    print(f"  Total Papers Analyzed: {report.get('total_papers', 0)}")
    print(f"  Themes Identified: {report.get('themes_identified', 0)}")
    print(f"  Word Count: {report.get('word_count', 0)}")

    if "search_queries" in result:
        queries = result["search_queries"]
        print_debug(0, f"Search Queries Generated ({len(queries)} queries)")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")

    if "thematic_clusters" in result and result["thematic_clusters"]:
        clusters = result["thematic_clusters"]
        print_debug(0, f"Thematic Clusters ({len(clusters)} themes)")
        for i, cluster in enumerate(clusters, 1):
            print(f"\n  Theme {i}: {cluster.get('theme', 'N/A')}")
            print(f"    Papers: {len(cluster.get('papers', []))}")
            insights = cluster.get('key_insights', [])
            if insights:
                print(f"    Key Insights: {insights[0][:200]}...")

    if "research_trends" in result and result["research_trends"]:
        trends = result["research_trends"]
        print_debug(0, f"Research Trends ({len(trends)} trends)")
        for i, trend in enumerate(trends, 1):
            print(f"  {i}. {trend}")

    if "research_gaps" in result and result["research_gaps"]:
        gaps = result["research_gaps"]
        print_debug(0, f"Research Gaps ({len(gaps)} gaps)")
        for i, gap in enumerate(gaps, 1):
            print(f"  {i}. {gap}")

    if "comprehensive_report" in report:
        print_debug(0, "Comprehensive Literature Review Report")
        print("\n" + "-" * 80)
        print(report["comprehensive_report"])
        print("-" * 80)

    print_section(f"LITERATURE REVIEW COMPLETE ({mode} mode)")
    print_debug(0, "✓ All stages completed successfully")


def main() -> None:
    """Run both lite and full mode examples."""
    print_section("LITERATURE REVIEW AGENT - MODE COMPARISON")
    print_debug(0, "This example demonstrates both lite and full modes")
    print_debug(0, "")
    print_debug(0, "Lite Mode (default):")
    print_debug(1, "- Uses parallel_tool_agent")
    print_debug(1, "- Faster execution (~60 seconds)")
    print_debug(1, "- Lower token usage")
    print_debug(1, "- Best for: Quick literature surveys, cost-aware workflows")
    print_debug(0, "")
    print_debug(0, "Full Mode:")
    print_debug(1, "- Uses parallel_react_agent")
    print_debug(1, "- More detailed results (~120 seconds)")
    print_debug(1, "- Higher token usage")
    print_debug(1, "- Best for: Comprehensive analysis, detailed traceability")

    # Run in lite mode (default, recommended for most use cases)
    print_section("EXAMPLE 1: Lite Mode (Fast & Efficient)")
    print_debug(0, "Running literature review agent in lite mode (default)...")
    result_lite = run_literature_review(mode="lite", verbose=True)
    
    if result_lite:
        print_results(result_lite, mode="lite")

    # Optionally run in full mode for comparison (uncomment to enable)
    # This will take longer but provide more detailed results
    # print_section("EXAMPLE 2: Full Mode (Comprehensive)")
    # print_debug(0, "Running literature review agent in full mode...")
    # result_full = run_literature_review(mode="full", verbose=True)
    # 
    # if result_full:
    #     print_results(result_full, mode="full")


if __name__ == "__main__":
    main()
