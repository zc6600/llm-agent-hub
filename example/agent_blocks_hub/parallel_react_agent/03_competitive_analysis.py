"""
Concrete example: Competitive Analysis using Parallel React Agent.

This example demonstrates how to analyze a company from multiple
competitive perspectives in parallel:
1. Product comparison with competitors
2. Pricing strategy analysis
3. Market positioning and messaging

Useful for: Business analysis, market research, competitive intelligence
"""

import sys
import json
from pathlib import Path

# Ensure project src directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_community.tools import DuckDuckGoSearchRun
from llm_provider import get_llm
from agent_blocks_hub.parallel_react_agent import create_parallel_react_agent


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    print("\n" + char * 80)
    print(f"  {title}")
    print(char * 80)


def print_subsection(title: str):
    """Print a formatted subsection."""
    print(f"\n{title}")
    print("-" * len(title))


def main():
    """Run competitive analysis example."""
    
    print_header("📊 COMPETITIVE ANALYSIS: LANGCHAIN ECOSYSTEM", "🔷")
    
    # Initialize LLM and Tools
    print("Initializing agents...")
    llm = get_llm(temperature=0.6, max_tokens=100000, enable_langsmith=False)
    search_tool = DuckDuckGoSearchRun()
    
    # Define competitive analysis queries
    analysis_queries = [
        "What are the main competitors to LangChain in the LLM application development space? Compare their core features, supported LLMs, and market positioning.",
        
        "What is the pricing model and licensing strategy of LangChain and its main competitors? Include open-source vs commercial offerings.",
        
        "What are the key differentiators and competitive advantages of LangChain? What are its weaknesses compared to competitors like LlamaIndex, AutoGen, and others?"
    ]
    
    analysis_instructions = """
Your task is to conduct competitive analysis research.

Research Guidelines:
1. Identify direct and indirect competitors
2. Compare key features, capabilities, and limitations
3. Include pricing models and licensing approaches
4. Analyze market positioning and target users
5. Identify unique strengths and weaknesses
6. Cite specific products, features, and market data

Be analytical and objective. Present balanced comparisons.
    """
    
    print_header("ANALYSIS DESIGN", "=")
    print(f"Competitive Analysis Points: {len(analysis_queries)}")
    print_subsection("Analysis Angle 1: Competitor Landscape")
    print(f"  {analysis_queries[0][:60]}...")
    print_subsection("Analysis Angle 2: Pricing & Licensing")
    print(f"  {analysis_queries[1][:60]}...")
    print_subsection("Analysis Angle 3: Competitive Advantages")
    print(f"  {analysis_queries[2][:60]}...")
    
    # Create agent
    print_header("PARALLEL ANALYSIS EXECUTION", "=")
    print("Creating and running parallel competitive analysts...\n")
    
    agent = create_parallel_react_agent(
        llm=llm,
        tools=[search_tool],
        system_prompt=analysis_instructions,
    )
    
    result = agent.invoke({
        "parallel_react_agent_messages": analysis_queries
    })
    
    # Display results
    print_header("COMPETITIVE ANALYSIS FINDINGS", "=")
    
    analysis_titles = [
        "Competitor Landscape",
        "Pricing & Licensing Strategy",
        "Competitive Positioning & Advantages"
    ]
    
    agent_results = result.get("agent_results", {})
    
    for idx in range(len(analysis_queries)):
        agent_result = agent_results.get(idx)
        print_subsection(f"Analysis {idx + 1}: {analysis_titles[idx]}")
        
        if agent_result and agent_result['success']:
            result_text = agent_result['result']
            if len(result_text) > 600:
                print(result_text[:600] + "\n   [... truncated ...]")
            else:
                print(result_text)
        else:
            error_msg = agent_result.get('error', 'Unknown error') if agent_result else 'No result'
            print(f"   Error: {error_msg}")
    
    # Display integrated summary
    print_header("INTEGRATED COMPETITIVE SUMMARY", "=")
    print("\n🧬 Synthesized Analysis across all competitive dimensions:\n")
    
    final_summary = result.get("final_summary", "")
    if final_summary:
        print(final_summary)
    else:
        print("⚠️  No summary generated")
    
    print_header("ANALYSIS COMPLETE", "🔷")


if __name__ == "__main__":
    main()
