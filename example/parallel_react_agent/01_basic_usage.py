"""
Basic usage example of the Parallel React Agent.

This example demonstrates how to create and use a Parallel React Agent
to process multiple queries in parallel and synthesize the results.
"""

import sys
import json
from pathlib import Path

# Ensure project src directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_community.tools import DuckDuckGoSearchRun
from llm_provider import get_llm
from agent_blocks_hub.parallel_react_agent import create_parallel_react_agent


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
    """Run a basic Parallel React Agent example."""
    print_section("INITIALIZING PARALLEL REACT AGENT")
    
    print_debug(0, "Configuring Language Model (LLM)")
    # Initialize LLM from environment variables
    llm = get_llm(
        temperature=0.7,
        max_tokens=100000,
        enable_langsmith=False,
    )
    print_debug(1, "✓ LLM initialized")
    
    print_debug(0, "Initializing Tools")
    # Initialize tools
    search_tool = DuckDuckGoSearchRun()
    print_debug(1, "✓ DuckDuckGo Search tool initialized")
    
    # System prompt for research
    research_instructions = """
    You are a thorough research assistant. When answering questions:
    1. Search for current and relevant information
    2. Provide evidence-based answers
    3. Cite sources when possible
    4. Be critical and thorough
    """
    
    print_debug(0, "Creating Parallel React Agent")
    # Create the agent
    agent = create_parallel_react_agent(
        llm=llm,
        tools=[search_tool],
        system_prompt=research_instructions,
    )
    print_debug(1, "✓ Agent created and compiled")
    
    # Define multiple queries
    queries = [
        "What is LangGraph and how does it work?",
        "What is the ReAct pattern in AI agents?",
        "What are the latest trends in LLM agents?",
    ]
    
    print_section("STARTING PARALLEL AGENT EXECUTION")
    print_debug(0, f"Processing {len(queries)} queries in parallel")
    for i, q in enumerate(queries):
        print_debug(1, f"Query {i+1}: {q}")
    
    print_debug(0, "Invoking agent...")
    # Invoke the agent
    result = agent.invoke({
        "parallel_react_agent_messages": queries
    })
    print_debug(1, "✓ Agent execution completed")
    
    # Print results
    print_section("INDIVIDUAL AGENT RESULTS")
    for idx, agent_result in result.get("agent_results", {}).items():
        print(f"\n[Agent {idx}] Query: {agent_result['query']}")
        if agent_result['success']:
            print(f"Result: {agent_result['result'][:500]}...")
        else:
            print(f"Error: {agent_result['error']}")
    
    print_section("FINAL SUMMARY")
    final_summary = result.get("final_summary", "")
    print(final_summary)
    
    print_section("EXECUTION COMPLETE")
    print("✓ All agents completed successfully!")


if __name__ == "__main__":
    main()
