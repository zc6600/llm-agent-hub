"""
Example usage of Related Paper Searcher Lite Agent

This example demonstrates how to use the agent to search for and evaluate
related papers based on a research query.
"""

import sys
from pathlib import Path

# Ensure project src directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_provider import get_llm
from agent_blocks_hub.scientific_research.related_paper_searcher_lite import (
    create_related_paper_searcher_lite_agent
)


def main():
    """Run the Related Paper Searcher Lite Agent"""
    
    # Initialize LLM
    llm = get_llm(model="gpt-4", temperature=0)
    
    # Create agent
    agent = create_related_paper_searcher_lite_agent(llm=llm)
    
    # Example query
    query = "transformer neural networks attention mechanism"
    
    # Run agent
    result = agent.invoke({
        "query": query,
        "messages": []
    })
    
    # Display results
    print("\n" + "=" * 80)
    print(f"Search Query: {query}")
    print("=" * 80)
    
    print(f"\nTotal papers found: {len(result.get('evaluation_results', []))}")
    print(f"Related papers: {len(result.get('related_papers', []))}")
    
    print("\n" + "-" * 80)
    print("RELATED PAPERS:")
    print("-" * 80)
    
    for paper in result.get("related_papers", []):
        print(f"\nTitle: {paper.get('paper_title', 'N/A')}")
        print(f"Relevance: {paper.get('relevance', 'N/A')}")
        print(f"Comment: {paper.get('comment', 'N/A')}")


if __name__ == "__main__":
    main()
