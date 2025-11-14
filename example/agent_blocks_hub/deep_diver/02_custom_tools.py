"""
Example of using Deep Diver agent with custom tools.

This example shows how to provide custom tools for scientific research.
"""

import sys
from pathlib import Path

# Ensure project src directory is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from llm_provider import get_llm
from agent_blocks_hub.deep_diver import create_deepdiver_agent


@tool
def academic_search(query: str) -> str:
    """
    Search academic papers and scientific literature.
    
    Args:
        query: Search query for academic content
        
    Returns:
        Search results from academic sources
    """
    # TODO: Implement actual academic search
    return f"Academic search results for: {query}"


@tool
def verify_facts(claim: str) -> str:
    """
    Verify factual claims using multiple sources.
    
    Args:
        claim: The claim to verify
        
    Returns:
        Verification result with confidence level
    """
    # TODO: Implement fact verification
    return f"Verification result for: {claim}"


def main():
    """Run Deep Diver with custom tools."""
    
    # Initialize LLM
    llm = get_llm(model="gpt-4", temperature=0)
    
    # Prepare tools
    tools = [
        DuckDuckGoSearchRun(),
        academic_search,
        verify_facts
    ]
    
    # Create agent
    agent = create_deepdiver_agent(
        llm=llm,
        tools=tools,
        max_iterations=5
    )
    
    # Complex research question
    question = """
    What are the key differences between transformer and recurrent neural network 
    architectures, and which is more suitable for long-sequence tasks?
    """
    
    # Invoke agent
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    
    # Display results
    print("\n" + "=" * 80)
    print("RESEARCH RESULTS")
    print("=" * 80)
    print(f"\n{result.get('final_answer')}")


if __name__ == "__main__":
    main()
