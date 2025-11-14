"""
Example showing LLM-controlled iteration in Deep Diver agent.

This example demonstrates how the LLM makes intelligent decisions
about when to continue iterating vs. when to finish, rather than
using hard-coded rules.
"""

import sys
from pathlib import Path

# Ensure project src directory is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from llm_provider import get_llm
from agent_blocks_hub.deep_diver import create_deepdiver_agent


@tool
def knowledge_base_search(query: str) -> str:
    """
    Search internal knowledge base for information.
    
    Args:
        query: Search query
        
    Returns:
        Search results from knowledge base
    """
    # Mock implementation
    kb_data = {
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs, using graphs.",
        "agents": "Agents are systems that use LLMs to decide which actions to take.",
        "tools": "Tools are functions that agents can call to interact with external systems.",
    }
    
    for key, value in kb_data.items():
        if key.lower() in query.lower():
            return value
    
    return "No matching information found in knowledge base."


def main():
    """Run example with LLM-controlled iteration."""
    
    print("=" * 80)
    print("LLM-Controlled Iteration Example")
    print("=" * 80)
    
    # Initialize LLM with temperature for more dynamic decisions
    llm = get_llm(model="gpt-4", temperature=0.3)
    
    # Prepare tools
    tools = [
        DuckDuckGoSearchRun(),
        knowledge_base_search
    ]
    
    # Custom system prompt emphasizing quality over quantity
    system_prompt = """
    You are a research agent that follows the scientific method.
    
    Key principles:
    - Seek high-quality, well-verified hypotheses over quantity
    - Be willing to iterate if evidence is weak or contradictory
    - Stop iterating when you have strong, converging evidence
    - Balance thoroughness with efficiency
    - Trust your judgment on when enough research has been done
    
    Make intelligent decisions about when to continue vs. finish based on:
    1. Quality and consistency of evidence
    2. Confidence in your hypotheses
    3. Whether additional iteration would meaningfully improve the answer
    """
    
    # Create agent with higher max_iterations to let LLM decide
    agent = create_deepdiver_agent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=5  # High limit, let LLM decide when to stop
    )
    
    # Test with different questions to see iteration behavior
    questions = [
        "What is LangGraph and how does it differ from LangChain?",
        "Explain the concept of agent tools in AI systems.",
        "What are the key principles of Popper's scientific method?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 80}")
        print(f"Question {i}: {question}")
        print('=' * 80)
        
        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            
            print(f"\nIterations Used: {result.get('current_iteration', 0)}")
            print(f"Hypotheses Generated: {len(result.get('hypotheses', []))}")
            print(f"Experience Pool Size: {len(result.get('experience_pool', []))}")
            
            print(f"\nHypotheses Status:")
            for j, hyp in enumerate(result.get('hypotheses', []), 1):
                status = hyp.get('verification_result', 'unknown')
                content = hyp.get('content', '')[:80]
                print(f"  {j}. [{status}] {content}...")
            
            print(f"\nFinal Answer Preview:")
            print(f"{result.get('final_answer', 'N/A')[:300]}...")

            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Demonstration Complete!")
    print("=" * 80)
    print("\nNotice how the LLM decides when to stop iterating based on:")
    print("- Quality of verified hypotheses")
    print("- Confidence in the evidence")
    print("- Diminishing returns from additional iterations")


if __name__ == "__main__":
    main()
