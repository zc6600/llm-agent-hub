"""
Concrete example: Product Evaluation using Parallel React Agent.

This example demonstrates evaluating a technology product from
multiple evaluation dimensions:
1. Technical capabilities and performance
2. User experience and community feedback
3. Cost-effectiveness and ROI considerations

Useful for: Product selection, technology evaluation, procurement decisions
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
    """Run product evaluation example."""
    
    print_header("🎯 PRODUCT EVALUATION: POSTGRESQL VS MONGODB", "🔷")
    
    # Initialize
    print("🚀 Initializing evaluation framework...\n")
    llm = get_llm(temperature=0.5, max_tokens=100000, enable_langsmith=False)
    search_tool = DuckDuckGoSearchRun()
    
    # Evaluation queries from different dimensions
    evaluation_queries = [
        "What are the technical capabilities, performance characteristics, and scalability limits of PostgreSQL? Include benchmarks, query performance, and concurrency handling.",
        
        "What is the user experience with PostgreSQL? Include community size, available tools, ecosystem maturity, and ease of deployment. What are common pain points?",
        
        "What is the total cost of ownership for PostgreSQL? Include licensing, operational costs, infrastructure requirements, and ROI for different use cases."
    ]
    
    evaluation_system_prompt = """
You are a technology product evaluator. Conduct thorough evaluation research.

Evaluation Criteria:
1. Technical Analysis:
   - Capabilities and features
   - Performance metrics and benchmarks
   - Scalability and limitations
   - Architecture and design choices

2. User Experience Analysis:
   - Community size and ecosystem
   - Documentation quality
   - Learning curve
   - Deployment complexity
   - Real-world user feedback

3. Cost Analysis:
   - Licensing model
   - Infrastructure costs
   - Operational overhead
   - Training requirements
   - Total cost of ownership

Provide concrete numbers, specific features, and real-world examples.
    """
    
    # Display evaluation plan
    print_header("EVALUATION PLAN", "=")
    print(f"Evaluation Dimensions: {len(evaluation_queries)}\n")
    
    print_subsection("Dimension 1: Technical Capabilities")
    print(f"  Focus: Performance, scalability, features")
    print_subsection("Dimension 2: User Experience")
    print(f"  Focus: Community, ecosystem, ease of use")
    print_subsection("Dimension 3: Cost & ROI")
    print(f"  Focus: Licensing, operations, total cost")
    
    # Run evaluation
    print_header("PARALLEL EVALUATION EXECUTION", "=")
    print("Launching parallel evaluators across all dimensions...\n")
    
    agent = create_parallel_react_agent(
        llm=llm,
        tools=[search_tool],
        system_prompt=evaluation_system_prompt,
    )
    
    result = agent.invoke({
        "parallel_react_agent_messages": evaluation_queries
    })
    
    # Display evaluation findings
    print_header("EVALUATION FINDINGS", "=")
    
    dimension_titles = [
        "Technical Capabilities",
        "User Experience & Ecosystem",
        "Cost & Total Cost of Ownership"
    ]
    
    agent_results = result.get("agent_results", {})
    
    for idx in range(len(evaluation_queries)):
        agent_result = agent_results.get(idx)
        print_subsection(f"Dimension {idx + 1}: {dimension_titles[idx]}")
        
        if agent_result and agent_result['success']:
            result_text = agent_result['result']
            if len(result_text) > 700:
                print(result_text[:700] + "\n   [... truncated ...]")
            else:
                print(result_text)
            print(f"   ✓ Evaluation complete")
        else:
            error_msg = agent_result.get('error', 'Unknown error') if agent_result else 'No result'
            print(f"   ❌ Error: {error_msg}")
    
    # Integrated evaluation summary
    print_header("INTEGRATED EVALUATION REPORT", "=")
    print("\n📋 Comprehensive Product Evaluation Summary:\n")
    
    final_summary = result.get("final_summary", "")
    if final_summary:
        print(final_summary)
    else:
        print("⚠️  No summary generated")
    
    # Recommendation framework
    print_header("EVALUATION FRAMEWORK", "=")
    print("""
Based on this multi-dimensional evaluation, consider:

✓ Technical Fit:
  - Does it meet your performance requirements?
  - Can it scale to your data volume?
  - Does it support your use cases?

✓ Operational Fit:
  - Can your team easily manage it?
  - Is the ecosystem mature enough?
  - What is the learning curve?

✓ Financial Fit:
  - What is the total cost of ownership?
  - How does it compare to alternatives?
  - What is the ROI timeline?

✓ Strategic Fit:
  - Aligns with your tech strategy?
  - Compatible with existing systems?
  - Future-proof for growth?
    """)
    
    print_header("EVALUATION COMPLETE", "🔷")


if __name__ == "__main__":
    main()
