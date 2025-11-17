"""
Example 03: ReAct Mode - Full Reasoning Loops

This example demonstrates the "react" execution mode, which uses full ReAct
reasoning loops with observation-thought-action cycles. This mode is recommended for:
- Complex research tasks requiring multi-step reasoning
- When you need detailed thought processes
- Quality is more important than speed

Note: This is the slowest mode but provides the most comprehensive reasoning.
"""

import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from agent_blocks_hub.parallel_agent import create_parallel_agent
from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
from llm_provider import get_llm

def main():
    print("=" * 80)
    print("Example 03: ReAct Mode - Full Reasoning with Thought Chains")
    print("=" * 80)
    print()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("This example requires an OpenAI API key for GPT-4")
        print()
        print("Please set it with:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize LLM and tool
    print("🔧 Initializing components...")
    print("   LLM: GPT-4")
    print("   Tool: SearchSemanticScholar")
    print()
    
    llm = get_llm(model="gpt-4", temperature=0)
    search_tool = SearchSemanticScholar()
    # Create agent in REACT mode
    print("🤖 Creating parallel agent in REACT mode...")
    print("   Mode: react (full reasoning loops)")
    print("   Features: Observation → Thought → Action cycles")
    print()
    
    agent = create_parallel_agent(
        llm=llm,
        tools=[search_tool],
        mode="react",  # Full ReAct reasoning loops
        verbose=True,
    )
    
    # Define search queries (use fewer for ReAct since it's slower)
    queries = [
        "transformer neural networks",
        "BERT language model",
    ]
    
    print(f"📊 Running {len(queries)} searches with full ReAct reasoning...")
    print(f"Queries: {queries}")
    print()
    print("⚠️  Note: This will be slower as LLM performs full reasoning")
    print()
    
    # Execute searches in parallel
    result = agent.invoke({
        "parallel_react_agent_messages": queries,
        "enable_summarization": False,  # Disable summarization to focus on individual agent results
    })
    
    # Display results
    print()
    print("=" * 80)
    print("📋 RESULTS (Unified Keys)")
    print("=" * 80)
    print()
    
    tool_result = result.get("tool_result", {})
    remarks = result.get("remark", {})
    combined = result.get("tool_result_with_remark", {})
    queries_from_result = result.get("query", {})

    for idx, query in enumerate(queries):
        q = queries_from_result.get(idx, query)
        print(f"Query {idx + 1}: {q}")
        tr = tool_result.get(idx, "")
        preview = tr[:300] + "..." if len(tr) > 300 else tr
        print(f"Result preview: {preview}")
        rm = remarks.get(idx)
        if rm:
            print(f"Remark: {rm}")
        cr = combined.get(idx)
        if cr:
            print(f"Result+Remark: {cr[:400]}{'...' if len(cr) > 400 else ''}")
        print("-" * 80)
        print()
    
    print("=" * 80)
    print("✅ ReAct mode example completed!")
    summary = result.get("summary")
    if summary:
        print("Summary:\n" + summary)
    print()
    print("💡 Key features of react mode:")
    print("   - Full observation-thought-action reasoning cycles")
    print("   - Most comprehensive and detailed results")
    print("   - Best for complex research questions")
    print("   - Slowest execution (many LLM calls)")
    print("   - Highest quality reasoning")
    print("=" * 80)


if __name__ == "__main__":
    main()
