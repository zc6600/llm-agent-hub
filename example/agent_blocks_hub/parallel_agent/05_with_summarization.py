"""
Example 05: With LLM Summarization

This example demonstrates how to use LLM summarization to synthesize results
from multiple parallel searches into a coherent summary.

LLM summarization is useful when:
- You have many search results and need a synthesized overview
- You want to extract key insights across multiple queries
- You need a narrative summary rather than raw tool outputs
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
    print("Example 05: Parallel Agent with LLM Summarization")
    print("=" * 80)
    print()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("This example requires an OpenAI API key for LLM summarization")
        print()
        print("Please set it with:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize components
    print("🔧 Initializing components...")
    print("   LLM: GPT-4 (for summarization)")
    print("   Tool: SearchSemanticScholar")
    print()
    
    llm = get_llm(model="gpt-4", temperature=0)
    search_tool = SearchSemanticScholar()
    
    # Create agent with summarization ENABLED
    print("🤖 Creating parallel agent with summarization...")
    print("   Mode: direct (fast tool execution)")
    print("   Summarization: ENABLED (LLM synthesizes results)")
    print()
    
    agent = create_parallel_agent(
        llm=llm,  # Required for summarization
        tools=[search_tool],
        mode="direct",
        enable_summarization=True,  # Enable LLM summarization
        system_prompt="You are a research assistant. Synthesize the search results into a coherent summary highlighting key findings and connections.",
        verbose=True,
    )
    
    # Define research topic with related queries
    print("📊 Research Topic: Transformer Architecture Evolution")
    queries = [
        "original transformer architecture attention is all you need",
        "BERT bidirectional language model",
        "GPT generative pre-training",
        "vision transformer ViT",
    ]
    
    print(f"Queries ({len(queries)} total):")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    print()
    
    # Execute searches with summarization
    print("🔍 Executing parallel searches...")
    result = agent.invoke({
        "parallel_agent_message": queries,
    })
    
    # Display individual results
    print()
    print("=" * 80)
    print("📋 INDIVIDUAL RESULTS (Unified Keys)")
    print("=" * 80)
    print()
    
    tool_result = result.get("tool_result", {})
    remarks = result.get("remark", {})
    combined = result.get("tool_result_with_remark", {})

    for idx, query in enumerate(queries):
        print(f"Query {idx + 1}: {query}")
        tr = tool_result.get(idx, "")
        preview = tr[:150] + "..." if len(tr) > 150 else tr
        print(f"Preview: {preview}")
        rm = remarks.get(idx)
        if rm:
            print(f"Remark: {rm}")
        cr = combined.get(idx)
        if cr:
            print(f"Result+Remark: {cr[:300]}{'...' if len(cr) > 300 else ''}")
        print("-" * 80)
    
    # Display synthesized summary
    print()
    print("=" * 80)
    print("🎯 LLM-GENERATED SUMMARY")
    print("=" * 80)
    print()
    
    summary = result.get("summary", "")
    
    if summary:
        print(summary)
    else:
        print("⚠️  No summary generated")
    
    print()
    print("=" * 80)
    print("✅ Summarization example completed!")
    print()
    print("💡 Key benefits of summarization:")
    print("   - Coherent narrative across multiple searches")
    print("   - Highlights connections and patterns")
    print("   - Extracts key insights")
    print("   - Provides research overview")
    print()
    print("⚠️  Note: Summarization adds LLM overhead")
    print("   - Only use when you need synthesized insights")
    print("   - For raw data, disable summarization")
    print("=" * 80)


if __name__ == "__main__":
    main()
