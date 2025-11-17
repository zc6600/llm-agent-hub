"""
Example 02: Tool Calling Mode - LLM Decides Tool Usage

This example demonstrates the "tool_calling" execution mode, where the LLM
decides when and how to call tools using llm.bind_tools(). This mode is
recommended for:
- Models with good tool calling support (GPT-4, Claude, etc.)
- When you have multiple tools and need intelligent selection
- When the LLM should decide how to formulate tool queries

Note: Requires an LLM with good tool calling support. Not recommended for
DeepSeek V3 or other models with poor tool calling.
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
    print("Example 02: Tool Calling Mode - LLM-Driven Tool Usage")
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
    print("   LLM: GPT-4 (good tool calling support)")
    print("   Tool: SearchSemanticScholar")
    print()
    
    llm = get_llm(model="gpt-4", temperature=0)
    search_tool = SearchSemanticScholar()
    # Create agent in TOOL_CALLING mode
    print("🤖 Creating parallel agent in TOOL_CALLING mode...")
    print("   Mode: tool_calling (LLM decides when/how to call tools)")
    print("   Summarization: disabled")
    print()
    
    agent = create_parallel_agent(
        llm=llm,
        tools=[search_tool],
        mode="tool_calling",  # LLM decides tool usage
        enable_summarization=False,
        verbose=True,
    )
    
    # Define search queries
    queries = [
        "transformer neural networks",
        "attention mechanism deep learning",
        "BERT language model",
    ]
    
    print(f"📊 Running {len(queries)} parallel searches with LLM tool calling...")
    print(f"Queries: {queries}")
    print()
    print("⚠️  Note: LLM will decide how to formulate the tool queries")
    print()
    
    # Execute searches in parallel
    result = agent.invoke({
        "parallel_agent_message": queries,
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
        preview = tr[:400] + "..." if len(tr) > 400 else tr
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
    print("✅ Tool calling mode example completed!")
    summary = result.get("summary")
    if summary:
        print("Summary:\n" + summary)
    print()
    print("💡 Key features of tool_calling mode:")
    print("   - LLM intelligently decides how to use tools")
    print("   - Can reformulate queries for better results")
    print("   - Best with GPT-4, Claude (good tool calling support)")
    print("   - Slower than direct mode due to LLM overhead")
    print("=" * 80)


if __name__ == "__main__":
    main()
