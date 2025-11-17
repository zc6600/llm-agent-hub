"""
Example 01: Direct Mode - Fast and Reliable

This example demonstrates the "direct" execution mode, which directly calls
tool.run() without LLM tool calling. This is the RECOMMENDED mode for:
- Models with poor tool calling support (e.g., DeepSeek V3)
- Simple information gathering tasks
- When you need maximum speed

The direct mode bypasses LLM tool calling entirely and directly invokes the
tool's run() method, making it much faster and more reliable.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from agent_blocks_hub.parallel_agent import create_parallel_agent
from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
from llm_provider import get_llm
def main():
    print("=" * 80)
    print("Example 01: Direct Mode - Fast Tool Execution")
    print("=" * 80)
    print()
    llm = get_llm(
        temperature=0.7,
        max_tokens=100000,
        enable_langsmith=True,
    )
    # Initialize tool
    print("🔧 Initializing SearchSemanticScholar tool...")
    search_tool = SearchSemanticScholar()
    # Create agent in DIRECT mode (no LLM needed!)
    print("🤖 Creating parallel agent in DIRECT mode...")
    print("   Mode: direct (bypasses LLM tool calling)")
    print("   Summarization: disabled (maximum speed)")
    print()
    
    agent = create_parallel_agent(
        llm=llm,
        tools=[search_tool],
        mode="direct",  # Direct tool calling - NO LLM NEEDED
        enable_summarization=False,  # No LLM summarization for maximum speed
        enable_remark=True,
        verbose=True,
    )
    
    # Define search queries
    queries = [
        "transformer neural networks",
        "attention mechanism deep learning",
        "BERT language model",
    ]
    
    print(f"📊 Running {len(queries)} parallel searches...")
    print(f"Queries: {queries}")
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
        preview = tr[:20000] + "..." if len(tr) > 20000 else tr
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
    print("✅ Direct mode example completed successfully!")
    summary = result.get("summary")
    if summary:
        print("Summary:\n" + summary)
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
