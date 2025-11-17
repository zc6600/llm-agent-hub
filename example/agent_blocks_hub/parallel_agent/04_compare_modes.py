"""
Example 04: Compare All Modes - Side-by-Side Comparison

This example runs all three execution modes (direct, tool_calling, react)
side-by-side to compare their:
- Execution time
- Result quality
- LLM usage
- Reliability

This helps you choose the right mode for your use case.
"""

import sys
from pathlib import Path
import os
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from agent_blocks_hub.parallel_agent import create_parallel_agent
from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
from llm_provider import get_llm


def run_mode(mode: str, llm, tools, queries):
    """Run a specific mode and measure performance."""
    print(f"\n{'=' * 80}")
    print(f"🚀 Testing Mode: {mode.upper()}")
    print(f"{'=' * 80}\n")
    
    start_time = time.time()
    
    try:
        # Create agent based on mode
        if mode == "direct":
            agent = create_parallel_agent(
                tools=tools,
                mode="direct",
                enable_summarization=False,
                verbose=False,
            )
            input_key = "parallel_agent_message"
        elif mode == "tool_calling":
            agent = create_parallel_agent(
                llm=llm,
                tools=tools,
                mode="tool_calling",
                enable_summarization=False,
                verbose=False,
            )
            input_key = "parallel_agent_message"
        else:  # react
            agent = create_parallel_agent(
                llm=llm,
                tools=tools,
                mode="react",
                verbose=False,
            )
            input_key = "parallel_react_agent_messages"
        
        # Execute
        payload = {input_key: queries}
        if mode == "react":
            payload["enable_summarization"] = False  # Avoid known summarization issue in react nodes
        result = agent.invoke(payload)
        
        execution_time = time.time() - start_time
        
        # Analyze results
        agent_results = result.get("agent_results", {})
        success_count = sum(1 for r in agent_results.values() if r.get("success", False))
        
        print(f"✅ Mode: {mode}")
        print(f"⏱️  Execution time: {execution_time:.2f}s")
        print(f"📊 Success rate: {success_count}/{len(queries)} ({success_count/len(queries)*100:.0f}%)")
        
        # Show sample result (Unified keys)
        tool_result = result.get("tool_result", {})
        if tool_result:
            sample_text = tool_result.get(0, "")
            preview = sample_text[:150] + "..." if len(sample_text) > 150 else sample_text
            print(f"📝 Sample result: {preview}")
        
        return {
            "mode": mode,
            "time": execution_time,
            "success_rate": success_count / len(queries),
            "success_count": success_count,
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ Mode {mode} FAILED: {str(e)}")
        return {
            "mode": mode,
            "time": execution_time,
            "success_rate": 0,
            "success_count": 0,
            "error": str(e),
        }


def main():
    print("=" * 80)
    print("Example 04: Compare All Parallel Agent Modes")
    print("=" * 80)
    print()
    
    # Initialize components
    print("🔧 Initializing components...")
    search_tool = SearchSemanticScholar()
    
    # Initialize LLM (optional for direct mode)
    llm = None
    if os.getenv("OPENAI_API_KEY"):
        print("   LLM: GPT-4 (for tool_calling and react modes)")
        llm = get_llm(model="gpt-4", temperature=0)
    else:
        print("   ⚠️  No OPENAI_API_KEY found - skipping tool_calling and react modes")
    
    print("   Tool: SearchSemanticScholar")
    print()
    
    # Define test queries
    queries = [
        "transformer neural networks",
        "BERT language model",
    ]
    
    print(f"📊 Test queries: {queries}")
    print()
    
    # Run all modes
    results = []
    
    # 1. Direct mode (always available)
    print("\n" + "🔹" * 40)
    print("Mode 1/3: DIRECT (no LLM needed)")
    print("🔹" * 40)
    results.append(run_mode("direct", llm, [search_tool], queries))
    
    # 2. Tool calling mode (requires LLM)
    if llm:
        print("\n" + "🔹" * 40)
        print("Mode 2/3: TOOL_CALLING (LLM decides tool usage)")
        print("🔹" * 40)
        results.append(run_mode("tool_calling", llm, [search_tool], queries))
    
    # 3. ReAct mode (requires LLM)
    if llm:
        print("\n" + "🔹" * 40)
        print("Mode 3/3: REACT (full reasoning loops)")
        print("🔹" * 40)
        results.append(run_mode("react", llm, [search_tool], queries))
    
    # Summary comparison
    print("\n\n" + "=" * 80)
    print("📊 COMPARISON SUMMARY")
    print("=" * 80)
    print()
    
    print(f"{'Mode':<15} {'Time (s)':<12} {'Success Rate':<15} {'Speed':<10}")
    print("-" * 80)
    
    # Sort by time for speed ranking
    sorted_results = sorted([r for r in results if "error" not in r], key=lambda x: x["time"])
    
    for idx, result in enumerate(sorted_results, 1):
        mode = result["mode"]
        time_taken = result["time"]
        success_rate = result["success_rate"]
        
        # Speed indicator
        if idx == 1:
            speed = "⚡⚡⚡"
        elif idx == 2:
            speed = "⚡⚡"
        else:
            speed = "⚡"
        
        print(f"{mode:<15} {time_taken:<12.2f} {f'{success_rate*100:.0f}%':<15} {speed:<10}")
    
    print()
    print("=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("✅ Use DIRECT mode when:")
    print("   - Working with DeepSeek V3 or models with poor tool calling")
    print("   - Need maximum speed")
    print("   - Simple information gathering")
    print()
    print("✅ Use TOOL_CALLING mode when:")
    print("   - Working with GPT-4, Claude")
    print("   - LLM should decide tool usage")
    print("   - Have multiple tools")
    print()
    print("✅ Use REACT mode when:")
    print("   - Complex reasoning required")
    print("   - Quality more important than speed")
    print("   - Need detailed thought processes")
    print("=" * 80)


if __name__ == "__main__":
    main()
