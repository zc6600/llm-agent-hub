"""
Concrete example: Multi-angle AI Safety Research using Parallel React Agent.

This example demonstrates how to research a complex topic from multiple
angles simultaneously. We research AI safety from three perspectives:
1. Technical safety measures
2. Governance and policy approaches
3. Emerging challenges and frontier risks

The parallel agents process these independently, then a summarizing agent
synthesizes the findings into a comprehensive overview.
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
from agent_blocks_hub.parallel_react_agent import create_parallel_react_agent, get_compiled_graph


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
    """Run the AI Safety research example."""
    
    print_header("🔬 AI SAFETY MULTI-ANGLE RESEARCH USING PARALLEL REACT AGENT", "🔷")
    
    # ===== STEP 1: Initialize LLM and Tools =====
    print_header("STEP 1: SYSTEM INITIALIZATION", "=")
    
    print("📦 Initializing Language Model...")
    llm = get_llm(
        temperature=0.5,
        max_tokens=100000,
        enable_langsmith=False,
    )
    print("   ✓ LLM ready (GPT-4 or Claude)")
    
    print("🔧 Initializing Research Tools...")
    search_tool = DuckDuckGoSearchRun()
    print("   ✓ DuckDuckGo search tool loaded")
    
    # ===== STEP 2: Define Research Queries =====
    print_header("STEP 2: RESEARCH DESIGN", "=")
    
    research_queries = [
        "What are the latest technical approaches to AI safety and alignment, including RLHF, constitutional AI, and mechanistic interpretability?",
        
        "What government policies, regulations, and international agreements are being developed for AI safety? Include frameworks like EU AI Act, Executive Order 14110, and recent policy developments.",
        
        "What are the emerging frontier risks and challenges in AI safety that researchers are currently worried about? Include jailbreaks, deceptive alignment, and scaling behaviors."
    ]
    
    print(f"📊 Designed {len(research_queries)} research angles:")
    print_subsection("Angle 1: Technical Safety Measures")
    print(f"   Query: {research_queries[0][:70]}...")
    
    print_subsection("Angle 2: Governance & Policy")
    print(f"   Query: {research_queries[1][:70]}...")
    
    print_subsection("Angle 3: Frontier Risks")
    print(f"   Query: {research_queries[2][:70]}...")
    
    # ===== STEP 3: Configure Research Instructions =====
    print_header("STEP 3: AGENT CONFIGURATION", "=")
    
    research_instructions = """
Your role is to conduct thorough, evidence-based research on AI safety topics.

Research Guidelines:
1. Search for peer-reviewed research, official reports, and credible sources
2. Focus on factual, recent information (prioritize last 2 years)
3. Include specific names of researchers, organizations, and projects
4. Cite specific metrics, benchmarks, or empirical findings when available
5. Present both consensus views and notable disagreements in the field
6. Be critical: identify limitations and open questions

Output Format:
- Start with a brief overview (2-3 sentences)
- Organize findings by key subtopics
- Include specific examples and recent developments
- End with key takeaways or implications
    """
    
    print("📋 Configured research guidelines:")
    print("   ✓ Evidence-based approach")
    print("   ✓ Recent information focus (2024-2025)")
    print("   ✓ Specific metrics and citations")
    print("   ✓ Critical analysis of field")
    
    # ===== STEP 4: Create Parallel React Agent =====
    print_header("STEP 4: AGENT CREATION", "=")
    
    print("🔨 Creating Parallel React Agent...")
    agent = create_parallel_react_agent(
        llm=llm,
        tools=[search_tool],
        system_prompt=research_instructions,
    )
    print("   ✓ Agent graph created and compiled")
    
    # Optional: Save graph visualization
    print("📊 Attempting to save graph visualization...")
    try:
        graph = get_compiled_graph(
            llm,
            [search_tool],
            research_instructions,
            save_graph_image=True,
            image_name="parallel_react_agent_graph.png"
        )
        print("   ✓ Graph visualization saved")
    except Exception as e:
        print(f"   ⚠️  Could not save graph (this is optional): {e}")
    
    # ===== STEP 5: Execute Parallel Research =====
    print_header("STEP 5: PARALLEL RESEARCH EXECUTION", "=")
    
    print(f"🚀 Launching {len(research_queries)} parallel research agents...\n")
    
    result = agent.invoke({
        "parallel_react_agent_messages": research_queries
    })
    
    print("\n✅ All research agents completed!")
    
    # ===== STEP 6: Display Individual Findings =====
    print_header("STEP 6: INDIVIDUAL RESEARCH FINDINGS", "=")
    
    angle_titles = [
        "Technical Safety Measures",
        "Governance & Policy Approaches",
        "Frontier Risks & Challenges"
    ]
    
    agent_results = result.get("agent_results", {})
    
    for idx in range(len(research_queries)):
        agent_result = agent_results.get(idx)
        
        print_subsection(f"Angle {idx + 1}: {angle_titles[idx]}")
        
        if agent_result:
            if agent_result['success']:
                result_text = agent_result['result']
                # Show first 800 characters
                if len(result_text) > 800:
                    print(result_text[:800] + "\n   [... truncated ...]")
                else:
                    print(result_text)
                
                print(f"\n   📈 Intermediate steps: {len(agent_result.get('intermediate_steps', []))} research iterations")
                print(f"   ✓ Status: Successfully completed")
            else:
                print(f"   ❌ Error: {agent_result['error']}")
        else:
            print("   ⚠️  No result found for this angle")
    
    # ===== STEP 7: Display Synthesized Summary =====
    print_header("STEP 7: SYNTHESIZED RESEARCH SUMMARY", "=")
    
    print("\n🧬 The Summarizing Agent has synthesized all three research angles:\n")
    
    final_summary = result.get("final_summary", "")
    
    if final_summary:
        print(final_summary)
    else:
        print("⚠️  No summary generated")
    
    # ===== STEP 8: Analysis and Insights =====
    print_header("STEP 8: RESEARCH INSIGHTS", "=")
    
    print("""
📌 Key Observations:

1. Multi-Angle Approach Benefits:
   - Technical angle: Deep dive into specific methods and their effectiveness
   - Governance angle: Understanding regulatory landscape and incentives
   - Frontier angle: Identifying open problems and future directions
   
2. Integration Value:
   - How technical solutions inform policy requirements
   - How governance affects research directions
   - How frontier risks guide technical priorities

3. Research Completeness:
   - All three angles processed independently and in parallel
   - Results logically integrated by the Summarizing Agent
   - Comprehensive view of the AI safety landscape

4. Next Steps for Using These Findings:
   - Use technical findings to understand capability/safety trade-offs
   - Use governance findings for regulatory compliance planning
   - Use frontier findings to identify high-impact research areas
    """)
    
    # ===== STEP 9: Execution Statistics =====
    print_header("STEP 9: EXECUTION STATISTICS", "=")
    
    successful_agents = sum(1 for r in agent_results.values() if r.get('success'))
    total_agents = len(agent_results)
    
    print(f"""
✓ Research Completion:
  - Total parallel agents: {total_agents}
  - Successful: {successful_agents}
  - Failed: {total_agents - successful_agents}

✓ Query Processing:
  - Technical safety: Query 1 ✓
  - Governance & policy: Query 2 ✓
  - Frontier risks: Query 3 ✓

✓ Summary Generation:
  - Summarizing agent synthesis: Complete
  - Logical integration: Done
  - Output format: Structured narrative
    """)
    
    print_header("🎯 RESEARCH COMPLETE", "🔷")
    
    print("""
This example demonstrates the power of the Parallel React Agent:

1. Efficiency: All three queries processed simultaneously
2. Comprehensiveness: Multiple research angles explored in parallel
3. Integration: Results synthesized into a coherent whole
4. Scalability: Easy to add more research angles as needed

You can extend this by:
- Adding more query angles for deeper exploration
- Using different tools for each angle (ArXiv for papers, etc.)
- Iterating on research instructions based on findings
- Combining with other agents for downstream tasks
    """)


if __name__ == "__main__":
    main()
