"""
Basic usage example of the Deep Diver agent with detailed debugging.

This example demonstrates how to create and use a Deep Diver agent
with a simple research question, including step-by-step debugging output.
"""

import sys
import json
from pathlib import Path

# Ensure project src directory is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_community.tools import DuckDuckGoSearchRun

from llm_provider import get_llm
from agent_blocks_hub.deep_diver import create_deepdiver_agent


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_debug(level: int, message: str, data=None):
    """Print debug message with indentation based on level."""
    indent = "  " * level
    print(f"{indent}[DEBUG {level}] {message}")
    if data is not None:
        if isinstance(data, (dict, list)):
            print(f"{indent}    {json.dumps(data, ensure_ascii=False, indent=2)}")
        else:
            print(f"{indent}    {data}")


def main():
    """Run a basic Deep Diver agent example."""
    print_section("INITIALIZING DEEP DIVER AGENT")
    
    print_debug(0, "Configuring Language Model (LLM)")
    # Initialize LLM from environment variables with LangSmith monitoring
    llm = get_llm(
        temperature=0.7,
        max_tokens=100000,
        enable_langsmith=True,
    )
    print_debug(1, "✓ LLM initialized from environment configuration", {
        "source": ".env environment variables",
        "langsmith_enabled": True,
    })
    
    print_debug(0, "Initializing Tools")
    # Initialize tools
    internet_search = DuckDuckGoSearchRun()
    print_debug(1, "✓ DuckDuckGo Search tool initialized")
    
    # Research instructions (system prompt)
    research_instructions = """
    You are a thorough research agent. Follow the scientific method:
    1. Break down the question into testable components
    2. Gather comprehensive information
    3. Form hypotheses based on evidence
    4. Verify each hypothesis rigorously
    5. Provide a well-supported final answer
    
    Be critical, evidence-based, and thorough.
    """
    
    print_debug(0, "Creating Deep Diver Agent")
    # Create the agent
    agent = create_deepdiver_agent(
        llm=llm,
        tools=[internet_search],
        system_prompt=research_instructions,
    )
    print_debug(1, "✓ Agent graph created and compiled")
    
    # Define the question
    question = "What is langgraph?"
    print_section("STARTING AGENT EXECUTION")
    print_debug(0, f"User Question: {question}")
    
    print_debug(0, "Invoking agent...")
    # Invoke the agent
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    print_debug(1, "✓ Agent execution completed")
    
    # Print results with debugging information
    print_section("STEP 1: PROBLEM FORMULATION & DECOMPOSITION")
    original_q = result.get('original_question')
    print_debug(1, "Original Question captured:", original_q)
    
    decomposed = result.get('decomposed_problems', [])
    print_debug(1, f"Problems decomposed into {len(decomposed)} sub-problems:")
    for i, problem in enumerate(decomposed, 1):
        print_debug(2, f"Problem {i}: {problem}")
    
    print_section("STEP 2: INFORMATION GATHERING")
    info = result.get('gathered_information', [])
    print_debug(1, f"Information gathered from {len(info)} sources:")
    for i, item in enumerate(info, 1):
        if isinstance(item, dict):
            source = item.get('source', 'Unknown')
            content = item.get('content', '')[:100]  # First 100 chars
            print_debug(2, f"Source {i}: {source}", f"Content preview: {content}...")
        else:
            print_debug(2, f"Source {i}: {str(item)[:100]}...")
    
    print_section("STEP 3: HYPOTHESIS GENERATION & VERIFICATION")
    hypotheses = result.get('hypotheses', [])
    print_debug(1, f"Total hypotheses generated: {len(hypotheses)}")
    
    for i, hyp in enumerate(hypotheses, 1):
        print_debug(2, f"Hypothesis {i}:")
        print_debug(3, "Content:", hyp.get('content'))
        confidence = hyp.get('confidence')
        if confidence is not None:
            print_debug(3, "Confidence Score:", f"{confidence:.2%}" if isinstance(confidence, float) else confidence)
        verification = hyp.get('verification_result')
        if verification:
            print_debug(3, "Verification Result:", verification[:150] + ("..." if len(str(verification)) > 150 else ""))
        evidence = hyp.get('evidence', [])
        if evidence:
            print_debug(3, f"Supporting Evidence ({len(evidence)} items):")
            for j, ev in enumerate(evidence[:2], 1):  # Show first 2 evidences
                print_debug(4, f"Evidence {j}: {ev[:100]}...")
            if len(evidence) > 2:
                print_debug(4, f"... and {len(evidence) - 2} more evidence items")
    
    print_section("STEP 4: FINAL ANSWER & CONFIDENCE")
    final_answer = result.get('final_answer')
    if final_answer:
        print_debug(1, "Final Answer generated:")
        # Print answer in chunks to avoid overwhelming output
        answer_text = final_answer if isinstance(final_answer, str) else str(final_answer)
        print_debug(2, answer_text[:5000] + ("..." if len(answer_text) > 500 else ""))
    else:
        print_debug(1, "⚠ No final answer generated")
    
    print_section("EXECUTION SUMMARY")
    print_debug(0, "Agent Execution Completed Successfully ✓", {
        "decomposed_problems": len(decomposed),
        "information_sources": len(info),
        "hypotheses_generated": len(hypotheses),
        "final_answer_provided": bool(final_answer),
    })


if __name__ == "__main__":
    main()
