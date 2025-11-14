#!/usr/bin/env python3
"""
Quick test of the parallel research with LangSmith monitoring.

This script tests:
1. Problem decomposition
2. Parallel information gathering
3. Result synthesis
4. Final answer generation

All steps should be visible in LangSmith.
"""

import os
from src.agent_blocks_hub.deep_diver.agent import create_deepdiver_agent
from src.llm_tool_hub.scientific_research_tool import internet_search

def main():
    print("\n" + "="*80)
    print("DEEP DIVER - PARALLEL RESEARCH TEST")
    print("="*80)
    
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return
    
    # Get LangSmith config
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    langsmith_project = os.getenv("LANGSMITH_PROJECT", "llm-tool-hub")
    
    print(f"\n✓ LangSmith Project: {langsmith_project}")
    print(f"✓ LangSmith Enabled: {bool(langsmith_api_key)}")
    
    # Import here to use environment variables
    from langchain_openai import ChatOpenAI
    
    # Create LLM
    llm = ChatOpenAI(
        model="google/gemini-2.5-flash-lite-preview-09-2025",
        temperature=0.7,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    print("\n✓ LLM initialized: google/gemini-2.5-flash-lite-preview-09-2025")
    
    # Create tools
    tools = [internet_search]
    print("✓ Tools ready: duckduckgo_search")
    
    # Create agent
    agent = create_deepdiver_agent(
        llm=llm,
        tools=tools,
        task_type="simple",
        enable_task_classification=True
    )
    
    print("\n✓ Agent created with parallel research capability")
    
    # Test question
    question = "What is LangGraph and how is it used in LLM applications?"
    
    print("\n" + "="*80)
    print("EXECUTING AGENT")
    print("="*80)
    print(f"\nQuestion: {question}\n")
    
    # Invoke agent
    result = agent.invoke({"messages": [("human", question)]})
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Show decomposed problems
    if "decomposed_problems" in result:
        print("\n[DECOMPOSED PROBLEMS]")
        for i, problem in enumerate(result["decomposed_problems"], 1):
            print(f"  {i}. {problem}")
    
    # Show synthesis
    if "synthesis_text" in result:
        print("\n[SYNTHESIS]")
        synthesis = result["synthesis_text"]
        if len(synthesis) > 500:
            print(synthesis[:500] + "...")
        else:
            print(synthesis)
    
    # Show final answer
    if "final_answer" in result:
        print("\n[FINAL ANSWER]")
        print("-" * 80)
        answer = result["final_answer"]
        if len(answer) > 1000:
            print(answer[:1000] + "\n... (truncated)")
        else:
            print(answer)
        print("-" * 80)
    
    print("\n" + "="*80)
    print("✓ Execution Complete - Check LangSmith for detailed traces")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
