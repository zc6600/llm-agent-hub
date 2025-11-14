"""
Test script to verify LangSmith integration with llm_provider.

This script will:
1. Initialize the LLM with LangSmith enabled
2. Check if environment variables are properly set
3. Make a simple LLM call that should be traced in LangSmith
"""

import sys
import os
from pathlib import Path

# Ensure project src directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_provider import get_llm


def test_langsmith_integration():
    """Test LangSmith integration."""
    print("=" * 80)
    print("TESTING LANGSMITH INTEGRATION")
    print("=" * 80)
    
    # Initialize LLM with LangSmith enabled
    print("\n[STEP 1] Initializing LLM with LangSmith enabled...")
    llm = get_llm(
        temperature=0.7,
        max_tokens=1000,
        enable_langsmith=True,
    )
    
    # Check environment variables
    print("\n[STEP 2] Checking LangSmith environment variables...")
    env_vars = {
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2"),
        "LANGCHAIN_API_KEY": "***" if os.getenv("LANGCHAIN_API_KEY") else None,
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT"),
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT"),
    }
    
    for key, value in env_vars.items():
        if value:
            print(f"  ✓ {key}: {value}")
        else:
            print(f"  ✗ {key}: NOT SET")
    
    # Make a simple LLM call
    print("\n[STEP 3] Making a simple LLM call (should be traced in LangSmith)...")
    try:
        response = llm.invoke("What is 2+2? Answer in one sentence.")
        print(f"  ✓ LLM Response: {response.content}")
        print("\n[SUCCESS] LangSmith integration appears to be working!")
        print("\nTo view the trace:")
        print("  1. Go to https://smith.langchain.com/")
        print("  2. Navigate to project: " + os.getenv("LANGSMITH_PROJECT", "llm-tool-hub"))
        print("  3. You should see the trace for this LLM call")
    except Exception as e:
        print(f"  ✗ Error during LLM call: {e}")
        print("\n[FAILED] There was an issue with the LLM call.")
        return False
    
    return True


if __name__ == "__main__":
    success = test_langsmith_integration()
    sys.exit(0 if success else 1)
