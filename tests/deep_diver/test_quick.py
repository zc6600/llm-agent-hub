"""
Quick test script for Deep Diver agent.

Run this to verify the implementation works.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Mock implementations for testing without actual LLM
class MockLLM:
    """Mock LLM for testing."""
    
    def invoke(self, messages):
        """Return mock response."""
        from types import SimpleNamespace
        return SimpleNamespace(content="Mock LLM response: This is a test answer.")
    
    def bind_tools(self, tools):
        """Return self for testing."""
        return self


class MockTool:
    """Mock tool for testing."""
    
    def __init__(self, name):
        self.name = name
    
    def invoke(self, args):
        """Return mock result."""
        return f"Mock result from {self.name}: {args}"


def test_deep_diver():
    """Test the Deep Diver agent."""
    
    from agent_blocks_hub.deep_diver import create_deepdiver_agent
    
    print("=" * 80)
    print("Testing Deep Diver Agent")
    print("=" * 80)
    
    # Create mock LLM and tools
    llm = MockLLM()
    tools = [MockTool("mock_search")]
    
    # Create agent
    print("\n1. Creating agent...")
    try:
        agent = create_deepdiver_agent(
            llm=llm,
            tools=tools,
            system_prompt="You are a research agent.",
            max_iterations=2
        )
        print("✓ Agent created successfully!")
    except Exception as e:
        print(f"✗ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test invocation
    print("\n2. Testing agent invocation...")
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": "What is LangGraph?"}]
        })
        print("✓ Agent invoked successfully!")
        
        # Print results
        print("\n3. Results:")
        print(f"   Original Question: {result.get('original_question', 'N/A')}")
        print(f"   Decomposed Problems: {len(result.get('decomposed_problems', []))} problems")
        print(f"   Hypotheses: {len(result.get('hypotheses', []))} hypotheses")
        print(f"   Experience Pool: {len(result.get('experience_pool', []))} entries")
        print(f"   Final Answer: {result.get('final_answer', 'N/A')[:100]}...")
        
    except Exception as e:
        print(f"✗ Failed to invoke agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_deep_diver()
