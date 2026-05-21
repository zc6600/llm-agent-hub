"""
Test to verify that each parallel agent only sees its own query.

This test ensures that when running parallel_tool_agent with multiple queries,
each agent instance only receives its specific query and not the queries of other agents.
"""

import pytest
from typing import List, Any
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class MockLLMResponse:
    """Mock LLM response for testing."""
    def __init__(self, content: str, query_received: str):
        self.content = content
        self.query_received = query_received
        self.tool_calls = None


class QueryTrackerLLM:
    """Mock LLM that tracks which queries it receives."""
    
    def __init__(self):
        self.queries_received = []
        self.call_count = 0
    
    def bind_tools(self, tools: List[Any]):
        """Mock bind_tools method."""
        return self
    
    def invoke(self, messages: List[BaseMessage]) -> MockLLMResponse:
        """Track which query was received in this invocation."""
        self.call_count += 1
        
        # Extract the human message (query)
        query = None
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                query = msg.content
                break
        
        self.queries_received.append({
            'call_number': self.call_count,
            'query': query,
            'all_messages': [msg.content for msg in messages]
        })
        
        return MockLLMResponse(
            content=f"Response to: {query}",
            query_received=query
        )


class MockToolInput(BaseModel):
    """Mock tool input schema."""
    query: str = Field(description="The query string")


class MockTool(BaseTool):
    """Mock tool for testing."""
    name: str = "mock_search"
    description: str = "A mock search tool"
    args_schema: type[BaseModel] = MockToolInput
    
    def _run(self, query: str) -> str:
        return f"Mock result for: {query}"


def test_parallel_agents_see_only_own_query():
    """
    Test that each parallel agent only sees its own query.
    
    This test verifies that when processing multiple queries in parallel,
    each agent instance only receives the specific query it's supposed to handle,
    and not the queries assigned to other agents.
    """
    from agent_blocks_hub.parallel_tool_agent import create_parallel_tool_agent
    
    # Create a query tracker LLM
    tracker_llm = QueryTrackerLLM()
    
    # Create mock tool
    mock_tool = MockTool()
    
    # Create agent
    agent = create_parallel_tool_agent(
        llm=tracker_llm,
        tools=[mock_tool],
        system_prompt="You are a helpful assistant.",
        verbose=False,
        enable_summarization=False,
    )
    
    # Define test queries
    queries = [
        "What is machine learning?",
        "Explain quantum computing",
        "Describe blockchain technology"
    ]
    
    # Run the agent
    result = agent.invoke({
        "parallel_agent_message": queries,
        "verbose": False,
    })
    
    # Verify that each query was processed
    assert len(tracker_llm.queries_received) == len(queries), \
        f"Expected {len(queries)} LLM calls, got {len(tracker_llm.queries_received)}"
    
    # Verify that each agent only saw its own query
    for i, call_info in enumerate(tracker_llm.queries_received):
        received_query = call_info['query']
        expected_query = queries[i]
        
        # The received query should match the expected query for this index
        assert received_query == expected_query, \
            f"Agent {i} received wrong query. Expected: '{expected_query}', Got: '{received_query}'"
        
        # Verify that the agent didn't see other queries in its messages
        all_message_content = ' '.join(call_info['all_messages'])
        other_queries = [q for j, q in enumerate(queries) if j != i]
        
        for other_query in other_queries:
            assert other_query not in all_message_content, \
                f"Agent {i} should not see query from agent {queries.index(other_query)}: '{other_query}'"
    
    print("\n✓ Test passed: Each parallel agent only saw its own query")
    print(f"✓ Processed {len(queries)} queries independently")
    print("\nQuery isolation verified:")
    for i, call_info in enumerate(tracker_llm.queries_received):
        print(f"  Agent {i} saw only: '{call_info['query']}'")


def test_parallel_agents_system_prompt_same():
    """
    Test that all parallel agents receive the same system prompt.
    
    The system prompt should be shared across all agents, but each should
    still only see their own specific query.
    """
    from agent_blocks_hub.parallel_tool_agent import create_parallel_tool_agent
    
    tracker_llm = QueryTrackerLLM()
    mock_tool = MockTool()
    
    system_prompt = "You are a specialized research assistant."
    
    agent = create_parallel_tool_agent(
        llm=tracker_llm,
        tools=[mock_tool],
        system_prompt=system_prompt,
        verbose=False,
        enable_summarization=False,
    )
    
    queries = ["Query A", "Query B", "Query C"]
    
    result = agent.invoke({
        "parallel_agent_message": queries,
        "verbose": False,
    })
    
    # Verify that all agents received the system prompt
    for i, call_info in enumerate(tracker_llm.queries_received):
        messages_content = call_info['all_messages']
        
        # System prompt should be in the messages
        assert system_prompt in messages_content[0], \
            f"Agent {i} didn't receive system prompt"
        
        # But should NOT contain other queries
        other_queries = [q for j, q in enumerate(queries) if j != i]
        for other_query in other_queries:
            assert other_query not in ' '.join(messages_content), \
                f"Agent {i} should not see other query: '{other_query}'"
    
    print("\n✓ Test passed: All agents received system prompt but only their own query")


if __name__ == "__main__":
    test_parallel_agents_see_only_own_query()
    test_parallel_agents_system_prompt_same()
