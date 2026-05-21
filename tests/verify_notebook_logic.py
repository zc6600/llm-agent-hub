import os
import sys
import shutil

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memory.fs_mem import FMemClient
from src.agent_blocks_hub.base_agent import create_base_agent
from src.llm_provider import get_llm
from langchain_core.messages import HumanMessage

def test_notebook_logic():
    print("Testing notebook logic...")
    
    # Setup
    task_dir = "./demo_memory_test"
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
        
    try:
        # Initialize LLM using the provider
        # We mock the API key if not present to avoid errors during test if env not set
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "sk-mock-key"
            
        llm = get_llm()
        
        # Initialize Memory Client
        client = FMemClient(task_dir=task_dir, llm=llm)
        print(f"Memory initialized at: {task_dir}")
        
        # Create Agent
        print("Creating agent...")
        agent = create_base_agent(llm=llm)
        print("Agent created.")
        
        # Define a conversation state
        state = {
            "messages": [
                HumanMessage(content="My name is Frank and I am working on an AI project.")
            ]
        }
        
        # Invoke agent WITH memory
        print("Invoking agent...")
        response = agent.invoke(state, memory=client)
        print("Agent Response:", response.content)
        
        # Save Context
        state["messages"].append(response)
        
        # Mock memory writer response
        if client._mw:
            client._mw.write_memory = lambda: []
            
        client.save_context(state["messages"])
        print("Context saved to memory.")
        
        # Verify Memory Retrieval
        new_state = {
            "messages": [
                HumanMessage(content="What is my name?")
            ]
        }
        
        response_2 = agent.invoke(new_state, memory=client)
        print("Agent Response 2:", response_2.content)
        
        print("✓ Notebook logic verified")
        
    finally:
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)

if __name__ == "__main__":
    test_notebook_logic()
