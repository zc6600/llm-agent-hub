import os
import sys
import shutil
from typing import Any, List, Optional
from langchain_core.messages import AIMessage

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.memory.fs_mem import FMemClient

class MockLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages: List[Any]) -> AIMessage:
        self.calls.append(messages)
        # Return a mock JSON response for memory writer
        return AIMessage(content='```json\n[{"tool_name": "write_memory_direct", "arguments": {"dir": "knowledge", "introduction": "test_fact", "summary": "The sky is blue."}}]\n```')

    def bind_tools(self, tools):
        return self

def test_fmem_client():
    print("Testing FMemClient...")
    task_dir = "./test_fmem_client_dir"
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
    
    try:
        # 1. Initialize
        llm = MockLLM()
        client = FMemClient(task_dir=task_dir, llm=llm)
        print("✓ Initialization successful")
        
        # 2. Check structure
        assert os.path.exists(os.path.join(task_dir, "knowledge"))
        print("✓ Directory structure created")
        
        # 3. Test get_context
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        context = client.get_context(recent_messages=messages)
        assert "=== RECENT CONVERSATION" in context
        assert "- user: Hello" in context
        print("✓ get_context successful")
        
        # 4. Test save_context
        client.save_context(messages)
        # Check if file was created (MockLLM returns a write action)
        expected_file = os.path.join(task_dir, "knowledge", "test_fact.txt")
        assert os.path.exists(expected_file)
        with open(expected_file, "r") as f:
            content = f.read()
            assert "The sky is blue" in content
        print("✓ save_context successful")
        
    finally:
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)
            print("✓ Cleanup successful")

if __name__ == "__main__":
    test_fmem_client()
