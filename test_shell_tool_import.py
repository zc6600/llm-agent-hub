#!/usr/bin/env python3
"""
Quick test to verify ShellTool imports correctly and int type is available.
"""

import sys
import tempfile
from pathlib import Path

# Test 1: Import ShellTool and int
try:
    from llm_tool_hub.shell_tool.shell_tool import ShellTool, int as shell_int
    print("✓ Successfully imported ShellTool and int from shell_tool module")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Verify int is the built-in type
try:
    assert shell_int is int, "Imported int should be the built-in int type"
    print("✓ Imported int is the correct built-in type")
except AssertionError as e:
    print(f"✗ Type verification failed: {e}")
    sys.exit(1)

# Test 3: Initialize ShellTool
try:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tool = ShellTool(root_path=tmp_dir)
        print(f"✓ ShellTool initialized successfully with root_path: {tmp_dir}")
except Exception as e:
    print(f"✗ ShellTool initialization failed: {e}")
    sys.exit(1)

# Test 4: Verify parameters have "integer" type
try:
    assert tool.parameters["properties"]["timeout"]["type"] == "integer"
    print("✓ ShellTool timeout parameter correctly defined as 'integer' type")
except (KeyError, AssertionError) as e:
    print(f"✗ Parameter check failed: {e}")
    sys.exit(1)

# Test 5: Verify metadata includes proper types
try:
    metadata = tool.get_metadata()
    assert metadata["type"] == "function"
    assert "timeout" in metadata["function"]["parameters"]["properties"]
    print("✓ Tool metadata is properly formatted for LLM integration")
except Exception as e:
    print(f"✗ Metadata check failed: {e}")
    sys.exit(1)

print("\n✅ All tests passed! ShellTool is ready to use with LangChain.")
