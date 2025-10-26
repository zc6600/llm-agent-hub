# tests/filesystem_tool/test_create_file.py

import pytest
from pathlib import Path
from llm_tool_hub.filesystem_tool.create_file_tool import CreateFileTool

# Example content for testing file creation
TEST_CONTENT = "def main():\n    print('Hello LLM')\n\nif __name__ == '__main__':\n    main()\n"
EXPECTED_LINE_COUNT = 5

@pytest.fixture
def create_tool(tmp_path: Path) -> CreateFileTool:
    """Fixture to initialize CreateFileTool with a temporary root path."""
    return CreateFileTool(root_path=tmp_path)

# ==============================================================================
# A. Successful File Creation Tests (Happy Path)
# ==============================================================================

def test_create_file_success_with_content_return(create_tool: CreateFileTool, tmp_path: Path):
    """A.1: Tests successful creation of a new file with content, defaulting to return_content=True."""
    file_path = "app/new_file.py"
    
    # BaseFileSystemTool checks that the parent directory exists, so we must create it.
    (tmp_path / "app").mkdir(parents=True)
    
    result = create_tool.run(file_path=file_path, content=TEST_CONTENT)
    
    target_path = tmp_path / file_path
    
    # 1. Check returned message structure and content
    assert result.startswith("SUCCESS:")
    assert f"File '{file_path}' successfully created" in result
    assert f"with {EXPECTED_LINE_COUNT} lines of initial content." in result
    assert "Content with line numbers" in result
    
    # 2. Verify file was physically created
    assert target_path.is_file()
    
    # 3. Verify file content matches the input
    assert target_path.read_text(encoding='utf-8') == TEST_CONTENT

def test_create_file_nested_success_no_content_return(create_tool: CreateFileTool, tmp_path: Path):
    """A.2: Tests successful creation of a file, explicitly setting return_content=False (token saving mode)."""
    file_path = "config/settings/prod.json"
    
    # Ensure parent directory exists for safety checks to pass
    (tmp_path / "config/settings").mkdir(parents=True)
    
    result = create_tool.run(file_path=file_path, content='{"env": "prod"}', return_content=False)
    target_path = tmp_path / file_path
    
    # 1. Check returned message structure
    assert result.startswith("SUCCESS:")
    # Must NOT contain the content block
    assert "Content with line numbers" not in result
    
    # 2. Verify file was physically created
    assert target_path.is_file()
    
    # 3. Verify file content is correct
    assert target_path.read_text() == '{"env": "prod"}'

# ==============================================================================
# B. Safety and Error Checks (Testing BaseFileSystemTool Logic)
# ==============================================================================

def test_create_file_already_exists_fails(create_tool: CreateFileTool, tmp_path: Path):
    """B.1: CRITICAL: Attempts to create a file that already exists (must fail to prevent overwrite)."""
    file_path = "existing.txt"
    # Pre-create the file with initial content
    (tmp_path / file_path).write_text("Original content.")
    
    result = create_tool.run(file_path=file_path, content="New content that should not be written.")
    
    # 1. Check for expected error message
    assert result.startswith("ERROR:")
    # Checks for the error message thrown by BaseFileSystemTool (must_exist=False check)
    assert "Tool execution failed for" in result
    assert "Reason: File already exists at path:" in result
    
    # 2. Verify original file content was NOT overwritten
    assert (tmp_path / file_path).read_text() == "Original content."

def test_create_file_empty_path_fails(create_tool: CreateFileTool):
    """B.2: Attempts to create a file with an empty path."""
    result = create_tool.run(file_path="", content="dummy")
    
    assert result.startswith("ERROR:")
    # Checks for ValueError from BaseFileSystemTool
    assert "File path is empty" in result

def test_create_file_path_traversal_fails(tmp_path: Path):
    """B.3: Attempts to create a file outside the root path (../)."""
    # Set root_path to a subdirectory for a clear traversal test
    (tmp_path / "subdir").mkdir(parents=True)

    tool = CreateFileTool(root_path=tmp_path / "subdir")
    
    result = tool.run(file_path="../evil_file.txt", content="hack")
    
    # Checks for ValueError from BaseFileSystemTool (Path Traversal Check)
    assert result.startswith("ERROR:")
    assert "Access Denied: File path" in result
    assert "must be inside the configured root" in result
    
    # Verify the forbidden file was NOT created
    assert not (tmp_path / "evil_file.txt").exists()

def test_create_file_parent_directory_not_exist_fails(create_tool: CreateFileTool):
    """B.4: Attempts to create a file where the parent directory does not exist (must fail)."""
    file_path = "non_existent_dir/test.txt"
    
    # Checks for FileNotFoundError thrown by BaseFileSystemTool (Check 4)
    result = create_tool.run(file_path=file_path, content="test")
    
    assert result.startswith("ERROR:")
    assert "Parent directory does not exist" in result
    
    # Verify the file was NOT created
    assert not (create_tool.root_path / file_path).exists()

def test_create_file_is_directory_fails(create_tool: CreateFileTool, tmp_path: Path):
    """B.5: Attempts to create a file at a path that is an existing directory."""
    dir_path = "test_dir"
    (tmp_path / dir_path).mkdir()
    
    result = create_tool.run(file_path=dir_path, content="Should fail.")
    
    # Checks for ValueError from BaseFileSystemTool (Check 3)
    assert result.startswith("ERROR:")
    assert "File already exists at path:" in result
    assert "Cannot overwrite" in result

# ==============================================================================
# C. Function Calling Metadata Test
# ==============================================================================

def test_create_file_tool_metadata(create_tool: CreateFileTool):
    """C.1: Tests that the tool's metadata (for Function Calling) is correctly defined."""
    
    # 1. Check Tool Name and Description
    assert create_tool.name == "create_file"
    
    # 2. Use check for the core purpose of the tool's description
    description = create_tool.description
    assert "NEW file" in description 
    assert "fail if the file already exists" in description
    
    # 3. Check the messaging about modification/deletion to match your updated string
    assert "To modify existing files, try to use 'modify_tool' or delete the original tool" in description

    # 4. Check Parameters Schema (The essential part for LLM integration)
    parameters = create_tool.parameters
    
    # Verify top-level structure
    assert parameters["type"] == "object"
    assert "file_path" in parameters["required"]
    assert "content" in parameters["required"]
    
    # Verify 'file_path' property
    file_path_prop = parameters["properties"]["file_path"]
    assert file_path_prop["type"] == "string"
    assert "relative path to the New file" in file_path_prop["description"]
    
    # Verify 'content' property
    content_prop = parameters["properties"]["content"]
    assert content_prop["type"] == "string"
    assert "The content to write into the new file." in content_prop["description"]

    # Verify 'return_content' property (Optional boolean)
    return_content_prop = parameters["properties"]["return_content"]
    assert return_content_prop["type"] == "boolean"
    assert return_content_prop["default"] is True
    assert "If true, returns the content written to the file in the response. Defaults to false." in return_content_prop["description"]