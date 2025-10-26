import pytest
from pathlib import Path
import re

from llm_tool_hub.filesystem_tool.replace_content_tool import ReplaceContentTool
from llm_tool_hub.filesystem_tool.base_filesystem_tool import CONTEXT_WINDOW_SIZE 


# --- Fixtures and Setup ---

# 初始文件内容 (10 行内容)
INITIAL_CONTENT_LINES = [
    "def function_a():",        # L1
    "    # Implementation of A", # L2
    "    pass",                  # L3
    "",                          # L4 (Empty line)
    "def function_b():",        # L5 (Target 1: Single line)
    "    # Implementation of B", # L6 (Target 2: Multi-line block start)
    "    return True",           # L7 (Target 2: Multi-line block end)
    "def function_c():",        # L8
    "    print('start')",        # L9 (Target 3: Duplicate candidate)
    "    return False"           # L10
]
INITIAL_CONTENT_STR = "\n".join(INITIAL_CONTENT_LINES) + "\n"

# 一个用于测试重复匹配的文件 (L9, L11, L13 是重复内容)
DUPLICATE_CONTENT_LINES = [
    "def setup_db():",           # L1
    "    connect()",             # L2
    "    create_table()",        # L3
    "",                          # L4
    "def teardown_db():",        # L5
    "    drop_table()",          # L6
    "    close()",               # L7
    "    ",                      # L8
    "    log.info('Finished')",  # L9 (Match 1)
    "    return True",           # L10
    "    log.info('Finished')",  # L11 (Match 2)
    "    return True",           # L12
    "    log.info('Finished')",  # L13 (Match 3)
    "    return True",           # L14
]
DUPLICATE_CONTENT_STR = "\n".join(DUPLICATE_CONTENT_LINES) + "\n"


@pytest.fixture
def replace_tool(tmp_path: Path) -> ReplaceContentTool:
    """Fixture to initialize ReplaceContentTool with a temporary root path."""
    return ReplaceContentTool(root_path=tmp_path)

@pytest.fixture
def setup_file(tmp_path: Path) -> Path:
    """Fixture to create the standard file for replacement tests."""
    file_path = tmp_path / "target_file.py"
    file_path.write_text(INITIAL_CONTENT_STR, encoding='utf-8')
    return file_path

@pytest.fixture
def setup_duplicate_file(tmp_path: Path) -> Path:
    """Fixture to create a file with duplicate content."""
    file_path = tmp_path / "duplicate_test.py"
    file_path.write_text(DUPLICATE_CONTENT_STR, encoding='utf-8')
    return file_path


# --- Helper functions for verification ---

def check_file_content(file_path: Path, expected_lines: list[str]):
    """Checks the final file content line by line."""
    # Note: read_text().strip().splitlines() 确保去除尾随换行符后的行列表
    actual_content = file_path.read_text().strip().splitlines()
    
    assert len(actual_content) == len(expected_lines), (
        f"File line count mismatch. Expected {len(expected_lines)}, got {len(actual_content)}"
    )
    for i, (expected, actual) in enumerate(zip(expected_lines, actual_content)):
        assert actual == expected, f"Line {i+1} content mismatch. Expected: '{expected}', Got: '{actual}'"

def check_sync_window(result: str, modified_start: int, modified_end: int, new_total_lines: int):
    """
    Checks the structure and line numbers of the synchronized window.
    """
    assert "SYNCHRONIZED CONTENT WINDOW" in result
    assert f"Total lines in file now: {new_total_lines}." in result
    
    # 提取同步窗口的行号范围
    match = re.search(r"Lines (\d+) to (\d+)", result)
    assert match is not None, "Could not find synchronized line range in result."
    
    read_start = int(match.group(1))
    read_end = int(match.group(2))
    
    # 验证窗口范围是否合理
    # 窗口应该从 (修改起始行 - 上下文) 开始
    assert read_start == max(1, modified_start - CONTEXT_WINDOW_SIZE)
    
    # 窗口应该到 (修改结束行 + 上下文) 结束，且不超过文件总行数
    expected_window_end = min(new_total_lines, modified_end + CONTEXT_WINDOW_SIZE + 1)
    assert read_end == expected_window_end, (
        f"Synchronized window end line mismatch. Expected {expected_window_end}, got {read_end}"
    )


# ==============================================================================
# A. Success Cases (Exactly 1 Match)
# ==============================================================================

def test_replace_single_line_success(replace_tool: ReplaceContentTool, setup_file: Path):
    """A.1: Tests successful replacement of a single, unique line (L5 -> L5). Total lines 10."""
    
    old_content = "def function_b():" # L5
    new_content = "def new_function_b(arg):"
    
    result = replace_tool.run(str(setup_file.name), old_content=old_content, new_content=new_content)
    
    assert result.startswith("SUCCESS:")
    assert "Replaced 1 match" in result
    assert "lines 5-5" in result
    assert "Net change: +0 lines" in result
    
    # 验证文件内容
    expected_lines = INITIAL_CONTENT_LINES.copy()
    expected_lines[4] = new_content # L5 is index 4
    check_file_content(setup_file, expected_lines)
    
    # 验证同步窗口
    check_sync_window(result, modified_start=5, modified_end=5, new_total_lines=10)


def test_replace_multi_line_success(replace_tool: ReplaceContentTool, setup_file: Path):
    """A.2: Tests successful replacement of a multi-line block (L6-L7 -> L6-L7). Total lines 10."""
    
    old_content = "    # Implementation of B\n    return True" # L6 to L7 (2 lines)
    new_content = "    return False\n    # New check" # (2 lines)
    
    result = replace_tool.run(str(setup_file.name), old_content=old_content, new_content=new_content)
    
    assert result.startswith("SUCCESS:")
    assert "replaced 2 lines" in result
    assert "lines 6-7" in result
    assert "Net change: +0 lines" in result
    
    # 验证文件内容
    expected_lines = INITIAL_CONTENT_LINES.copy()
    expected_lines[5:7] = new_content.splitlines() # 替换 L6, L7 (index 5, 6)
    check_file_content(setup_file, expected_lines)
    
    # 验证同步窗口
    check_sync_window(result, modified_start=6, modified_end=7, new_total_lines=10)


def test_replace_and_change_line_count_success(replace_tool: ReplaceContentTool, setup_file: Path):
    """A.3: Tests replacement where line count changes (L8-L10 replaced with 1 line). Total lines 8."""
    
    old_content = "def function_c():\n    print('start')\n    return False" # L8 to L10 (3 lines)
    new_content = "def function_c_simple(): return True" # (1 line)
    
    result = replace_tool.run(str(setup_file.name), old_content=old_content, new_content=new_content)
    
    assert result.startswith("SUCCESS:")
    assert "lines 8-10" in result
    assert "replaced 3 lines" in result
    assert "with 1 new lines" in result
    assert "Net change: -2 lines" in result
    
    # 验证文件内容
    expected_lines = INITIAL_CONTENT_LINES[0:7].copy() # L1-L7
    expected_lines.append(new_content) # New L8 content
    
    check_file_content(setup_file, expected_lines)
    # 新的修改结束行是 L8
    check_sync_window(result, modified_start=8, modified_end=8, new_total_lines=8)


def test_delete_content_success(replace_tool: ReplaceContentTool, setup_file: Path):
    """A.4: Tests successful deletion of content (L6-L7 replaced with empty content). Total lines 8."""
    
    old_content = "    # Implementation of B\n    return True" # L6 to L7
    new_content = "" # Deletion
    
    result = replace_tool.run(str(setup_file.name), old_content=old_content, new_content=new_content)
    
    assert result.startswith("SUCCESS:")
    assert "replaced 2 lines" in result
    assert "with 0 new lines" in result
    assert "Net change: -2 lines" in result
    
    # 验证文件内容
    expected_lines = INITIAL_CONTENT_LINES[0:5].copy() + INITIAL_CONTENT_LINES[7:]
    
    check_file_content(setup_file, expected_lines)
    # L6, L7 被删除，L8 (def function_c()) 成为新的 L6
    # 窗口应该围绕新的 L6 (即原来的 L8) 展开。
    check_sync_window(result, modified_start=6, modified_end=5, new_total_lines=8)


# ==============================================================================
# B. Failure Cases (0 or >1 Matches)
# ==============================================================================

def test_replace_zero_match_fails(replace_tool: ReplaceContentTool, setup_file: Path):
    """B.1: Tests failure when no matching content is found."""
    
    old_content = "def non_existent_function():"
    new_content = "def dummy():"
    
    result = replace_tool.run(str(setup_file.name), old_content=old_content, new_content=new_content)
    
    assert result.startswith("ERROR:")
    assert "Found 0 match(es) (Expected exactly 1)" in result
    
    # 验证文件内容未改变
    check_file_content(setup_file, INITIAL_CONTENT_LINES)


def test_replace_multiple_match_fails(replace_tool: ReplaceContentTool, setup_duplicate_file: Path):
    """B.2: Tests failure when multiple matching content blocks are found."""
    
    # This content appears on L9, L11, L13
    old_content = "    log.info('Finished')" 
    new_content = "    log.info('DONE')"
    
    result = replace_tool.run(str(setup_duplicate_file.name), old_content=old_content, new_content=new_content)
    
    assert result.startswith("ERROR:")
    assert "Found 3 match(es) (Expected exactly 1)" in result
    assert "Multiple matches found at: lines 9-9, lines 11-11, lines 13-13" in result
    
    # 验证文件内容未改变
    check_file_content(setup_duplicate_file, DUPLICATE_CONTENT_LINES)

# ==============================================================================
# C. Edge Cases (Path/Empty Content)
# ==============================================================================

def test_replace_non_existent_file_fails(replace_tool: ReplaceContentTool):
    """C.1: Tests failure when file does not exist."""
    
    result = replace_tool.run("non_existent.py", old_content="a", new_content="b")
    
    assert result.startswith("ERROR:")
    assert "File not found" in result

def test_replace_empty_old_content_fails(replace_tool: ReplaceContentTool, setup_file: Path):
    """C.2: Tests failure when old_content is empty (cannot search for nothing)."""
    
    result = replace_tool.run(str(setup_file.name), old_content="", new_content="something")
    
    assert result.startswith("ERROR:")
    assert "'old_content' cannot be empty" in result
    
    # 验证文件内容未改变
    check_file_content(setup_file, INITIAL_CONTENT_LINES)

def test_replace_content_tool_metadata(replace_tool: ReplaceContentTool):
    """C.3: Tests that the tool's metadata is correctly defined."""
    
    assert replace_tool.name == "replace_content"
    assert "Replaces ONE AND ONLY ONE exact block" in replace_tool.description
    
    # 检查关键参数是否存在
    parameters = replace_tool.parameters
    assert parameters["type"] == "object"
    assert "file_path" in parameters["required"]
    assert "old_content" in parameters["required"]
    assert "new_content" in parameters["required"]