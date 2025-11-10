# 📚 llm-tool-hub Examples

This directory contains comprehensive Jupyter notebooks demonstrating how to use each tool in **llm-tool-hub**.

## 📁 Directory Structure

```bash
example/
├── filesystem_tool/
│   ├── 01_create_file_tool.ipynb       # Create files with content
│   ├── 02_read_file_tool.ipynb         # Read files with line ranges
│   └── 03_modify_file_tool.ipynb       # Modify file content
├── shell_tool/
│   └── 01_shell_tool.ipynb             # Execute shell commands
└── integrations/
    └── 01_langchain_integration.ipynb   # Use tools with LangChain
```

## 🚀 Quick Start

1. **Filesystem Tools** - For file manipulation:
   - [CreateFileTool](filesystem_tool/01_create_file_tool.ipynb) - Create new files
   - [ReadFileTool](filesystem_tool/02_read_file_tool.ipynb) - Read file content
   - [ModifyFileTool](filesystem_tool/03_modify_file_tool.ipynb) - Edit files

2. **Shell Tools** - For command execution:
   - [ShellTool](shell_tool/01_shell_tool.ipynb) - Run shell commands

3. **Integrations** - For AI agent workflows:
   - [LangChain Integration](integrations/01_langchain_integration.ipynb) - Use with LangChain agents

## 📖 Learning Path

### Beginner

Start with individual tool notebooks to understand basic functionality:

1. CreateFileTool
2. ReadFileTool
3. ShellTool

### Intermediate

Learn file modification and combining tools:

1. ModifyFileTool
2. Run multiple tools in sequence

### Advanced

Build AI agent workflows:

1. LangChain Integration
2. Custom agent patterns

## 🎯 Tool Overview

### CreateFileTool

**Purpose**: Create new files with content

- Create text files, code files, configuration files
- Auto-numbered output for reference
- Example: `example/filesystem_tool/01_create_file_tool.ipynb`

### ReadFileTool

**Purpose**: Read files with optional line range specification

- Read entire files
- Read specific line ranges
- Efficient for large files
- Example: `example/filesystem_tool/02_read_file_tool.ipynb`

### ModifyFileTool

**Purpose**: Modify existing files

- Replace lines
- Delete lines
- Insert new lines
- Example: `example/filesystem_tool/03_modify_file_tool.ipynb`

### ShellTool

**Purpose**: Execute shell commands

- Run arbitrary commands
- Capture output
- Execute scripts
- Example: `example/shell_tool/01_shell_tool.ipynb`

### LangChain Integration

**Purpose**: Use tools with LangChain AI agents

- Convert tools to LangChain StructuredTools
- Build agent workflows
- Chain multiple tools
- Example: `example/integrations/01_langchain_integration.ipynb`

## 🔧 Running the Notebooks

### Prerequisites

```bash
pip install jupyter ipython
pip install -e .  # Install llm-tool-hub in development mode
```

### Launch Jupyter

```bash
jupyter notebook
```

### Run Individual Notebooks

```bash
# Open specific notebook
jupyter notebook example/filesystem_tool/01_create_file_tool.ipynb
```

## 📝 Key Concepts

### 1. File Path References

- All paths are **relative to the tool's `root_path`**
- Example: `file_path="docs/readme.md"` creates `{root_path}/docs/readme.md`

### 2. Line Numbers

- All tools return output with **line numbers**
- Line numbers are **1-indexed** (start from 1, not 0)
- Used for precise modifications with ModifyFileTool

### 3. Return Format

All tools return **formatted strings**:

```
1:line content
2:more content
3:...
```

### 4. Error Handling

- Check if response starts with "ERROR:" or "UNEXPECTED ERROR:"
- All operations are atomic (succeed or fail completely)

## 💡 Common Patterns

### Pattern 1: Create and Verify

```python
# Create file
create_tool.run(file_path="app.py", content=code)

# Read it back
read_tool.run(file_path="app.py")
```

### Pattern 2: Modify and Verify

```python
# Read original
original = read_tool.run(file_path="app.py")

# Modify specific lines
modify_tool.run(file_path="app.py", start_line=5, end_line=5, new_content="...")

# Verify changes
updated = read_tool.run(file_path="app.py", start_line=5, end_line=10)
```

### Pattern 3: Execute and Parse

```python
# Run command
result = shell_tool.run(command="python script.py")

# Parse output
if "ERROR" not in result:
    # Process success output
else:
    # Handle error
```

### Pattern 4: Agent Workflow

```python
# Convert to LangChain tools
tools = [
    LangchainToolAdapter.to_langchain_structured_tool(create_tool),
    LangchainToolAdapter.to_langchain_structured_tool(read_tool),
]

# Use with agent
agent = initialize_agent(tools, llm)
result = agent.run("Create and verify a hello.py file")
```

## 📚 Additional Resources

- [Main README](../README.md) - Project overview
- [Source Code](../src/llm_tool_hub/) - Tool implementations
- [Tests](../tests/) - More usage examples
- [Documentation](../docs/) - Detailed API documentation

## 🤝 Contributing

Found a useful pattern or example? Consider adding it:

1. Create a new notebook
2. Add it to the appropriate directory
3. Update this README
4. Submit a pull request

## ❓ Questions?

- Check the [Main README](../README.md)
- Review tool-specific notebooks
- Check source code examples in [tests](../tests/)
- Open an issue on GitHub

---

Happy learning! 🚀
