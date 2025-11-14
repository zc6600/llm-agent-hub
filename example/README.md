# 📚 llm-tool-hub Examples

This directory contains comprehensive examples and tutorials demonstrating how to use each component in **llm-tool-hub**. The structure mirrors the main `src/` directory organization.

## 📁 Directory Structure

```bash
example/
├── agent_blocks_hub/              # Agent building blocks examples
│   ├── deep_diver/                # DeepDiver agent examples
│   │   ├── 01_basic_usage.py
│   │   ├── 01_ideation_agent.py
│   │   ├── 02_custom_tools.py
│   │   ├── 02_paper_planner.py
│   │   ├── 03_llm_controlled_iteration.py
│   │   ├── 04_flexible_workflow.py
│   │   ├── 06_parallel_research.py
│   │   └── test_*.py
│   └── parallel_react_agent/      # ParallelReactAgent examples
│       ├── 01_basic_usage.py
│       ├── 02_ai_safety_research.py
│       ├── 03_competitive_analysis.py
│       ├── 04_product_evaluation.py
│       └── EXAMPLES_GUIDE.py
├── llm_tool_hub/                  # Core tools and integrations
│   ├── filesystem_tool/           # File manipulation tools
│   │   ├── 01_create_file_tool.ipynb
│   │   ├── 02_read_file_tool.ipynb
│   │   └── 03_modify_file_tool.ipynb
│   ├── shell_tool/                # Shell execution examples
│   │   └── 01_shell_tool.ipynb
│   ├── integrations/              # Integration examples
│   │   ├── 01_langchain_integration.ipynb
│   │   └── mcp/                   # Model Context Protocol examples
│   │       ├── 01_basic_stdio_server.py
│   │       ├── 02_simple_test.py
│   │       └── 03_mcp_client.py
│   └── function_adapter/          # Function adapter examples
│       └── 01_basic_usage.py
└── README.md
```

## 🚀 Quick Start

### Agent Building Blocks

1. **DeepDiver Agent** - Multi-stage research and reasoning
   - [01_basic_usage.py](agent_blocks_hub/deep_diver/01_basic_usage.py) - Get started with DeepDiver
   - [01_ideation_agent.py](agent_blocks_hub/deep_diver/01_ideation_agent.py) - Ideation workflows
   - [02_custom_tools.py](agent_blocks_hub/deep_diver/02_custom_tools.py) - Add custom tools

2. **ParallelReactAgent** - Concurrent reasoning and research
   - [01_basic_usage.py](agent_blocks_hub/parallel_react_agent/01_basic_usage.py) - Basic usage
   - [02_ai_safety_research.py](agent_blocks_hub/parallel_react_agent/02_ai_safety_research.py) - Research example
   - [03_competitive_analysis.py](agent_blocks_hub/parallel_react_agent/03_competitive_analysis.py) - Competitive analysis

### Core Tools

1. **Filesystem Tools** - File manipulation:
   - [01_create_file_tool.ipynb](llm_tool_hub/filesystem_tool/01_create_file_tool.ipynb) - Create files
   - [02_read_file_tool.ipynb](llm_tool_hub/filesystem_tool/02_read_file_tool.ipynb) - Read file content
   - [03_modify_file_tool.ipynb](llm_tool_hub/filesystem_tool/03_modify_file_tool.ipynb) - Edit files

2. **Shell Tools** - Command execution:
   - [01_shell_tool.ipynb](llm_tool_hub/shell_tool/01_shell_tool.ipynb) - Execute shell commands

### Integrations

1. **LangChain** - AI agent workflows:
   - [01_langchain_integration.ipynb](llm_tool_hub/integrations/01_langchain_integration.ipynb) - Use tools with LangChain

2. **Model Context Protocol (MCP)** - MCP server examples:
   - [01_basic_stdio_server.py](llm_tool_hub/integrations/mcp/01_basic_stdio_server.py) - Basic MCP server
   - [03_mcp_client.py](llm_tool_hub/integrations/mcp/03_mcp_client.py) - MCP client example

### Function Adapter

- [01_basic_usage.py](llm_tool_hub/function_adapter/01_basic_usage.py) - Function to tool adapter

## 📖 Learning Path

### Beginner

Start with basic tool examples:

1. Filesystem Tools (Create, Read)
2. Shell Tools
3. Basic agent examples

### Intermediate

Learn advanced agent features:

1. Custom tools with DeepDiver
2. Parallel research workflows
3. Tool integration patterns

### Advanced

Master complex workflows:

1. Multi-agent orchestration
2. LangChain integrations
3. MCP server development

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
