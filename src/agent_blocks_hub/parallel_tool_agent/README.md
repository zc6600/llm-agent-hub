# Parallel Tool Agent - 轻量级并行工具执行框架

## 概述

`parallel_tool_agent` 是一个轻量级的并行执行框架，专为快速信息收集任务（如论文搜索）优化。它是 `parallel_react_agent` 的简化替代方案。

## 主要特点

### 与 parallel_react_agent 的对比

| 特性 | parallel_tool_agent | parallel_react_agent |
|------|---------------------|----------------------|
| **执行方式** | 直接工具调用 | ReAct 推理循环 |
| **速度** | ⚡️ 非常快 | 🐢 较慢（多轮推理） |
| **复杂度** | 简单 | 复杂 |
| **适用场景** | 简单信息收集 | 需要推理的复杂任务 |
| **Summarization** | 可选（默认关闭） | 总是开启 |
| **LLM 调用** | 仅 summarization 时 | 每个 agent 多次 |

## 使用方法

### 快速模式（无 Summarization）

最快的执行模式，直接获取工具结果：

```python
from agent_blocks_hub.parallel_tool_agent import create_parallel_tool_agent
from llm_tool_hub.scientific_research_tool import SearchSemanticScholar

# 创建轻量级 agent（无需 LLM）
agent = create_parallel_tool_agent(
    tools=[SearchSemanticScholar()],
    enable_summarization=False,  # 最快模式
    verbose=True
)

# 执行并行查询
result = agent.invoke({
    "parallel_tool_agent_messages": [
        "transformer neural networks",
        "attention mechanism deep learning",
        "BERT language model"
    ]
})

# 访问每个工具的结果
for idx, tool_result in result["tool_results"].items():
    print(f"Query {idx}: {tool_result['query']}")
    print(f"Result: {tool_result['result'][:200]}...")
    print(f"Success: {tool_result['success']}")
```

### Summarization 模式

启用 LLM 智能合成结果：

```python
from langchain_openai import ChatOpenAI
from agent_blocks_hub.parallel_tool_agent import create_parallel_tool_agent
from llm_tool_hub.scientific_research_tool import SearchSemanticScholar

llm = ChatOpenAI(model="gpt-4")

agent = create_parallel_tool_agent(
    llm=llm,  # 需要 LLM 进行 summarization
    tools=[SearchSemanticScholar()],
    enable_summarization=True,  # 启用智能合成
    system_prompt="Focus on recent breakthroughs in AI",
    verbose=True
)

result = agent.invoke({
    "parallel_tool_agent_messages": [
        "transformer neural networks",
        "attention mechanism deep learning"
    ]
})

# 获取合成后的摘要
print(result["final_summary"])

# 仍可访问原始工具结果
for idx, tool_result in result["tool_results"].items():
    print(f"Original result {idx}: {tool_result['result']}")
```

## 在 Ideation Agent 中的集成

Ideation Agent 已更新为使用 `parallel_tool_agent`：

```python
from langchain_openai import ChatOpenAI
from llm_tool_hub.scientific_research_tool import SearchSemanticScholar
from multi_agent_hub.scientific_research.ideation import create_ideation_agent

llm = ChatOpenAI(model="gpt-4")
tools = [SearchSemanticScholar()]

# Ideation agent 现在使用轻量级 parallel_tool_agent
agent = create_ideation_agent(
    llm=llm,
    tools=tools,
    system_prompt="Focus on AI safety research",
    verbose=True
)

result = agent.invoke({
    "ideation_message": "How can we improve AI alignment?"
})

print(result["final_idea_report"]["comprehensive_report"])
```

### Ideation Agent 中的改进

- **Stage 1**: 初始信息收集现在使用 `parallel_tool_agent`（快 3-5 倍）
- **Stage 3**: Gap-driven 信息收集也使用 `parallel_tool_agent`
- **默认启用 Summarization**: 保证结果质量的同时提升速度

## Parallel React Agent 的更新

`parallel_react_agent` 也已更新，添加了可选的 summarization：

```python
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from agent_blocks_hub.parallel_react_agent import create_parallel_react_agent

llm = ChatOpenAI(model="gpt-4")
tools = [DuckDuckGoSearchRun()]

# 模式 1: 带 summarization（默认）
agent_with_summary = create_parallel_react_agent(
    llm=llm,
    tools=tools,
    enable_summarization=True,  # 默认值
    verbose=True
)

# 模式 2: 不带 summarization（更快）
agent_fast = create_parallel_react_agent(
    llm=llm,
    tools=tools,
    enable_summarization=False,  # 跳过 summarization
    verbose=True
)

result = agent_fast.invoke({
    "parallel_react_agent_messages": ["What is LangGraph?"]
})

# 访问各个 agent 的结果
for idx, res in result["agent_results"].items():
    print(f"Agent {idx}: {res['result']}")
```

## 性能对比

基于论文搜索任务的测试（3 个查询）：

| Agent 类型 | 平均执行时间 | LLM 调用次数 | 适用场景 |
|-----------|------------|------------|---------|
| **parallel_tool_agent (no summary)** | ~3-5 秒 | 0 | 快速信息收集 |
| **parallel_tool_agent (with summary)** | ~8-10 秒 | 1 | 需要合成的信息收集 |
| **parallel_react_agent (no summary)** | ~15-20 秒 | 9-15 | 复杂任务（快速模式） |
| **parallel_react_agent (with summary)** | ~20-30 秒 | 10-16 | 需要推理的复杂任务 |

## API 参考

### create_parallel_tool_agent

```python
def create_parallel_tool_agent(
    llm: Optional[BaseChatModel] = None,
    tools: List[BaseTool] = None,
    system_prompt: Optional[str] = None,
    verbose: bool = False,
    enable_summarization: bool = False,
    tool_name: Optional[str] = None,
) -> Any:
```

**参数:**
- `llm`: 语言模型（仅在 `enable_summarization=True` 时需要）
- `tools`: 可用工具列表
- `system_prompt`: 用户提供的系统提示词（用于 summarization）
- `verbose`: 是否打印详细执行日志
- `enable_summarization`: 是否启用 LLM 合成（默认: False）
- `tool_name`: 指定使用的工具名称（默认使用第一个工具）

**返回:** 编译好的 LangGraph agent

### State 结构

```python
class ParallelToolAgentState(TypedDict, total=False):
    # 输入查询
    parallel_tool_agent_messages: List[str]
    
    # 配置
    llm: Any
    tools: List[BaseTool]
    system_prompt: str
    verbose: bool
    enable_summarization: bool
    tool_name: Optional[str]
    
    # 工具执行结果
    tool_results: Dict[int, ToolResult]
    
    # 最终摘要（仅在 enable_summarization=True 时）
    final_summary: str
```

## 最佳实践

1. **简单信息收集**: 使用 `parallel_tool_agent` 不带 summarization
2. **需要合成**: 使用 `parallel_tool_agent` 带 summarization
3. **复杂推理任务**: 使用 `parallel_react_agent`
4. **调试**: 始终设置 `verbose=True` 查看详细日志

## 迁移指南

### 从 parallel_react_agent 迁移到 parallel_tool_agent

**之前:**
```python
from agent_blocks_hub.parallel_react_agent import create_parallel_react_agent

agent = create_parallel_react_agent(
    llm=llm,
    tools=tools,
    verbose=True
)

result = agent.invoke({
    "parallel_react_agent_messages": queries
})
```

**之后:**
```python
from agent_blocks_hub.parallel_tool_agent import create_parallel_tool_agent

agent = create_parallel_tool_agent(
    llm=llm,  # 可选，仅 summarization 需要
    tools=tools,
    verbose=True,
    enable_summarization=False  # 快速模式
)

result = agent.invoke({
    "parallel_tool_agent_messages": queries  # 注意 key 的变化
})
```

## 贡献

欢迎提交问题和改进建议！

## 许可证

遵循项目主许可证
