# Multi-Agent Orchestration for Deep Diver

## 概述

`MultiAgentOrchestrator` 是一个新的架构，改进了原有的单 agent 问题分解方式：

### 原有方式 (单 Agent)
```
用户问题
    ↓
问题分解 (生成子问题)
    ↓
单个 Agent 处理所有子问题
    ↓
最终答案
```

### 新方式 (多 Agent 编排)
```
用户问题
    ↓
问题分解 (生成子问题)
    ↓
为每个子问题创建一个独立 Agent
    ↓
并行执行所有 Agent
    ↓
汇总所有搜索结果
    ↓
最终综合答案
```

## 核心组件

### 1. `SubTaskAgent` 类

每个子任务都由一个独立的 agent 处理，该 agent 拥有：
- 完整的 Deep Diver 功能（分类、搜索、假设生成等）
- 独立的执行上下文
- 自己的工具集和 LLM

```python
class SubTaskAgent:
    def __init__(
        self,
        task_id: int,              # 任务编号
        task_description: str,     # 任务描述
        agent_graph: Any,          # 完整的 agent 图
        llm: BaseChatModel,
        system_prompt: Optional[str] = None
    ):
        pass
    
    def execute(self) -> Dict[str, Any]:
        # 执行该任务的完整 Deep Diver 流程
        pass
```

### 2. `MultiAgentOrchestrator` 类

编排管理所有子任务 agent：

```python
class MultiAgentOrchestrator:
    def decompose_problem(self, question: str) -> List[str]
        # 将问题分解为子任务
        pass
    
    def create_sub_task_agents(self, sub_tasks: List[str]) -> List[SubTaskAgent]
        # 为每个子任务创建一个独立 agent
        pass
    
    def execute_agents_parallel(self, agents: List[SubTaskAgent]) -> List[Dict[str, Any]]
        # 并行执行所有 agent
        pass
    
    def aggregate_results(
        self,
        original_question: str,
        sub_task_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]
        # 汇总所有结果
        pass
    
    def execute(self, question: str) -> Dict[str, Any]
        # 执行完整的编排流程
        pass
```

## 工作流详解

### 步骤 1: 问题分解 (Decomposition)

```python
orchestrator = MultiAgentOrchestrator(
    llm=llm,
    tools=[internet_search],
    system_prompt=research_instructions,
    max_workers=3  # 最多同时运行 3 个 agent
)

sub_tasks = orchestrator.decompose_problem(
    "What is langgraph and how does it compare with other LLM frameworks?"
)

# 输出示例:
# 1. What is LangGraph and what are its core features?
# 2. How does LangGraph compare with LangChain?
# 3. What are the use cases and advantages of LangGraph?
# 4. How does LangGraph integrate with other frameworks?
```

### 步骤 2: Agent 创建 (Agent Creation)

```python
agents = orchestrator.create_sub_task_agents(sub_tasks)

# 每个 agent 都有:
# - 完整的 Deep Diver 工作流 (分类→搜索→假设→验证)
# - 独立的上下文和状态
# - 相同的工具和 LLM
```

### 步骤 3: 并行执行 (Parallel Execution)

```python
results = orchestrator.execute_agents_parallel(agents)

# 多线程执行:
# - Agent 1 独立处理子任务 1
# - Agent 2 独立处理子任务 2
# - Agent 3 独立处理子任务 3
# 
# 同时运行，加速整个流程
```

### 步骤 4: 结果汇总 (Aggregation)

```python
final_result = orchestrator.aggregate_results(
    original_question,
    sub_task_results
)

# 汇总包括:
# - 所有子任务的信息来源
# - 所有生成的假设
# - 子任务答案
# - 最终综合答案
```

## 使用示例

### 基础使用

```python
from langchain_community.tools import DuckDuckGoSearchRun
from multi_agent_hub.deep_diver import MultiAgentOrchestrator
from multi_agent_hub.llm_provider import get_llm

# 初始化
llm = get_llm(temperature=0.7)
internet_search = DuckDuckGoSearchRun()

# 创建编排器
orchestrator = MultiAgentOrchestrator(
    llm=llm,
    tools=[internet_search],
    system_prompt="You are a thorough research agent...",
    max_workers=3,
    max_iterations=3
)

# 执行
result = orchestrator.execute(
    "What is langgraph and how does it compare with other frameworks?"
)

# 查看结果
print(f"Sub-tasks: {result['num_sub_tasks']}")
print(f"Total sources: {result['total_gathered_sources']}")
print(f"Final answer:\n{result['final_answer']}")
```

### 高级用法

```python
# 自定义控制流程

# 1. 手动分解
sub_tasks = orchestrator.decompose_problem(question)
print(f"Generated {len(sub_tasks)} sub-tasks")

# 2. 创建 agents
agents = orchestrator.create_sub_task_agents(sub_tasks)

# 3. 自定义执行（例如：顺序而非并行）
results = orchestrator.execute_agents_parallel(agents)

# 4. 自定义汇总逻辑
custom_result = orchestrator.aggregate_results(question, results)
```

## 输出结构

### 完整结果格式

```python
{
    "original_question": "用户的原始问题",
    "num_sub_tasks": 4,
    "total_gathered_sources": 25,
    "total_hypotheses": 12,
    
    "sub_task_results": [
        {
            "task_id": 1,
            "task_description": "子任务 1 的描述",
            "status": "completed",  # 或 "failed"
            "gathered_information": [
                {
                    "tool": "duckduckgo",
                    "query": "搜索查询",
                    "result": "搜索结果"
                },
                ...
            ],
            "final_answer": "该子任务的答案",
            "hypotheses": [...]
        },
        ...
    ],
    
    "final_answer": "综合所有子任务的最终答案",
    "gathered_information": [...],  # 所有信息来源的合并
    "hypotheses": [...]  # 所有假设的合并
}
```

## 优势

### 1. 并行化 (Parallelization)
- 多个 agent 同时工作，加速整个流程
- 通过 `max_workers` 参数控制并行度

### 2. 专业化 (Specialization)
- 每个 agent 专注于单个子任务
- 可以针对不同类型的任务定制 agent（未来功能）

### 3. 容错性 (Fault Tolerance)
- 单个 agent 失败不影响其他 agent
- 结果汇总时可以跳过失败的任务

### 4. 可追溯性 (Traceability)
- 清晰的任务分割
- 每个信息来源都能追溯到对应的子任务

### 5. 灵活性 (Flexibility)
- 可以手动控制分解、创建、执行、汇总每个步骤
- 易于实现自定义逻辑

## 配置参数

```python
orchestrator = MultiAgentOrchestrator(
    llm=llm,                      # LLM 实例
    tools=[search_tool, ...],      # 可用工具列表
    system_prompt="...",           # 系统提示词
    max_workers=3,                 # 最大并行 agent 数
    max_iterations=3               # 每个 agent 的最大迭代次数
)
```

## 性能考虑

### 并行度 (max_workers)
- 默认: 3
- 增加可提高吞吐量，但增加 API 调用
- 建议: 根据可用的 API 配额调整

### 迭代次数 (max_iterations)
- 默认: 3
- 每个 agent 在验证假设时的最大循环次数
- 增加可提高准确性，但增加成本

### 成本估算
- N 个子任务，每个 agent 可能进行 M 次 LLM 调用
- 总体成本: ~N × M × (LLM 调用成本)

## 何时使用

### 推荐使用多 Agent 编排:
- ✅ 问题有多个独立的研究方向
- ✅ 子任务之间没有强依赖关系
- ✅ 希望加快整体研究时间
- ✅ 需要针对不同方面的深入研究

### 使用单 Agent 更合适:
- ✅ 问题相对简单，不需要深度分解
- ✅ 子任务之间有强依赖关系
- ✅ API 配额有限
- ✅ 需要严格控制成本

## 运行示例

```bash
# 基础使用
python example/deep_diver/05_multi_agent_orchestration.py

# 使用 LangSmith 追踪
export LANGSMITH_API_KEY=your_key
python example/deep_diver/05_multi_agent_orchestration.py
```

## 未来扩展

1. **动态 Agent 特化**: 为不同类型的子任务创建不同配置的 agent
2. **结果依赖**: 支持子任务之间的依赖关系（例如：任务 2 依赖任务 1 的结果）
3. **智能汇总**: 使用 LLM 识别和解决子任务之间的矛盾
4. **进度监控**: 实时监控 agent 执行进度
5. **A/B 测试**: 比较不同的分解策略

## 常见问题

**Q: 如何增加 agent 的搜索深度?**
A: 增加 `max_iterations` 参数，每次迭代会进行更多的假设验证。

**Q: 并行数增加会怎样?**
A: 会加快执行速度，但同时增加 API 调用数和成本。建议监控和测试。

**Q: 如何处理失败的 agent?**
A: 自动记录为 `"status": "failed"`，最终汇总会跳过或标记失败的任务。

**Q: 能否改变问题分解的方式?**
A: 可以，继承 `MultiAgentOrchestrator` 并覆盖 `decompose_problem()` 方法。
