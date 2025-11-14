# 接口一致性对比

## Deep Diver vs Paper Planner

paper_planner 已经集成到 `src/multi_agent_hub/scientific_research/paper_planner` 下，
并与 Deep Diver 保持一致的接口风格。

## 接口对比

### 创建 Agent

#### Deep Diver
```python
from langchain_openai import ChatOpenAI
from multi_agent_hub.deep_diver import create_deepdiver_agent

llm = ChatOpenAI(model="gpt-4", api_key="...")

agent = create_deepdiver_agent(
    llm=llm,
    tools=[internet_search],
    system_prompt=research_instructions,
)
```

#### Paper Planner
```python
from langchain_openai import ChatOpenAI
from multi_agent_hub.scientific_research.paper_planner import create_paper_planner_agent

llm = ChatOpenAI(model="gpt-4", api_key="...")

agent = create_paper_planner_agent(
    llm=llm,
    system_prompt=research_instructions,
)
```

### 调用 Agent

#### Deep Diver
```python
result = agent.invoke({
    "messages": [{"role": "user", "content": question}]
})
```

#### Paper Planner
```python
result = agent.invoke({
    "messages": [{"role": "user", "content": question}],
    "original_request": research_request
})
```

## 主要差异

### 1. 导入路径
- Deep Diver: `from multi_agent_hub.deep_diver import create_deepdiver_agent`
- Paper Planner: `from multi_agent_hub.scientific_research.paper_planner import create_paper_planner_agent`

### 2. 工厂函数参数
| 参数 | Deep Diver | Paper Planner |
|------|-----------|---------------|
| llm | ✓ 必需 | ✓ 必需 |
| tools | ✓ 必需 | ✗ (内置工具) |
| system_prompt | ✓ 可选 | ✓ 可选 |
| max_iterations | ✓ (default=3) | ✗ max_literature_iterations (default=5) |
| | | ✗ max_refinement_iterations (default=3) |

### 3. Agent 调用输入
| 字段 | Deep Diver | Paper Planner |
|------|-----------|---------------|
| messages | ✓ 必需 | ✓ 必需 |
| other | (工具会添加) | original_request 推荐提供 |

### 4. 返回结果

#### Deep Diver 返回字段
- `messages`: 消息历史
- `original_question`: 原始问题
- `decomposed_problems`: 分解后的子问题
- `gathered_information`: 收集的信息
- `hypotheses`: 生成的假设
- `final_answer`: 最终答案


#### Paper Planner 返回字段
- `messages`: 消息历史
- `original_request`: 原始研究请求
- `current_plan`: 所有计划迭代
- `final_plan`: 最终研究计划
- `literature_note`: 论文笔记列表
- `searched_papers`: 搜索过的论文列表
- `counter`: 文献综述迭代计数
- `counter_2`: 计划细化迭代计数

## 功能对比

| 功能 | Deep Diver | Paper Planner |
|------|-----------|---------------|
| 问题分解 | ✓ | ✓ (隐式) |
| 信息收集 | ✓ (通用工具) | ✓ (arXiv 搜索) |
| 假设生成 | ✓ | ✗ |
| 假设验证 | ✓ 迭代 | ✗ |
| 计划生成 | ✗ | ✓ 初始 |
| 计划细化 | ✗ | ✓ 迭代 |
| 论文笔记 | ✗ | ✓ |
| 文件导出 | ✗ | ✓ |

## 工作流对比

### Deep Diver 工作流
```
问题分解
  ↓
信息收集
  ↓
假设生成 → 假设验证 (迭代)
  ↓
最终答案
```

### Paper Planner 工作流
```
初始计划
  ↓
文献综述 (搜索 → 检索 → 笔记) (迭代)
  ↓
计划细化 (迭代)
  ↓
最终计划
```

## 使用场景

### Deep Diver 适用场景
- 快速问题解答
- 信息验证和事实检查
- 多轮假设-验证循环
- 需要高置信度答案的场景

### Paper Planner 适用场景
- 学术研究计划制定
- 文献综述和背景调研
- 研究假设和方法设计
- 生成结构化研究报告

## 示例文件位置

- Deep Diver 示例: `example/deep_diver/01_basic_usage.py`
- Paper Planner 示例: `example/deep_diver/02_paper_planner.py`

## 集成说明

### 在同一项目中使用两个 Agent

```python
from langchain_openai import ChatOpenAI
from multi_agent_hub.deep_diver import create_deepdiver_agent
from multi_agent_hub.scientific_research.paper_planner import create_paper_planner_agent
from langchain_community.tools import DuckDuckGoSearchRun

llm = ChatOpenAI(model="gpt-4", api_key="...")

# 创建 Deep Diver - 用于快速问题解答
deep_diver = create_deepdiver_agent(
    llm=llm,
    tools=[DuckDuckGoSearchRun()],
    system_prompt="You are a thorough researcher..."
)

# 创建 Paper Planner - 用于研究计划
paper_planner = create_paper_planner_agent(
    llm=llm,
    system_prompt="You are a research planning expert..."
)

# 使用场景 1: 快速查询
result1 = deep_diver.invoke({
    "messages": [{"role": "user", "content": "What is transformer?"}]
})

# 使用场景 2: 研究计划制定
result2 = paper_planner.invoke({
    "messages": [{"role": "user", "content": ""}],
    "original_request": "I want to research..."
})
```

## 下一步

1. 运行示例: `python example/deep_diver/02_paper_planner.py`
2. 查看输出文件夹获取生成的研究计划
3. 根据需要调整 `max_literature_iterations` 和 `max_refinement_iterations`
4. 结合 Deep Diver 和 Paper Planner 进行完整的研究工作流
