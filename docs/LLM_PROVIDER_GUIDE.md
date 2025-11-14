# LLM Provider 使用指南

## 概述

`llm_provider` 是 `multi_agent_hub` 中的一个新模块，提供了一个集中化的方式来初始化和配置 LLM 实例。它支持：

- 从 `.env` 文件读取环境变量
- 自动配置 OpenRouter 或 OpenAI API
- 集成 LangSmith 监控和追踪
- 灵活的配置选项

## 快速开始

### 1. 设置环境变量 (.env)

在项目根目录的 `.env` 文件中配置以下变量：

```properties
# API 密钥 (选择其一)
OPENROUTER_API_KEY=your_openrouter_key_here
# 或
OPENAI_API_KEY=your_openai_key_here

# 模型配置
OPENROUTER_MODEL=google/gemini-2.5-flash-lite-preview-09-2025

# LangSmith 监控 (可选)
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=llm-tool-hub
```

### 2. 在代码中使用

#### 方法 1: 使用默认配置

```python
from multi_agent_hub.llm_provider import get_llm

# 使用 .env 中的所有设置
llm = get_llm()
```

#### 方法 2: 自定义参数

```python
from multi_agent_hub.llm_provider import get_llm

# 覆盖某些参数
llm = get_llm(
    temperature=0.5,
    max_tokens=2000,
    enable_langsmith=True,
)
```

#### 方法 3: 完全自定义

```python
from multi_agent_hub.llm_provider import get_llm

llm = get_llm(
    model="gpt-4o",
    api_key="sk-xxx",
    base_url="https://api.openai.com/v1",
    temperature=0.7,
    max_tokens=100000,
    enable_langsmith=True,
)
```

#### 方法 4: 使用配置字典

```python
from multi_agent_hub.llm_provider import get_llm_with_custom_config

config = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 50000,
    "api_key": "sk-xxx",
    "base_url": "https://api.openai.com/v1",
}

llm = get_llm_with_custom_config(config, enable_langsmith=True)
```

## 完整示例

查看 `example/deep_diver/01_basic_usage.py` 了解完整的使用示例：

```python
import sys
from pathlib import Path

# 添加 src 到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_community.tools import DuckDuckGoSearchRun
from multi_agent_hub.deep_diver import create_deepdiver_agent
from multi_agent_hub.llm_provider import get_llm

# 初始化 LLM
llm = get_llm(
    temperature=0.7,
    max_tokens=100000,
    enable_langsmith=True,
)

# 初始化工具
internet_search = DuckDuckGoSearchRun()

# 创建代理
agent = create_deepdiver_agent(
    llm=llm,
    tools=[internet_search],
    system_prompt="You are a thorough research agent...",
)

# 使用代理
result = agent.invoke({
    "messages": [{"role": "user", "content": "Your question here"}]
})
```

## 功能说明

### `get_llm()` 函数

```python
def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 100000,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    enable_langsmith: bool = True,
) -> ChatOpenAI:
```

**参数:**
- `model`: 要使用的模型。如果为 None，使用 `.env` 中的 `OPENROUTER_MODEL`
- `temperature`: 模型的温度参数 (0-1)，控制生成的随机性
- `max_tokens`: 响应的最大 token 数
- `api_key`: API 密钥。如果为 None，自动从 `.env` 读取
- `base_url`: API 基础 URL。如果为 None 且使用 OpenRouter key，自动设置为 OpenRouter 地址
- `enable_langsmith`: 是否启用 LangSmith 监控

**返回值:**
- 配置完成的 `ChatOpenAI` 实例

### LangSmith 监控

当 `enable_langsmith=True` 时，会自动配置以下环境变量：

- `LANGCHAIN_TRACING_V2`: 设置为 `"true"`
- `LANGCHAIN_API_KEY`: 从 `.env` 的 `LANGSMITH_API_KEY` 读取
- `LANGCHAIN_PROJECT`: 从 `.env` 的 `LANGSMITH_PROJECT` 读取（默认为 `"llm-tool-hub"`）

所有 LLM 调用都会被自动追踪到 LangSmith 项目中。

## 最佳实践

1. **不要硬编码 API 密钥**: 始终使用 `.env` 文件
2. **启用 LangSmith**: 用于监控和调试生成过程
3. **设置合理的 temperature**: 
   - 0-0.3: 保守，输出确定性强（适合准确性要求高的任务）
   - 0.5-0.7: 平衡（推荐用于大多数任务）
   - 0.8-1.0: 创意，输出多样化（适合创意任务）
4. **按需调整 max_tokens**: 根据任务的预期输出长度调整

## 环境变量参考

| 变量名 | 用途 | 示例 |
|-------|------|------|
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | `sk-or-v1-xxx` |
| `OPENAI_API_KEY` | OpenAI API 密钥 | `sk-xxx` |
| `OPENROUTER_MODEL` | 默认使用的模型 | `google/gemini-2.5-flash-lite-preview-09-2025` |
| `LANGSMITH_API_KEY` | LangSmith 追踪密钥 | `lsv2_pt_xxx` |
| `LANGSMITH_PROJECT` | LangSmith 项目名称 | `llm-tool-hub` |

## 错误处理

如果未提供 API 密钥，会抛出 `ValueError`：

```python
try:
    llm = get_llm()
except ValueError as e:
    print(f"配置错误: {e}")
    # 确保 .env 文件中设置了 OPENROUTER_API_KEY 或 OPENAI_API_KEY
```

## 更多示例

查看 `example/` 目录中的其他示例，了解如何在不同场景中使用 `llm_provider`。
