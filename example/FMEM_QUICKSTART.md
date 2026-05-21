# F-Mem 快速开始指南

## 🚀 最简单的使用方式

只需要在`invoke`时传入`memory`参数，就能自动获得记忆功能！

```python
from agent_blocks_hub import BaseAgent
from memory import FMemClient
from llm_provider import get_llm

# 1. 创建 agent 和 memory
llm = get_llm()
agent = BaseAgent(llm=llm)  # auto_save_memory=True 是默认的
memory = FMemClient(task_dir="./my_memory", llm=llm)

# 2. 直接使用 - 就这么简单！
state = {"messages": [{"role": "user", "content": "你好"}]}
response = agent.invoke(state, memory=memory)

# 搞定！memory 会自动：
# ✅ 读取之前的对话记忆作为上下文
# ✅ 保存这次对话到记忆中
```

---

## 📋 功能说明

### 自动完成的事情

当你调用`agent.invoke(state, memory=memory)`时，系统会：

1. **自动读取记忆**：从`memory`中读取相关的历史信息和上下文
2. **增强上下文**：将记忆内容添加到系统提示词中
3. **调用LLM**：用增强后的上下文调用LLM
4. **自动保存**：将新的对话保存到记忆中（默认开启）

### 控制选项

**默认：自动保存开启**
```python
agent = BaseAgent(llm=llm)  # auto_save_memory=True
response = agent.invoke(state, memory=memory)  # 自动保存
```

**全局关闭自动保存**
```python
agent = BaseAgent(llm=llm, auto_save_memory=False)
response = agent.invoke(state, memory=memory)  # 不自动保存
```

**单次控制**
```python
agent = BaseAgent(llm=llm)  # 默认开启

# 这次保存
response1 = agent.invoke(state, memory=memory, auto_save=True)

# 这次不保存（比如临时查询）
response2 = agent.invoke(state, memory=memory, auto_save=False)
```

---

## 💡 实际使用示例

### 示例1：多轮对话

```python
agent = BaseAgent(llm=get_llm())
memory = FMemClient(task_dir="./chat_memory", llm=get_llm())

state = {"messages": []}

# 第一轮
state["messages"].append({"role": "user", "content": "我的名字是张三"})
resp1 = agent.invoke(state, memory=memory)
state["messages"].append({"role": "assistant", "content": resp1.content})

# 第二轮 - memory会记住"张三"
state["messages"].append({"role": "user", "content": "我叫什么名字？"})
resp2 = agent.invoke(state, memory=memory)
# resp2 会正确回答"张三"!
```

### 示例2：项目助手

```python
agent = BaseAgent(
    llm=get_llm(),
    system_prompt="你是一个编程助手"
)
memory = FMemClient(task_dir="./project_memory", llm=get_llm())

# 所有对话都会记住项目相关信息
response = agent.invoke(
    {"messages": [{"role": "user", "content": "如何优化认证模块?"}]},
    memory=memory
)
# agent可以访问之前关于认证模块的所有讨论
```

### 示例3：带工具的Agent

```python
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    """搜索文档"""
    return f"找到关于 {query} 的文档"

agent = BaseAgent(llm=get_llm(), tools=[search_docs])
memory = FMemClient(task_dir="./memory", llm=get_llm())

# Agent可以使用工具，同时访问记忆
response = agent.invoke(
    {"messages": [{"role": "user", "content": "查找API文档"}]},
    memory=memory
)
```

---

## ⚙️ 高级配置

### 自定义记忆配置

```python
from memory import FMemConfig
import logging

config = FMemConfig(
    buffer_size=20,              # 每20条消息保存一次
    max_recent_messages=15,       # 上下文包含最近15条消息
    max_file_size=5*1024*1024,   # 最大读取5MB文件
    log_level=logging.DEBUG,      # 调试日志
    enable_validation=True        # 验证LLM的工具调用
)

memory = FMemClient(task_dir="./memory", llm=llm, config=config)
```

### 语义搜索

```python
# 搜索特定主题的记忆
context = memory.get_context(
    messages=messages,
    query="认证相关的讨论"  # 触发语义搜索
)
```

---

## 🔍 调试和日志

查看记忆操作的详细日志：

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# 现在会看到详细的记忆操作日志
agent.invoke(state, memory=memory)
```

---

## 📁 记忆存储结构

F-Mem在文件系统中存储记忆：

```
my_memory/
├── rules/          # 规则和约束
├── preference/     # 用户偏好
├── state/          # 当前状态
└── knowledge/      # 知识和事实
```

每个目录下可以有多个`.txt`文件，agent会自动读取和管理。

---

## 🎯 最佳实践

1. **一个项目一个memory实例**：为每个项目或对话创建独立的`task_dir`
2. **合理使用auto_save**：临时查询可以关闭自动保存
3. **配置合适的buffer_size**：根据对话频率调整保存频率
4. **查看日志**：开启DEBUG日志了解记忆操作

---

## 🆚 对比：使用前 vs 使用后

### 没有记忆时
```python
agent = BaseAgent(llm=llm)
response = agent.invoke(state)
# ❌ Agent不记得之前的对话
# ❌ 每次都要重新提供上下文
```

### 有记忆时
```python
agent = BaseAgent(llm=llm)
memory = FMemClient(task_dir="./memory", llm=llm)
response = agent.invoke(state, memory=memory)
# ✅ 自动记住所有重要信息
# ✅ 上下文持久化
# ✅ 可以跨会话访问
```

---

## ❓ 常见问题

**Q: 记忆什么时候被保存？**
A: 默认情况下，每次`invoke`之后会自动保存。也可以通过`buffer_size`配置每N条消息保存一次。

**Q: 如何清空记忆？**
A: 删除`task_dir`目录即可。

**Q: 记忆会不会越来越大？**
A: FMemClient只读取文件摘要（默认500字符），大文件会被跳过，所以不会有问题。

**Q: 可以手动搜索记忆吗？**
A: 可以！使用`memory.search("查询内容")`。

**Q: 记忆存储在哪里？**
A: 存储在你指定的`task_dir`本地文件系统中，完全在你的控制下。

---

## 📚 更多示例

查看完整示例：[fmem_usage_examples.py](./fmem_usage_examples.py)
