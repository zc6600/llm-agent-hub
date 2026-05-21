import os
import sys
sys.path.append(os.path.abspath('../../src'))
from memory import FsMem
from memory.fs_mem import FMemAgent
from memory.fs_mem import search_memory
from agent_blocks_hub.base_agent import create_base_agent
from llm_provider import get_llm
from langchain_core.messages import HumanMessage

task_llm = get_llm(temperature=0.2)
mem_llm = get_llm(temperature=0.0)

fs_mem_for_dir = FsMem(task_dir='./example_output/fs_mem_quick', llm=mem_llm)
task_state = {'messages': [{'role': 'user', 'content': 'Plan a small web app to track tasks.'}]}
task_dir = fs_mem_for_dir.create_task_dir(task_state, base_path='./example_output/fs_mem_quick')

fs_mem = FsMem(task_dir=task_dir, llm=mem_llm)
state = {'messages': [
    {'role': 'user', 'content': 'Plan a small web app to track tasks.'},
    {'role': 'assistant', 'content': 'Use FastAPI and SQLite.'}
]}
state = fs_mem.update_state(state, max_messages=2)
print(state['context'][:400])

print(search_memory('FastAPI', task_dir=task_dir)[:200])

agent = create_base_agent(llm=task_llm, tools=None, system_prompt='You are a helpful assistant.')
resp1 = agent.invoke({'messages': [{'role': 'user', 'content': 'Propose a database choice.'}]}, memory=fs_mem)
print(getattr(resp1, 'content', resp1))

fmem_agent = FMemAgent(task_dir=task_dir)
resp2 = fmem_agent.call_with_memory({'messages': [HumanMessage(content='Suggest a stack for the app')]}, llm=task_llm, system_prompt='You are a helpful assistant.')
print(getattr(resp2, 'content', resp2))