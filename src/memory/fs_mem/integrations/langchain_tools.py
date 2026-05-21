import os
from typing import Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .. import FMem, ensure_task_structure


class SearchMemoryInput(BaseModel):
    query: str = Field(description="")


class SearchMemoryTool(BaseTool):
    name: str = "search_memory"
    description: str = ""
    args_schema: Type[BaseModel] = SearchMemoryInput
    fmem: FMem
    def __init__(self, task_dir: str, summary_length: int = 500):
        fmem = FMem(task_dir, summary_length=summary_length)
        super().__init__(fmem=fmem)
    def _run(self, query: str) -> str:
        return self.fmem.search_memory(query)
    async def _arun(self, query: str) -> str:
        return self._run(query)


class GetContextInput(BaseModel):
    query: str = Field(description="")


class GetContextTool(BaseTool):
    name: str = "get_fmem_context"
    description: str = ""
    args_schema: Type[BaseModel] = GetContextInput
    fmem: FMem
    max_messages: int = 10
    def __init__(self, task_dir: str, summary_length: int = 500, max_messages: int = 10):
        fmem = FMem(task_dir, summary_length=summary_length)
        super().__init__(fmem=fmem, max_messages=max_messages)
    def _run(self, query: str) -> str:
        state = {"task_dir": self.fmem.task_dir, "messages": [], "query": query}
        state = self.fmem.update_state(state, max_messages=self.max_messages)
        return state.get("context", "")
    async def _arun(self, query: str) -> str:
        return self._run(query)


def create_fmem_tools(task_dir: str, *, summary_length: int = 500, max_messages: int = 10, auto_ensure_structure: bool = True) -> list[BaseTool]:
    if auto_ensure_structure:
        ensure_task_structure(task_dir)
    return [
        SearchMemoryTool(task_dir=task_dir, summary_length=summary_length),
        GetContextTool(task_dir=task_dir, summary_length=summary_length, max_messages=max_messages),
    ]