from typing import Any, List

class LLMBasedComponent:
    def __init__(self, task_dir: str, llm: Any | None = None):
        self.task_dir = task_dir
        self.llm = llm

    def _initialize_llm_from_env(self):
        from llm_provider import get_llm
        return get_llm()

    def get_llm(self):
        if self.llm is not None:
            return self.llm
        self.llm = self._initialize_llm_from_env()
        return self.llm

def call_llm_with_retry(llm: Any, messages: List[Any], tries: int = 2):
    last_err = None
    for _ in range(max(1, tries)):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_err = e
    raise last_err