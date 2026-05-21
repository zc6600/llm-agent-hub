import os
from .llm_support import LLMBasedComponent, call_llm_with_retry


class StateUpdater(LLMBasedComponent):
    def __init__(self, task_dir: str, llm=None):
        super().__init__(task_dir, llm)

    def update_state(self, conversation_context: str, current_state: str) -> str:
        prompt = (
            "Update the task state based on conversation. Include done, in-progress, next, blockers.\n"
            + "Current State:\n" + (current_state or "") + "\n\nRecent Conversation:\n" + (conversation_context or "")
        )
        try:
            llm = self.get_llm()
            response = call_llm_with_retry(llm, [
                {"role": "system", "content": "You track task progress."},
                {"role": "user", "content": prompt},
            ])
            new_state = str(getattr(response, "content", response)).strip()
        except Exception:
            new_state = current_state or ""
        state_file = os.path.join(self.task_dir, "state", "current_state.txt")
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        with open(state_file, "w", encoding="utf-8") as f:
            f.write(new_state)
        return new_state