from pathlib import Path
import time
import re
import json
from .llm_support import LLMBasedComponent, call_llm_with_retry


class DirectoryManager(LLMBasedComponent):
    def __init__(self, task_dir: str = "", base_path: str = "./memory_filesystem", llm=None):
        super().__init__(task_dir, llm)
        self.base_path = base_path

    def create_task_directory(self, state, llm_response=None) -> str:
        desc = state.get("task_description") or "Task"
        slug = re.sub(r"[^\w\-]+", "_", str(desc).strip())[:40] or "task"
        ts = int(time.time())
        root = Path(self.base_path).resolve()
        root.mkdir(parents=True, exist_ok=True)
        td = root / f"{slug}_{ts}"
        td.mkdir(parents=True, exist_ok=True)
        for d in ["rules", "preference", "state", "knowledge"]:
            Path(td, d).mkdir(parents=True, exist_ok=True)
        if llm_response is None:
            llm_response = self.ask_llm_for_directory_structure(desc)
        allowed_categories = ["knowledge", "preference"]
        directory_structure = llm_response.get("directory_structure", {}) if isinstance(llm_response, dict) else {}
        for category, subdirs in directory_structure.items():
            if category in allowed_categories and isinstance(subdirs, list):
                for subdir in subdirs:
                    Path(td, category, subdir).mkdir(parents=True, exist_ok=True)
        return str(td)

    def ask_llm_for_directory_structure(self, task_description: str) -> dict:
        prompt = (
            "Suggest a memory directory structure for this task. Top-level fixed: rules, preference, state, knowledge. Only create subdirs under knowledge/preference. Return JSON with directory_structure.\nTask: "
            + str(task_description)
        )
        try:
            llm = self.get_llm()
            response = call_llm_with_retry(llm, [
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ])
            content = str(getattr(response, "content", response)).strip()
            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in content:
                content = content.split("```", 1)[1].split("```", 1)[0].strip()
            return json.loads(content)
        except Exception:
            return {"directory_structure": {"knowledge": ["general"]}}