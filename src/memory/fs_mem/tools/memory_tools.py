import os
from typing import Dict, Any, Optional, List

from ..core.memory_writer import MemoryWriter
from ..core.memory_searcher import MemorySearcher
from ..core.context_builder import ContextBuilder
from ..core.state_updater import StateUpdater
from ..core.directory_manager import DirectoryManager


def read_file_content_tool(file_path: str, encoding: str = "utf-8") -> dict:
    import os
    if not os.path.isfile(file_path):
        return {"status": "error", "content": "", "message": f"File does not exist: {file_path}"}
    try:
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        return {"status": "success", "content": content, "message": ""}
    except Exception as e:
        return {"status": "error", "content": "", "message": str(e)}


def read_directory_contents_tool(dir_path: str, recursive: bool = True, encoding: str = "utf-8") -> Dict[str, Any]:
    import os
    files_content = {}
    if not os.path.isdir(dir_path):
        return {"status": "error", "files": {}, "message": f"Directory does not exist: {dir_path}"}
    try:
        for root, _, files in os.walk(dir_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, dir_path)
                try:
                    with open(fpath, "r", encoding=encoding) as f:
                        files_content[rel_path] = f.read()
                except Exception as fe:
                    files_content[rel_path] = f"[Read failed: {fe}]"
            if not recursive:
                break
        return {"status": "success", "files": files_content, "message": f"{len(files_content)} files read"}
    except Exception as e:
        return {"status": "error", "files": {}, "message": str(e)}


class MemorySystem:
    def __init__(self, task_dir: str):
        self.task_dir = task_dir
        self.memory_writer = MemoryWriter(task_dir)
        self.memory_searcher = MemorySearcher(task_dir)
        self.context_builder = ContextBuilder(task_dir)
        self.state_updater = StateUpdater(task_dir)


_memory_systems: Dict[str, MemorySystem] = {}


def create_memory_system_tool(task_description: str, base_path: str = "./memory_filesystem") -> Dict[str, Any]:
    try:
        dm = DirectoryManager(base_path=base_path)
        td = dm.create_task_directory({"task_description": task_description})
        system = MemorySystem(td)
        system_id = os.path.basename(td)
        _memory_systems[system_id] = system
        return {"status": "success", "system_id": system_id, "task_dir": td, "task_description": task_description}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def write_memory_tool(system_id: str, role: str, content: str, force_write: bool = False) -> Dict[str, Any]:
    if system_id not in _memory_systems:
        return {"status": "error", "message": f"Memory system '{system_id}' not found"}
    try:
        system = _memory_systems[system_id]
        system.memory_writer.add_message(role, content)
        files_written = None
        if force_write:
            files_written = system.memory_writer.write_memory()
        return {"status": "success", "message": "Message added", "files_written": files_written}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def search_memory_tool(system_id: str, question: str) -> Dict[str, Any]:
    if system_id not in _memory_systems:
        return {"status": "error", "message": f"Memory system '{system_id}' not found"}
    try:
        system = _memory_systems[system_id]
        content = system.memory_searcher.search_memory(question)
        return {"status": "success", "content": content}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def update_state_tool(system_id: str, conversation_context: Optional[str] = None) -> Dict[str, Any]:
    if system_id not in _memory_systems:
        return {"status": "error", "message": f"Memory system '{system_id}' not found"}
    try:
        system = _memory_systems[system_id]
        if conversation_context is None:
            conversation_context = system.memory_writer._format_conversation_context()
        current_state = ""
        new_state = system.state_updater.update_state(conversation_context, current_state)
        return {"status": "success", "new_state": new_state}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_context_tool(system_id: str, question: str) -> Dict[str, Any]:
    if system_id not in _memory_systems:
        return {"status": "error", "message": f"Memory system '{system_id}' not found"}
    try:
        system = _memory_systems[system_id]
        context = system.context_builder.build_context(question)
        return {"status": "success", "context": context}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_memory_systems_tool() -> Dict[str, Any]:
    systems = [{"system_id": sid, "task_dir": system.task_dir, "task_description": "N/A"} for sid, system in _memory_systems.items()]
    return {"status": "success", "systems": systems, "count": len(systems)}


def delete_memory_system_tool(system_id: str, delete_files: bool = True) -> Dict[str, Any]:
    if system_id not in _memory_systems:
        return {"status": "error", "message": f"Memory system '{system_id}' not found"}
    try:
        system = _memory_systems[system_id]
        task_dir = system.task_dir
        del _memory_systems[system_id]
        if delete_files:
            import shutil, os
            if os.path.exists(task_dir):
                shutil.rmtree(task_dir)
        return {"status": "success", "files_deleted": delete_files}
    except Exception as e:
        return {"status": "error", "message": str(e)}