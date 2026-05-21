import os
import re
import time
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

from .llm_support import LLMBasedComponent, call_llm_with_retry

logger = logging.getLogger(__name__)


class MemoryWriter(LLMBasedComponent):
    def __init__(self, task_dir: str, llm=None, buffer_size: int = 10, config=None):
        super().__init__(task_dir, llm)
        self.conversation_buffer: List[Dict[str, str]] = []
        self.buffer_size = buffer_size
        self.config = config
        
        if config:
            logger.setLevel(config.log_level)
        
        logger.info(f"MemoryWriter initialized for: {task_dir}")

    def _get_available_subdirs(self) -> Dict[str, list]:
        subdirs = {"knowledge": [], "preference": []}
        for category in ["knowledge", "preference"]:
            category_path = os.path.join(self.task_dir, category)
            if os.path.exists(category_path):
                for item in os.listdir(category_path):
                    item_path = os.path.join(category_path, item)
                    if os.path.isdir(item_path):
                        subdirs[category].append(item)
        return subdirs

    def add_message(self, role: str, content: str) -> None:
        self.conversation_buffer.append({"role": role, "content": content})
        if len(self.conversation_buffer) >= self.buffer_size:
            self.write_memory()

    def _format_conversation_context(self) -> str:
        lines = []
        for msg in self.conversation_buffer:
            lines.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(lines)

    def write_memory(self) -> list:
        if not self.conversation_buffer:
            logger.debug("No messages in buffer, skipping write")
            return []
        
        context = self._format_conversation_context()
        logger.info(f"Writing memory with {len(self.conversation_buffer)} buffered messages")
        
        files_written = self._summarize_conversations(context)
        self.conversation_buffer = []
        
        logger.info(f"Memory write complete: {len(files_written)} files modified")
        return files_written

    def _summarize_conversations(self, context: str) -> Dict[str, Any]:
        tools_description = (
            "1. write_memory_direct(dir: str, introduction: str, summary: str, mode: str = 'append')\n"
            "2. write_memory_with_context(dir: str, introduction: str, first_10_words: str, last_10_words: str, mode: str = 'append')\n"
        )
        available_subdirs = self._get_available_subdirs()
        subdir_info = ""
        if available_subdirs["knowledge"]:
            subdir_info += "\nAvailable knowledge subdirectories: " + ", ".join(available_subdirs["knowledge"])
        if available_subdirs["preference"]:
            subdir_info += "\nAvailable preference subdirectories: " + ", ".join(available_subdirs["preference"])
        prompt = (
            "Analyze the following conversation and write important information to memory using the provided tools.\n\n"
            "Memory directories:\n- rules\n- preference\n- state\n- knowledge\n"
            f"{subdir_info}\n\n{tools_description}\n\nConversation:\n{context}\n\n"
            "Return ONLY a JSON array of tool calls."
        )
        try:
            llm = self.get_llm()
            logger.debug("Calling LLM for memory summarization")
            
            response = call_llm_with_retry(llm, [
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ])
            content = str(getattr(response, "content", response)).strip()
            
            # Extract JSON from code blocks
            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif content.startswith("```") and content.endswith("```"):
                content = content.strip("`")
            
            # Parse tool calls
            try:
                tool_calls = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"LLM returned invalid JSON: {e}")
                logger.debug(f"Raw response: {content}")
                return []
            
            # Validate and execute tool calls
            if self.config and self.config.enable_validation:
                tool_calls = self._validate_tool_calls(tool_calls)
                logger.debug(f"Validated {len(tool_calls)} tool calls")
            
            files_written = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("tool_name", "")
                args = tool_call.get("arguments", {})
                dir_name = args.get("dir", "")
                introduction = args.get("introduction", "")
                mode = args.get("mode", "append")
                
                if tool_name == "write_memory_direct":
                    summary = args.get("summary", "")
                    content_to_write = summary
                elif tool_name == "write_memory_with_context":
                    first_10 = args.get("first_10_words", "")
                    last_10 = args.get("last_10_words", "")
                    extracted_content = self._extract_context_from_conversation(first_10, last_10, context)
                    content_to_write = extracted_content or ("[Context Start]: " + first_10 + "\n...\n[Context End]: " + last_10)
                else:
                    logger.warning(f"Unknown tool name: {tool_name}")
                    continue
                
                try:
                    self._write_to_file(dir_name, introduction, content_to_write, mode)
                    files_written.append({
                        "dir": dir_name, 
                        "introduction": introduction, 
                        "type": tool_name, 
                        "mode": mode
                    })
                    logger.debug(f"Wrote to: {dir_name}/{introduction}")
                except Exception as e:
                    logger.error(f"Failed to write file: {dir_name}/{introduction}, error: {e}")
            
            return files_written
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return []
        except Exception as e:
            logger.exception(f"Memory write failed: {e}")
            return []

    def _validate_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """
        Validate LLM-generated tool calls before execution.
        
        Args:
            tool_calls: List of tool call dicts from LLM.
        
        Returns:
            List of validated tool calls (invalid ones are filtered out).
        """
        valid_calls = []
        
        for call in tool_calls:
            tool_name = call.get("tool_name", "")
            args = call.get("arguments", {})
            dir_name = args.get("dir", "")
            
            # Validate directory is legal
            if not self._is_valid_directory(dir_name):
                logger.warning(f"Skipping invalid directory: {dir_name}")
                continue
            
            # Validate required fields
            if tool_name == "write_memory_direct":
                if not args.get("summary"):
                    logger.warning(f"Skipping tool call without summary: {dir_name}")
                    continue
            elif tool_name == "write_memory_with_context":
                if not args.get("first_10_words") or not args.get("last_10_words"):
                    logger.warning(f"Skipping tool call without context markers: {dir_name}")
                    continue
            else:
                logger.warning(f"Skipping unknown tool: {tool_name}")
                continue
            
            # Validate introduction
            if not args.get("introduction"):
                logger.warning(f"Skipping tool call without introduction: {dir_name}")
                continue
            
            valid_calls.append(call)
        
        return valid_calls
    
    def _is_valid_directory(self, dir_name: str) -> bool:
        """
        Check if directory name is valid.
        
        Args:
            dir_name: Directory path to validate.
        
        Returns:
            True if valid, False otherwise.
        """
        if not dir_name:
            return False
        
        # Get allowed categories from config or use defaults
        allowed_categories = [
            "rules", "preference", "state", "knowledge"
        ]
        if self.config and self.config.memory_categories:
            allowed_categories = self.config.memory_categories
        
        # Check if path starts with allowed category
        parts = dir_name.split("/")
        if len(parts) == 0:
            return False
        
        category = parts[0]
        return category in allowed_categories
    
    def _write_to_file(self, dir_name: str, introduction: str, content: str, mode: str = "append"):
        dir_path = os.path.join(self.task_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        
        if introduction.endswith(".txt"):
            filename = introduction
        else:
            filename = f"{introduction}.txt"
        
        filepath = os.path.join(dir_path, filename)
        file_exists = os.path.exists(filepath)
        
        if mode == "append" and file_exists:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write("\n\n---[New Entry]---\n\n")
                f.write(content)
            logger.debug(f"Appended to file: {filepath}")
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.debug(f"Created/overwrote file: {filepath}")
        
        return filepath

    def _extract_context_from_conversation(self, first_10: str, last_10: str, full_context: str) -> str:
        context_lower = full_context.lower()
        first_lower = first_10.lower()
        last_lower = last_10.lower()
        start_idx = context_lower.find(first_lower)
        if start_idx == -1:
            first_words = [w for w in first_lower.split() if len(w) > 3]
            if first_words:
                for i in range(0, len(context_lower), 5):
                    window_end = min(i + 150, len(context_lower))
                    window = context_lower[i:window_end]
                    match_count = sum(1 for w in first_words if w in window)
                    if match_count >= max(2, len(first_words) // 2):
                        start_idx = i
                        break
        end_idx = -1
        search_start = start_idx if start_idx != -1 else 0
        end_idx = context_lower.find(last_lower, search_start)
        if end_idx == -1:
            last_words = [w for w in last_lower.split() if len(w) > 3]
            if last_words:
                best_pos = -1
                best_score = 0
                for i in range(search_start, len(context_lower), 5):
                    window_end = min(i + 150, len(context_lower))
                    window = context_lower[i:window_end]
                    match_count = sum(1 for w in last_words if w in window)
                    if match_count >= max(2, len(last_words) // 2) and match_count > best_score:
                        best_score = match_count
                        best_pos = window_end
                if best_pos > search_start:
                    end_idx = best_pos
        else:
            end_idx += len(last_lower)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            extracted = full_context[start_idx:end_idx].strip()
            return extracted
        return ""