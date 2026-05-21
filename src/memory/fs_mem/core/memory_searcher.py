import os
import logging
import json
from .llm_support import LLMBasedComponent, call_llm_with_retry

logger = logging.getLogger(__name__)


class MemorySearcher(LLMBasedComponent):
    def __init__(self, task_dir: str, llm=None, summary_length: int = 500, config=None):
        super().__init__(task_dir, llm)
        self.summary_length = summary_length
        self.config = config
        self.file_summaries = self._build_file_summaries()
        self._last_messages = None
        self._last_prompt_text = None
        
        if config:
            logger.setLevel(config.log_level)
        
        logger.info(f"MemorySearcher initialized for: {task_dir}")

    def _build_file_summaries(self) -> dict:
        summaries = {}
        file_count = 0
        
        for root, _, files in os.walk(self.task_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, self.task_dir)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read(self.summary_length)
                    summaries[rel_path] = content
                    file_count += 1
                except UnicodeDecodeError:
                    logger.warning(f"Skipping binary file: {rel_path}")
                    summaries[rel_path] = "[binary file]"
                except Exception as e:
                    logger.error(f"Failed to read file: {rel_path}, error: {e}")
                    summaries[rel_path] = f"[read failed: {e}]"
        
        logger.info(f"Built summaries for {file_count} files")
        return summaries

    def search_memory(self, question: str, *_, **__) -> str:
        from ..tools.memory_tools import read_file_content_tool
        
        file_names = list(self.file_summaries.keys())
        allowed_paths = {name: os.path.abspath(os.path.join(self.task_dir, name)) for name in file_names}
        context = "\n\n".join([f"[{fname}]\n{summary}" for fname, summary in self.file_summaries.items()])
        
        system_prompt = (
            "You are a memory_searcher_agent. Do not fabricate. Use summaries first. "
            "If insufficient, output only JSON tool call "
            '{"tool":{"name":"read_file_content_tool","args":{"file_path":<one of allowed>}}} '
            "or final_answer JSON."
        )
        user_content = (
            f"Question: {question}\n\n"
            f"Options: {file_names}\n"
            f"Allowed paths: {list(allowed_paths.values())}\n"
            f"Summaries (first {self.summary_length} chars):\n{context}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        self._last_messages = messages
        self._last_prompt_text = f"SYSTEM\n{system_prompt}\n\nUSER\n{user_content}"
        
        try:
            llm = self.get_llm()
            logger.debug(f"Calling LLM for search: {question}")
            response = call_llm_with_retry(llm, messages)
            content = str(getattr(response, "content", response)).strip()
            
            # Parse JSON from response
            json_text = content
            if "```json" in content:
                try:
                    json_text = content.split("```json", 1)[1].split("```", 1)[0].strip()
                except Exception as e:
                    logger.warning(f"Failed to extract JSON from code block: {e}")
                    json_text = content
            elif content.startswith("```") and content.endswith("```"):
                json_text = content.strip("`")
            
            # Try to parse as JSON
            obj = None
            try:
                obj = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.warning(f"LLM returned non-JSON response: {e}")
                # Return raw content as fallback
                return content
            
            # Handle final_answer
            if isinstance(obj, dict) and "final_answer" in obj:
                answer = str(obj["final_answer"]).strip()
                logger.debug(f"Got final answer: {len(answer)} characters")
                return answer
            
            # Handle tool call
            if isinstance(obj, dict) and "tool" in obj:
                tool = obj["tool"] or {}
                if tool.get("name") == "read_file_content_tool":
                    args = tool.get("args") or {}
                    fpath = args.get("file_path")
                    
                    if not fpath:
                        logger.error("Tool call missing file_path")
                        return "Error: missing file_path in tool call"
                    
                    fpath_abs = os.path.abspath(fpath)
                    if fpath_abs not in allowed_paths.values():
                        logger.warning(f"Tool call requested illegal path: {fpath}")
                        return f"Error: illegal file_path: {fpath}"
                    
                    logger.debug(f"Reading full file: {fpath}")
                    res = read_file_content_tool(fpath_abs)
                    
                    if res.get("status") == "success":
                        content = res.get("content", "")
                        logger.debug(f"File read successfully: {len(content)} characters")
                        return content
                    else:
                        error_msg = str(res.get("message", "unknown error"))
                        logger.error(f"File read failed: {error_msg}")
                        return f"Error reading file: {error_msg}"
            
            # Fallback: return raw content
            logger.warning("Unexpected LLM response format, returning raw content")
            return content
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return f"Search error: Invalid JSON response from LLM"
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return f"Search error: File not found"
        except Exception as e:
            logger.exception(f"Search failed for question: {question}")
            return f"Search error: {str(e)}"

    def get_last_prompt(self) -> str:
        return self._last_prompt_text or ""