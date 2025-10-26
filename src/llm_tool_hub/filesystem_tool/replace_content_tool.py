import logging
from typing import Dict, Any
from pathlib import Path
from .base_filesystem_tool import BaseFileSystemTool # Assuming BaseFileSystemTool is the base class

logger = logging.getLogger(__name__)

class ReplaceContentTool(BaseFileSystemTool):
    """
    Tool for atomically finding and replacing a single exact block of code 
    or content within a file. It strictly requires exactly one match.
    """
    
    # --- 1. Required Metadata ---
    name: str = "replace_content"
    description: str = (
        "**[SINGLE FILE OPERATION]** Replaces ONE AND ONLY ONE exact block of 'old_content' "
        "with 'new_content' within the specified file. "
        "If 0 or >1 matches are found, the operation will FAIL and return an ERROR. "
        "The tool returns a **SYNCHRONIZED CONTENT WINDOW** with the new, correct line numbers. "
        "Use this for highly targeted, single-instance code replacements and refactoring."
    )
    
    # JSON Schema for the 'run' method parameters
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The relative path to the EXISTING file to be modified. Example: 'src/main.py'",
            },
            "old_content": {
                "type": "string",
                "description": (
                    "The EXACT code or text block to be found and replaced. "
                    "Use '\\n' for multi-line content. It MUST match content exactly."
                ),
            },
            "new_content": {
                "type": "string",
                "description": (
                    "The new content to replace the 'old_content'. "
                    "Use '\\n' for multi-line content. Use an empty string (\"\") for deletion."
                ),
                "default": "",
            },
        },
        "required": ["file_path", "old_content", "new_content"]
    }
    
    # --- 2. Core Replacement Logic ---

    def _write_file_content(self, target_path: Path, modified_lines: list[str]):
        """Helper to write the modified lines back to the file."""
        # This implementation matches the implied logic from your ModifyFileTool run() method:
        # Join lines back with newline character for writing.
        new_content_to_write = '\n'.join(modified_lines)
        
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(new_content_to_write)

    def run(self, 
            file_path: str, 
            old_content: str, 
            new_content: str) -> str:
        """
        Finds exactly one instance of old_content and replaces it with new_content.
        """
        try:
            # 2.1 Path Validation & Read File
            target_path = self._check_path_safety(
                file_path, 
                must_exist=True, 
                allow_dir=False
            )
            
            original_lines = self._get_lines(target_path)
            old_content_lines = old_content.splitlines()
            new_content_lines = new_content.splitlines()
            
            len_old = len(old_content_lines)
            if len_old == 0:
                return "ERROR: 'old_content' cannot be empty. Please specify content to find."

            # 2.2 Find matches for old_content
            matches = []
            
            # Search over the lines list (0-indexed)
            for i in range(len(original_lines) - len_old + 1):
                # Extract a window from the file matching the length of old_content
                window = original_lines[i : i + len_old]
                
                if window == old_content_lines:
                    # Found an exact match!
                    matches.append({
                        "start_line": i + 1,  # 1-based index
                        "end_line": i + len_old
                    })

            # 2.3 Error Check (0 or >1 matches)
            if len(matches) != 1:
                # Prepare detailed error message for LLM
                error_reason = (
                    f"Found {len(matches)} match(es) (Expected exactly 1). "
                )
                if len(matches) > 1:
                    match_details = [f"lines {m['start_line']}-{m['end_line']}" for m in matches]
                    error_reason += f"Multiple matches found at: {', '.join(match_details)}."
                
                return (
                    f"ERROR: Failed to replace content in '{file_path}'. Reason: {error_reason} "
                    f"No modifications were made. Please refine your 'old_content' to be unique."
                )
                
            # 2.4 Execute Replacement (Exactly 1 match)
            match = matches[0]
            start_index = match['start_line'] - 1  # 0-based start index
            end_index = match['end_line']          # 0-based exclusive end index

            # Construct the modified lines list
            modified_lines = (
                original_lines[:start_index] +  # Lines before the match
                new_content_lines +             # The new content to insert/replace
                original_lines[end_index:]      # Lines after the match
            )

            # 2.5 Write Back to File (Atomic operation)
            self._write_file_content(target_path, modified_lines)
            
            # 2.6 Success Message and Synchronization Return
            
            new_total_lines = len(modified_lines)
            len_new = len(new_content_lines)
            lines_replaced = len_old
            lines_added = len_new - lines_replaced
            
            base_success_message = (
                f"SUCCESS: Replaced 1 match of content in file '{file_path}'. "
                f"Operation: replaced {lines_replaced} lines (lines {match['start_line']}-{match['end_line']}) "
                f"with {len_new} new lines (Net change: {lines_added:+d} lines)."
            )
            
            # We must determine the new start and end lines for the synchronization window
            modified_start_line = match['start_line']
            new_lines_count = len_new
            
            sync_content = self._format_modified_content(
                target_path=target_path,
                new_total_lines=new_total_lines,
                modified_start_line=modified_start_line,
                new_lines_count=new_lines_count,
            )
            
            return f"{base_success_message}\n{sync_content}"

        # 2.7 Error Handling (Unified)
        except Exception as e:
            # Catch all exceptions, including those from _check_path_safety
            return f"ERROR: Tool execution failed for '{file_path}'. System message: {e}"

