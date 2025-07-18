import streamlit as st
import pandas as pd
import copy
import datetime
import sys
import io
import traceback
import signal
import numpy as np
from typing import Dict, Optional, Tuple, Any
from langchain.tools import Tool
from src.core.utils import sync_dataframe_versions
from src.core.backup_manager import backup_manager
# from docker_sandbox import DockerSandbox  # Temporarily disabled


class CheckpointManager:
    """Manages checkpoints for undo/redo functionality"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self._ensure_session_state()
    
    def _ensure_session_state(self):
        """Ensure all required session state variables are initialized"""
        if not hasattr(st, 'session_state'):
            return
            
        if "checkpoints" not in st.session_state:
            st.session_state.checkpoints = []
        if "checkpoint_index" not in st.session_state:
            st.session_state.checkpoint_index = -1
    
    def save_checkpoint(self, df: pd.DataFrame, operation: str) -> None:
        """Save current state as checkpoint with automatic backup"""
        self._ensure_session_state()
        
        if df is None:
            return
            
        checkpoint = {
            "df": df.copy(),
            "operation": operation,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create automatic backup for significant operations
        if self._is_significant_operation(operation):
            try:
                backup_id = backup_manager.create_automatic_backup(
                    operation=operation,
                    before_df=df
                )
                checkpoint["backup_id"] = backup_id
            except Exception as e:
                st.warning(f"Failed to create automatic backup: {e}")
        
        # Remove any checkpoints after current index (for redo functionality)
        if st.session_state.checkpoint_index < len(st.session_state.checkpoints) - 1:
            st.session_state.checkpoints = st.session_state.checkpoints[:st.session_state.checkpoint_index + 1]
        
        # Add new checkpoint
        st.session_state.checkpoints.append(checkpoint)
        st.session_state.checkpoint_index += 1
        
        # Maintain max history
        if len(st.session_state.checkpoints) > self.max_history:
            st.session_state.checkpoints.pop(0)
            st.session_state.checkpoint_index -= 1
    
    def undo(self) -> Tuple[bool, str]:
        """Undo last operation"""
        self._ensure_session_state()
        
        if st.session_state.checkpoint_index > 0:
            st.session_state.checkpoint_index -= 1
            checkpoint = st.session_state.checkpoints[st.session_state.checkpoint_index]
            
            # Restore both df and df_original
            restored_df = checkpoint["df"].copy()
            st.session_state.df_original = restored_df.copy()
            st.session_state.df = restored_df.copy()
            
            return True, f"✅ Undid: {checkpoint['operation']}"
        return False, "❌ No operations to undo"
    
    def redo(self) -> Tuple[bool, str]:
        """Redo last undone operation"""
        self._ensure_session_state()
        
        if st.session_state.checkpoint_index < len(st.session_state.checkpoints) - 1:
            st.session_state.checkpoint_index += 1
            checkpoint = st.session_state.checkpoints[st.session_state.checkpoint_index]
            
            # Restore both df and df_original
            restored_df = checkpoint["df"].copy()
            st.session_state.df_original = restored_df.copy()
            st.session_state.df = restored_df.copy()
            
            return True, f"✅ Redid: {checkpoint['operation']}"
        return False, "❌ No operations to redo"
    
    def _is_significant_operation(self, operation: str) -> bool:
        """Determine if an operation is significant enough to warrant automatic backup"""
        significant_keywords = [
            "drop", "delete", "remove", "split", "merge", "transform", 
            "apply", "replace", "fillna", "dropna", "pivot", "melt",
            "groupby", "aggregate", "join", "concat", "assign"
        ]
        
        operation_lower = operation.lower()
        return any(keyword in operation_lower for keyword in significant_keywords)


class ExecutionLogger:
    """Logs all code executions with results"""
    
    def __init__(self):
        self._ensure_session_state()
    
    def _ensure_session_state(self):
        """Ensure all required session state variables are initialized"""
        if not hasattr(st, 'session_state'):
            return
            
        if "execution_log" not in st.session_state:
            st.session_state.execution_log = []
        
        if "checkpoints" not in st.session_state:
            st.session_state.checkpoints = []
            
        if "checkpoint_index" not in st.session_state:
            st.session_state.checkpoint_index = -1
    
    def log_execution(self, code: str, success: bool, result: str, execution_time: float = None) -> None:
        """Log code execution result"""
        self._ensure_session_state()
        
        log_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "code": code,
            "success": success,
            "result": result,
            "execution_time": execution_time
        }
        
        st.session_state.execution_log.append(log_entry)
        
        # Keep only last 50 entries
        if len(st.session_state.execution_log) > 50:
            st.session_state.execution_log.pop(0)
    
    def get_recent_logs(self, n: int = 10) -> list:
        """Get n most recent log entries"""
        self._ensure_session_state()
        return st.session_state.execution_log[-n:]


# Initialize global objects
checkpoint_manager = CheckpointManager()
execution_logger = ExecutionLogger()


def get_indentation(line: str) -> str:
    """Get the indentation (whitespace) from the beginning of a line"""
    return line[:len(line) - len(line.lstrip())]


def needs_block_content(lines: list, current_index: int) -> bool:
    """
    Check if a comment-only line needs to be replaced with 'pass' to maintain block structure
    """
    if current_index == 0:
        return False
    
    # Find the previous line that ends with colon
    colon_line_index = -1
    for i in range(current_index - 1, -1, -1):
        prev_line = lines[i].strip()
        if prev_line and prev_line.endswith(':'):
            colon_line_index = i
            break
        elif prev_line and not prev_line.startswith('#'):
            # Found non-comment, non-colon line, stop looking
            break
    
    if colon_line_index == -1:
        return False  # No colon line found
    
    # Check if this comment is the first line after the colon
    current_indent = len(get_indentation(lines[current_index]))
    
    # Look for any non-comment line between colon and current line
    for k in range(colon_line_index + 1, current_index):
        check_line = lines[k].strip()
        if check_line and not check_line.startswith('#'):
            return False  # There's already code content, no need for pass
    
    # Check if there's any code content after this line at the same indentation level
    for j in range(current_index + 1, len(lines)):
        next_line = lines[j].strip()
        if next_line and not next_line.startswith('#'):
            next_indent = len(get_indentation(lines[j]))
            if next_indent >= current_indent:
                return False  # There's code content at this level, no need for pass
            elif next_indent < current_indent:
                break  # End of block
    
    # This is the first content after colon and there's no code content, need pass
    return True


def clean_code_input(code: str) -> str:
    """
    Enhanced clean up code input from the agent to remove quotes, comments, and markdown formatting
    Handles Vietnamese comments and various markdown patterns
    """
    if not code:
        return code
    
    # Remove surrounding whitespace
    code = code.strip()
    
    # Remove surrounding quotes first (they might wrap the entire markdown block)
    if (code.startswith('"') and code.endswith('"')) or (code.startswith("'") and code.endswith("'")):
        code = code[1:-1].strip()
    
    # Remove markdown code blocks (```python and ```)
    if code.startswith('```python'):
        code = code[9:].strip()  # Remove ```python
    elif code.startswith('```py'):
        code = code[5:].strip()  # Remove ```py
    elif code.startswith('```'):
        code = code[3:].strip()  # Remove ```
    
    if code.endswith('```'):
        code = code[:-3].strip()  # Remove closing ```
    
    # Remove any remaining surrounding quotes after markdown removal
    if (code.startswith('"') and code.endswith('"')) or (code.startswith("'") and code.endswith("'")):
        code = code[1:-1].strip()
    
    # Handle cases where the agent might add extra quotes around the entire code
    # Example: '"st.session_state.df.iloc[18]"  # comment'
    if code.startswith('"') and '"' in code[1:]:
        # Find the closing quote before any comment
        quote_pos = code.find('"', 1)
        if quote_pos != -1:
            code = code[1:quote_pos]
    
    # Aggressive comment removal with better string handling
    if '#' in code:
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            
            # Skip empty lines or comment-only lines
            if not stripped_line or stripped_line.startswith('#'):
                continue
            
            # Handle inline comments more carefully
            if '#' in line:
                # Split on # but be careful about strings
                parts = line.split('#')
                code_part = parts[0].rstrip()
                
                # Check if # might be inside a string (basic check)
                quote_count_single = code_part.count("'")
                quote_count_double = code_part.count('"')
                
                # If quotes are unbalanced, the # might be inside a string
                if (quote_count_single % 2 == 0 and quote_count_double % 2 == 0):
                    # Quotes are balanced, safe to remove comment
                    if code_part.strip():
                        cleaned_lines.append(code_part)
                else:
                    # Quotes unbalanced, keep original line (might be # in string)
                    cleaned_lines.append(line)
            else:
                # No comment, keep the line as-is
                cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines).strip()
    
    # Additional safety check - reject if still contains suspicious comment patterns
    if code.strip().startswith('#') or '\n#' in code:
        raise ValueError("❌ Code contains comments. Only pure Python code is allowed.")
    
    # Additional cleanup for common agent artifacts
    # Remove any remaining markdown artifacts
    code = code.replace('```python', '').replace('```py', '').replace('```', '')
    
    # Fix common variable reference issues
    # Replace st.session_state.df with df for code execution
    code = code.replace('st.session_state.df', 'df')
    
    # Only remove surrounding quotes if they wrap the ENTIRE code block
    # and the code inside is syntactically valid
    code = code.strip()
    if len(code) >= 2:
        # Check if the entire code is wrapped in quotes
        if ((code.startswith('"') and code.endswith('"')) or 
            (code.startswith("'") and code.endswith("'"))):
            # Extract the inner code
            inner_code = code[1:-1].strip()
            # Only unwrap if the inner code doesn't contain the same quote type
            if code.startswith('"') and '"' not in inner_code:
                code = inner_code
            elif code.startswith("'") and "'" not in inner_code:
                code = inner_code
    
    return code.strip()


def run_code_safely(code: str) -> str:
    """
    Execute pandas code safely with direct execution (fallback mode)
    """
    if not code or not code.strip():
        return "❌ Empty code provided"
    
    # Store original code for debugging
    original_code = code
    
    # Pre-validation: Reject code with comments immediately
    if '#' in code and (code.strip().startswith('#') or '\n#' in code or code.count('#') > code.count('"#"') + code.count("'#'")):
        return f"❌ Code contains comments. Only pure Python code is allowed.\n🔍 Original input: {repr(original_code)}"
    
    # Clean up the code input - remove quotes and comments that might be added by the agent
    code = clean_code_input(code)
    
    if not code or not code.strip():
        return f"❌ Empty code provided after cleaning\n🔍 Original input: {repr(original_code)}"
    
    # Log the code transformation for debugging
    if original_code != code:
        print(f"🔍 Code transformation:")
        print(f"   Original: {repr(original_code)}")
        print(f"   Cleaned:  {repr(code)}")
        
        # Log comment filtering specifically
        if '#' in original_code and '#' not in code:
            print(f"✅ Comments successfully filtered from code")
        elif '#' in original_code and '#' in code:
            print(f"⚠️ Comments detected but not fully filtered")
        
        # Add to execution log for user visibility
        if 'execution_log' not in st.session_state:
            st.session_state.execution_log = []
        
        st.session_state.execution_log.append({
            'timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
            'original_code': original_code,
            'cleaned_code': code,
            'comment_filtered': '#' in original_code and '#' not in code
        })
    
    # Check if dataframe exists - prioritize df_original (real data)
    if (not hasattr(st.session_state, 'df_original') or st.session_state.df_original is None) and st.session_state.df is None:
        return "❌ No dataframe loaded. Please upload a file first."
    
    try:
        # Save checkpoint before execution - use df_original (real data) if available
        if hasattr(st.session_state, 'df_original') and st.session_state.df_original is not None:
            checkpoint_df = st.session_state.df_original
        else:
            checkpoint_df = st.session_state.df
        
        checkpoint_manager.save_checkpoint(checkpoint_df, f"Before: {code[:50]}...")
        
        # Record start time
        start_time = datetime.datetime.now()
        
        # Execute code directly
        success, output, updated_df = run_code_directly(code)
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        if success and updated_df is not None:
            # Get shape before and after for accurate reporting
            old_shape = checkpoint_df.shape if checkpoint_df is not None else (0, 0)
            new_shape = updated_df.shape
            
            # Synchronize all DataFrame versions
            sync_dataframe_versions(updated_df)
            
            result_msg = f"✅ Code executed successfully!\n"
            result_msg += f"📊 Shape changed: {old_shape} → {new_shape}\n"
            result_msg += f"⏱️ Execution time: {execution_time:.2f}s\n"
            if output:
                result_msg += f"🔧 Output: {output}"
            
            # Log successful execution
            execution_logger.log_execution(code, True, result_msg, execution_time)
            
            return result_msg
        else:
            error_msg = f"❌ Code execution failed: {output}"
            execution_logger.log_execution(code, False, error_msg, execution_time)
            return error_msg
            
    except Exception as e:
        error_msg = f"❌ Execution error: {str(e)}"
        execution_logger.log_execution(code, False, error_msg)
        return error_msg


def format_result(result) -> str:
    """
    Format the result of code execution for display
    """
    if result is None:
        return ""
    
    try:
        # Handle pandas objects
        if hasattr(result, '__module__') and result.__module__ == 'pandas.core.series':
            # This is a pandas Series
            if len(result) > 20:
                # For long series, show first 10 and last 10
                result_str = str(result.head(10))
                result_str += "\n...\n"
                result_str += str(result.tail(10))
                result_str += f"\n\nLength: {len(result)}, dtype: {result.dtype}"
            else:
                result_str = str(result)
            return f"📊 Result:\n{result_str}"
        
        elif hasattr(result, '__module__') and result.__module__ == 'pandas.core.frame':
            # This is a pandas DataFrame
            if len(result) > 20:
                result_str = str(result.head(10))
                result_str += "\n...\n"
                result_str += str(result.tail(10))
                result_str += f"\n\n[{len(result)} rows x {len(result.columns)} columns]"
            else:
                result_str = str(result)
            return f"📊 Result:\n{result_str}"
        
        elif isinstance(result, (list, tuple)) and len(result) > 20:
            # For long lists/tuples, truncate
            result_str = str(result[:10])
            result_str += f"\n... ({len(result)} items total)"
            return f"📊 Result:\n{result_str}"
        
        elif isinstance(result, dict) and len(result) > 20:
            # For large dictionaries, show first few items
            items = list(result.items())[:10]
            result_str = "{\n"
            for k, v in items:
                result_str += f"  {repr(k)}: {repr(v)},\n"
            result_str += f"  ... ({len(result)} items total)\n}}"
            return f"📊 Result:\n{result_str}"
        
        else:
            # For other objects, use string representation
            result_str = str(result)
            # Truncate very long strings
            if len(result_str) > 1000:
                result_str = result_str[:1000] + "... (truncated)"
            return f"📊 Result:\n{result_str}"
    
    except Exception as e:
        return f"📊 Result: {type(result).__name__} (display error: {str(e)})"


def run_code_directly(code: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Execute pandas code directly in current process with safety measures
    Maintains persistent execution environment across calls
    """
    try:
        # Basic safety checks
        if not is_code_safe(code):
            return False, "❌ Code contains potentially dangerous operations", None
        
        # Initialize persistent execution environment if not exists
        if not hasattr(st.session_state, 'code_execution_env'):
            st.session_state.code_execution_env = {}
        
        # Prepare execution environment - use df_original (real data) if available
        # This ensures CodeRunner works with actual data types, not display-optimized data
        if hasattr(st.session_state, 'df_original') and st.session_state.df_original is not None:
            working_df = st.session_state.df_original.copy()
        else:
            working_df = st.session_state.df.copy()
        
        # Update persistent environment with current dataframe
        st.session_state.code_execution_env.update({
            'df': working_df,
            'pd': pd,
            'np': np,
            'st': st,
            'datetime': datetime,
        })
        
        # If there are other dataframes in session, add them
        if "dfs" in st.session_state:
            st.session_state.code_execution_env.update(st.session_state.dfs)
        
        # Use persistent environment for execution
        local_vars = st.session_state.code_execution_env
        
        # Capture stdout
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Execute as statement directly to avoid eval/exec issues
            # This handles complex statements and expressions uniformly
            exec(code, {"__builtins__": __builtins__}, local_vars)
            
            # Get captured output from print statements
            output = captured_output.getvalue()
            
            # Return the modified dataframe
            return True, output, local_vars.get('df', working_df)
            
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            
    except Exception as e:
        # Get detailed error information
        error_info = traceback.format_exc()
        return False, f"Error: {str(e)}\n\nDetails:\n{error_info}", None


def is_code_safe(code: str) -> bool:
    """
    Basic safety check for code execution
    """
    # List of dangerous operations to block
    dangerous_patterns = [
        'import os',
        'import sys',
        'import subprocess',
        'import shutil',
        'from os',
        'from sys',
        'from subprocess',
        'from shutil',
        'exec(',
        'eval(',
        'open(',
        'file(',
        '__import__',
        'globals(',
        'locals(',
        'vars(',
        'dir(',
        'getattr(',
        'setattr(',
        'delattr(',
        'hasattr(',
    ]
    
    # Check for dangerous patterns
    code_lower = code.lower()
    for pattern in dangerous_patterns:
        if pattern in code_lower:
            return False
    
    return True


def undo_operation(_: str) -> str:
    """Undo last operation"""
    success, message = checkpoint_manager.undo()
    return message


def redo_operation(_: str) -> str:
    """Redo last undone operation"""
    success, message = checkpoint_manager.redo()
    return message


def show_execution_log(n: str = "5") -> str:
    """Show recent execution logs"""
    try:
        num_logs = int(n) if n.isdigit() else 5
        logs = execution_logger.get_recent_logs(num_logs)
        
        if not logs:
            return "📝 No execution logs available"
        
        result = f"📝 **Recent Execution Logs ({len(logs)} entries):**\n\n"
        
        for i, log in enumerate(reversed(logs), 1):
            status = "✅" if log["success"] else "❌"
            result += f"**{i}. {log['timestamp']}** {status}\n"
            result += f"Code: `{log['code'][:100]}...`\n"
            result += f"Result: {log['result'][:200]}...\n"
            if log.get("execution_time"):
                result += f"Time: {log['execution_time']:.2f}s\n"
            result += "---\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error showing logs: {str(e)}"


def reset_execution_environment(_: str = "") -> str:
    """Reset the persistent execution environment"""
    try:
        if hasattr(st.session_state, 'code_execution_env'):
            st.session_state.code_execution_env = {}
        return "✅ Execution environment reset successfully. All imported modules and variables cleared."
    except Exception as e:
        return f"❌ Error resetting environment: {str(e)}"


# Create tools
CodeRunnerTool = Tool(
    name="CodeRunner",
    func=run_code_safely,
    description="Execute pandas code safely with persistent execution environment. Updates the main dataframe. Maintains all imports and variables between calls. Input: Clean Python code WITHOUT comments (#) or markdown (```). Use 'df' variable to access dataframe. Send only executable code, no explanations or comments."
)

UndoTool = Tool(
    name="Undo",
    func=undo_operation,
    description="Undo the last operation performed on the dataframe. No input required."
)

RedoTool = Tool(
    name="Redo", 
    func=redo_operation,
    description="Redo the last undone operation. No input required."
)

ExecutionLogTool = Tool(
    name="ExecutionLog",
    func=show_execution_log,
    description="Show recent code execution logs. Input: number of logs to show (default: 5)."
)

ResetEnvironmentTool = Tool(
    name="ResetEnvironment",
    func=reset_execution_environment,
    description="Reset the persistent execution environment, clearing all imported modules and variables. Use this when you need a clean slate or encounter import conflicts. No input required."
)