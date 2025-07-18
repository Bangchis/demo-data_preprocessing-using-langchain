"""
Comprehensive Backup Manager for Data Preprocessing MVP
Handles all backup operations including DataFrames, session state, and recovery.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import os
import datetime
import shutil
import gzip
import hashlib
import uuid
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings


class BackupManager:
    """
    Comprehensive backup manager for the data preprocessing system.
    Handles persistent storage, recovery, and backup management.
    """
    
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.ensure_backup_directory()
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        self.session_dir = self.backup_dir / "sessions"
        self.dataframe_dir = self.backup_dir / "dataframes"
        self.checkpoint_dir = self.backup_dir / "checkpoints"
        
        # Create subdirectories
        self.session_dir.mkdir(exist_ok=True)
        self.dataframe_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize session state and session ID
        self._ensure_session_state()
        self._initialize_session_id()
        
        # Load existing metadata
        self.metadata = self._load_metadata()
        
        # Configuration for session cleanup (manual only)
        self.session_timeout_hours = 24  # Default 24 hours
        self.cleanup_frequency_minutes = 30  # Check every 30 minutes
        self.max_backups_per_session = 50  # Maximum backups per session
    
    def ensure_backup_directory(self):
        """Ensure backup directory exists"""
        self.backup_dir.mkdir(exist_ok=True)
        
        # Create .gitignore to exclude backup files from git
        gitignore_path = self.backup_dir / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, 'w') as f:
                f.write("# Backup files\n*\n!.gitignore\n")
    
    def _ensure_session_state(self):
        """Ensure all required session state variables are initialized"""
        if not hasattr(st, 'session_state'):
            return
        
        try:
            # Initialize backup-related session state
            if "backup_status" not in st.session_state:
                st.session_state.backup_status = {
                    "last_backup": None,
                    "last_restore": None,
                    "auto_backup_enabled": True,
                    "backup_count": 0
                }
            
            if "backup_history" not in st.session_state:
                st.session_state.backup_history = []
        except Exception:
            # If session state is not available, silently continue
            pass
    
    def _initialize_session_id(self):
        """Initialize unique session ID for backup tracking"""
        if not hasattr(st, 'session_state'):
            return
        
        try:
            if "session_id" not in st.session_state:
                # Generate unique session ID
                session_id = str(uuid.uuid4())[:8]
                st.session_state.session_id = session_id
                st.session_state.session_start_time = datetime.datetime.now().isoformat()
                
                # Store session info
                if "backup_status" in st.session_state:
                    st.session_state.backup_status["session_id"] = session_id
                    st.session_state.backup_status["session_start_time"] = st.session_state.session_start_time
                    
        except Exception:
            # If session state is not available, silently continue
            pass
    
    def _load_metadata(self) -> Dict:
        """Load backup metadata from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {"backups": [], "sessions": [], "dataframes": []}
        except Exception as e:
            st.error(f"Error loading backup metadata: {e}")
            return {"backups": [], "sessions": [], "dataframes": []}
    
    def _save_metadata(self):
        """Save backup metadata to disk"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Error saving backup metadata: {e}")
    
    def create_automatic_backup(self, operation: str, before_df: pd.DataFrame = None) -> str:
        """
        Create automatic backup before risky operations
        
        Args:
            operation: Description of the operation about to be performed
            before_df: DataFrame state before the operation
            
        Returns:
            backup_id: Unique identifier for the backup
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"auto_{timestamp}_{hashlib.md5(operation.encode()).hexdigest()[:8]}"
        
        try:
            # Create backup
            backup_info = {
                "id": backup_id,
                "type": "automatic",
                "operation": operation,
                "timestamp": timestamp,
                "created_at": datetime.datetime.now().isoformat(),
                "dataframe_shape": before_df.shape if before_df is not None else None,
                "session_state_keys": list(st.session_state.keys()) if hasattr(st, 'session_state') else [],
                "session_id": getattr(st.session_state, 'session_id', None) if hasattr(st, 'session_state') else None,
                "session_start_time": getattr(st.session_state, 'session_start_time', None) if hasattr(st, 'session_state') else None
            }
            
            # Save DataFrame if provided
            if before_df is not None:
                df_path = self.dataframe_dir / f"{backup_id}_df.pkl.gz"
                self._save_dataframe(before_df, df_path)
                backup_info["dataframe_path"] = str(df_path)
            
            # Save session state
            session_path = self.session_dir / f"{backup_id}_session.pkl.gz"
            self._save_session_state(session_path)
            backup_info["session_path"] = str(session_path)
            
            # Update metadata
            self.metadata["backups"].append(backup_info)
            self._save_metadata()
            
            # Update session state
            if hasattr(st.session_state, 'backup_status'):
                st.session_state.backup_status["last_backup"] = backup_id
                st.session_state.backup_status["backup_count"] += 1
            
            if hasattr(st.session_state, 'backup_history'):
                st.session_state.backup_history.append({
                    "id": backup_id,
                    "type": "automatic",
                    "operation": operation,
                    "timestamp": timestamp
                })
            
            return backup_id
            
        except Exception as e:
            st.error(f"Error creating automatic backup: {e}")
            return None
    
    def create_manual_backup(self, name: str, description: str = "") -> str:
        """
        Create manual backup with user-defined name and description
        
        Args:
            name: User-defined name for the backup
            description: Optional description of the backup
            
        Returns:
            backup_id: Unique identifier for the backup
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"manual_{timestamp}_{name.replace(' ', '_')}"
        
        try:
            # Create backup
            backup_info = {
                "id": backup_id,
                "type": "manual",
                "name": name,
                "description": description,
                "timestamp": timestamp,
                "created_at": datetime.datetime.now().isoformat(),
                "dataframe_shape": st.session_state.df.shape if hasattr(st.session_state, 'df') and st.session_state.df is not None else None,
                "session_state_keys": list(st.session_state.keys()) if hasattr(st, 'session_state') else [],
                "session_id": getattr(st.session_state, 'session_id', None) if hasattr(st, 'session_state') else None,
                "session_start_time": getattr(st.session_state, 'session_start_time', None) if hasattr(st, 'session_state') else None
            }
            
            # Save DataFrame if available
            if hasattr(st.session_state, 'df') and st.session_state.df is not None:
                df_path = self.dataframe_dir / f"{backup_id}_df.pkl.gz"
                self._save_dataframe(st.session_state.df, df_path)
                backup_info["dataframe_path"] = str(df_path)
            
            # Save original DataFrame if available
            if hasattr(st.session_state, 'df_original') and st.session_state.df_original is not None:
                df_orig_path = self.dataframe_dir / f"{backup_id}_df_original.pkl.gz"
                self._save_dataframe(st.session_state.df_original, df_orig_path)
                backup_info["dataframe_original_path"] = str(df_orig_path)
            
            # Save session state
            session_path = self.session_dir / f"{backup_id}_session.pkl.gz"
            self._save_session_state(session_path)
            backup_info["session_path"] = str(session_path)
            
            # Update metadata
            self.metadata["backups"].append(backup_info)
            self._save_metadata()
            
            # Update session state
            if hasattr(st.session_state, 'backup_status'):
                st.session_state.backup_status["last_backup"] = backup_id
                st.session_state.backup_status["backup_count"] += 1
            
            if hasattr(st.session_state, 'backup_history'):
                st.session_state.backup_history.append({
                    "id": backup_id,
                    "type": "manual",
                    "name": name,
                    "description": description,
                    "timestamp": timestamp
                })
            
            return backup_id
            
        except Exception as e:
            st.error(f"Error creating manual backup: {e}")
            return None
    
    def restore_backup(self, backup_id: str) -> bool:
        """
        Restore a backup by ID with full DataFrame synchronization
        
        Args:
            backup_id: Unique identifier of the backup to restore
            
        Returns:
            bool: True if restoration successful, False otherwise
        """
        try:
            # Ensure session state is initialized
            self._ensure_session_state()
            # Find backup in metadata
            backup_info = None
            for backup in self.metadata["backups"]:
                if backup["id"] == backup_id:
                    backup_info = backup
                    break
            
            if not backup_info:
                st.error(f"Backup {backup_id} not found")
                return False
            
            # Store original shape for comparison
            original_shape = None
            if hasattr(st.session_state, 'df') and st.session_state.df is not None:
                original_shape = st.session_state.df.shape
            
            restored_df = None
            restored_df_original = None
            
            # Restore DataFrame
            if "dataframe_path" in backup_info:
                df_path = Path(backup_info["dataframe_path"])
                if df_path.exists():
                    df = self._load_dataframe(df_path)
                    if df is not None:
                        restored_df = df
                        st.session_state.df = df
                        # Also update dfs dictionary
                        if "dfs" not in st.session_state:
                            st.session_state.dfs = {}
                        st.session_state.dfs["df"] = df.copy()
            
            # Restore original DataFrame if available
            if "dataframe_original_path" in backup_info:
                df_orig_path = Path(backup_info["dataframe_original_path"])
                if df_orig_path.exists():
                    df_original = self._load_dataframe(df_orig_path)
                    if df_original is not None:
                        restored_df_original = df_original
                        st.session_state.df_original = df_original
                        # Also update dfs_original dictionary
                        if "dfs_original" not in st.session_state:
                            st.session_state.dfs_original = {}
                        st.session_state.dfs_original["df"] = df_original.copy()
            
            # Synchronize all DataFrame versions using the available restored data
            # Priority: df_original > df
            if restored_df_original is not None:
                self._sync_all_dataframe_versions(restored_df_original)
            elif restored_df is not None:
                self._sync_all_dataframe_versions(restored_df)
            
            # Restore session state (selective restoration)
            if "session_path" in backup_info:
                session_path = Path(backup_info["session_path"])
                if session_path.exists():
                    self._restore_session_state(session_path)
            
            # Update session state
            if hasattr(st.session_state, 'backup_status'):
                st.session_state.backup_status["last_restore"] = backup_id
            
            # Clear persistent execution environment to ensure fresh start
            if hasattr(st.session_state, 'code_execution_env'):
                st.session_state.code_execution_env = {}
            
            # Trigger UI refresh
            st.rerun()
            
            st.success(f"✅ Backup {backup_id} restored successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error restoring backup {backup_id}: {e}")
            return False
    
    def _sync_all_dataframe_versions(self, df: pd.DataFrame):
        """
        Synchronize all DataFrame versions after restore
        
        Args:
            df: The restored DataFrame to sync across all versions
        """
        try:
            # Import the sync function from utils
            from src.core.utils import sync_dataframe_versions
            
            # Use the existing sync function
            sync_dataframe_versions(df)
            
        except Exception as e:
            st.error(f"Error synchronizing DataFrame versions: {e}")
    
    def _save_dataframe(self, df: pd.DataFrame, path: Path):
        """Save DataFrame to compressed pickle file"""
        try:
            with gzip.open(path, 'wb') as f:
                pickle.dump(df, f)
        except Exception as e:
            st.error(f"Error saving DataFrame to {path}: {e}")
    
    def _load_dataframe(self, path: Path) -> Optional[pd.DataFrame]:
        """Load DataFrame from compressed pickle file"""
        try:
            with gzip.open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading DataFrame from {path}: {e}")
            return None
    
    def _save_session_state(self, path: Path):
        """Save session state to compressed pickle file"""
        try:
            if not hasattr(st, 'session_state'):
                return
            
            # Extract important session state components
            session_data = {
                "checkpoints": getattr(st.session_state, 'checkpoints', []),
                "checkpoint_index": getattr(st.session_state, 'checkpoint_index', -1),
                "execution_log": getattr(st.session_state, 'execution_log', []),
                "react_chat_history": getattr(st.session_state, 'react_chat_history', []),
                "chat_history": getattr(st.session_state, 'chat_history', []),
                "web_search_log": getattr(st.session_state, 'web_search_log', [])
            }
            
            with gzip.open(path, 'wb') as f:
                pickle.dump(session_data, f)
                
        except Exception as e:
            st.error(f"Error saving session state to {path}: {e}")
    
    def _restore_session_state(self, path: Path):
        """Restore session state from compressed pickle file"""
        try:
            with gzip.open(path, 'rb') as f:
                session_data = pickle.load(f)
            
            # Restore important session state components
            if "checkpoints" in session_data:
                st.session_state.checkpoints = session_data["checkpoints"]
            if "checkpoint_index" in session_data:
                st.session_state.checkpoint_index = session_data["checkpoint_index"]
            if "execution_log" in session_data:
                st.session_state.execution_log = session_data["execution_log"]
            if "react_chat_history" in session_data:
                st.session_state.react_chat_history = session_data["react_chat_history"]
            if "chat_history" in session_data:
                st.session_state.chat_history = session_data["chat_history"]
            if "web_search_log" in session_data:
                st.session_state.web_search_log = session_data["web_search_log"]
                
        except Exception as e:
            st.error(f"Error restoring session state from {path}: {e}")
    
    def get_backup_list(self) -> List[Dict]:
        """Get list of all available backups"""
        return self.metadata.get("backups", [])
    
    def get_backup_info(self, backup_id: str) -> Optional[Dict]:
        """Get detailed information about a specific backup"""
        for backup in self.metadata["backups"]:
            if backup["id"] == backup_id:
                return backup
        return None
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup by ID"""
        try:
            # Find backup in metadata
            backup_info = None
            for i, backup in enumerate(self.metadata["backups"]):
                if backup["id"] == backup_id:
                    backup_info = backup
                    backup_index = i
                    break
            
            if not backup_info:
                st.error(f"Backup {backup_id} not found")
                return False
            
            # Delete files
            files_to_delete = []
            if "dataframe_path" in backup_info:
                files_to_delete.append(Path(backup_info["dataframe_path"]))
            if "dataframe_original_path" in backup_info:
                files_to_delete.append(Path(backup_info["dataframe_original_path"]))
            if "session_path" in backup_info:
                files_to_delete.append(Path(backup_info["session_path"]))
            
            for file_path in files_to_delete:
                if file_path.exists():
                    file_path.unlink()
            
            # Remove from metadata
            self.metadata["backups"].pop(backup_index)
            self._save_metadata()
            
            st.success(f"✅ Backup {backup_id} deleted successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error deleting backup {backup_id}: {e}")
            return False
    
    
    def cleanup_session_backups(self, session_id: str = None) -> int:
        """Clean up backups from a specific session"""
        try:
            if session_id is None:
                # Use current session ID
                session_id = getattr(st.session_state, 'session_id', None)
                if not session_id:
                    return 0
            
            # Find backups for this session
            session_backups = [
                backup for backup in self.metadata["backups"]
                if backup.get("session_id") == session_id
            ]
            
            # Delete all backups for this session
            deleted_count = 0
            for backup in session_backups:
                if self.delete_backup(backup["id"]):
                    deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            st.error(f"Error cleaning up session backups: {e}")
            return 0
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions based on timeout"""
        try:
            current_time = datetime.datetime.now()
            timeout_delta = datetime.timedelta(hours=self.session_timeout_hours)
            
            # Find expired sessions
            expired_sessions = []
            for backup in self.metadata["backups"]:
                if "session_start_time" in backup:
                    try:
                        session_start = datetime.datetime.fromisoformat(backup["session_start_time"])
                        if current_time - session_start > timeout_delta:
                            expired_sessions.append(backup["session_id"])
                    except (ValueError, KeyError):
                        # Skip backups with invalid timestamps
                        continue
            
            # Remove duplicates
            expired_sessions = list(set(expired_sessions))
            
            # Clean up expired sessions
            total_deleted = 0
            for session_id in expired_sessions:
                deleted = self.cleanup_session_backups(session_id)
                total_deleted += deleted
            
            if total_deleted > 0:
                st.info(f"🧹 Cleaned up {total_deleted} backups from {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            # Silent failure for background cleanup
            pass
    
    def cleanup_current_session(self) -> int:
        """Clean up all backups from current session"""
        try:
            session_id = getattr(st.session_state, 'session_id', None)
            if not session_id:
                return 0
            
            deleted_count = self.cleanup_session_backups(session_id)
            
            if deleted_count > 0:
                st.success(f"🧹 Cleaned up {deleted_count} backups from current session")
            else:
                st.info("ℹ️ No backups found in current session")
            
            return deleted_count
            
        except Exception as e:
            st.error(f"Error cleaning up current session: {e}")
            return 0
    
    def delete_all_backups(self) -> Dict[str, int]:
        """Delete all backups in the system"""
        try:
            # Get current statistics
            stats_before = self.get_backup_statistics()
            total_backups = stats_before['total_backups']
            
            if total_backups == 0:
                return {"deleted_count": 0, "total_before": 0}
            
            # Get all backup IDs
            all_backups = self.metadata.get("backups", [])
            backup_ids = [backup["id"] for backup in all_backups]
            
            # Delete all backups
            deleted_count = 0
            for backup_id in backup_ids:
                try:
                    if self.delete_backup(backup_id):
                        deleted_count += 1
                except Exception as e:
                    # Continue deleting even if one fails
                    st.warning(f"Failed to delete backup {backup_id}: {e}")
                    continue
            
            # Clear the metadata list
            self.metadata["backups"] = []
            self._save_metadata()
            
            return {
                "deleted_count": deleted_count,
                "total_before": total_backups,
                "success": deleted_count > 0
            }
            
        except Exception as e:
            st.error(f"Error deleting all backups: {e}")
            return {"deleted_count": 0, "total_before": 0, "success": False, "error": str(e)}
    
    def get_backup_statistics(self) -> Dict:
        """Get statistics about backups"""
        # Ensure session state is initialized
        self._ensure_session_state()
        backups = self.metadata.get("backups", [])
        
        total_backups = len(backups)
        manual_backups = len([b for b in backups if b.get("type") == "manual"])
        automatic_backups = len([b for b in backups if b.get("type") == "automatic"])
        
        # Calculate total size
        total_size = 0
        for backup in backups:
            for path_key in ["dataframe_path", "dataframe_original_path", "session_path"]:
                if path_key in backup:
                    file_path = Path(backup[path_key])
                    if file_path.exists():
                        total_size += file_path.stat().st_size
        
        return {
            "total_backups": total_backups,
            "manual_backups": manual_backups,
            "automatic_backups": automatic_backups,
            "total_size_mb": total_size / (1024 * 1024),
            "last_backup": st.session_state.backup_status.get("last_backup") if hasattr(st.session_state, 'backup_status') else None,
            "last_restore": st.session_state.backup_status.get("last_restore") if hasattr(st.session_state, 'backup_status') else None
        }
    
    def export_backup(self, backup_id: str, export_path: str) -> bool:
        """Export a backup to a ZIP file"""
        try:
            import zipfile
            
            backup_info = self.get_backup_info(backup_id)
            if not backup_info:
                st.error(f"Backup {backup_id} not found")
                return False
            
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add metadata
                zipf.writestr("metadata.json", json.dumps(backup_info, indent=2))
                
                # Add files
                for path_key in ["dataframe_path", "dataframe_original_path", "session_path"]:
                    if path_key in backup_info:
                        file_path = Path(backup_info[path_key])
                        if file_path.exists():
                            zipf.write(file_path, file_path.name)
            
            st.success(f"✅ Backup {backup_id} exported to {export_path}")
            return True
            
        except Exception as e:
            st.error(f"Error exporting backup {backup_id}: {e}")
            return False
    
    def import_backup(self, import_path: str) -> bool:
        """Import a backup from a ZIP file"""
        try:
            import zipfile
            
            with zipfile.ZipFile(import_path, 'r') as zipf:
                # Extract metadata
                metadata_content = zipf.read("metadata.json")
                backup_info = json.loads(metadata_content)
                
                # Generate new backup ID to avoid conflicts
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                original_id = backup_info["id"]
                backup_info["id"] = f"imported_{timestamp}_{original_id}"
                backup_info["imported_at"] = datetime.datetime.now().isoformat()
                backup_info["original_id"] = original_id
                
                # Extract files
                for path_key in ["dataframe_path", "dataframe_original_path", "session_path"]:
                    if path_key in backup_info:
                        original_path = Path(backup_info[path_key])
                        if original_path.name in zipf.namelist():
                            # Create new path with new backup ID
                            new_path = original_path.parent / f"{backup_info['id']}_{original_path.name.split('_', 1)[1]}"
                            
                            # Extract file
                            with zipf.open(original_path.name) as src:
                                with open(new_path, 'wb') as dst:
                                    dst.write(src.read())
                            
                            backup_info[path_key] = str(new_path)
                
                # Add to metadata
                self.metadata["backups"].append(backup_info)
                self._save_metadata()
            
            st.success(f"✅ Backup imported successfully as {backup_info['id']}")
            return True
            
        except Exception as e:
            st.error(f"Error importing backup from {import_path}: {e}")
            return False


# Global backup manager instance
backup_manager = BackupManager()