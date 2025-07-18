"""
Backup Tools for ReAct Agent
Provides backup-related tools for the ReAct agent system.
"""

import streamlit as st
import pandas as pd
import datetime
from typing import Dict, List, Optional
from langchain.tools import Tool

from src.core.backup_manager import backup_manager


def create_manual_backup_tool(name_and_description: str) -> str:
    """
    Create a manual backup with name and description
    
    Args:
        name_and_description: Format "backup_name | description" or just "backup_name"
        
    Returns:
        str: Result message
    """
    try:
        # Clean and parse input - remove backticks and quotes
        name_and_description = name_and_description.strip().strip('`').strip('"').strip("'")
        parts = name_and_description.split("|")
        backup_name = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else ""
        
        if not backup_name:
            return "❌ Backup name is required"
        
        # Check if data is available
        if not hasattr(st.session_state, 'df') or st.session_state.df is None:
            return "❌ No data available to backup. Please upload a dataset first."
        
        # Create backup
        backup_id = backup_manager.create_manual_backup(
            name=backup_name,
            description=description
        )
        
        if backup_id:
            return f"✅ Manual backup created successfully!\n📁 Backup ID: {backup_id}\n📊 Data: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns"
        else:
            return "❌ Failed to create backup"
            
    except Exception as e:
        return f"❌ Error creating backup: {str(e)}"


def list_available_backups(limit: str = "10") -> str:
    """
    List available backups
    
    Args:
        limit: Maximum number of backups to list (default: 10)
        
    Returns:
        str: Formatted list of backups
    """
    try:
        # Clean input
        limit = limit.strip().strip('`').strip('"').strip("'")
        max_backups = int(limit) if limit.isdigit() else 10
        backups = backup_manager.get_backup_list()
        
        if not backups:
            return "📭 No backups available"
        
        # Sort by timestamp (newest first)
        backups_sorted = sorted(backups, key=lambda x: x["timestamp"], reverse=True)
        backups_to_show = backups_sorted[:max_backups]
        
        result = f"📋 **Available Backups ({len(backups_to_show)}/{len(backups)}):**\n\n"
        
        for i, backup in enumerate(backups_to_show, 1):
            backup_type = "🔄 Auto" if backup["type"] == "automatic" else "📝 Manual"
            name = backup.get("name", backup.get("operation", "Unknown"))
            timestamp = backup["timestamp"]
            shape_info = ""
            
            if backup.get("dataframe_shape"):
                shape = backup["dataframe_shape"]
                shape_info = f" | {shape[0]} rows × {shape[1]} cols"
            
            result += f"**{i}. {backup_type}** | {name} | {timestamp}{shape_info}\n"
            result += f"   ID: {backup['id']}\n"
            
            if backup["type"] == "manual" and backup.get("description"):
                result += f"   📝 {backup['description']}\n"
            
            result += "\n"
        
        if len(backups) > max_backups:
            result += f"... and {len(backups) - max_backups} more backups\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error listing backups: {str(e)}"


def restore_backup_tool(backup_id: str) -> str:
    """
    Restore a backup by ID with full DataFrame synchronization
    
    Args:
        backup_id: Unique identifier of the backup to restore
        
    Returns:
        str: Result message
    """
    try:
        # Clean the backup ID - remove backticks and whitespace
        backup_id = backup_id.strip().strip('`').strip()
        if not backup_id:
            return "❌ Backup ID is required"
        
        # Get backup info first
        backup_info = backup_manager.get_backup_info(backup_id)
        if not backup_info:
            return f"❌ Backup '{backup_id}' not found"
        
        # Store current shape for comparison
        current_shape = None
        if hasattr(st.session_state, 'df') and st.session_state.df is not None:
            current_shape = st.session_state.df.shape
        
        # Restore the backup
        success = backup_manager.restore_backup(backup_id)
        
        if success:
            # Get restored shape
            restored_shape = None
            if hasattr(st.session_state, 'df') and st.session_state.df is not None:
                restored_shape = st.session_state.df.shape
            
            # Build detailed success message
            result_msg = f"✅ Backup '{backup_id}' restored successfully!\n"
            
            # Add shape comparison
            if backup_info.get("dataframe_shape"):
                backup_shape = backup_info["dataframe_shape"]
                result_msg += f"📊 Restored Data: {backup_shape[0]} rows × {backup_shape[1]} columns\n"
                
                if current_shape and restored_shape:
                    result_msg += f"📈 Shape Change: {current_shape} → {restored_shape}\n"
            
            # Add synchronization info
            result_msg += "🔄 All DataFrame versions synchronized (df, df_original, display)\n"
            result_msg += "🔄 Session state restored to backup point\n"
            result_msg += "🔄 Execution environment reset\n"
            result_msg += "✨ UI refreshed - Preview now shows restored data"
            
            return result_msg
        else:
            return f"❌ Failed to restore backup '{backup_id}'"
            
    except Exception as e:
        return f"❌ Error restoring backup: {str(e)}"


def get_backup_statistics(query: str = "") -> str:
    """
    Get backup statistics and status
    
    Args:
        query: Optional query parameter (not used)
        
    Returns:
        str: Formatted statistics
    """
    try:
        stats = backup_manager.get_backup_statistics()
        
        result = "📊 **Backup Statistics:**\n\n"
        result += f"📁 Total Backups: {stats['total_backups']}\n"
        result += f"📝 Manual Backups: {stats['manual_backups']}\n"
        result += f"🔄 Automatic Backups: {stats['automatic_backups']}\n"
        result += f"💾 Storage Used: {stats['total_size_mb']:.1f} MB\n"
        
        if stats.get('last_backup'):
            result += f"🕐 Last Backup: {stats['last_backup']}\n"
        
        if stats.get('last_restore'):
            result += f"🔄 Last Restore: {stats['last_restore']}\n"
        
        # Auto backup status
        if hasattr(st.session_state, 'backup_status') and st.session_state.backup_status:
            auto_backup_enabled = st.session_state.backup_status.get("auto_backup_enabled", True)
            status = "🟢 Enabled" if auto_backup_enabled else "🔴 Disabled"
            result += f"⚙️ Auto Backup: {status}\n"
        else:
            result += f"⚙️ Auto Backup: 🔄 Initializing...\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error getting backup statistics: {str(e)}"


def delete_backup_tool(backup_id: str) -> str:
    """
    Delete a backup by ID
    
    Args:
        backup_id: Unique identifier of the backup to delete
        
    Returns:
        str: Result message
    """
    try:
        # Clean the backup ID - remove backticks and whitespace
        backup_id = backup_id.strip().strip('`').strip()
        if not backup_id:
            return "❌ Backup ID is required"
        
        # Get backup info first
        backup_info = backup_manager.get_backup_info(backup_id)
        if not backup_info:
            return f"❌ Backup '{backup_id}' not found"
        
        # Delete the backup
        success = backup_manager.delete_backup(backup_id)
        
        if success:
            backup_name = backup_info.get("name", backup_info.get("operation", "Unknown"))
            return f"✅ Backup '{backup_name}' ({backup_id}) deleted successfully!"
        else:
            return f"❌ Failed to delete backup '{backup_id}'"
            
    except Exception as e:
        return f"❌ Error deleting backup: {str(e)}"


def quick_backup_tool(query: str = "") -> str:
    """
    Create a quick backup with timestamp name
    
    Args:
        query: Optional query parameter (not used)
        
    Returns:
        str: Result message
    """
    try:
        # Check if data is available
        if not hasattr(st.session_state, 'df') or st.session_state.df is None:
            return "❌ No data available to backup. Please upload a dataset first."
        
        # Generate timestamp name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"Quick_Backup_{timestamp}"
        
        # Create backup
        backup_id = backup_manager.create_manual_backup(
            name=backup_name,
            description="Quick backup created by ReAct agent"
        )
        
        if backup_id:
            return f"✅ Quick backup created successfully!\n📁 Backup ID: {backup_id}\n📊 Data: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns"
        else:
            return "❌ Failed to create quick backup"
            
    except Exception as e:
        return f"❌ Error creating quick backup: {str(e)}"




def cleanup_session_backups_tool(query: str = "") -> str:
    """
    Clean up all backups from current session
    
    Args:
        query: Optional query parameter (not used)
        
    Returns:
        str: Result message
    """
    try:
        # Check if session ID exists
        if not hasattr(st.session_state, 'session_id'):
            return "❌ No session ID found. Session cleanup not available."
        
        session_id = st.session_state.session_id
        
        # Get current session stats
        stats_before = backup_manager.get_backup_statistics()
        total_before = stats_before['total_backups']
        
        # Clean up current session
        deleted_count = backup_manager.cleanup_current_session()
        
        # Get updated stats
        stats_after = backup_manager.get_backup_statistics()
        total_after = stats_after['total_backups']
        
        result_msg = f"🧹 **Session Cleanup Completed**\n"
        result_msg += f"📋 Session ID: {session_id}\n"
        result_msg += f"🗑️ Deleted backups: {deleted_count}\n"
        result_msg += f"📊 Total backups: {total_before} → {total_after}\n"
        
        if deleted_count > 0:
            result_msg += f"✅ Successfully cleaned up {deleted_count} backups from current session!"
        else:
            result_msg += "ℹ️ No backups found in current session to clean up."
            
        return result_msg
        
    except Exception as e:
        return f"❌ Error cleaning up session backups: {str(e)}"


def delete_all_backups_tool(confirmation: str = "") -> str:
    """
    Delete ALL backups in the system (WARNING: This is irreversible!)
    
    Args:
        confirmation: Must be "CONFIRM DELETE ALL" to proceed
        
    Returns:
        str: Result message
    """
    try:
        # Safety check - require exact confirmation
        if confirmation.strip().upper() != "CONFIRM DELETE ALL":
            return """❌ **SAFETY CHECK FAILED**
            
⚠️ **This action will DELETE ALL BACKUPS in the system!**

To proceed, you must provide the exact confirmation: "CONFIRM DELETE ALL"

📋 **What will be deleted:**
- All manual backups from all sessions
- All automatic backups from all sessions
- All backup metadata and files
- This action is IRREVERSIBLE

💡 **Alternative options:**
- Use SessionCleanup to delete only current session backups
- Use DeleteBackup to delete individual backups

Are you sure you want to delete ALL backups? If so, use: "CONFIRM DELETE ALL"
"""
        
        # Get current statistics
        stats_before = backup_manager.get_backup_statistics()
        total_before = stats_before['total_backups']
        
        if total_before == 0:
            return "ℹ️ No backups found in the system to delete."
        
        # Delete all backups
        result = backup_manager.delete_all_backups()
        
        if result.get("success", False):
            deleted_count = result["deleted_count"]
            total_before = result["total_before"]
            
            result_msg = f"🗑️ **COMPLETE SYSTEM CLEANUP**\n"
            result_msg += f"📊 Total backups deleted: {deleted_count}/{total_before}\n"
            result_msg += f"💾 All backup files and metadata removed\n"
            result_msg += f"🔄 System reset to clean state\n"
            result_msg += f"✅ **All backups successfully deleted!**\n"
            result_msg += f"⚠️ This action was irreversible - all backup data is permanently lost"
            
            return result_msg
        else:
            error_msg = result.get("error", "Unknown error")
            deleted_count = result.get("deleted_count", 0)
            total_before = result.get("total_before", 0)
            
            result_msg = f"❌ **PARTIAL CLEANUP COMPLETED**\n"
            result_msg += f"📊 Deleted: {deleted_count}/{total_before} backups\n"
            result_msg += f"⚠️ Some backups may still remain due to errors\n"
            result_msg += f"🔧 Error details: {error_msg}\n"
            result_msg += f"💡 Try running BackupStats to see remaining backups"
            
            return result_msg
            
    except Exception as e:
        return f"❌ Error deleting all backups: {str(e)}"


# Create LangChain tools
ManualBackupTool = Tool(
    name="ManualBackup",
    func=create_manual_backup_tool,
    description="Create a manual backup with name and description. Input format: 'backup_name | description' or just 'backup_name'. Use this before performing risky operations."
)

ListBackupsTool = Tool(
    name="ListBackups",
    func=list_available_backups,
    description="List available backups. Input: number of backups to show (default: 10). Shows backup ID, type, name, and timestamp."
)

RestoreBackupTool = Tool(
    name="RestoreBackup",
    func=restore_backup_tool,
    description="Restore a backup by ID with full DataFrame synchronization. Input: backup_id. This will overwrite current data, synchronize all DataFrame versions (df, df_original, display), reset execution environment, and refresh the UI to show restored data."
)

BackupStatsTool = Tool(
    name="BackupStats",
    func=get_backup_statistics,
    description="Get backup statistics including total backups, storage used, and last backup/restore info. No input required."
)

DeleteBackupTool = Tool(
    name="DeleteBackup",
    func=delete_backup_tool,
    description="Delete a backup by ID. Input: backup_id. Use with caution - this cannot be undone."
)

QuickBackupTool = Tool(
    name="QuickBackup",
    func=quick_backup_tool,
    description="Create a quick backup with timestamp name. No input required. Use this for emergency backups."
)


SessionCleanupTool = Tool(
    name="SessionCleanup",
    func=cleanup_session_backups_tool,
    description="Clean up all backups from current session. No input required. Use this to clean up session-specific backups when finishing work."
)

DeleteAllBackupsTool = Tool(
    name="DeleteAllBackups",
    func=delete_all_backups_tool,
    description="Delete ALL backups in the system (WARNING: Irreversible!). Input: 'CONFIRM DELETE ALL' to proceed. This will delete all manual and automatic backups from all sessions. Use with extreme caution."
)