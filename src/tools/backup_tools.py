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
    Restore a backup by ID
    
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
        
        # Restore the backup
        success = backup_manager.restore_backup(backup_id)
        
        if success:
            shape_info = ""
            if backup_info.get("dataframe_shape"):
                shape = backup_info["dataframe_shape"]
                shape_info = f"\n📊 Restored Data: {shape[0]} rows × {shape[1]} columns"
            
            return f"✅ Backup '{backup_id}' restored successfully!{shape_info}\n🔄 Session state has been restored to the backup point."
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


def cleanup_old_backups_tool(max_backups: str = "20") -> str:
    """
    Clean up old automatic backups
    
    Args:
        max_backups: Maximum number of automatic backups to keep (default: 20)
        
    Returns:
        str: Result message
    """
    try:
        # Clean input
        max_backups = max_backups.strip().strip('`').strip('"').strip("'")
        max_count = int(max_backups) if max_backups.isdigit() else 20
        
        # Get current backup count
        stats_before = backup_manager.get_backup_statistics()
        auto_backups_before = stats_before['automatic_backups']
        
        # Cleanup
        backup_manager.cleanup_old_backups(max_count)
        
        # Get new backup count
        stats_after = backup_manager.get_backup_statistics()
        auto_backups_after = stats_after['automatic_backups']
        
        cleaned_count = auto_backups_before - auto_backups_after
        
        if cleaned_count > 0:
            return f"✅ Cleaned up {cleaned_count} old automatic backups!\n📊 Remaining automatic backups: {auto_backups_after}"
        else:
            return f"ℹ️ No old backups to clean up. Current automatic backups: {auto_backups_after}"
            
    except Exception as e:
        return f"❌ Error cleaning up backups: {str(e)}"


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
    description="Restore a backup by ID. Input: backup_id. This will overwrite current data and session state with the backup."
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

CleanupBackupsTool = Tool(
    name="CleanupBackups",
    func=cleanup_old_backups_tool,
    description="Clean up old automatic backups. Input: maximum number of automatic backups to keep (default: 20)."
)