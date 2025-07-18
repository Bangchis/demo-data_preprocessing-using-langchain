"""
Backup UI Components for Data Preprocessing MVP
Provides user interface for backup management and restoration.
"""

import streamlit as st
import pandas as pd
import datetime
from typing import Dict, List, Optional
from pathlib import Path
import json
import tempfile
import os

from src.core.backup_manager import backup_manager
from src.core.utils import clean_dataframe_for_display


def render_backup_control_panel():
    """Render the main backup control panel"""
    st.subheader("💾 Backup Management")
    
    # Get backup statistics
    stats = backup_manager.get_backup_statistics()
    
    # Display backup status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Backups", stats["total_backups"])
    
    with col2:
        st.metric("Manual Backups", stats["manual_backups"])
    
    with col3:
        st.metric("Auto Backups", stats["automatic_backups"])
    
    with col4:
        st.metric("Storage Used", f"{stats['total_size_mb']:.1f} MB")
    
    # Create tabs for different backup operations
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Create Backup", "🔄 Restore Backup", "📋 Backup History", "⚙️ Settings"])
    
    with tab1:
        render_create_backup_tab()
    
    with tab2:
        render_restore_backup_tab()
    
    with tab3:
        render_backup_history_tab()
    
    with tab4:
        render_backup_settings_tab()


def render_create_backup_tab():
    """Render the create backup tab"""
    st.subheader("Create New Backup")
    
    # Check if there's data to backup
    if not hasattr(st.session_state, 'df') or st.session_state.df is None:
        st.warning("⚠️ No data available to backup. Please upload a dataset first.")
        return
    
    # Manual backup form
    with st.form("create_backup_form"):
        backup_name = st.text_input(
            "Backup Name",
            placeholder="Enter a descriptive name for this backup",
            help="Choose a name that describes the current state of your data"
        )
        
        backup_description = st.text_area(
            "Description (Optional)",
            placeholder="Describe what this backup contains or why you're creating it",
            height=100
        )
        
        # Show current data info
        st.info(f"📊 Current Data: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
        
        # Backup options
        col1, col2 = st.columns(2)
        
        with col1:
            include_session = st.checkbox(
                "Include Session State",
                value=True,
                help="Include chat history, execution log, and other session data"
            )
        
        with col2:
            include_original = st.checkbox(
                "Include Original Data",
                value=True,
                help="Include the original DataFrame with proper data types"
            )
        
        submitted = st.form_submit_button("🔄 Create Backup", type="primary")
        
        if submitted:
            if not backup_name.strip():
                st.error("❌ Please enter a backup name")
            else:
                with st.spinner("Creating backup..."):
                    backup_id = backup_manager.create_manual_backup(
                        name=backup_name.strip(),
                        description=backup_description.strip()
                    )
                    
                    if backup_id:
                        st.success(f"✅ Backup created successfully! ID: {backup_id}")
                        st.balloons()
                        # Refresh the page to show updated stats
                        st.rerun()
    
    # Quick backup button
    st.markdown("---")
    st.subheader("Quick Backup")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Create a quick backup with timestamp as name:")
    
    with col2:
        if st.button("⚡ Quick Backup"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"Quick_Backup_{timestamp}"
            
            with st.spinner("Creating quick backup..."):
                backup_id = backup_manager.create_manual_backup(
                    name=backup_name,
                    description="Quick backup created automatically"
                )
                
                if backup_id:
                    st.success(f"✅ Quick backup created: {backup_id}")
                    st.rerun()


def render_restore_backup_tab():
    """Render the restore backup tab"""
    st.subheader("Restore from Backup")
    
    backups = backup_manager.get_backup_list()
    
    if not backups:
        st.info("📭 No backups available. Create a backup first.")
        return
    
    # Sort backups by timestamp (newest first)
    backups_sorted = sorted(backups, key=lambda x: x["timestamp"], reverse=True)
    
    # Backup selection
    backup_options = []
    for backup in backups_sorted:
        backup_type = "🔄 Auto" if backup["type"] == "automatic" else "📝 Manual"
        name = backup.get("name", backup.get("operation", "Unknown"))
        timestamp = backup["timestamp"]
        backup_options.append(f"{backup_type} | {name} | {timestamp}")
    
    selected_backup_display = st.selectbox(
        "Select Backup to Restore",
        options=backup_options,
        help="Choose a backup to restore your data and session state"
    )
    
    if selected_backup_display:
        # Get the selected backup
        selected_index = backup_options.index(selected_backup_display)
        selected_backup = backups_sorted[selected_index]
        
        # Display backup details
        st.subheader("Backup Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**ID:** {selected_backup['id']}")
            st.write(f"**Type:** {selected_backup['type'].title()}")
            st.write(f"**Created:** {selected_backup['timestamp']}")
            
            if selected_backup['type'] == 'manual':
                st.write(f"**Name:** {selected_backup.get('name', 'N/A')}")
                if selected_backup.get('description'):
                    st.write(f"**Description:** {selected_backup['description']}")
            else:
                st.write(f"**Operation:** {selected_backup.get('operation', 'N/A')}")
        
        with col2:
            if 'dataframe_shape' in selected_backup and selected_backup['dataframe_shape']:
                shape = selected_backup['dataframe_shape']
                st.write(f"**Data Shape:** {shape[0]} rows × {shape[1]} columns")
            
            # Show what will be restored
            st.write("**Will Restore:**")
            restore_items = []
            if 'dataframe_path' in selected_backup:
                restore_items.append("📊 Main DataFrame")
            if 'dataframe_original_path' in selected_backup:
                restore_items.append("📋 Original DataFrame")
            if 'session_path' in selected_backup:
                restore_items.append("🔄 Session State")
            
            for item in restore_items:
                st.write(f"- {item}")
        
        # Restoration warning
        st.warning(
            "⚠️ **Warning:** Restoring will overwrite your current data and session state. "
            "Consider creating a backup of your current state first."
        )
        
        # Restore button
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 Restore Backup", type="primary"):
                with st.spinner("Restoring backup..."):
                    success = backup_manager.restore_backup(selected_backup['id'])
                    
                    if success:
                        st.success("✅ Backup restored successfully!")
                        st.info("🔄 Please refresh the page to see the restored data.")
                        st.rerun()
        
        with col2:
            # Export backup option
            if st.button("📤 Export Backup"):
                with st.spinner("Exporting backup..."):
                    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                        export_path = tmp.name
                    
                    success = backup_manager.export_backup(selected_backup['id'], export_path)
                    
                    if success:
                        # Read the exported file
                        with open(export_path, 'rb') as f:
                            backup_data = f.read()
                        
                        # Clean up temp file
                        os.unlink(export_path)
                        
                        # Provide download
                        st.download_button(
                            label="💾 Download Backup",
                            data=backup_data,
                            file_name=f"{selected_backup['id']}.zip",
                            mime="application/zip"
                        )


def render_backup_history_tab():
    """Render the backup history tab"""
    st.subheader("Backup History")
    
    backups = backup_manager.get_backup_list()
    
    if not backups:
        st.info("📭 No backup history available.")
        return
    
    # Sort backups by timestamp (newest first)
    backups_sorted = sorted(backups, key=lambda x: x["timestamp"], reverse=True)
    
    # Display backups in a table format
    for i, backup in enumerate(backups_sorted):
        with st.expander(f"{'🔄' if backup['type'] == 'automatic' else '📝'} {backup.get('name', backup.get('operation', 'Backup'))} - {backup['timestamp']}"):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**ID:** {backup['id']}")
                st.write(f"**Type:** {backup['type'].title()}")
                st.write(f"**Created:** {backup['timestamp']}")
                
                if backup['type'] == 'manual':
                    if backup.get('name'):
                        st.write(f"**Name:** {backup['name']}")
                    if backup.get('description'):
                        st.write(f"**Description:** {backup['description']}")
                else:
                    st.write(f"**Operation:** {backup.get('operation', 'N/A')}")
            
            with col2:
                if 'dataframe_shape' in backup and backup['dataframe_shape']:
                    shape = backup['dataframe_shape']
                    st.write(f"**Data Shape:** {shape[0]} rows × {shape[1]} columns")
                
                # Show backup components
                components = []
                if 'dataframe_path' in backup:
                    components.append("📊 Main DataFrame")
                if 'dataframe_original_path' in backup:
                    components.append("📋 Original DataFrame")
                if 'session_path' in backup:
                    components.append("🔄 Session State")
                
                if components:
                    st.write("**Components:**")
                    for component in components:
                        st.write(f"- {component}")
            
            with col3:
                # Action buttons
                if st.button("🔄 Restore", key=f"restore_{backup['id']}"):
                    with st.spinner("Restoring..."):
                        success = backup_manager.restore_backup(backup['id'])
                        if success:
                            st.rerun()
                
                if st.button("🗑️ Delete", key=f"delete_{backup['id']}"):
                    if st.session_state.get(f"confirm_delete_{backup['id']}", False):
                        with st.spinner("Deleting..."):
                            success = backup_manager.delete_backup(backup['id'])
                            if success:
                                st.rerun()
                    else:
                        st.session_state[f"confirm_delete_{backup['id']}"] = True
                        st.warning("Click again to confirm deletion")
                        st.rerun()
    


def render_backup_settings_tab():
    """Render the backup settings tab"""
    st.subheader("Backup Settings")
    
    # Auto backup settings
    st.write("### Automatic Backup Settings")
    
    current_status = st.session_state.backup_status.get("auto_backup_enabled", True)
    
    auto_backup_enabled = st.checkbox(
        "Enable automatic backups",
        value=current_status,
        help="Automatically create backups before significant operations"
    )
    
    if auto_backup_enabled != current_status:
        st.session_state.backup_status["auto_backup_enabled"] = auto_backup_enabled
        if auto_backup_enabled:
            st.success("✅ Automatic backups enabled")
        else:
            st.warning("⚠️ Automatic backups disabled")
    
    # Backup directory info
    st.write("### Storage Information")
    
    backup_dir = backup_manager.backup_dir
    st.write(f"**Backup Directory:** `{backup_dir}`")
    
    # Storage statistics
    stats = backup_manager.get_backup_statistics()
    st.write(f"**Total Storage Used:** {stats['total_size_mb']:.1f} MB")
    
    # Import/Export settings
    st.write("### Import/Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Import Backup**")
        uploaded_file = st.file_uploader(
            "Choose backup file",
            type=['zip'],
            help="Upload a previously exported backup file"
        )
        
        if uploaded_file is not None:
            with st.spinner("Importing backup..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    success = backup_manager.import_backup(tmp_path)
                    if success:
                        st.success("✅ Backup imported successfully!")
                        st.rerun()
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)
    
    with col2:
        st.write("**Export All Settings**")
        if st.button("📤 Export Configuration"):
            # Export backup configuration
            config = {
                "backup_settings": st.session_state.backup_status,
                "backup_statistics": stats,
                "export_timestamp": datetime.datetime.now().isoformat()
            }
            
            config_json = json.dumps(config, indent=2)
            
            st.download_button(
                label="💾 Download Configuration",
                data=config_json,
                file_name="backup_configuration.json",
                mime="application/json"
            )
    
    # Dangerous Operations Section
    st.markdown("---")
    st.write("### 🚨 Dangerous Operations")
    st.warning("⚠️ **These operations are irreversible and will permanently delete data!**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Delete All Backups**")
        st.write("This will permanently delete ALL backups in the system.")
        
        # Show current backup count
        if stats['total_backups'] > 0:
            st.write(f"📊 **{stats['total_backups']} backups** will be deleted")
            st.write(f"💾 **{stats['total_size_mb']:.1f} MB** will be freed")
        else:
            st.write("📭 No backups to delete")
    
    with col2:
        # Delete all backups button with confirmation
        if stats['total_backups'] > 0:
            if st.session_state.get("confirm_delete_all_backups", False):
                st.error("⚠️ **FINAL CONFIRMATION**")
                st.write("Click again to permanently delete ALL backups!")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if st.button("🗑️ DELETE ALL", type="primary"):
                        with st.spinner("Deleting all backups..."):
                            result = backup_manager.delete_all_backups()
                            
                            if result.get("success", False):
                                st.success(f"✅ Deleted {result['deleted_count']} backups successfully!")
                                st.balloons()
                            else:
                                st.error(f"❌ Failed to delete all backups: {result.get('error', 'Unknown error')}")
                            
                            # Reset confirmation state
                            st.session_state.confirm_delete_all_backups = False
                            st.rerun()
                
                with col_b:
                    if st.button("❌ Cancel"):
                        st.session_state.confirm_delete_all_backups = False
                        st.rerun()
            else:
                if st.button("🗑️ Delete All Backups"):
                    st.session_state.confirm_delete_all_backups = True
                    st.rerun()
        else:
            st.write("🚫 No backups to delete")


def render_backup_status_indicator():
    """Render a compact backup status indicator for the sidebar"""
    if not hasattr(st.session_state, 'backup_status'):
        return
    
    stats = backup_manager.get_backup_statistics()
    
    # Compact status display
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Backup Status")
    
    # Status indicators
    backup_enabled = st.session_state.backup_status.get("auto_backup_enabled", True)
    status_icon = "🟢" if backup_enabled else "🔴"
    status_text = "Enabled" if backup_enabled else "Disabled"
    
    st.sidebar.write(f"{status_icon} Auto Backup: {status_text}")
    st.sidebar.write(f"📊 Total Backups: {stats['total_backups']}")
    st.sidebar.write(f"💾 Storage: {stats['total_size_mb']:.1f} MB")
    
    # Last backup info
    if stats.get('last_backup'):
        st.sidebar.write(f"🕐 Last Backup: {stats['last_backup']}")
    
    # Quick actions
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("⚡ Quick Backup", key="sidebar_quick_backup"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"Quick_{timestamp}"
            
            with st.spinner("Creating backup..."):
                backup_id = backup_manager.create_manual_backup(
                    name=backup_name,
                    description="Quick backup from sidebar"
                )
                
                if backup_id:
                    st.sidebar.success("✅ Backup created!")
    
    with col2:
        if st.button("🔄 Restore", key="sidebar_restore"):
            st.session_state.show_backup_panel = True
            st.rerun()


def render_backup_notification():
    """Render backup notifications and alerts"""
    if not hasattr(st.session_state, 'backup_status'):
        return
    
    # Check for important notifications
    stats = backup_manager.get_backup_statistics()
    
    # Storage warning
    if stats['total_size_mb'] > 100:  # 100MB threshold
        st.warning(f"⚠️ **Storage Warning:** Backup storage is using {stats['total_size_mb']:.1f} MB. Consider cleaning up old backups.")
    
    # No recent backups warning
    if stats['total_backups'] == 0:
        st.info("💡 **Tip:** Create your first backup to protect your data against accidental loss.")
    
    # Auto backup disabled warning
    if not st.session_state.backup_status.get("auto_backup_enabled", True):
        st.warning("⚠️ **Automatic backups are disabled.** Enable them in backup settings for better data protection.")