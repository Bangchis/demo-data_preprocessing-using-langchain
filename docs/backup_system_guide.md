# Comprehensive Backup System Guide

## Overview

The Data Preprocessing MVP now includes a comprehensive backup system that protects your data from loss and allows you to safely experiment with different data transformations. This guide explains how to use the backup system effectively.

## Key Features

### 🔄 Automatic Backups
- **Smart Detection**: Automatically creates backups before risky operations
- **Seamless Integration**: Works with the ReAct agent workflow
- **No Interruption**: Backups are created in the background

### 📝 Manual Backups
- **User Control**: Create backups at any time with custom names
- **Descriptive**: Add descriptions to remember what each backup contains
- **Quick Backup**: One-click backup creation for emergency situations

### 🔙 Easy Restoration
- **Point-in-Time Recovery**: Restore to any previous backup
- **Complete Restoration**: Includes both data and session state
- **Backup Browser**: Easy-to-use interface to browse and select backups

### 📊 Backup Management
- **Statistics**: Track backup usage and storage
- **Management**: Easy backup management and deletion
- **Export/Import**: Share backups between sessions or users

## Getting Started

### Accessing the Backup System

1. **Sidebar Status**: Check backup status in the sidebar
2. **Backup Panel**: Click "💾 Backup Management" to open the full control panel
3. **ReAct Agent**: Use backup tools directly in chat

### Creating Your First Backup

#### Method 1: Manual Backup (Recommended)
1. Open the backup control panel
2. Go to the "📝 Create Backup" tab
3. Enter a descriptive name (e.g., "Clean raw data")
4. Add a description explaining what the backup contains
5. Click "🔄 Create Backup"

#### Method 2: Quick Backup
1. Click "⚡ Quick Backup" in the sidebar
2. A backup with timestamp name will be created instantly

#### Method 3: Using ReAct Agent
Ask the agent to create a backup:
```
"Tạo backup với tên 'before_cleaning' trước khi tôi làm sạch dữ liệu"
```

## Understanding Backup Types

### 🔄 Automatic Backups
- **Created**: Before risky operations like drop, merge, transform
- **Naming**: Auto-generated with timestamp and operation description
- **Retention**: Managed manually through backup panel

### 📝 Manual Backups
- **Created**: When you explicitly create them
- **Naming**: Custom names you provide
- **Retention**: Kept indefinitely until manually deleted

## Best Practices

### 🛡️ Data Protection Strategy

1. **Before Major Operations**:
   - Always create a backup before dropping columns
   - Backup before merging datasets
   - Create checkpoint before complex transformations

2. **During Experimentation**:
   - Create named backups for different approaches
   - Use descriptive names like "approach_1_fillna" or "method_2_outliers"

3. **Regular Checkpoints**:
   - Create manual backups at key milestones
   - Use descriptions to document your reasoning

### 📋 Naming Conventions

**Good Examples**:
- `raw_data_uploaded` - Original data after upload
- `cleaned_missing_values` - After handling missing data
- `outliers_removed` - After outlier treatment
- `feature_engineering_complete` - After creating new features

**Poor Examples**:
- `backup1` - Not descriptive
- `test` - Unclear purpose
- `temp` - Temporary nature unclear

## Using the Backup Control Panel

### 📝 Create Backup Tab
- **Backup Name**: Enter descriptive name
- **Description**: Optional detailed description
- **Options**: Choose what to include in backup
- **Quick Backup**: One-click backup creation

### 🔄 Restore Backup Tab
- **Backup List**: Browse available backups
- **Backup Details**: View backup information
- **Restore**: One-click restoration
- **Export**: Export backup to file

### 📋 Backup History Tab
- **Timeline View**: See all backups chronologically
- **Expandable Details**: Click to see backup contents
- **Actions**: Restore or delete individual backups
- **Management**: Delete individual backups as needed

### ⚙️ Settings Tab
- **Auto Backup**: Enable/disable automatic backups
- **Storage Info**: View disk usage
- **Import/Export**: Backup file management

## ReAct Agent Integration

### Available Backup Tools

The ReAct agent has 8 backup-related tools:

1. **ManualBackup**: Create named backup
   ```
   "Tạo backup với tên 'experiment_1' và mô tả 'Testing different approach'"
   ```

2. **QuickBackup**: Create quick backup
   ```
   "Tạo backup nhanh trước khi tôi thực hiện thao tác này"
   ```

3. **ListBackups**: Show available backups
   ```
   "Hiển thị danh sách các backup hiện có"
   ```

4. **RestoreBackup**: Restore specific backup
   ```
   "Khôi phục backup có ID 'manual_20240117_142530_experiment_1'"
   ```

5. **BackupStats**: Show backup statistics
   ```
   "Hiển thị thống kê backup"
   ```

6. **DeleteBackup**: Delete specific backup
   ```
   "Xóa backup có ID 'auto_20240117_141200_abcd1234'"
   ```

7. **SessionCleanup**: Clean up backups from current session
   ```
   "Dọn dẹp tất cả backup trong session hiện tại"
   ```

8. **DeleteAllBackups**: Delete all backups in the system (use with caution)
   ```
   "Xóa tất cả backup trong hệ thống" (requires confirmation)
   ```

### Agent Recommendations

The ReAct agent is programmed to:
- **Suggest backups** before risky operations
- **Create automatic backups** for dangerous operations
- **Recommend restoration** if errors occur
- **Help with backup management** tasks

## Error Recovery Scenarios

### Scenario 1: Data Corruption
**Problem**: Agent executed incorrect code that corrupted data
**Solution**:
1. Use `ListBackups` to see available backups
2. Identify the backup before the problematic operation
3. Use `RestoreBackup` with the backup ID
4. Verify data integrity after restoration

### Scenario 2: Accidental Data Loss
**Problem**: Accidentally dropped important columns
**Solution**:
1. Check backup history in the control panel
2. Find the backup from before the drop operation
3. Restore the backup
4. Re-implement the intended operation correctly

### Scenario 3: Experiment Gone Wrong
**Problem**: Complex transformation produced unexpected results
**Solution**:
1. Restore to the checkpoint before the experiment
2. Analyze what went wrong
3. Try a different approach
4. Create a new backup when satisfied

## Storage Management

### Understanding Storage Usage

Backups are stored in compressed format:
- **DataFrames**: Compressed pickle files (.pkl.gz)
- **Session State**: Compressed session data
- **Metadata**: JSON files with backup information

### Managing Storage

1. **Check Usage**: View storage statistics in settings
2. **Delete Unnecessary Backups**: Use backup management interface
3. **Remove Experimental Backups**: Delete failed experiments
4. **Export Important Backups**: Save critical backups externally

### Storage Recommendations

- **Keep**: Important milestones and clean datasets
- **Delete**: Failed experiments and redundant backups
- **Export**: Final results and sharable states
- **Monitor**: Keep track of backup usage and storage

## Advanced Features

### Export and Import

#### Exporting Backups
1. Select backup in the history tab
2. Click "📤 Export Backup"
3. Download the .zip file
4. Share with team members or save externally

#### Importing Backups
1. Go to Settings tab
2. Upload a .zip backup file
3. Backup will be imported with new ID
4. All data and session state will be available

### Backup Metadata

Each backup contains:
- **DataFrame Data**: Your processed data
- **Session State**: Chat history, execution log
- **Metadata**: Creation time, description, data shape
- **Provenance**: What operation triggered the backup

### Integration with Other Systems

The backup system integrates with:
- **Checkpoint Manager**: Undo/redo functionality
- **Execution Logger**: Code execution history
- **Web Search Log**: Search history preservation
- **Chat Memory**: Conversation context

## Troubleshooting

### Common Issues

#### Backup Creation Fails
- **Cause**: Insufficient disk space or permissions
- **Solution**: Check storage usage, delete unnecessary backups

#### Restoration Doesn't Work
- **Cause**: Corrupted backup files
- **Solution**: Try different backup, check file integrity

#### Missing Backups
- **Cause**: Manual deletion or system errors
- **Solution**: Check deletion history, restore from export

#### Performance Issues
- **Cause**: Too many backups or large datasets
- **Solution**: Delete unnecessary backups, optimize storage

### Best Practices for Troubleshooting

1. **Check Backup Statistics**: Understand current usage
2. **Test Restoration**: Verify critical backups work
3. **Export Important Backups**: Create external copies
4. **Monitor Storage**: Keep an eye on disk usage

## FAQ

### Q: How often should I create backups?
**A**: Create backups before any major operation, after successful experiments, and at natural breakpoints in your workflow.

### Q: Do automatic backups impact performance?
**A**: Minimal impact. Backups are created in the background and use compression.

### Q: Can I disable automatic backups?
**A**: Yes, in the Settings tab. However, this is not recommended for safety.

### Q: What happens if I run out of disk space?
**A**: New backups will fail. Delete unnecessary backups or increase storage.

### Q: Can I backup multiple DataFrames?
**A**: Yes, the system backs up all DataFrames in your session.

### Q: How do I share backups with others?
**A**: Export backups to .zip files and share the files.

### Q: Can I restore just the data without session state?
**A**: Currently no, but this feature may be added in the future.

### Q: What happens to backups when I restart the application?
**A**: Backups are persisted to disk and available after restart.

## Conclusion

The backup system provides comprehensive protection for your data preprocessing work. By following the best practices in this guide, you can:

- **Work Confidently**: Experiment knowing you can always revert
- **Collaborate Effectively**: Share states with team members
- **Maintain Data Integrity**: Protect against accidental loss
- **Iterate Efficiently**: Try different approaches without fear

Remember: **A backup today saves hours of work tomorrow!**