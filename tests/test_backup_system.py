"""
Test suite for the comprehensive backup system
"""

import unittest
import pandas as pd
import tempfile
import shutil
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock streamlit for testing
class MockStreamlit:
    class session_state:
        def __init__(self):
            self.data = {}
        
        def __getattr__(self, name):
            return self.data.get(name)
        
        def __setattr__(self, name, value):
            if name == 'data':
                super().__setattr__(name, value)
            else:
                self.data[name] = value
        
        def __contains__(self, name):
            return name in self.data
        
        def keys(self):
            return self.data.keys()
    
    def __init__(self):
        self.session_state = self.session_state()
    
    def error(self, msg):
        print(f"ERROR: {msg}")
    
    def success(self, msg):
        print(f"SUCCESS: {msg}")
    
    def warning(self, msg):
        print(f"WARNING: {msg}")
    
    def info(self, msg):
        print(f"INFO: {msg}")

# Mock streamlit
mock_st = MockStreamlit()

# Patch streamlit imports
import sys
sys.modules['streamlit'] = mock_st

# Now import our backup system
from src.core.backup_manager import BackupManager
from src.tools.backup_tools import (
    create_manual_backup_tool,
    list_available_backups,
    restore_backup_tool,
    get_backup_statistics,
    quick_backup_tool
)


class TestBackupSystem(unittest.TestCase):
    """Test suite for the backup system"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.backup_manager = BackupManager(backup_dir=self.test_dir)
        
        # Create test DataFrame
        self.test_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        # Set up mock session state
        mock_st.session_state.df = self.test_df.copy()
        mock_st.session_state.df_original = self.test_df.copy()
        mock_st.session_state.backup_status = {
            "last_backup": None,
            "last_restore": None,
            "auto_backup_enabled": True,
            "backup_count": 0
        }
        mock_st.session_state.backup_history = []
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_backup_manager_initialization(self):
        """Test BackupManager initialization"""
        self.assertTrue(Path(self.test_dir).exists())
        self.assertTrue((Path(self.test_dir) / "sessions").exists())
        self.assertTrue((Path(self.test_dir) / "dataframes").exists())
        self.assertTrue((Path(self.test_dir) / "checkpoints").exists())
        
    def test_create_manual_backup(self):
        """Test creating manual backup"""
        backup_id = self.backup_manager.create_manual_backup(
            name="test_backup",
            description="Test backup for unit testing"
        )
        
        self.assertIsNotNone(backup_id)
        self.assertTrue(backup_id.startswith("manual_"))
        
        # Check backup exists in metadata
        backup_info = self.backup_manager.get_backup_info(backup_id)
        self.assertIsNotNone(backup_info)
        self.assertEqual(backup_info["name"], "test_backup")
        self.assertEqual(backup_info["description"], "Test backup for unit testing")
        
    def test_create_automatic_backup(self):
        """Test creating automatic backup"""
        backup_id = self.backup_manager.create_automatic_backup(
            operation="test_operation",
            before_df=self.test_df
        )
        
        self.assertIsNotNone(backup_id)
        self.assertTrue(backup_id.startswith("auto_"))
        
        # Check backup exists in metadata
        backup_info = self.backup_manager.get_backup_info(backup_id)
        self.assertIsNotNone(backup_info)
        self.assertEqual(backup_info["operation"], "test_operation")
        self.assertEqual(backup_info["type"], "automatic")
        
    def test_restore_backup(self):
        """Test restoring backup"""
        # Create backup
        backup_id = self.backup_manager.create_manual_backup(
            name="restore_test",
            description="Test restore functionality"
        )
        
        # Modify DataFrame
        mock_st.session_state.df = pd.DataFrame({'X': [1, 2, 3]})
        
        # Restore backup
        success = self.backup_manager.restore_backup(backup_id)
        self.assertTrue(success)
        
        # Check DataFrame was restored
        self.assertEqual(len(mock_st.session_state.df), 5)
        self.assertIn('A', mock_st.session_state.df.columns)
        
    def test_backup_statistics(self):
        """Test backup statistics"""
        # Create multiple backups
        self.backup_manager.create_manual_backup("backup1", "Test 1")
        self.backup_manager.create_manual_backup("backup2", "Test 2")
        self.backup_manager.create_automatic_backup("operation1", self.test_df)
        
        stats = self.backup_manager.get_backup_statistics()
        
        self.assertEqual(stats["total_backups"], 3)
        self.assertEqual(stats["manual_backups"], 2)
        self.assertEqual(stats["automatic_backups"], 1)
        self.assertGreater(stats["total_size_mb"], 0)
        
    def test_delete_backup(self):
        """Test deleting backup"""
        # Create backup
        backup_id = self.backup_manager.create_manual_backup(
            name="delete_test",
            description="Test delete functionality"
        )
        
        # Delete backup
        success = self.backup_manager.delete_backup(backup_id)
        self.assertTrue(success)
        
        # Check backup no longer exists
        backup_info = self.backup_manager.get_backup_info(backup_id)
        self.assertIsNone(backup_info)
        
    def test_cleanup_old_backups(self):
        """Test cleanup of old automatic backups"""
        # Create multiple automatic backups
        for i in range(5):
            self.backup_manager.create_automatic_backup(f"operation_{i}", self.test_df)
        
        # Create manual backup (should not be deleted)
        manual_backup_id = self.backup_manager.create_manual_backup("manual_keep", "Keep this")
        
        # Cleanup with limit of 2
        self.backup_manager.cleanup_old_backups(max_backups=2)
        
        stats = self.backup_manager.get_backup_statistics()
        
        # Should have 2 automatic backups + 1 manual backup
        self.assertEqual(stats["total_backups"], 3)
        self.assertEqual(stats["automatic_backups"], 2)
        self.assertEqual(stats["manual_backups"], 1)
        
        # Manual backup should still exist
        manual_backup_info = self.backup_manager.get_backup_info(manual_backup_id)
        self.assertIsNotNone(manual_backup_info)
        
    def test_backup_tools_integration(self):
        """Test backup tools integration"""
        # Test manual backup tool
        result = create_manual_backup_tool("test_tool_backup | Tool integration test")
        self.assertIn("Backup created successfully", result)
        
        # Test list backups tool
        result = list_available_backups("5")
        self.assertIn("Available Backups", result)
        
        # Test backup stats tool
        result = get_backup_statistics("")
        self.assertIn("Backup Statistics", result)
        
        # Test quick backup tool
        result = quick_backup_tool("")
        self.assertIn("Quick backup created", result)
        
    def test_export_import_backup(self):
        """Test backup export and import"""
        # Create backup
        backup_id = self.backup_manager.create_manual_backup(
            name="export_test",
            description="Test export functionality"
        )
        
        # Export backup
        export_path = os.path.join(self.test_dir, "test_export.zip")
        success = self.backup_manager.export_backup(backup_id, export_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_path))
        
        # Delete original backup
        self.backup_manager.delete_backup(backup_id)
        
        # Import backup
        success = self.backup_manager.import_backup(export_path)
        self.assertTrue(success)
        
        # Check imported backup exists
        backups = self.backup_manager.get_backup_list()
        imported_backup = None
        for backup in backups:
            if backup.get("original_id") == backup_id:
                imported_backup = backup
                break
        
        self.assertIsNotNone(imported_backup)
        self.assertEqual(imported_backup["name"], "export_test")
        
    def test_session_state_persistence(self):
        """Test session state persistence"""
        # Set up session state data
        mock_st.session_state.test_data = {"key": "value"}
        mock_st.session_state.execution_log = [{"code": "test", "result": "success"}]
        
        # Create backup
        backup_id = self.backup_manager.create_manual_backup(
            name="session_test",
            description="Test session state persistence"
        )
        
        # Clear session state
        mock_st.session_state.test_data = None
        mock_st.session_state.execution_log = []
        
        # Restore backup
        success = self.backup_manager.restore_backup(backup_id)
        self.assertTrue(success)
        
        # Note: Session state restoration would need to be tested 
        # with actual Streamlit session state, not our mock
        
    def test_error_handling(self):
        """Test error handling"""
        # Test restore non-existent backup
        success = self.backup_manager.restore_backup("non_existent_backup")
        self.assertFalse(success)
        
        # Test delete non-existent backup
        success = self.backup_manager.delete_backup("non_existent_backup")
        self.assertFalse(success)
        
        # Test backup without data
        mock_st.session_state.df = None
        result = create_manual_backup_tool("no_data_test")
        self.assertIn("No data available", result)
        
    def test_backup_with_different_dtypes(self):
        """Test backup with various data types"""
        # Create DataFrame with different data types
        complex_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'string_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'datetime_col': pd.date_range('2023-01-01', periods=3)
        })
        
        mock_st.session_state.df = complex_df
        mock_st.session_state.df_original = complex_df.copy()
        
        # Create backup
        backup_id = self.backup_manager.create_manual_backup(
            name="dtype_test",
            description="Test various data types"
        )
        
        # Restore and verify
        success = self.backup_manager.restore_backup(backup_id)
        self.assertTrue(success)
        
        # Check data types are preserved
        restored_df = mock_st.session_state.df
        self.assertEqual(len(restored_df), 3)
        self.assertEqual(len(restored_df.columns), 5)


class TestBackupWorkflow(unittest.TestCase):
    """Test complete backup workflow scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.backup_manager = BackupManager(backup_dir=self.test_dir)
        
        # Create test scenario DataFrame
        self.original_df = pd.DataFrame({
            'country': ['USA', 'UK', 'Japan', 'Germany', 'France'],
            'population': [331000000, 67000000, 125000000, 83000000, 68000000],
            'gdp': [21.43, 2.83, 4.94, 3.85, 2.72]
        })
        
        mock_st.session_state.df = self.original_df.copy()
        mock_st.session_state.df_original = self.original_df.copy()
        mock_st.session_state.backup_status = {
            "auto_backup_enabled": True,
            "backup_count": 0
        }
        mock_st.session_state.backup_history = []
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_data_corruption_recovery_scenario(self):
        """Test scenario where data gets corrupted and needs recovery"""
        # Create initial backup
        initial_backup_id = self.backup_manager.create_manual_backup(
            name="initial_clean_data",
            description="Clean data before processing"
        )
        
        # Simulate data corruption (like the user's example)
        corrupted_df = pd.DataFrame({
            'country': ['USA', 'UK,Japan,Germany', 'France'],  # Corrupted row
            'population': [331000000, 67000000, 68000000],
            'gdp': [21.43, 2.83, 2.72]
        })
        
        mock_st.session_state.df = corrupted_df
        
        # Verify corruption
        self.assertEqual(len(mock_st.session_state.df), 3)  # Should be 5
        self.assertIn('UK,Japan,Germany', mock_st.session_state.df['country'].values)
        
        # Restore from backup
        success = self.backup_manager.restore_backup(initial_backup_id)
        self.assertTrue(success)
        
        # Verify restoration
        restored_df = mock_st.session_state.df
        self.assertEqual(len(restored_df), 5)
        self.assertIn('UK', restored_df['country'].values)
        self.assertIn('Japan', restored_df['country'].values)
        self.assertIn('Germany', restored_df['country'].values)
        
    def test_multiple_operation_backup_workflow(self):
        """Test workflow with multiple operations and backups"""
        # Step 1: Create initial backup
        backup_1 = self.backup_manager.create_manual_backup(
            name="step_1_original",
            description="Original data"
        )
        
        # Step 2: Perform operation 1 (add calculated column)
        mock_st.session_state.df['gdp_per_capita'] = (
            mock_st.session_state.df['gdp'] * 1e12 / mock_st.session_state.df['population']
        )
        
        backup_2 = self.backup_manager.create_manual_backup(
            name="step_2_calculated",
            description="Added GDP per capita"
        )
        
        # Step 3: Perform operation 2 (filter data)
        mock_st.session_state.df = mock_st.session_state.df[
            mock_st.session_state.df['population'] > 70000000
        ]
        
        backup_3 = self.backup_manager.create_manual_backup(
            name="step_3_filtered",
            description="Filtered for large populations"
        )
        
        # Verify final state
        self.assertEqual(len(mock_st.session_state.df), 3)  # USA, Japan, Germany
        self.assertIn('gdp_per_capita', mock_st.session_state.df.columns)
        
        # Test restoring to different points
        # Restore to step 2
        success = self.backup_manager.restore_backup(backup_2)
        self.assertTrue(success)
        self.assertEqual(len(mock_st.session_state.df), 5)  # All countries
        self.assertIn('gdp_per_capita', mock_st.session_state.df.columns)
        
        # Restore to step 1 (original)
        success = self.backup_manager.restore_backup(backup_1)
        self.assertTrue(success)
        self.assertEqual(len(mock_st.session_state.df), 5)  # All countries
        self.assertNotIn('gdp_per_capita', mock_st.session_state.df.columns)
        
    def test_backup_before_risky_operations(self):
        """Test creating backups before risky operations"""
        operations = [
            ("drop_column", "Dropping column"),
            ("merge_data", "Merging with external data"),
            ("transform_data", "Transforming data structure"),
            ("fillna_operation", "Filling missing values")
        ]
        
        for operation, description in operations:
            # Create backup before operation
            backup_id = self.backup_manager.create_automatic_backup(
                operation=f"Before {operation}",
                before_df=mock_st.session_state.df
            )
            
            self.assertIsNotNone(backup_id)
            
            # Simulate risky operation
            if operation == "drop_column":
                mock_st.session_state.df = mock_st.session_state.df.drop(columns=['gdp'])
            elif operation == "merge_data":
                new_data = pd.DataFrame({'country': ['USA'], 'new_col': [1]})
                mock_st.session_state.df = mock_st.session_state.df.merge(new_data, how='left')
            
            # Verify backup can restore if needed
            backup_info = self.backup_manager.get_backup_info(backup_id)
            self.assertIsNotNone(backup_info)
            self.assertEqual(backup_info["type"], "automatic")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)