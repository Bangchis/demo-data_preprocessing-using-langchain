#!/usr/bin/env python3
"""
Test script for backup system with ReAct agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.tools.backup_tools import (
    create_manual_backup_tool,
    list_available_backups,
    restore_backup_tool,
    delete_backup_tool,
    quick_backup_tool,
    get_backup_statistics
)

def test_backup_id_formatting():
    """Test backup ID formatting and parsing"""
    print("🧪 Testing backup ID formatting...")
    
    # Test with backticks (simulating agent input)
    test_cases = [
        "`manual_20250717_210545_test`",
        "manual_20250717_210545_test",
        "  `manual_20250717_210545_test`  ",
        '"`manual_20250717_210545_test`"',
        "'`manual_20250717_210545_test`'"
    ]
    
    for test_id in test_cases:
        print(f"   Input: {repr(test_id)}")
        
        # Clean the ID like our tools do
        cleaned_id = test_id.strip().strip('`').strip('"').strip("'")
        print(f"   Cleaned: {repr(cleaned_id)}")
        
        # Test restore with this ID (will fail since backup doesn't exist, but we can see the error)
        result = restore_backup_tool(test_id)
        print(f"   Result: {result}")
        print()

def test_backup_workflow():
    """Test complete backup workflow"""
    print("🧪 Testing backup workflow...")
    
    # Test creating a backup
    print("1. Creating a manual backup...")
    result = create_manual_backup_tool("test_backup | Test backup for agent")
    print(f"   Result: {result}")
    print()
    
    # Test listing backups
    print("2. Listing available backups...")
    result = list_available_backups("5")
    print(f"   Result: {result}")
    print()
    
    # Test getting statistics
    print("3. Getting backup statistics...")
    result = get_backup_statistics("")
    print(f"   Result: {result}")
    print()
    
    # Test quick backup
    print("4. Creating quick backup...")
    result = quick_backup_tool("")
    print(f"   Result: {result}")
    print()

def test_input_sanitization():
    """Test input sanitization for all tools"""
    print("🧪 Testing input sanitization...")
    
    # Test various malformed inputs
    malformed_inputs = [
        "`test_backup | description`",
        "test_backup | description",
        "  `test_backup | description`  ",
        '"test_backup | description"',
        "'test_backup | description'"
    ]
    
    for input_str in malformed_inputs:
        print(f"   Input: {repr(input_str)}")
        result = create_manual_backup_tool(input_str)
        print(f"   Result: {result[:100]}...")  # Truncate for readability
        print()

if __name__ == "__main__":
    print("🔧 Backup System Agent Test")
    print("=" * 50)
    
    test_backup_id_formatting()
    test_input_sanitization()
    
    # Only run workflow test if we have a real environment
    try:
        import streamlit as st
        # Mock basic session state for testing
        if not hasattr(st, 'session_state'):
            st.session_state = type('MockSessionState', (), {})()
        
        print("⚠️  Note: Full workflow test requires Streamlit environment")
        print("   Run this in the actual application to test complete functionality")
        
    except ImportError:
        print("⚠️  Streamlit not available - skipping workflow test")
    
    print("\n✅ Test completed!")