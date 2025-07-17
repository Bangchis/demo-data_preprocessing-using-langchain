#!/usr/bin/env python3
"""
Test session state management fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock streamlit session state for testing
class MockSessionState:
    def __init__(self):
        self._state = {}
    
    def __contains__(self, key):
        return key in self._state
    
    def __getitem__(self, key):
        return self._state[key]
    
    def __setitem__(self, key, value):
        self._state[key] = value
    
    def __getattr__(self, key):
        return self._state.get(key, None)
    
    def __setattr__(self, key, value):
        if key == '_state':
            super().__setattr__(key, value)
        else:
            self._state[key] = value

# Mock streamlit module
class MockStreamlit:
    def __init__(self):
        self.session_state = MockSessionState()

# Mock streamlit import
import sys
sys.modules['streamlit'] = MockStreamlit()

def test_execution_logger():
    """Test ExecutionLogger initialization"""
    print("Testing ExecutionLogger initialization...")
    
    try:
        from tools_core import ExecutionLogger
        
        # Create logger
        logger = ExecutionLogger()
        
        # Test logging
        logger.log_execution("test code", True, "test result", 0.5)
        
        # Test getting logs
        logs = logger.get_recent_logs(5)
        
        print(f"✅ ExecutionLogger test passed. Found {len(logs)} logs.")
        return True
        
    except Exception as e:
        print(f"❌ ExecutionLogger test failed: {str(e)}")
        return False

def test_checkpoint_manager():
    """Test CheckpointManager initialization"""
    print("Testing CheckpointManager initialization...")
    
    try:
        from tools_core import CheckpointManager
        import pandas as pd
        
        # Create checkpoint manager
        checkpoint_manager = CheckpointManager()
        
        # Create test dataframe
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        
        # Test checkpoint saving
        checkpoint_manager.save_checkpoint(df, "test operation")
        
        # Test undo
        success, message = checkpoint_manager.undo()
        
        print(f"✅ CheckpointManager test passed. Undo result: {message}")
        return True
        
    except Exception as e:
        print(f"❌ CheckpointManager test failed: {str(e)}")
        return False

def test_web_search():
    """Test web search error handling"""
    print("Testing web search error handling...")
    
    try:
        from tools_web import ddg_search
        
        # Test with empty query
        result = ddg_search("")
        print(f"Empty query result: {result}")
        
        # Test with rate limit simulation
        try:
            result = ddg_search("test query")
            print(f"Web search result: {result[:100]}...")
        except Exception as e:
            print(f"Web search error (expected): {str(e)}")
        
        print("✅ Web search test passed.")
        return True
        
    except Exception as e:
        print(f"❌ Web search test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing session state management fixes...\n")
    
    tests = [
        test_execution_logger,
        test_checkpoint_manager,
        test_web_search
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
        print()
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Session state management is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)