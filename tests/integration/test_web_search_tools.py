#!/usr/bin/env python3
"""
Test script for Web Search Tools
"""

import os
import sys
from unittest.mock import MagicMock

# Mock streamlit for testing
class MockSessionState:
    def __init__(self):
        self.data = {}
        self.data.setdefault('web_search_log', [])
        self.data.setdefault('current_time', '2024-01-01 10:00:00')
    
    def __contains__(self, key):
        return key in self.data
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    def __getattr__(self, key):
        return self.data.get(key)
    
    def __setattr__(self, key, value):
        if key == 'data':
            super().__setattr__(key, value)
        else:
            self.data[key] = value
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def setdefault(self, key, default=None):
        return self.data.setdefault(key, default)

class MockStreamlit:
    def __init__(self):
        self.session_state = MockSessionState()

# Setup mock
mock_st = MockStreamlit()
sys.modules['streamlit'] = mock_st
sys.modules['langchain'] = MagicMock()
sys.modules['langchain.tools'] = MagicMock()

# Import the tools
from tools_web import (
    WebSearchTool, 
    PandasHelpTool, 
    DataScienceHelpTool, 
    ErrorSolutionTool, 
    SearchHistoryTool,
    ddg_search,
    search_pandas_help,
    search_data_science_help,
    search_error_solution,
    get_search_history
)

def test_web_search_tool():
    """Test WebSearchTool functionality"""
    print("🔍 Testing WebSearchTool...")
    
    # Test 1: Basic search
    print("\n1. Testing basic search...")
    query = "pandas dataframe merge"
    result = ddg_search(query)
    print(f"   Query: {query}")
    print(f"   Result length: {len(result)} characters")
    print(f"   Contains '🔍': {'🔍' in result}")
    print(f"   Contains 'pandas': {'pandas' in result.lower()}")
    
    if "❌" in result:
        print(f"   ❌ Error: {result[:100]}...")
    else:
        print(f"   ✅ Success: {result[:100]}...")
    
    # Test 2: Domain-specific search
    print("\n2. Testing domain-specific search...")
    query = "pandas pivot table | domains=pandas.pydata.org,stackoverflow.com"
    result = ddg_search(query)
    print(f"   Query: {query}")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result[:100]}...")
    else:
        print(f"   ✅ Success: {result[:100]}...")
    
    # Test 3: Empty query
    print("\n3. Testing empty query...")
    result = ddg_search("")
    print(f"   Expected error: {result}")
    
    return result

def test_pandas_help_tool():
    """Test PandasHelpTool functionality"""
    print("\n📊 Testing PandasHelpTool...")
    
    query = "pivot table"
    result = search_pandas_help(query)
    print(f"   Query: {query}")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result[:100]}...")
    else:
        print(f"   ✅ Success: {result[:100]}...")
    
    return result

def test_data_science_help_tool():
    """Test DataScienceHelpTool functionality"""
    print("\n🔬 Testing DataScienceHelpTool...")
    
    query = "machine learning"
    result = search_data_science_help(query)
    print(f"   Query: {query}")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result[:100]}...")
    else:
        print(f"   ✅ Success: {result[:100]}...")
    
    return result

def test_error_solution_tool():
    """Test ErrorSolutionTool functionality"""
    print("\n🚨 Testing ErrorSolutionTool...")
    
    query = "KeyError pandas"
    result = search_error_solution(query)
    print(f"   Query: {query}")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result[:100]}...")
    else:
        print(f"   ✅ Success: {result[:100]}...")
    
    return result

def test_search_history_tool():
    """Test SearchHistoryTool functionality"""
    print("\n📝 Testing SearchHistoryTool...")
    
    # Should show previous searches
    result = get_search_history("")
    print(f"   Result: {result}")
    
    # Check if log has entries
    search_log = mock_st.session_state.get("web_search_log", [])
    print(f"   Search log entries: {len(search_log)}")
    
    for i, entry in enumerate(search_log):
        print(f"   {i+1}. {entry['query']}")
    
    return result

def test_error_handling():
    """Test error handling capabilities"""
    print("\n🛠️ Testing Error Handling...")
    
    # Test with None input
    print("\n1. Testing None input...")
    result = ddg_search(None)
    print(f"   Result: {result}")
    
    # Test with empty string
    print("\n2. Testing empty string...")
    result = ddg_search("")
    print(f"   Result: {result}")
    
    # Test with very long query
    print("\n3. Testing very long query...")
    long_query = "a" * 1000
    result = ddg_search(long_query)
    print(f"   Result length: {len(result)} characters")
    print(f"   Contains error: {'❌' in result}")

def check_dependencies():
    """Check if required dependencies are available"""
    print("📦 Checking Dependencies...")
    
    try:
        from duckduckgo_search import DDGS
        print("   ✅ duckduckgo-search is available")
        return True
    except ImportError:
        print("   ❌ duckduckgo-search is NOT available")
        print("   Install with: pip install duckduckgo-search")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Web Search Tools")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Cannot run tests - missing dependencies")
        return
    
    # Test each tool
    try:
        web_result = test_web_search_tool()
        pandas_result = test_pandas_help_tool()
        ds_result = test_data_science_help_tool()
        error_result = test_error_solution_tool()
        history_result = test_search_history_tool()
        test_error_handling()
        
        # Summary
        print("\n📊 Test Summary:")
        print("=" * 30)
        
        results = [
            ("WebSearchTool", web_result),
            ("PandasHelpTool", pandas_result),
            ("DataScienceHelpTool", ds_result),
            ("ErrorSolutionTool", error_result),
            ("SearchHistoryTool", history_result)
        ]
        
        working_count = 0
        for tool_name, result in results:
            if result and "❌" not in result and len(result) > 50:
                print(f"   ✅ {tool_name}: Working")
                working_count += 1
            else:
                print(f"   ❌ {tool_name}: Issues detected")
        
        print(f"\n📈 Overall Status: {working_count}/{len(results)} tools working")
        
        if working_count == len(results):
            print("🎉 All web search tools are working correctly!")
        elif working_count > 0:
            print("⚠️ Some tools are working, but there may be issues")
        else:
            print("💥 No tools are working - check network connection and DuckDuckGo availability")
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()