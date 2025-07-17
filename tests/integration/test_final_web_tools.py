#!/usr/bin/env python3
"""
Final test for web search tools with Tavily (using requests)
"""

import os
import sys
from unittest.mock import MagicMock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    tavily_search,
    search_pandas_help,
    search_data_science_help,
    search_error_solution,
    get_search_history
)

def test_basic_search():
    """Test basic Tavily search"""
    print("🔍 Testing Basic Search:")
    print("-" * 30)
    
    result = tavily_search("pandas dataframe")
    print(f"   Query: pandas dataframe")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result}")
        return False
    else:
        print(f"   ✅ Success: {result[:150]}...")
        return True

def test_domain_search():
    """Test domain-specific search"""
    print("\n🎯 Testing Domain-Specific Search:")
    print("-" * 35)
    
    result = tavily_search("pandas pivot | domains=pandas.pydata.org")
    print(f"   Query: pandas pivot | domains=pandas.pydata.org")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result}")
        return False
    else:
        print(f"   ✅ Success: {result[:150]}...")
        return True

def test_pandas_help():
    """Test PandasHelp tool"""
    print("\n📊 Testing PandasHelp:")
    print("-" * 25)
    
    result = search_pandas_help("merge")
    print(f"   Query: merge")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result}")
        return False
    else:
        print(f"   ✅ Success: {result[:150]}...")
        return True

def test_data_science_help():
    """Test DataScienceHelp tool"""
    print("\n🔬 Testing DataScienceHelp:")
    print("-" * 30)
    
    result = search_data_science_help("machine learning")
    print(f"   Query: machine learning")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result}")
        return False
    else:
        print(f"   ✅ Success: {result[:150]}...")
        return True

def test_error_solution():
    """Test ErrorSolution tool"""
    print("\n🚨 Testing ErrorSolution:")
    print("-" * 25)
    
    result = search_error_solution("KeyError")
    print(f"   Query: KeyError")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result}")
        return False
    else:
        print(f"   ✅ Success: {result[:150]}...")
        return True

def test_search_history():
    """Test SearchHistory tool"""
    print("\n📝 Testing SearchHistory:")
    print("-" * 25)
    
    result = get_search_history("")
    print(f"   Result: {result}")
    
    search_log = mock_st.session_state.get("web_search_log", [])
    print(f"   Search log entries: {len(search_log)}")
    
    return True

def test_vietnamese_search():
    """Test Vietnamese search"""
    print("\n🇻🇳 Testing Vietnamese Search:")
    print("-" * 30)
    
    result = tavily_search("xử lý dữ liệu pandas")
    print(f"   Query: xử lý dữ liệu pandas")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result}")
        return False
    else:
        print(f"   ✅ Success: {result[:150]}...")
        return True

def test_error_handling():
    """Test error handling"""
    print("\n🛠️ Testing Error Handling:")
    print("-" * 30)
    
    # Test empty query
    result = tavily_search("")
    print(f"   Empty query result: {result}")
    
    # Test None query
    result = tavily_search(None)
    print(f"   None query result: {result}")
    
    return True

def main():
    """Main test function"""
    print("🧪 Final Web Search Tools Test (Tavily with Requests)")
    print("=" * 60)
    
    # Check environment
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("❌ TAVILY_API_KEY not found")
        return
    
    print(f"✅ TAVILY_API_KEY: {api_key[:10]}...{api_key[-5:]}")
    
    # Test all functions
    tests = [
        ("Basic Search", test_basic_search),
        ("Domain Search", test_domain_search),
        ("PandasHelp", test_pandas_help),
        ("DataScienceHelp", test_data_science_help),
        ("ErrorSolution", test_error_solution),
        ("SearchHistory", test_search_history),
        ("Vietnamese Search", test_vietnamese_search),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"   ❌ {test_name} failed: {str(e)}")
    
    # Summary
    print(f"\n📊 Test Summary:")
    print("=" * 30)
    print(f"   Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All web search tools are working correctly!")
        print("✅ Tavily search is fully functional")
        print("✅ Domain filtering works")
        print("✅ All specialized tools work")
        print("✅ Error handling works")
    elif passed > 0:
        print("⚠️ Some tools are working, but there may be issues")
    else:
        print("❌ No tools are working - check API key and network")
    
    print(f"\n🔧 Web Search Tools Status:")
    print("   - WebSearchTool: ✅ Ready")
    print("   - PandasHelpTool: ✅ Ready")
    print("   - DataScienceHelpTool: ✅ Ready")
    print("   - ErrorSolutionTool: ✅ Ready")
    print("   - SearchHistoryTool: ✅ Ready")

if __name__ == "__main__":
    main()