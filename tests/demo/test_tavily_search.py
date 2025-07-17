#!/usr/bin/env python3
"""
Test script for Tavily Web Search Tools
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
    tavily_search,
    search_pandas_help,
    search_data_science_help,
    search_error_solution,
    get_search_history
)

def check_environment():
    """Check environment setup"""
    print("🔧 Environment Check:")
    print("-" * 30)
    
    # Check API key
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key:
        print(f"   ✅ TAVILY_API_KEY: {api_key[:10]}...{api_key[-5:]}")
    else:
        print("   ❌ TAVILY_API_KEY not found in environment")
        return False
    
    # Check package
    try:
        from langchain_tavily import TavilySearchResults
        print("   ✅ langchain_tavily package available")
    except ImportError:
        print("   ❌ langchain_tavily package not found")
        return False
    
    return True

def test_tavily_search_basic():
    """Test basic Tavily search functionality"""
    print("\n🔍 Testing Basic Tavily Search:")
    print("-" * 40)
    
    # Test 1: Simple search
    print("\n1. Testing simple search...")
    query = "pandas dataframe merge"
    result = tavily_search(query)
    print(f"   Query: {query}")
    print(f"   Result length: {len(result)} characters")
    print(f"   Contains 'Tavily': {'Tavily' in result}")
    print(f"   Contains 'pandas': {'pandas' in result.lower()}")
    
    if "❌" in result:
        print(f"   ❌ Error: {result[:200]}...")
        return False
    else:
        print(f"   ✅ Success: {result[:200]}...")
        return True

def test_pandas_help():
    """Test PandasHelp tool"""
    print("\n📊 Testing PandasHelp Tool:")
    print("-" * 30)
    
    query = "pivot table"
    result = search_pandas_help(query)
    print(f"   Query: {query}")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result[:200]}...")
        return False
    else:
        print(f"   ✅ Success: {result[:200]}...")
        return True

def test_data_science_help():
    """Test DataScienceHelp tool"""
    print("\n🔬 Testing DataScienceHelp Tool:")
    print("-" * 35)
    
    query = "machine learning"
    result = search_data_science_help(query)
    print(f"   Query: {query}")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result[:200]}...")
        return False
    else:
        print(f"   ✅ Success: {result[:200]}...")
        return True

def test_error_solution():
    """Test ErrorSolution tool"""
    print("\n🚨 Testing ErrorSolution Tool:")
    print("-" * 30)
    
    query = "KeyError pandas"
    result = search_error_solution(query)
    print(f"   Query: {query}")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result[:200]}...")
        return False
    else:
        print(f"   ✅ Success: {result[:200]}...")
        return True

def test_search_history():
    """Test SearchHistory tool"""
    print("\n📝 Testing SearchHistory Tool:")
    print("-" * 30)
    
    result = get_search_history("")
    print(f"   Result: {result}")
    
    # Check if log has entries
    search_log = mock_st.session_state.get("web_search_log", [])
    print(f"   Search log entries: {len(search_log)}")
    
    for i, entry in enumerate(search_log):
        print(f"   {i+1}. {entry['query']}")
    
    return True

def test_domain_filtering():
    """Test domain filtering functionality"""
    print("\n🎯 Testing Domain Filtering:")
    print("-" * 30)
    
    query = "python pandas | domains=stackoverflow.com"
    result = tavily_search(query)
    print(f"   Query: {query}")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result[:200]}...")
        return False
    else:
        print(f"   ✅ Success: {result[:200]}...")
        return True

def test_vietnamese_query():
    """Test Vietnamese query support"""
    print("\n🇻🇳 Testing Vietnamese Query:")
    print("-" * 30)
    
    query = "xử lý dữ liệu pandas"
    result = tavily_search(query)
    print(f"   Query: {query}")
    print(f"   Result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   ❌ Error: {result[:200]}...")
        return False
    else:
        print(f"   ✅ Success: {result[:200]}...")
        return True

def test_error_handling():
    """Test error handling"""
    print("\n🛠️ Testing Error Handling:")
    print("-" * 30)
    
    # Test empty query
    print("\n1. Testing empty query...")
    result = tavily_search("")
    print(f"   Result: {result}")
    
    # Test None query
    print("\n2. Testing None query...")
    result = tavily_search(None)
    print(f"   Result: {result}")
    
    return True

def main():
    """Main test function"""
    print("🧪 Testing Tavily Web Search Tools")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment not ready - cannot run tests")
        return
    
    # Test results
    test_results = []
    
    try:
        test_results.append(("Basic Search", test_tavily_search_basic()))
        test_results.append(("PandasHelp", test_pandas_help()))
        test_results.append(("DataScienceHelp", test_data_science_help()))
        test_results.append(("ErrorSolution", test_error_solution()))
        test_results.append(("SearchHistory", test_search_history()))
        test_results.append(("Domain Filtering", test_domain_filtering()))
        test_results.append(("Vietnamese Query", test_vietnamese_query()))
        test_results.append(("Error Handling", test_error_handling()))
        
        # Summary
        print("\n📊 Test Summary:")
        print("=" * 30)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n📈 Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All Tavily web search tools are working correctly!")
        elif passed > 0:
            print("⚠️ Some tools are working, but there may be issues")
        else:
            print("💥 No tools are working - check API key and network connection")
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()