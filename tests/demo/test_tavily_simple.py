#!/usr/bin/env python3
"""
Simple test for Tavily search functionality
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

def test_tavily_direct():
    """Test Tavily search directly"""
    print("🧪 Testing Tavily Search Directly")
    print("=" * 40)
    
    # Check environment
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("❌ TAVILY_API_KEY not found")
        return
    
    print(f"✅ API Key: {api_key[:10]}...{api_key[-5:]}")
    
    # Test direct import and usage
    try:
        from langchain_tavily import TavilySearchResults
        print("✅ langchain_tavily imported successfully")
        
        # Test basic search
        tavily_tool = TavilySearchResults(k=3)
        result = tavily_tool.run({"query": "pandas dataframe"})
        
        print(f"✅ Search successful, got {len(result)} results")
        
        # Display first result
        if result and len(result) > 0:
            first_result = result[0]
            print(f"   Title: {first_result.get('title', 'No title')[:50]}...")
            print(f"   Content: {first_result.get('content', 'No content')[:100]}...")
            print(f"   URL: {first_result.get('url', 'No URL')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

def test_tavily_with_tools():
    """Test Tavily through our tools"""
    print("\n🔧 Testing Tavily Through Our Tools")
    print("=" * 40)
    
    # Test our tavily_search function
    try:
        # Import our function
        from tools_web import tavily_search
        
        # Test search
        result = tavily_search("pandas dataframe")
        print(f"✅ Function call successful")
        print(f"   Result length: {len(result)} characters")
        
        if "❌" in result:
            print(f"   Error: {result}")
            return False
        else:
            print(f"   Success: {result[:200]}...")
            return True
            
    except Exception as e:
        print(f"❌ Function test error: {e}")
        return False

def main():
    print("🚀 Simple Tavily Test")
    print("=" * 30)
    
    # Test direct usage
    direct_result = test_tavily_direct()
    
    # Test through tools
    if direct_result:
        tools_result = test_tavily_with_tools()
        
        if tools_result:
            print("\n🎉 Tavily search is working!")
            print("Web search tools should now be functional.")
        else:
            print("\n⚠️ Direct API works but tools have issues")
    else:
        print("\n❌ Tavily API not working")
        print("Check API key and langchain_tavily installation")

if __name__ == "__main__":
    main()