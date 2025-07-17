#!/usr/bin/env python3
"""
Test Tavily API directly with requests
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_tavily_api_direct():
    """Test Tavily API directly with requests"""
    print("🧪 Testing Tavily API Directly with Requests")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("❌ TAVILY_API_KEY not found")
        return False
    
    print(f"✅ API Key: {api_key[:10]}...{api_key[-5:]}")
    
    # Tavily API endpoint
    url = "https://api.tavily.com/search"
    
    # Request payload
    payload = {
        "api_key": api_key,
        "query": "pandas dataframe merge",
        "search_depth": "basic",
        "max_results": 3
    }
    
    try:
        # Make request
        response = requests.post(url, json=payload, timeout=10)
        
        print(f"✅ Request sent, status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            
            print(f"✅ Got {len(results)} results")
            
            # Display first result
            if results:
                first_result = results[0]
                print(f"   Title: {first_result.get('title', 'No title')[:50]}...")
                print(f"   Content: {first_result.get('content', 'No content')[:100]}...")
                print(f"   URL: {first_result.get('url', 'No URL')}")
                
                return True
            else:
                print("❌ No results returned")
                return False
        else:
            print(f"❌ API error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def create_custom_tavily_search():
    """Create custom Tavily search function using requests"""
    print("\n🔧 Creating Custom Tavily Search Function")
    print("=" * 45)
    
    def custom_tavily_search(query: str) -> str:
        """Custom Tavily search using requests"""
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "❌ TAVILY_API_KEY not found"
        
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 3
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if not results:
                    return f"❌ No results found for query: '{query}'"
                
                # Format results
                formatted_results = []
                for i, result in enumerate(results, 1):
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    url = result.get('url', 'No URL')
                    
                    formatted_result = f"**{i}. {title}**\n"
                    formatted_result += f"{content}\n"
                    formatted_result += f"🔗 Source: {url}\n"
                    
                    formatted_results.append(formatted_result)
                
                # Format response
                response_text = f"🔍 **Web Search Results (Tavily) for:** `{query}`\n\n"
                response_text += "\n".join(formatted_results)
                
                return response_text
                
            else:
                return f"❌ API error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"❌ Search error: {str(e)}"
    
    # Test the custom function
    result = custom_tavily_search("pandas dataframe")
    print(f"✅ Custom function created")
    print(f"   Test result length: {len(result)} characters")
    
    if "❌" in result:
        print(f"   Error: {result}")
        return False
    else:
        print(f"   Success: {result[:200]}...")
        return True

def main():
    print("🚀 Tavily Direct API Test")
    print("=" * 30)
    
    # Test direct API
    api_result = test_tavily_api_direct()
    
    # Test custom function
    if api_result:
        custom_result = create_custom_tavily_search()
        
        if custom_result:
            print("\n🎉 Tavily API is working!")
            print("We can create custom search functions using requests")
            print("This can be used as fallback if langchain_tavily has issues")
        else:
            print("\n⚠️ API works but custom function has issues")
    else:
        print("\n❌ Tavily API not accessible")
        print("Check API key and internet connection")

if __name__ == "__main__":
    main()