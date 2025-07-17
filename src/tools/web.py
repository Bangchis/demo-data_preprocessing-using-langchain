import streamlit as st
from langchain.tools import Tool
from typing import Optional
import os
import requests


def tavily_search(query: str) -> str:
    """
    Search the web using Tavily for domain knowledge
    
    Usage:
    - Simple search: "pandas pivot table"
    - Domain-specific: "pandas pivot table | domains=stackoverflow.com,pandas.pydata.org"
    
    Returns max 3 results with snippets for LLM consumption
    """
    if not query or not query.strip():
        return "❌ Empty search query provided"
    
    try:
        # Parse domain filters if present
        search_query = query.strip()
        domain_filter = ""
        
        if "| domains=" in query:
            parts = query.split("| domains=")
            if len(parts) == 2:
                search_query = parts[0].strip()
                domains = parts[1].strip()
                
                # Convert domains to site: filters
                domain_list = [d.strip() for d in domains.split(",") if d.strip()]
                if domain_list:
                    domain_filter = " " + " ".join([f"site:{domain}" for domain in domain_list])
        
        # Combine query with domain filters
        final_query = search_query + domain_filter
        
        # Log search attempt
        st.session_state.setdefault("web_search_log", []).append({
            "query": final_query,
            "timestamp": st.session_state.get("current_time", "unknown")
        })
        
        # Get API key
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "❌ TAVILY_API_KEY not found in environment variables"
        
        # Tavily API endpoint
        url = "https://api.tavily.com/search"
        
        # Request payload
        payload = {
            "api_key": api_key,
            "query": final_query,
            "search_depth": "basic",
            "max_results": 3
        }
        
        # Make request
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            search_results = data.get('results', [])
            
            if not search_results or len(search_results) == 0:
                return f"❌ No results found for query: '{search_query}'"
            
            # Format results
            results = []
            for i, result in enumerate(search_results, 1):
                title = result.get('title', 'No title')
                content = result.get('content', 'No content')
                url = result.get('url', 'No URL')
                
                # Format result
                formatted_result = f"**{i}. {title}**\n"
                formatted_result += f"{content}\n"
                formatted_result += f"🔗 Source: {url}\n"
                
                results.append(formatted_result)
            
            # Format response
            response_text = f"🔍 **Web Search Results (Tavily) for:** `{search_query}`\n\n"
            response_text += "\n".join(results)
            
            # Limit response length to prevent token overflow
            if len(response_text) > 1200:
                response_text = response_text[:1200] + "...\n\n*[Results truncated for brevity]*"
            
            return response_text
            
        else:
            return f"❌ API error: {response.status_code} - {response.text}"
        
    except requests.exceptions.RequestException as e:
        return f"❌ Network error: {str(e)}"
    except Exception as e:
        error_msg = str(e)
        if "api" in error_msg.lower() or "key" in error_msg.lower():
            return f"⚠️ **API Key Error**\n\nTavily API key may be invalid or missing. Please check TAVILY_API_KEY in .env file."
        elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
            return f"⚠️ **Rate limit exceeded**\n\nTavily is temporarily limiting searches. Try again in a few minutes, or use a more specific query."
        else:
            return f"❌ Search error: {error_msg}"


def search_pandas_help(topic: str) -> str:
    """
    Search for pandas-specific help and documentation
    
    Automatically searches pandas documentation and Stack Overflow
    """
    if not topic or not topic.strip():
        return "❌ Empty topic provided"
    
    # Create targeted search query
    search_query = f"pandas {topic.strip()} | domains=pandas.pydata.org,stackoverflow.com"
    
    return tavily_search(search_query)


def search_data_science_help(topic: str) -> str:
    """
    Search for data science and statistics help
    
    Automatically searches relevant data science sites
    """
    if not topic or not topic.strip():
        return "❌ Empty topic provided"
    
    # Create targeted search query
    domains = "stackoverflow.com,kaggle.com,towardsdatascience.com,docs.python.org"
    search_query = f"data science {topic.strip()} | domains={domains}"
    
    return tavily_search(search_query)


def search_error_solution(error_message: str) -> str:
    """
    Search for solutions to specific error messages
    
    Optimized for finding programming error solutions
    """
    if not error_message or not error_message.strip():
        return "❌ Empty error message provided"
    
    # Clean up error message for better search
    cleaned_error = error_message.strip()
    
    # Create targeted search query for error solutions
    search_query = f'python pandas "{cleaned_error}" solution | domains=stackoverflow.com,github.com'
    
    return tavily_search(search_query)


def get_search_history(_: str) -> str:
    """
    Get recent web search history
    """
    search_log = st.session_state.get("web_search_log", [])
    
    if not search_log:
        return "📝 No web searches performed yet"
    
    result = "📝 **Recent Web Searches:**\n\n"
    
    # Show last 5 searches
    for i, search in enumerate(reversed(search_log[-5:]), 1):
        result += f"{i}. `{search['query']}`\n"
        if search.get('timestamp'):
            result += f"   🕐 {search['timestamp']}\n"
    
    return result


# Create tools
WebSearchTool = Tool(
    name="WebSearch",
    func=tavily_search,
    description="Real-time web search via Tavily (100 req/day free). Input: 'search query' or 'search query | domains=site1.com,site2.com' for domain-specific search. Supports Vietnamese & English."
)

PandasHelpTool = Tool(
    name="PandasHelp",
    func=search_pandas_help,
    description="Search for pandas-specific help and documentation. Input: pandas topic or function name."
)

DataScienceHelpTool = Tool(
    name="DataScienceHelp", 
    func=search_data_science_help,
    description="Search for data science and statistics help. Input: data science topic or concept."
)

ErrorSolutionTool = Tool(
    name="ErrorSolution",
    func=search_error_solution,
    description="Search for solutions to specific error messages. Input: error message text."
)

SearchHistoryTool = Tool(
    name="SearchHistory",
    func=get_search_history,
    description="Get recent web search history. No input required."
)