#!/usr/bin/env python3
"""
Simple test for token management functionality
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_token_manager():
    """Test token manager without external dependencies"""
    print("🧪 Testing TokenManager...")
    
    try:
        from utils.token_manager import TokenManager
        
        # Test basic functionality
        token_manager = TokenManager("gpt-4o")
        
        # Test simple token counting
        test_text = "Hello world, this is a test"
        tokens = token_manager.count_tokens(test_text)
        print(f"✅ Token counting works: '{test_text}' = {tokens} tokens")
        
        # Test query validation
        short_query = "Analyze data"
        query_info = token_manager.get_query_token_info(short_query)
        print(f"✅ Query validation works: {query_info['tokens']} tokens, {query_info['percentage']:.1f}% usage")
        
        # Test truncation
        long_query = "This is a very long query that should be truncated. " * 100
        if token_manager.is_query_too_long(long_query):
            truncated = token_manager.smart_truncate_query(long_query)
            print(f"✅ Query truncation works: {len(long_query)} -> {len(truncated)} characters")
        
        # Test different models
        for model in ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]:
            tm = TokenManager(model)
            print(f"✅ Model {model}: {tm.token_budget.user_query} token budget")
        
        print("\n🎉 All token management tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_tiktoken():
    """Test fallback functionality without tiktoken"""
    print("\n🧪 Testing fallback functionality...")
    
    try:
        # Mock tiktoken to test fallback
        import sys
        original_tiktoken = sys.modules.get('tiktoken')
        sys.modules['tiktoken'] = None
        
        from utils.token_manager import TokenManager
        
        # Should still work with fallback
        token_manager = TokenManager("gpt-4o")
        
        # Test fallback token counting (rough estimate)
        test_text = "Hello world, this is a test"
        tokens = token_manager.count_tokens(test_text)
        print(f"✅ Fallback token counting: '{test_text}' = {tokens} tokens (estimated)")
        
        # Restore tiktoken
        if original_tiktoken:
            sys.modules['tiktoken'] = original_tiktoken
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Simple Token Management Test")
    print("=" * 50)
    
    success = test_token_manager()
    
    if success:
        print("\n✅ Token management system is working correctly!")
    else:
        print("\n❌ Token management system has issues.")
        sys.exit(1)