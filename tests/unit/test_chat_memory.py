#!/usr/bin/env python3
"""
Test script for chat memory functionality
"""

import os
import sys
import pandas as pd
from unittest.mock import MagicMock

# Mock streamlit for testing
class MockSessionState:
    def __init__(self):
        self.data = {}
        self.data.setdefault('react_chat_history', [])
        self.data.setdefault('df', None)
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

class MockStreamlit:
    def __init__(self):
        self.session_state = MockSessionState()

# Mock all required modules
mock_st = MockStreamlit()
sys.modules['streamlit'] = mock_st
sys.modules['langchain'] = MagicMock()
sys.modules['langchain.agents'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langchain.tools'] = MagicMock()
sys.modules['tools_core'] = MagicMock()
sys.modules['tools_basic'] = MagicMock()
sys.modules['tools_web'] = MagicMock()

# Now we can import our agent_manager
from agent_manager import AgentManager

def test_chat_memory():
    """Test chat memory functionality"""
    print("🧪 Testing Chat Memory Functionality...")
    
    # Create mock API key
    api_key = "test-key-123"
    
    # Create agent manager
    agent_manager = AgentManager(api_key, "gpt-4o")
    
    # Test 1: Empty chat history
    print("\n1. Testing empty chat history...")
    context = agent_manager.get_chat_context()
    print(f"   Empty context: '{context}'")
    assert context == "", "Empty chat history should return empty context"
    
    # Test 2: Add some chat history
    print("\n2. Testing with chat history...")
    mock_st.session_state['react_chat_history'] = [
        {"role": "user", "content": "Hãy hiển thị thông tin cơ bản về dữ liệu"},
        {"role": "assistant", "content": "Tôi sẽ sử dụng QuickInfo để hiển thị thông tin cơ bản..."},
        {"role": "user", "content": "Tìm các cột có giá trị thiếu"},
        {"role": "assistant", "content": "Tôi sẽ sử dụng MissingReport để tìm các cột có giá trị thiếu..."}
    ]
    
    context = agent_manager.get_chat_context()
    print(f"   Context with history: {len(context)} characters")
    assert "📚 **LỊCH SỬ CUỘC HỘI THOẠI GẦN ĐÂY:**" in context, "Context should contain history header"
    assert "👤 Người dùng:" in context, "Context should contain user messages"
    assert "🤖 Agent:" in context, "Context should contain agent messages"
    
    # Test 3: Get chat stats
    print("\n3. Testing chat stats...")
    stats = agent_manager.get_chat_stats()
    print(f"   Stats: {stats}")
    assert stats["total_messages"] == 4, f"Expected 4 messages, got {stats['total_messages']}"
    assert stats["context_messages"] == 4, f"Expected 4 context messages, got {stats['context_messages']}"
    assert stats["memory_active"] == True, "Memory should be active"
    
    # Test 4: Test max context limit
    print("\n4. Testing max context limit...")
    agent_manager.set_max_context_messages(2)
    context = agent_manager.get_chat_context()
    stats = agent_manager.get_chat_stats()
    print(f"   Stats after limit: {stats}")
    assert stats["context_messages"] == 2, f"Expected 2 context messages after limit, got {stats['context_messages']}"
    
    # Test 5: Clear chat history
    print("\n5. Testing clear chat history...")
    result = agent_manager.clear_chat_history()
    print(f"   Clear result: {result}")
    assert result == True, "Clear should return True"
    
    stats = agent_manager.get_chat_stats()
    print(f"   Stats after clear: {stats}")
    assert stats["total_messages"] == 0, f"Expected 0 messages after clear, got {stats['total_messages']}"
    assert stats["memory_active"] == False, "Memory should be inactive after clear"
    
    # Test 6: Test context with long messages
    print("\n6. Testing context with long messages...")
    long_message = "X" * 500  # 500 characters
    mock_st.session_state['react_chat_history'] = [
        {"role": "user", "content": long_message},
        {"role": "assistant", "content": "Short response"}
    ]
    
    context = agent_manager.get_chat_context()
    print(f"   Long message context length: {len(context)} characters")
    assert "..." in context, "Long messages should be truncated"
    
    print("\n✅ All tests passed! Chat memory functionality working correctly.")

def test_enhanced_query():
    """Test enhanced query with context integration"""
    print("\n🧪 Testing Enhanced Query with Context...")
    
    # Setup mock data
    mock_st.session_state['df'] = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    mock_st.session_state['react_chat_history'] = [
        {"role": "user", "content": "Hiển thị thông tin cơ bản"},
        {"role": "assistant", "content": "Đây là thông tin cơ bản về dữ liệu..."}
    ]
    
    # This would normally call the actual LLM, but we'll just test the query construction
    api_key = "test-key-123"
    agent_manager = AgentManager(api_key, "gpt-4o")
    
    # Test query enhancement (we can't actually run the LLM, but we can test the setup)
    print("   Enhanced query structure should include chat context")
    context = agent_manager.get_chat_context()
    assert context != "", "Context should be included in enhanced query"
    
    print("✅ Enhanced query test passed!")

if __name__ == "__main__":
    try:
        test_chat_memory()
        test_enhanced_query()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()