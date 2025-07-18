#!/usr/bin/env python3
"""
Demo script showing chat memory functionality
"""

import os
import sys
import pandas as pd
from unittest.mock import MagicMock

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock all required modules
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

mock_st = MockStreamlit()
sys.modules['streamlit'] = mock_st
sys.modules['langchain'] = MagicMock()
sys.modules['langchain.agents'] = MagicMock()
sys.modules['langchain_openai'] = MagicMock()
sys.modules['langchain.tools'] = MagicMock()
sys.modules['tools_core'] = MagicMock()
sys.modules['tools_basic'] = MagicMock()
sys.modules['tools_web'] = MagicMock()

from agents.react_agent import AgentManager

def demo_chat_memory():
    """Demo chat memory functionality"""
    print("🤖 Demo: Chat Memory Functionality")
    print("="*50)
    
    # Create agent manager
    agent_manager = AgentManager("test-key", "gpt-4o")
    
    # Simulate a conversation
    conversation = [
        {"role": "user", "content": "Hãy hiển thị thông tin cơ bản về dữ liệu"},
        {"role": "assistant", "content": "Tôi sẽ sử dụng QuickInfo để hiển thị thông tin cơ bản về DataFrame. Đây là thông tin về shape, dtypes và các thống kê cơ bản..."},
        {"role": "user", "content": "Tìm các cột có giá trị thiếu"},
        {"role": "assistant", "content": "Tôi đã sử dụng MissingReport và tìm thấy 3 cột có giá trị thiếu: 'age' (15%), 'income' (8%), 'education' (3%)..."},
        {"role": "user", "content": "Có bao nhiêu dòng trùng lặp?"},
        {"role": "assistant", "content": "Sau khi sử dụng DuplicateCheck, tôi tìm thấy 25 dòng trùng lặp (2.5% tổng dữ liệu)..."},
        {"role": "user", "content": "Xóa các dòng trùng lặp và hiển thị thông tin mới"},
    ]
    
    # Add conversation to session state
    mock_st.session_state.react_chat_history = conversation
    
    print("📚 Conversation History:")
    for i, msg in enumerate(conversation):
        role = "👤 User" if msg["role"] == "user" else "🤖 Agent"
        print(f"{i+1}. {role}: {msg['content'][:60]}...")
    
    print(f"\n🧠 Memory Stats:")
    stats = agent_manager.get_chat_stats()
    print(f"   Total messages: {stats['total_messages']}")
    print(f"   Context messages: {stats['context_messages']}")
    print(f"   Memory active: {stats['memory_active']}")
    print(f"   Max context: {stats['max_context_messages']}")
    
    print(f"\n📝 Generated Context for Next Query:")
    context = agent_manager.get_chat_context()
    print(context)
    
    print("🔧 Simulating Enhanced Query:")
    query = "Xóa các dòng trùng lặp và hiển thị thông tin mới"
    
    # Mock the enhanced query format
    enhanced_query = f"""
{context}🎯 YÊU CẦU CỦA NGƯỜI DÙNG: {query}

📊 THÔNG TIN NGỮ CẢNH:
- DataFrame hiện tại: (1000, 5)
- Thời gian: 2024-01-01 10:00:00

🔍 HƯỚNG DẪN:
1. Phân tích yêu cầu kỹ lưỡng (tham khảo lịch sử cuộc hội thoại nếu có)
2. Sử dụng công cụ phù hợp để khám phá/xử lý dữ liệu
3. Khi cần thực thi code, sử dụng CodeRunner
4. Khi cần tìm hiểu thêm, sử dụng WebSearch hoặc PandasHelp
5. Giải thích rõ ràng từng bước

Hãy bắt đầu với Thought để phân tích yêu cầu:
"""
    
    print(f"Enhanced Query Length: {len(enhanced_query)} characters")
    print(f"Context included: {'📚 **LỊCH SỬ CUỘC HỘI THOẠI GẦN ĐÂY:**' in enhanced_query}")
    
    print(f"\n⚙️ Testing Configuration:")
    print(f"   Setting max context to 3 messages...")
    agent_manager.set_max_context_messages(3)
    
    stats = agent_manager.get_chat_stats()
    print(f"   New context messages: {stats['context_messages']}")
    
    print(f"\n🗑️ Testing Clear History:")
    agent_manager.clear_chat_history()
    stats = agent_manager.get_chat_stats()
    print(f"   Messages after clear: {stats['total_messages']}")
    print(f"   Memory active: {stats['memory_active']}")
    
    print(f"\n✅ Demo completed! The agent can now:")
    print("   1. 🧠 Remember previous conversations")
    print("   2. 📚 Include chat history in new queries")
    print("   3. ⚙️ Configure context window size")
    print("   4. 🗑️ Clear conversation history")
    print("   5. 📊 Provide memory statistics")

if __name__ == "__main__":
    demo_chat_memory()