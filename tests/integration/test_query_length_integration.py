"""
Integration tests for query length handling with ReAct agent
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agents.react_agent import AgentManager
from utils.token_manager import TokenManager, get_token_manager
import streamlit as st


class TestQueryLengthIntegration:
    """Integration tests for query length handling with ReAct agent"""
    
    def setup_method(self):
        """Setup test environment"""
        # Mock streamlit session state
        self.mock_session_state = Mock()
        self.mock_session_state.df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        self.mock_session_state.react_chat_history = []
        self.mock_session_state.current_time = "2024-01-01 12:00:00"
        
        # Mock API key
        self.api_key = "test-api-key"
        
    @patch('streamlit.session_state')
    @patch('streamlit.warning')
    @patch('streamlit.info')
    @patch('streamlit.expander')
    def test_agent_manager_query_validation(self, mock_expander, mock_info, mock_warning, mock_session_state):
        """Test AgentManager query validation"""
        mock_session_state.__dict__ = self.mock_session_state.__dict__
        mock_session_state.__contains__ = lambda key: hasattr(self.mock_session_state, key)
        mock_session_state.__getitem__ = lambda key: getattr(self.mock_session_state, key)
        mock_session_state.__setitem__ = lambda key, value: setattr(self.mock_session_state, key, value)
        
        # Mock the expander context manager
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        # Create agent manager
        agent_manager = AgentManager(self.api_key, "gpt-4o")
        
        # Test normal query
        normal_query = "Analyze the data quality"
        
        # Mock the agent response
        with patch.object(agent_manager, 'create_agent') as mock_create_agent:
            mock_agent = Mock()
            mock_agent.run.return_value = "Analysis complete"
            mock_create_agent.return_value = mock_agent
            
            # This should not trigger warnings
            response = agent_manager.process_query(normal_query)
            
            # Should not call warning
            mock_warning.assert_not_called()
            assert response == "Analysis complete"
        
        # Test long query
        long_query = "Please provide a comprehensive analysis " * 200
        
        with patch.object(agent_manager, 'create_agent') as mock_create_agent:
            mock_agent = Mock()
            mock_agent.run.return_value = "Analysis complete"
            mock_create_agent.return_value = mock_agent
            
            # This should trigger warnings
            response = agent_manager.process_query(long_query)
            
            # Should call warning and info
            mock_warning.assert_called()
            mock_info.assert_called()
    
    def test_token_manager_with_agent_context(self):
        """Test token manager with agent context"""
        token_manager = TokenManager("gpt-4o")
        
        # Test system instructions token count
        system_instructions = """
        You are a data analysis assistant with access to multiple tools.
        You can analyze data, generate visualizations, and provide insights.
        Please follow the guidelines and provide structured responses.
        """
        
        system_tokens = token_manager.count_tokens(system_instructions)
        assert system_tokens > 0
        assert system_tokens < token_manager.token_budget.system_instructions
        
        # Test realistic query scenarios
        queries = [
            "Analyze the data",
            "Please provide a comprehensive analysis of the dataset including missing values, outliers, and statistical summaries",
            "Can you help me understand the data quality issues and suggest preprocessing steps for machine learning?",
            "I need a detailed report on data distribution, correlation analysis, and recommendations for feature engineering" * 10
        ]
        
        for query in queries:
            query_info = token_manager.get_query_token_info(query)
            
            if query_info["is_over_limit"]:
                # Test truncation
                truncated = token_manager.smart_truncate_query(query)
                truncated_info = token_manager.get_query_token_info(truncated)
                assert not truncated_info["is_over_limit"]
                assert len(truncated) < len(query)
    
    def test_context_optimization_scenarios(self):
        """Test context optimization in various scenarios"""
        token_manager = TokenManager("gpt-4o")
        
        # Scenario 1: Normal conversation
        normal_context = [
            {"role": "user", "content": "Hello, can you help me with data analysis?"},
            {"role": "assistant", "content": "Of course! I'd be happy to help with your data analysis."},
            {"role": "user", "content": "What should I do about missing values?"},
            {"role": "assistant", "content": "There are several approaches to handle missing values..."}
        ]
        
        query = "Please analyze the correlation between variables"
        system_instructions = "You are a data analysis assistant"
        
        optimized_query, optimized_context = token_manager.optimize_context_for_query(
            query, normal_context, system_instructions
        )
        
        # Should not need much optimization
        assert optimized_query == query
        assert len(optimized_context) == len(normal_context)
        
        # Scenario 2: Heavy context
        heavy_context = []
        for i in range(50):
            heavy_context.extend([
                {"role": "user", "content": f"Question {i}: " + "Long question about data analysis " * 20},
                {"role": "assistant", "content": f"Answer {i}: " + "Detailed analysis response " * 50}
            ])
        
        optimized_query, optimized_context = token_manager.optimize_context_for_query(
            query, heavy_context, system_instructions
        )
        
        # Should optimize context
        assert len(optimized_context) < len(heavy_context)
        
        # Should preserve recent messages
        if optimized_context:
            recent_original = [msg["content"] for msg in heavy_context[-10:]]
            optimized_contents = [msg["content"] for msg in optimized_context]
            
            # At least some recent messages should be preserved
            assert any(content in recent_original for content in optimized_contents)
    
    def test_token_budget_allocation(self):
        """Test token budget allocation for different models"""
        models = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        
        for model in models:
            token_manager = TokenManager(model)
            
            # Check budget allocation makes sense
            budget = token_manager.token_budget
            
            # Total should be reasonable
            assert budget.total_limit > 1000
            
            # Components should sum to less than total (leave room for response)
            component_sum = (budget.system_instructions + 
                           budget.user_query + 
                           budget.chat_context + 
                           budget.tools_description)
            
            assert component_sum < budget.total_limit
            
            # Query budget should be reasonable
            assert budget.user_query > 500  # Should allow decent queries
            assert budget.user_query < budget.total_limit / 2  # But not too much
    
    def test_real_world_query_patterns(self):
        """Test with real-world query patterns"""
        token_manager = TokenManager("gpt-4o")
        
        # Common data analysis queries
        real_queries = [
            "Show me the basic statistics of this dataset",
            "What are the missing values and how should I handle them?",
            "Can you identify outliers in the numerical columns?",
            "Please analyze the correlation between all variables and create a heatmap",
            "I need to prepare this data for machine learning. What preprocessing steps do you recommend?",
            "Generate a comprehensive data quality report including missing values, duplicates, outliers, and data types",
            "Create visualizations showing the distribution of each variable and identify any patterns or anomalies that might affect model performance",
            # Vietnamese queries
            "Hãy phân tích dữ liệu thiếu trong dataset",
            "Tôi cần một báo cáo chi tiết về chất lượng dữ liệu",
            "Vui lòng đưa ra khuyến nghị về cách tiền xử lý dữ liệu cho machine learning"
        ]
        
        for query in real_queries:
            query_info = token_manager.get_query_token_info(query)
            
            # Most real queries should be reasonable
            assert query_info["tokens"] > 0
            
            # If over limit, truncation should work
            if query_info["is_over_limit"]:
                truncated = token_manager.smart_truncate_query(query)
                truncated_info = token_manager.get_query_token_info(truncated)
                assert not truncated_info["is_over_limit"]
    
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling"""
        token_manager = TokenManager("gpt-4o")
        
        # Empty inputs
        assert token_manager.count_tokens("") == 0
        assert not token_manager.is_query_too_long("")
        
        # Very short inputs
        assert not token_manager.is_query_too_long("Hi")
        
        # Special characters and unicode
        special_query = "Análisis de datos con acentos y símbolos: 🔥📊💻"
        assert token_manager.count_tokens(special_query) > 0
        
        # Very long single word (edge case for truncation)
        long_word = "a" * 10000
        truncated = token_manager.smart_truncate_query(long_word)
        assert len(truncated) < len(long_word)
        
        # Empty context
        optimized_query, optimized_context = token_manager.optimize_context_for_query(
            "test query", [], "system instructions"
        )
        assert optimized_query == "test query"
        assert optimized_context == []


def test_performance_with_large_contexts():
    """Test performance with large contexts"""
    token_manager = TokenManager("gpt-4o")
    
    # Create large context
    large_context = []
    for i in range(100):
        large_context.extend([
            {"role": "user", "content": f"User message {i} " * 100},
            {"role": "assistant", "content": f"Assistant response {i} " * 200}
        ])
    
    # Performance test - should complete in reasonable time
    import time
    start_time = time.time()
    
    optimized_query, optimized_context = token_manager.optimize_context_for_query(
        "Analyze the data", large_context, "You are an assistant"
    )
    
    end_time = time.time()
    
    # Should complete within 5 seconds
    assert end_time - start_time < 5
    
    # Should produce valid output
    assert isinstance(optimized_query, str)
    assert isinstance(optimized_context, list)
    assert len(optimized_context) < len(large_context)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])