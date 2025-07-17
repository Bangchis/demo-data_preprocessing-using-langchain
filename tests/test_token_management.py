"""
Comprehensive tests for query length handling and token management system
"""

import pytest
from unittest.mock import Mock, patch
import streamlit as st
from src.utils.token_manager import TokenManager, get_token_manager, count_tokens, is_query_too_long


class TestTokenManager:
    """Test TokenManager functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.token_manager = TokenManager("gpt-4o")
        
    def test_token_counting_basic(self):
        """Test basic token counting functionality"""
        # Test empty string
        assert self.token_manager.count_tokens("") == 0
        
        # Test simple text
        text = "Hello world"
        tokens = self.token_manager.count_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)
        
        # Test longer text should have more tokens
        longer_text = "This is a much longer text that should have more tokens than the simple hello world example"
        longer_tokens = self.token_manager.count_tokens(longer_text)
        assert longer_tokens > tokens
    
    def test_token_counting_batch(self):
        """Test batch token counting"""
        texts = ["Hello", "World", "This is a longer text"]
        token_counts = self.token_manager.count_tokens_batch(texts)
        
        assert len(token_counts) == len(texts)
        assert all(isinstance(count, int) for count in token_counts)
        assert all(count > 0 for count in token_counts)
        
        # Longer text should have more tokens
        assert token_counts[2] > token_counts[0]
    
    def test_query_length_validation(self):
        """Test query length validation"""
        # Short query should be fine
        short_query = "Analyze the data"
        assert not self.token_manager.is_query_too_long(short_query)
        
        # Create a very long query that exceeds the budget
        long_query = "Please analyze the data " * 1000  # Repeat to make it long enough
        assert self.token_manager.is_query_too_long(long_query)
    
    def test_query_token_info(self):
        """Test detailed query token information"""
        query = "Analyze the dataset and provide insights"
        info = self.token_manager.get_query_token_info(query)
        
        assert "tokens" in info
        assert "limit" in info
        assert "percentage" in info
        assert "is_over_limit" in info
        assert "remaining" in info
        
        assert isinstance(info["tokens"], int)
        assert isinstance(info["limit"], int)
        assert isinstance(info["percentage"], float)
        assert isinstance(info["is_over_limit"], bool)
        assert isinstance(info["remaining"], int)
    
    def test_query_truncation(self):
        """Test query truncation functionality"""
        # Create a query that's too long
        long_query = "Please analyze the dataset and provide detailed insights about the data quality, missing values, outliers, and recommendations for preprocessing. " * 200
        
        # Test basic truncation
        truncated = self.token_manager.truncate_query(long_query)
        assert len(truncated) < len(long_query)
        assert truncated.endswith("... [truncated]")
        assert self.token_manager.count_tokens(truncated) <= self.token_manager.token_budget.user_query
        
        # Test smart truncation
        smart_truncated = self.token_manager.smart_truncate_query(long_query)
        assert len(smart_truncated) < len(long_query)
        assert smart_truncated.endswith(" [truncated]")
    
    def test_context_management(self):
        """Test context management functionality"""
        # Create sample context messages
        context_messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "Can you analyze my data?"},
            {"role": "assistant", "content": "Sure, I'd be happy to help with data analysis."}
        ]
        
        # Test context budget calculation
        current_tokens = sum(self.token_manager.count_tokens(msg["content"]) for msg in context_messages)
        budget = self.token_manager.get_context_budget(current_tokens)
        
        assert budget <= self.token_manager.token_budget.chat_context
        assert budget > 0
        
        # Test context truncation
        truncated = self.token_manager.truncate_context(context_messages, 100)
        assert len(truncated) <= len(context_messages)
        
        # Verify truncated context fits in budget
        truncated_tokens = sum(self.token_manager.count_tokens(msg["content"]) for msg in truncated)
        assert truncated_tokens <= 100
    
    def test_total_token_estimation(self):
        """Test total token estimation"""
        system_instructions = "You are a data analysis assistant"
        user_query = "Analyze the data"
        context_messages = [{"role": "user", "content": "Previous question"}]
        tools_description = "Available tools: DataAnalyzer, ChartGenerator"
        
        estimation = self.token_manager.estimate_total_tokens(
            system_instructions, user_query, context_messages, tools_description
        )
        
        assert "system_instructions" in estimation
        assert "user_query" in estimation
        assert "context" in estimation
        assert "tools" in estimation
        assert "total" in estimation
        
        # Total should be sum of parts
        expected_total = (estimation["system_instructions"] + 
                         estimation["user_query"] + 
                         estimation["context"] + 
                         estimation["tools"])
        assert estimation["total"] == expected_total
    
    def test_query_and_context_optimization(self):
        """Test query and context optimization"""
        # Create a long query
        long_query = "Please provide a comprehensive analysis " * 200
        
        # Create context messages
        context_messages = [
            {"role": "user", "content": "Previous analysis request " * 100},
            {"role": "assistant", "content": "Here's the analysis " * 100}
        ]
        
        system_instructions = "You are an AI assistant"
        
        # Test optimization
        optimized_query, optimized_context = self.token_manager.optimize_context_for_query(
            long_query, context_messages, system_instructions
        )
        
        # Query should be truncated if too long
        assert self.token_manager.count_tokens(optimized_query) <= self.token_manager.token_budget.user_query
        
        # Context should be truncated if necessary
        context_tokens = sum(self.token_manager.count_tokens(msg["content"]) for msg in optimized_context)
        assert context_tokens <= self.token_manager.token_budget.chat_context
    
    def test_token_usage_summary(self):
        """Test comprehensive token usage summary"""
        system_instructions = "You are a helpful assistant"
        user_query = "Analyze the data"
        context_messages = [{"role": "user", "content": "Hello"}]
        
        summary = self.token_manager.get_token_usage_summary(
            system_instructions, user_query, context_messages
        )
        
        assert "breakdown" in summary
        assert "budget" in summary
        assert "usage" in summary
        assert "warnings" in summary
        
        # Check budget information
        budget = summary["budget"]
        assert "total_limit" in budget
        assert "user_query_limit" in budget
        assert "context_limit" in budget
        
        # Check usage information
        usage = summary["usage"]
        assert "total_used" in usage
        assert "remaining" in usage
        assert "percentage" in usage
        
        # Warnings should be a list
        assert isinstance(summary["warnings"], list)
    
    def test_different_model_limits(self):
        """Test different model configurations"""
        models = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        
        for model in models:
            tm = TokenManager(model)
            
            # Each model should have different limits
            assert tm.model_name == model
            assert tm.token_budget.total_limit > 0
            assert tm.token_budget.user_query > 0
            assert tm.token_budget.chat_context > 0
            
            # Test encoding works
            tokens = tm.count_tokens("Test message")
            assert tokens > 0


class TestTokenManagerIntegration:
    """Integration tests for token manager with the application"""
    
    def test_get_token_manager_session_state(self):
        """Test get_token_manager creates instances correctly"""
        # Test that get_token_manager creates instances
        tm1 = get_token_manager("gpt-4o")
        assert isinstance(tm1, TokenManager)
        
        # Test different models
        tm2 = get_token_manager("gpt-4")
        assert isinstance(tm2, TokenManager)
        
        # Test same model returns consistent instance
        tm3 = get_token_manager("gpt-4o")
        assert isinstance(tm3, TokenManager)
    
    def test_utility_functions(self):
        """Test utility functions"""
        # Test count_tokens utility
        tokens = count_tokens("Hello world", "gpt-4o")
        assert tokens > 0
        
        # Test is_query_too_long utility
        short_query = "Short query"
        long_query = "Very long query " * 1000
        
        assert not is_query_too_long(short_query, "gpt-4o")
        assert is_query_too_long(long_query, "gpt-4o")


class TestQueryHandlingScenarios:
    """Test real-world query handling scenarios"""
    
    def setup_method(self):
        self.token_manager = TokenManager("gpt-4o")
    
    def test_normal_query_scenario(self):
        """Test normal query processing"""
        query = "Please analyze the missing values in my dataset and suggest preprocessing steps"
        
        # Should not be too long
        assert not self.token_manager.is_query_too_long(query)
        
        # Should have reasonable token count
        tokens = self.token_manager.count_tokens(query)
        assert tokens < 100  # Should be well under limit
    
    def test_very_long_query_scenario(self):
        """Test very long query processing"""
        query = ("Please provide a comprehensive analysis of my dataset including "
                "data quality assessment, missing value analysis, outlier detection, "
                "correlation analysis, statistical summaries, recommendations for "
                "data preprocessing, feature engineering suggestions, and visualization "
                "recommendations. " * 100)  # Repeat to make it very long
        
        # Should be too long
        assert self.token_manager.is_query_too_long(query)
        
        # Truncation should work
        truncated = self.token_manager.smart_truncate_query(query)
        assert not self.token_manager.is_query_too_long(truncated)
        assert len(truncated) < len(query)
    
    def test_context_heavy_scenario(self):
        """Test scenario with heavy context"""
        query = "Continue the previous analysis"
        
        # Create heavy context
        context_messages = []
        for i in range(20):
            context_messages.extend([
                {"role": "user", "content": f"Question {i}: Please analyze this data aspect" * 10},
                {"role": "assistant", "content": f"Analysis {i}: Here are the results" * 50}
            ])
        
        # Optimization should handle this
        optimized_query, optimized_context = self.token_manager.optimize_context_for_query(
            query, context_messages, "You are an assistant"
        )
        
        # Should fit within budget
        context_tokens = sum(self.token_manager.count_tokens(msg["content"]) for msg in optimized_context)
        assert context_tokens <= self.token_manager.token_budget.chat_context
        
        # Should preserve most recent messages
        assert len(optimized_context) < len(context_messages)
        if optimized_context:
            # Last message should be from original context
            assert optimized_context[-1]["content"] in [msg["content"] for msg in context_messages[-5:]]
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Empty query
        assert self.token_manager.count_tokens("") == 0
        assert not self.token_manager.is_query_too_long("")
        
        # Very short query
        assert not self.token_manager.is_query_too_long("Hi")
        
        # Special characters
        special_query = "Analyze: data with símbolos, émojis 🔥, and números 123"
        assert self.token_manager.count_tokens(special_query) > 0
        
        # Unicode text
        unicode_query = "Phân tích dữ liệu với tiếng Việt và các ký tự đặc biệt"
        assert self.token_manager.count_tokens(unicode_query) > 0


def test_token_manager_error_handling():
    """Test error handling in token management"""
    tm = TokenManager("gpt-4o")
    
    # Test with None input
    assert tm.count_tokens(None) == 0
    
    # Test with invalid model (should fallback)
    tm_invalid = TokenManager("invalid-model")
    assert tm_invalid.count_tokens("test") > 0  # Should still work with fallback
    
    # Test truncation with empty string
    truncated = tm.truncate_query("")
    assert truncated == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])