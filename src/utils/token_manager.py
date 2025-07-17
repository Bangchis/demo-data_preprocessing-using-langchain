"""
Token Management Utilities for OpenAI Models

This module provides utilities for:
- Token counting for different OpenAI models
- Query length validation and truncation
- Context management based on token limits
- Dynamic token budget allocation
"""

import tiktoken
from typing import Dict, List, Tuple, Optional, Union
import streamlit as st
from dataclasses import dataclass


@dataclass
class TokenBudget:
    """Token budget allocation for different parts of the prompt"""
    total_limit: int
    system_instructions: int
    user_query: int
    chat_context: int
    tools_description: int
    buffer: int  # Safety buffer for response


class TokenManager:
    """Manages token counting and limits for OpenAI models"""
    
    # Model token limits (input + output)
    MODEL_LIMITS = {
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 4096,
        "o1-preview": 128000
    }
    
    # Default token budgets for different models
    DEFAULT_BUDGETS = {
        "gpt-4o": TokenBudget(
            total_limit=120000,  # Leave some buffer
            system_instructions=8000,
            user_query=3000,
            chat_context=2000,
            tools_description=1000,
            buffer=4000
        ),
        "gpt-4-turbo": TokenBudget(
            total_limit=120000,
            system_instructions=8000,
            user_query=3000,
            chat_context=2000,
            tools_description=1000,
            buffer=4000
        ),
        "gpt-4": TokenBudget(
            total_limit=7000,
            system_instructions=2000,
            user_query=1500,
            chat_context=1000,
            tools_description=500,
            buffer=2000
        ),
        "gpt-3.5-turbo": TokenBudget(
            total_limit=3500,
            system_instructions=1000,
            user_query=800,
            chat_context=500,
            tools_description=200,
            buffer=1000
        ),
        "o1-preview": TokenBudget(
            total_limit=120000,
            system_instructions=8000,
            user_query=3000,
            chat_context=2000,
            tools_description=1000,
            buffer=4000
        )
    }
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.encoding = self._get_encoding(model_name)
        self.token_budget = self.DEFAULT_BUDGETS.get(model_name, self.DEFAULT_BUDGETS["gpt-4o"])
    
    def _get_encoding(self, model_name: str) -> tiktoken.Encoding:
        """Get appropriate encoding for model"""
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base for GPT-4 family
            return tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        if not text:
            return 0
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts"""
        return [self.count_tokens(text) for text in texts]
    
    def is_query_too_long(self, query: str) -> bool:
        """Check if user query exceeds the allocated budget"""
        query_tokens = self.count_tokens(query)
        return query_tokens > self.token_budget.user_query
    
    def get_query_token_info(self, query: str) -> Dict[str, Union[int, float, bool]]:
        """Get detailed token information for a query"""
        query_tokens = self.count_tokens(query)
        limit = self.token_budget.user_query
        
        return {
            "tokens": query_tokens,
            "limit": limit,
            "percentage": (query_tokens / limit) * 100,
            "is_over_limit": query_tokens > limit,
            "remaining": max(0, limit - query_tokens)
        }
    
    def truncate_query(self, query: str, preserve_ratio: float = 0.9) -> str:
        """
        Truncate query to fit within token budget
        
        Args:
            query: Original query text
            preserve_ratio: Ratio of budget to use (0.9 = 90% of budget)
        """
        target_tokens = int(self.token_budget.user_query * preserve_ratio)
        
        if self.count_tokens(query) <= target_tokens:
            return query
        
        # Simple truncation approach - encode, truncate, decode
        encoded = self.encoding.encode(query)
        truncated = encoded[:target_tokens]
        truncated_text = self.encoding.decode(truncated)
        
        return truncated_text + "... [truncated]"
    
    def smart_truncate_query(self, query: str, preserve_ratio: float = 0.9) -> str:
        """
        Smart truncation that tries to preserve sentence boundaries
        """
        target_tokens = int(self.token_budget.user_query * preserve_ratio)
        
        if self.count_tokens(query) <= target_tokens:
            return query
        
        # Try to truncate at sentence boundaries
        sentences = query.split('.')
        truncated_query = ""
        
        for sentence in sentences:
            test_query = truncated_query + sentence + "."
            if self.count_tokens(test_query) <= target_tokens:
                truncated_query = test_query
            else:
                break
        
        if not truncated_query:
            # Fallback to simple truncation
            return self.truncate_query(query, preserve_ratio)
        
        return truncated_query + " [truncated]"
    
    def get_context_budget(self, current_context_tokens: int) -> int:
        """Get available tokens for chat context"""
        return min(current_context_tokens, self.token_budget.chat_context)
    
    def truncate_context(self, context_messages: List[Dict], target_tokens: int) -> List[Dict]:
        """
        Truncate context messages to fit within token budget
        Keeps most recent messages and removes older ones
        """
        if not context_messages:
            return []
        
        # Calculate tokens for each message
        message_tokens = []
        for msg in context_messages:
            tokens = self.count_tokens(msg.get("content", ""))
            message_tokens.append(tokens)
        
        # Start from the most recent messages
        selected_messages = []
        current_tokens = 0
        
        for i in range(len(context_messages) - 1, -1, -1):
            msg_tokens = message_tokens[i]
            if current_tokens + msg_tokens <= target_tokens:
                selected_messages.insert(0, context_messages[i])
                current_tokens += msg_tokens
            else:
                break
        
        return selected_messages
    
    def estimate_total_tokens(self, 
                            system_instructions: str,
                            user_query: str,
                            context_messages: List[Dict],
                            tools_description: str = "") -> Dict[str, int]:
        """
        Estimate total tokens for complete prompt
        """
        return {
            "system_instructions": self.count_tokens(system_instructions),
            "user_query": self.count_tokens(user_query),
            "context": sum(self.count_tokens(msg.get("content", "")) for msg in context_messages),
            "tools": self.count_tokens(tools_description),
            "total": (self.count_tokens(system_instructions) + 
                     self.count_tokens(user_query) + 
                     sum(self.count_tokens(msg.get("content", "")) for msg in context_messages) + 
                     self.count_tokens(tools_description))
        }
    
    def optimize_context_for_query(self, 
                                  query: str, 
                                  context_messages: List[Dict],
                                  system_instructions: str = "",
                                  tools_description: str = "") -> Tuple[str, List[Dict]]:
        """
        Optimize query and context to fit within token budget
        """
        # First, check if query needs truncation
        optimized_query = query
        if self.is_query_too_long(query):
            optimized_query = self.smart_truncate_query(query)
        
        # Calculate remaining budget for context
        used_tokens = (self.count_tokens(system_instructions) + 
                      self.count_tokens(optimized_query) + 
                      self.count_tokens(tools_description))
        
        remaining_budget = self.token_budget.total_limit - used_tokens - self.token_budget.buffer
        context_budget = min(remaining_budget, self.token_budget.chat_context)
        
        # Optimize context
        optimized_context = self.truncate_context(context_messages, context_budget)
        
        return optimized_query, optimized_context
    
    def get_token_usage_summary(self, 
                               system_instructions: str,
                               user_query: str,
                               context_messages: List[Dict],
                               tools_description: str = "") -> Dict:
        """
        Get comprehensive token usage summary
        """
        token_breakdown = self.estimate_total_tokens(
            system_instructions, user_query, context_messages, tools_description
        )
        
        return {
            "breakdown": token_breakdown,
            "budget": {
                "total_limit": self.token_budget.total_limit,
                "user_query_limit": self.token_budget.user_query,
                "context_limit": self.token_budget.chat_context,
                "system_limit": self.token_budget.system_instructions,
                "tools_limit": self.token_budget.tools_description
            },
            "usage": {
                "total_used": token_breakdown["total"],
                "remaining": self.token_budget.total_limit - token_breakdown["total"],
                "percentage": (token_breakdown["total"] / self.token_budget.total_limit) * 100
            },
            "warnings": self._get_usage_warnings(token_breakdown)
        }
    
    def _get_usage_warnings(self, token_breakdown: Dict[str, int]) -> List[str]:
        """Generate warnings for token usage"""
        warnings = []
        
        if token_breakdown["total"] > self.token_budget.total_limit:
            warnings.append(f"🚨 Total tokens ({token_breakdown['total']}) exceed model limit ({self.token_budget.total_limit})")
        
        if token_breakdown["user_query"] > self.token_budget.user_query:
            warnings.append(f"⚠️ Query tokens ({token_breakdown['user_query']}) exceed budget ({self.token_budget.user_query})")
        
        if token_breakdown["context"] > self.token_budget.chat_context:
            warnings.append(f"⚠️ Context tokens ({token_breakdown['context']}) exceed budget ({self.token_budget.chat_context})")
        
        if token_breakdown["system_instructions"] > self.token_budget.system_instructions:
            warnings.append(f"⚠️ System instructions ({token_breakdown['system_instructions']}) exceed budget ({self.token_budget.system_instructions})")
        
        # Usage percentage warnings
        usage_percentage = (token_breakdown["total"] / self.token_budget.total_limit) * 100
        if usage_percentage > 90:
            warnings.append(f"⚠️ High token usage: {usage_percentage:.1f}% of limit")
        elif usage_percentage > 70:
            warnings.append(f"💡 Moderate token usage: {usage_percentage:.1f}% of limit")
        
        return warnings


# Global token manager instance
def get_token_manager(model_name: str = "gpt-4o") -> TokenManager:
    """Get or create token manager instance"""
    session_key = f"token_manager_{model_name}"
    if session_key not in st.session_state:
        st.session_state[session_key] = TokenManager(model_name)
    return st.session_state[session_key]


# Utility functions for easy integration
def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """Quick token counting function"""
    return get_token_manager(model_name).count_tokens(text)


def is_query_too_long(query: str, model_name: str = "gpt-4o") -> bool:
    """Quick query length check"""
    return get_token_manager(model_name).is_query_too_long(query)


def optimize_query_and_context(query: str, 
                              context_messages: List[Dict],
                              model_name: str = "gpt-4o",
                              system_instructions: str = "",
                              tools_description: str = "") -> Tuple[str, List[Dict]]:
    """Quick optimization function"""
    return get_token_manager(model_name).optimize_context_for_query(
        query, context_messages, system_instructions, tools_description
    )