#!/usr/bin/env python3
"""
Demo script for testing query length management system
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), 'config', '.env'))

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.token_manager import TokenManager, get_token_manager
import pandas as pd


def demo_token_counting():
    """Demo token counting functionality"""
    print("🔢 Token Counting Demo")
    print("=" * 50)
    
    token_manager = TokenManager("gpt-4o")
    
    # Test queries of different lengths
    test_queries = [
        "Analyze data",
        "Please analyze the data quality in my dataset",
        "Can you provide a comprehensive analysis of data quality including missing values, outliers, duplicates, and statistical summaries?",
        "I need a detailed analysis of my dataset that includes data quality assessment, missing value analysis, outlier detection, correlation analysis, statistical summaries, data type validation, duplicate detection, and preprocessing recommendations for machine learning applications. Please also provide visualizations and actionable insights." * 10
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        query_info = token_manager.get_query_token_info(query)
        print(f"   Tokens: {query_info['tokens']}")
        print(f"   Limit: {query_info['limit']}")
        print(f"   Percentage: {query_info['percentage']:.1f}%")
        print(f"   Over limit: {'❌' if query_info['is_over_limit'] else '✅'}")
        
        if query_info['is_over_limit']:
            truncated = token_manager.smart_truncate_query(query)
            truncated_info = token_manager.get_query_token_info(truncated)
            print(f"   Truncated: {truncated[:100]}...")
            print(f"   Truncated tokens: {truncated_info['tokens']}")


def demo_context_optimization():
    """Demo context optimization"""
    print("\n\n🎯 Context Optimization Demo")
    print("=" * 50)
    
    token_manager = TokenManager("gpt-4o")
    
    # Create sample context
    context_messages = []
    for i in range(20):
        context_messages.extend([
            {"role": "user", "content": f"User question {i}: Please analyze this aspect of the data " * 20},
            {"role": "assistant", "content": f"Assistant response {i}: Here's the detailed analysis " * 50}
        ])
    
    query = "Continue the previous analysis with more details"
    system_instructions = "You are a data analysis assistant with access to multiple tools."
    
    print(f"📊 Original context: {len(context_messages)} messages")
    original_context_tokens = sum(token_manager.count_tokens(msg["content"]) for msg in context_messages)
    print(f"   Context tokens: {original_context_tokens}")
    
    # Optimize
    optimized_query, optimized_context = token_manager.optimize_context_for_query(
        query, context_messages, system_instructions
    )
    
    optimized_context_tokens = sum(token_manager.count_tokens(msg["content"]) for msg in optimized_context)
    
    print(f"✅ Optimized context: {len(optimized_context)} messages")
    print(f"   Context tokens: {optimized_context_tokens}")
    print(f"   Reduction: {((original_context_tokens - optimized_context_tokens) / original_context_tokens * 100):.1f}%")
    
    # Token usage summary
    summary = token_manager.get_token_usage_summary(
        system_instructions, optimized_query, optimized_context
    )
    
    print(f"\n📈 Token Usage Summary:")
    print(f"   Total used: {summary['usage']['total_used']}")
    print(f"   Total limit: {summary['budget']['total_limit']}")
    print(f"   Usage percentage: {summary['usage']['percentage']:.1f}%")
    
    if summary['warnings']:
        print(f"⚠️  Warnings:")
        for warning in summary['warnings']:
            print(f"   - {warning}")


def demo_model_comparison():
    """Demo different model configurations"""
    print("\n\n🔧 Model Comparison Demo")
    print("=" * 50)
    
    models = ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
    test_query = "Please analyze the data quality and provide recommendations"
    
    for model in models:
        print(f"\n🤖 Model: {model}")
        token_manager = TokenManager(model)
        
        print(f"   Total limit: {token_manager.token_budget.total_limit:,}")
        print(f"   Query budget: {token_manager.token_budget.user_query:,}")
        print(f"   Context budget: {token_manager.token_budget.chat_context:,}")
        
        query_info = token_manager.get_query_token_info(test_query)
        print(f"   Test query tokens: {query_info['tokens']}")
        print(f"   Query usage: {query_info['percentage']:.1f}%")


def demo_real_world_scenarios():
    """Demo real-world scenarios"""
    print("\n\n🌟 Real-World Scenarios Demo")
    print("=" * 50)
    
    token_manager = TokenManager("gpt-4o")
    
    # Vietnamese queries
    vietnamese_queries = [
        "Hãy phân tích dữ liệu thiếu trong dataset",
        "Tôi cần một báo cáo chi tiết về chất lượng dữ liệu bao gồm missing values, outliers, duplicates",
        "Vui lòng đưa ra khuyến nghị về cách tiền xử lý dữ liệu cho machine learning, bao gồm feature engineering và data cleaning"
    ]
    
    print("🇻🇳 Vietnamese Queries:")
    for i, query in enumerate(vietnamese_queries, 1):
        query_info = token_manager.get_query_token_info(query)
        print(f"   {i}. Tokens: {query_info['tokens']}, Usage: {query_info['percentage']:.1f}%")
    
    # English queries
    english_queries = [
        "Analyze missing values in the dataset",
        "Generate a comprehensive data quality report including missing values, outliers, and duplicates",
        "Provide preprocessing recommendations for machine learning including feature engineering and data cleaning steps"
    ]
    
    print("\n🇺🇸 English Queries:")
    for i, query in enumerate(english_queries, 1):
        query_info = token_manager.get_query_token_info(query)
        print(f"   {i}. Tokens: {query_info['tokens']}, Usage: {query_info['percentage']:.1f}%")


def demo_performance_test():
    """Demo performance testing"""
    print("\n\n⚡ Performance Test Demo")
    print("=" * 50)
    
    import time
    
    token_manager = TokenManager("gpt-4o")
    
    # Create large context
    large_context = []
    for i in range(100):
        large_context.extend([
            {"role": "user", "content": f"User message {i} " * 50},
            {"role": "assistant", "content": f"Assistant response {i} " * 100}
        ])
    
    query = "Analyze the data with detailed insights"
    system_instructions = "You are a comprehensive data analysis assistant"
    
    print(f"📊 Testing with {len(large_context)} context messages")
    
    start_time = time.time()
    optimized_query, optimized_context = token_manager.optimize_context_for_query(
        query, large_context, system_instructions
    )
    end_time = time.time()
    
    print(f"⏱️  Optimization time: {end_time - start_time:.3f} seconds")
    print(f"✅ Optimized to {len(optimized_context)} messages")
    
    # Test token counting performance
    test_texts = [f"Test message {i} " * 100 for i in range(1000)]
    
    start_time = time.time()
    token_counts = token_manager.count_tokens_batch(test_texts)
    end_time = time.time()
    
    print(f"⏱️  Token counting for 1000 texts: {end_time - start_time:.3f} seconds")
    print(f"📈 Average: {sum(token_counts) / len(token_counts):.1f} tokens per text")


def main():
    """Main demo function"""
    print("🚀 Query Length Management System Demo")
    print("=" * 60)
    
    try:
        demo_token_counting()
        demo_context_optimization()
        demo_model_comparison()
        demo_real_world_scenarios()
        demo_performance_test()
        
        print("\n\n✅ All demos completed successfully!")
        print("🎉 Query length management system is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()