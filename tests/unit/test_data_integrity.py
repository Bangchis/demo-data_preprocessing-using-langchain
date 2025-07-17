#!/usr/bin/env python3
"""
Test data integrity improvements
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock streamlit for testing
class MockStreamlit:
    def __init__(self):
        self.session_state = {}
    
    def error(self, msg):
        print(f"ERROR: {msg}")

# Mock streamlit module
sys.modules['streamlit'] = MockStreamlit()

from utils import _apply_intelligent_type_inference, _create_display_version

def create_test_excel_data():
    """Create test data that mimics Excel data types"""
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Amount': [1500.50, 2000.75, 1750.00, 3000.25, 1200.00],
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'Active': ['TRUE', 'FALSE', 'True', 'False', 'TRUE'],
        'Score': [85.5, 92.0, 78.5, 95.0, 88.0],
        'Category': ['A', 'B', 'A', 'C', 'B']
    }
    
    df = pd.DataFrame(data)
    
    # Convert to object types to simulate Excel import issues
    df['Amount'] = df['Amount'].astype(str)
    df['Date'] = df['Date'].astype(str)
    df['Active'] = df['Active'].astype(str)
    df['Score'] = df['Score'].astype(str)
    
    return df

def test_type_inference():
    """Test intelligent type inference"""
    print("Testing intelligent type inference...")
    
    # Create test data
    df_raw = create_test_excel_data()
    
    print(f"Original data types:")
    print(df_raw.dtypes)
    
    # Apply type inference
    df_typed = _apply_intelligent_type_inference(df_raw)
    
    print(f"\nAfter type inference:")
    print(df_typed.dtypes)
    
    # Check if types are correct
    results = {
        'ID': pd.api.types.is_numeric_dtype(df_typed['ID']),
        'Name': df_typed['Name'].dtype == 'object',
        'Amount': pd.api.types.is_numeric_dtype(df_typed['Amount']),
        'Date': pd.api.types.is_datetime64_any_dtype(df_typed['Date']),
        'Active': df_typed['Active'].dtype == 'bool',
        'Score': pd.api.types.is_numeric_dtype(df_typed['Score']),
        'Category': df_typed['Category'].dtype == 'object'
    }
    
    print(f"\nType inference results:")
    for col, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {col}: {success}")
    
    return all(results.values())

def test_display_version():
    """Test display version creation"""
    print("Testing display version creation...")
    
    # Create test data with proper types
    df_raw = create_test_excel_data()
    df_typed = _apply_intelligent_type_inference(df_raw)
    
    # Create display version
    df_display = _create_display_version(df_typed)
    
    print(f"Original types: {df_typed.dtypes.to_dict()}")
    print(f"Display types: {df_display.dtypes.to_dict()}")
    
    # Check if display version is safe for Streamlit
    safe_for_streamlit = True
    issues = []
    
    for col in df_display.columns:
        if pd.api.types.is_datetime64_any_dtype(df_display[col]):
            safe_for_streamlit = False
            issues.append(f"Datetime column {col} not converted to string")
        elif df_display[col].dtype == 'bool':
            safe_for_streamlit = False
            issues.append(f"Boolean column {col} not converted to string")
        elif pd.api.types.is_numeric_dtype(df_display[col]):
            # Check for inf values
            if np.isinf(df_display[col]).any():
                safe_for_streamlit = False
                issues.append(f"Numeric column {col} contains inf values")
    
    if safe_for_streamlit:
        print("✅ Display version is safe for Streamlit")
    else:
        print("❌ Display version has issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    return safe_for_streamlit

def test_mathematical_operations():
    """Test that mathematical operations work on original data"""
    print("Testing mathematical operations...")
    
    # Create test data
    df_raw = create_test_excel_data()
    df_typed = _apply_intelligent_type_inference(df_raw)
    
    try:
        # Test mathematical operations
        total_amount = df_typed['Amount'].sum()
        avg_score = df_typed['Score'].mean()
        count_active = df_typed['Active'].sum()
        
        print(f"✅ Mathematical operations work:")
        print(f"  Total amount: ${total_amount:,.2f}")
        print(f"  Average score: {avg_score:.1f}")
        print(f"  Active count: {count_active}")
        
        # Test date operations
        df_typed['Date'] = pd.to_datetime(df_typed['Date'])
        latest_date = df_typed['Date'].max()
        print(f"  Latest date: {latest_date.strftime('%Y-%m-%d')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mathematical operations failed: {str(e)}")
        return False

def test_correlation_analysis():
    """Test that correlation analysis works"""
    print("Testing correlation analysis...")
    
    # Create test data
    df_raw = create_test_excel_data()
    df_typed = _apply_intelligent_type_inference(df_raw)
    
    try:
        # Get numeric columns
        numeric_df = df_typed.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) >= 2:
            # Calculate correlation
            corr_matrix = numeric_df.corr()
            print(f"✅ Correlation analysis works:")
            print(f"  Numeric columns: {list(numeric_df.columns)}")
            print(f"  Correlation shape: {corr_matrix.shape}")
            
            return True
        else:
            print("❌ Not enough numeric columns for correlation")
            return False
            
    except Exception as e:
        print(f"❌ Correlation analysis failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Data Integrity Improvements...\n")
    
    tests = [
        ("Type Inference", test_type_inference),
        ("Display Version", test_display_version),
        ("Mathematical Operations", test_mathematical_operations),
        ("Correlation Analysis", test_correlation_analysis)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED\n")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED\n")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED: {str(e)}\n")
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Data integrity is preserved.")
        print("\n🔍 **Summary of Improvements:**")
        print("✅ Excel data types are now properly preserved")
        print("✅ Numeric operations work on actual numbers")
        print("✅ Date operations work on actual dates")
        print("✅ Boolean logic works on actual booleans")
        print("✅ Correlation analysis works with proper numeric data")
        print("✅ Streamlit display compatibility maintained")
        
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementations.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)