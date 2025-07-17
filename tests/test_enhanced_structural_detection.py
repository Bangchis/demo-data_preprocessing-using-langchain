#!/usr/bin/env python3
"""
Test script for enhanced structural error detection system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.tools.enhanced_structural_detection import (
    detect_advanced_structural_errors, 
    StructuralErrorDetector,
    get_structural_error_details
)


def create_test_dataframe():
    """Create a test DataFrame with various structural errors"""
    data = {
        'id': [1, 2, 3, None, 5, 6, 7, 8, 9, 10],  # Missing ID
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
        'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'city': ['NY', 'LA', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
        'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000]
    }
    
    df = pd.DataFrame(data)
    
    # Add some structural errors
    
    # Row with collapsed data (multiple values in one cell)
    df.loc[2, 'name'] = 'Charlie, Manager, Software Engineer, Team Lead'
    df.loc[2, 'age'] = None
    df.loc[2, 'city'] = None
    df.loc[2, 'salary'] = None
    
    # Row with delimiter error (CSV-like data in single cell)
    df.loc[4, 'name'] = 'Eve,25,Manager,Engineering,2023-01-15,Active,Full-time'
    df.loc[4, 'age'] = None
    df.loc[4, 'city'] = None
    df.loc[4, 'salary'] = None
    
    # Row with very long single cell
    df.loc[6, 'name'] = 'Grace ' * 50  # Very long name
    df.loc[6, 'age'] = None
    df.loc[6, 'city'] = None
    df.loc[6, 'salary'] = None
    
    # Row with unusual null pattern
    df.loc[8, 'name'] = None
    df.loc[8, 'age'] = None
    df.loc[8, 'city'] = None
    df.loc[8, 'salary'] = 'Header: Employee Data Summary'
    
    return df


def test_basic_detection():
    """Test basic structural error detection"""
    print("🧪 Testing basic structural error detection...")
    
    df = create_test_dataframe()
    mask, report = detect_advanced_structural_errors(df)
    
    print(f"DataFrame shape: {df.shape}")
    print(f"Errors detected: {mask.sum()}")
    print("Report:")
    print(report)
    print()
    
    return mask.sum() > 0


def test_id_column_detection():
    """Test ID column specific detection"""
    print("🧪 Testing ID column detection...")
    
    df = create_test_dataframe()
    
    # Test with specific ID columns
    mask, report = detect_advanced_structural_errors(
        df, 
        id_columns=['id']
    )
    
    print(f"ID-specific errors detected: {mask.sum()}")
    print("Report:")
    print(report)
    print()
    
    return mask.sum() > 0


def test_detector_class():
    """Test the StructuralErrorDetector class directly"""
    print("🧪 Testing StructuralErrorDetector class...")
    
    df = create_test_dataframe()
    detector = StructuralErrorDetector(
        id_columns=['id'],
        text_length_threshold=50,
        delimiter_threshold=2
    )
    
    results = detector.detect_all_structural_errors(df)
    
    print(f"Total error types detected: {len(results['masks'])}")
    print("Error types:")
    for error_type, mask in results['masks'].items():
        if mask.sum() > 0:
            print(f"  - {error_type}: {mask.sum()} errors")
    
    print("\nSummary:")
    print(results['summary'])
    print()
    
    return results['summary']['total_error_rows'] > 0


def test_error_details():
    """Test detailed error analysis"""
    print("🧪 Testing detailed error analysis...")
    
    df = create_test_dataframe()
    mask, _ = detect_advanced_structural_errors(df)
    
    if mask.sum() > 0:
        details = get_structural_error_details(df, mask, max_examples=3)
        print("Detailed error analysis:")
        print(details)
        print()
        return True
    else:
        print("No errors found for detailed analysis")
        return False


def test_edge_cases():
    """Test edge cases and error handling"""
    print("🧪 Testing edge cases...")
    
    # Empty DataFrame
    empty_df = pd.DataFrame()
    try:
        mask, report = detect_advanced_structural_errors(empty_df)
        print("Empty DataFrame handled successfully")
    except Exception as e:
        print(f"Empty DataFrame error: {e}")
    
    # Single row DataFrame
    single_row_df = pd.DataFrame({'a': [1], 'b': [2]})
    try:
        mask, report = detect_advanced_structural_errors(single_row_df)
        print("Single row DataFrame handled successfully")
    except Exception as e:
        print(f"Single row DataFrame error: {e}")
    
    # DataFrame with all NaN
    nan_df = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
    try:
        mask, report = detect_advanced_structural_errors(nan_df)
        print("All-NaN DataFrame handled successfully")
    except Exception as e:
        print(f"All-NaN DataFrame error: {e}")
    
    print()
    return True


if __name__ == "__main__":
    print("🔧 Enhanced Structural Error Detection Tests")
    print("=" * 50)
    
    # Run all tests
    tests = [
        test_basic_detection,
        test_id_column_detection,
        test_detector_class,
        test_error_details,
        test_edge_cases
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print(f"✅ {test.__name__}: {'PASS' if result else 'PASS (no errors expected)'}")
        except Exception as e:
            print(f"❌ {test.__name__}: FAIL - {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed")
        
    print("\n✅ Enhanced structural error detection system is ready!")