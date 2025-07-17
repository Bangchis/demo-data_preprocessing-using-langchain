"""
Enhanced Structural Error Detection System
Provides advanced detection of structural errors in DataFrames including:
- Missing ID columns
- Column collapse issues
- Inconsistent row patterns
- Header-like rows in data body
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import warnings


class StructuralErrorDetector:
    """Advanced structural error detection for DataFrames"""
    
    def __init__(self, 
                 id_columns: Optional[List[str]] = None,
                 text_length_threshold: int = 100,
                 delimiter_threshold: int = 3,
                 pattern_similarity_threshold: float = 0.7,
                 min_id_fill_rate: float = 0.8):
        """
        Initialize the structural error detector
        
        Args:
            id_columns: List of columns that should act as IDs (critical columns)
            text_length_threshold: Max length for single cell content
            delimiter_threshold: Min delimiter count to flag as structural error
            pattern_similarity_threshold: Threshold for row pattern similarity
            min_id_fill_rate: Minimum fill rate required for ID columns
        """
        self.id_columns = id_columns or []
        self.text_length_threshold = text_length_threshold
        self.delimiter_threshold = delimiter_threshold
        self.pattern_similarity_threshold = pattern_similarity_threshold
        self.min_id_fill_rate = min_id_fill_rate
        
    def detect_all_structural_errors(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive structural error detection
        
        Returns:
            Dict containing all detected error types and their masks
        """
        results = {
            'masks': {},
            'reports': {},
            'summary': {}
        }
        
        # 1. Original detection methods (enhanced)
        results['masks']['long_single_cell'] = self._detect_long_single_cells(df)
        results['masks']['delimiter_errors'] = self._detect_delimiter_errors(df)
        
        # 2. New enhanced detection methods
        results['masks']['missing_id_columns'] = self._detect_missing_id_columns(df)
        results['masks']['column_collapse'] = self._detect_column_collapse(df)
        results['masks']['inconsistent_patterns'] = self._detect_inconsistent_patterns(df)
        results['masks']['header_like_rows'] = self._detect_header_like_rows(df)
        results['masks']['data_type_inconsistency'] = self._detect_data_type_inconsistency(df)
        results['masks']['unusual_null_patterns'] = self._detect_unusual_null_patterns(df)
        
        # 3. Combine all masks
        all_masks = list(results['masks'].values())
        results['masks']['combined'] = pd.Series(False, index=df.index)
        for mask in all_masks:
            results['masks']['combined'] = results['masks']['combined'] | mask
        
        # 4. Generate reports for each error type
        for error_type, mask in results['masks'].items():
            if error_type != 'combined':
                results['reports'][error_type] = self._generate_error_report(
                    df, mask, error_type
                )
        
        # 5. Generate summary
        results['summary'] = self._generate_summary(df, results['masks'])
        
        return results
    
    def _detect_long_single_cells(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced detection of rows with abnormally long single cells"""
        n_cols = df.shape[1]
        non_null_count = df.notna().sum(axis=1)
        
        # Check for single cells with excessive length
        mask = pd.Series(False, index=df.index)
        
        for idx, row in df.iterrows():
            if non_null_count[idx] == 1:
                non_null_val = str(row.dropna().iloc[0])
                if len(non_null_val) > self.text_length_threshold:
                    mask[idx] = True
        
        return mask
    
    def _detect_delimiter_errors(self, df: pd.DataFrame) -> pd.Series:
        """Enhanced detection of delimiter-related structural errors"""
        n_cols = df.shape[1]
        non_null_count = df.notna().sum(axis=1)
        mask = pd.Series(False, index=df.index)
        
        delimiters = [',', ';', '|', '\t', ':', '  ']  # Extended delimiter list
        
        for idx, row in df.iterrows():
            if non_null_count[idx] <= 2:  # Focus on rows with very few filled cells
                for cell in row.dropna():
                    cell_str = str(cell)
                    for delimiter in delimiters:
                        if cell_str.count(delimiter) >= self.delimiter_threshold:
                            mask[idx] = True
                            break
                    if mask[idx]:
                        break
        
        return mask
    
    def _detect_missing_id_columns(self, df: pd.DataFrame) -> pd.Series:
        """Detect rows missing critical ID column values"""
        mask = pd.Series(False, index=df.index)
        
        if not self.id_columns:
            # Auto-detect potential ID columns
            potential_id_cols = self._auto_detect_id_columns(df)
            self.id_columns = potential_id_cols
        
        for col in self.id_columns:
            if col in df.columns:
                # Check for missing values in ID columns
                missing_mask = df[col].isna() | (df[col].astype(str).str.strip() == '')
                mask = mask | missing_mask
        
        return mask
    
    def _detect_column_collapse(self, df: pd.DataFrame) -> pd.Series:
        """Detect rows where multiple column values are collapsed into single cells"""
        mask = pd.Series(False, index=df.index)
        
        for idx, row in df.iterrows():
            non_null_cells = row.dropna()
            
            if len(non_null_cells) < len(df.columns) * 0.3:  # Very few filled cells
                for cell in non_null_cells:
                    cell_str = str(cell)
                    
                    # Check for patterns indicating collapsed data
                    patterns = [
                        r'\w+\s+\w+\s+\w+\s+\w+',  # Multiple words/values
                        r'\d+\s+\d+\s+\d+',        # Multiple numbers
                        r'[A-Z]+\s+[A-Z]+\s+[A-Z]+',  # Multiple codes
                        r'[a-zA-Z]+:\s*[^,]+,\s*[a-zA-Z]+:\s*[^,]+',  # Key-value pairs
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, cell_str):
                            mask[idx] = True
                            break
                    
                    if mask[idx]:
                        break
        
        return mask
    
    def _detect_inconsistent_patterns(self, df: pd.DataFrame) -> pd.Series:
        """Detect rows with inconsistent column fill patterns"""
        mask = pd.Series(False, index=df.index)
        
        # Calculate fill pattern for each row
        fill_patterns = []
        for idx, row in df.iterrows():
            pattern = tuple(row.notna().astype(int))
            fill_patterns.append(pattern)
        
        # Find the most common pattern
        pattern_counts = Counter(fill_patterns)
        most_common_pattern = pattern_counts.most_common(1)[0][0]
        
        # Mark rows that deviate significantly from the common pattern
        for idx, pattern in enumerate(fill_patterns):
            similarity = sum(a == b for a, b in zip(pattern, most_common_pattern)) / len(pattern)
            if similarity < self.pattern_similarity_threshold:
                mask.iloc[idx] = True
        
        return mask
    
    def _detect_header_like_rows(self, df: pd.DataFrame) -> pd.Series:
        """Detect rows that look like headers in the middle of data"""
        mask = pd.Series(False, index=df.index)
        
        for idx, row in df.iterrows():
            row_str = ' '.join(str(cell) for cell in row.dropna())
            
            # Check for header-like patterns
            header_indicators = [
                'total', 'sum', 'average', 'mean', 'count',
                'subtotal', 'grand total', 'summary',
                'header', 'column', 'field', 'variable'
            ]
            
            # Check if row contains mostly text and header-like words
            if any(indicator in row_str.lower() for indicator in header_indicators):
                # Additional check: mostly text, few numbers
                words = row_str.split()
                numeric_count = sum(1 for word in words if word.replace('.', '').isdigit())
                if numeric_count < len(words) * 0.3:
                    mask[idx] = True
        
        return mask
    
    def _detect_data_type_inconsistency(self, df: pd.DataFrame) -> pd.Series:
        """Detect rows with unexpected data types in key columns"""
        mask = pd.Series(False, index=df.index)
        
        for col in df.columns:
            if df[col].dtype in ['object', 'string']:
                continue
                
            # For numeric columns, check for text values
            if pd.api.types.is_numeric_dtype(df[col]):
                for idx, value in df[col].items():
                    if pd.notna(value):
                        try:
                            float(value)
                        except (ValueError, TypeError):
                            mask[idx] = True
        
        return mask
    
    def _detect_unusual_null_patterns(self, df: pd.DataFrame) -> pd.Series:
        """Detect rows with unusual null value patterns"""
        mask = pd.Series(False, index=df.index)
        
        # Calculate null rate for each row
        null_rates = df.isna().sum(axis=1) / len(df.columns)
        
        # Find rows with extremely high null rates
        high_null_threshold = 0.8
        high_null_mask = null_rates > high_null_threshold
        
        # Find rows with unusual null patterns compared to similar rows
        for idx, row in df.iterrows():
            if high_null_mask[idx]:
                # Check if the non-null values are in unexpected columns
                non_null_cols = row.dropna().index
                if len(non_null_cols) == 1:
                    # Single non-null value in unexpected column
                    mask[idx] = True
        
        return mask
    
    def _auto_detect_id_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect potential ID columns based on patterns"""
        potential_id_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check column name patterns
            id_patterns = ['id', 'key', 'code', 'number', 'ref', 'identifier']
            if any(pattern in col_lower for pattern in id_patterns):
                potential_id_cols.append(col)
                continue
            
            # Check data characteristics
            if df[col].dtype in ['object', 'string']:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.8:  # High uniqueness
                    potential_id_cols.append(col)
        
        return potential_id_cols
    
    def _generate_error_report(self, df: pd.DataFrame, mask: pd.Series, error_type: str) -> str:
        if mask.sum() == 0:
            return f"✅ No {error_type.replace('_', ' ')} errors detected."

        error_count = mask.sum()
        error_percentage = (error_count / len(df)) * 100

        report = f"⚠️ **{error_type.replace('_', ' ').title()} Errors**\n"
        report += f"   Count: {error_count:,} rows ({error_percentage:.1f}% of data)\n"

        # 🔸 Thay vì chỉ lấy head(3) → lấy **toàn bộ** chỉ số dòng lỗi
        error_indices = list(df[mask].index)
        report += f"   Rows: {error_indices}\n"        # <- Dòng mới

        # (Tùy chọn) nêu chi tiết từng dòng
        for idx in error_indices:                      # <- Lặp toàn bộ
            row_data = df.loc[idx]
            non_null_data = row_data.dropna()
            report += f"   Row {idx}: {len(non_null_data)} non‑null values\n"

        # Khuyến nghị
        recommendations = self._get_error_recommendations(error_type, error_count, error_percentage)
        if recommendations:
            report += f"\n💡 **Recommended Actions:**\n"
            for rec in recommendations:
                report += f"   • {rec}\n"

        return report

    
    def _get_error_recommendations(self, error_type: str, error_count: int, error_percentage: float) -> List[str]:
        """Get actionable recommendations for specific error types"""
        recommendations = []
        
        if error_type == "long_single_cell":
            recommendations.extend([
                "Check if CSV delimiter is correct (comma, semicolon, tab)",
                "Verify that quoted fields are properly escaped",
                "Consider using 'Use the ReAct agent to examine specific rows with: QuickInfo'",
                "If data contains long text, consider splitting into multiple columns"
            ])
        
        elif error_type == "delimiter_errors":
            recommendations.extend([
                "Re-upload file with correct delimiter detection",
                "Check for embedded delimiters in text fields",
                "Use 'CodeRunner' to manually specify delimiter: pd.read_csv(file, delimiter=';')",
                "Consider using text qualifiers (quotes) around text fields"
            ])
        
        elif error_type == "missing_id_columns":
            recommendations.extend([
                "Review data source to ensure ID columns are populated",
                "Use 'Remove rows with missing ID values' or fill with default values",
                "Check if ID columns are in the correct format",
                "Consider using row index as ID if no natural ID exists"
            ])
        
        elif error_type == "column_collapse":
            recommendations.extend([
                "Check original data source for proper column separation",
                "Use text-to-columns functionality to split collapsed data",
                "Review CSV export settings from source system",
                "Consider manual data entry correction for critical records"
            ])
        
        elif error_type == "inconsistent_patterns":
            if error_percentage > 20:
                recommendations.extend([
                    "High inconsistency suggests data source issues - review original data",
                    "Consider standardizing data collection processes",
                    "Use 'Auto-Clean' feature to fix common inconsistencies"
                ])
            else:
                recommendations.extend([
                    "Remove inconsistent rows if they represent data entry errors",
                    "Standardize row patterns through data transformation",
                    "Use 'Missing value imputation' for incomplete rows"
                ])
        
        elif error_type == "header_like_rows":
            recommendations.extend([
                "Remove header rows that appear in the middle of data",
                "Check data export settings to avoid embedded headers",
                "Use 'Remove specific rows' functionality",
                "Review data source for proper formatting"
            ])
        
        elif error_type == "data_type_inconsistency":
            recommendations.extend([
                "Use 'Data type optimization' to fix type inconsistencies",
                "Check for text values in numeric columns",
                "Consider data validation rules at source",
                "Use 'CodeRunner' to manually convert problematic values"
            ])
        
        elif error_type == "unusual_null_patterns":
            recommendations.extend([
                "Investigate source of unusual missing data patterns",
                "Consider if missing data is meaningful (e.g., 'Not Applicable')",
                "Use 'Missing data analysis' to understand patterns",
                "Review data collection process for improvements"
            ])
        
        # Add severity-based recommendations
        if error_percentage > 30:
            recommendations.insert(0, "⚠️ HIGH SEVERITY: Consider reviewing data source quality")
        elif error_percentage > 10:
            recommendations.insert(0, "⚠️ MEDIUM SEVERITY: Address before analysis")
        else:
            recommendations.insert(0, "ℹ️ LOW SEVERITY: Minor issues, can proceed with caution")
        
        return recommendations
    
    def _generate_summary(self, df: pd.DataFrame, masks: Dict) -> Dict:
        """Generate summary statistics for all detected errors"""
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'errors_by_type': {},
            'total_error_rows': masks['combined'].sum(),
            'error_percentage': (masks['combined'].sum() / len(df)) * 100
        }
        
        for error_type, mask in masks.items():
            if error_type != 'combined':
                summary['errors_by_type'][error_type] = {
                    'count': mask.sum(),
                    'percentage': (mask.sum() / len(df)) * 100
                }
        
        return summary


def detect_advanced_structural_errors(df: pd.DataFrame,
                                     id_columns: Optional[List[str]] = None,
                                     text_length_threshold: int = 100,
                                     delimiter_threshold: int = 3,
                                     pattern_similarity_threshold: float = 0.7,
                                     min_id_fill_rate: float = 0.8) -> Tuple[pd.Series, str]:
    """
    Main function for advanced structural error detection
    
    Args:
        df: DataFrame to analyze
        id_columns: List of critical ID columns
        text_length_threshold: Max length for single cell content
        delimiter_threshold: Min delimiter count to flag as error
        pattern_similarity_threshold: Threshold for row pattern similarity
        min_id_fill_rate: Minimum fill rate required for ID columns
    
    Returns:
        Tuple of (combined_mask, detailed_report)
    """
    detector = StructuralErrorDetector(
        id_columns=id_columns,
        text_length_threshold=text_length_threshold,
        delimiter_threshold=delimiter_threshold,
        pattern_similarity_threshold=pattern_similarity_threshold,
        min_id_fill_rate=min_id_fill_rate
    )
    
    results = detector.detect_all_structural_errors(df)
    
    # Generate comprehensive report
    report = "🔍 **ENHANCED STRUCTURAL ERROR ANALYSIS**\n\n"
    
    # Summary section
    summary = results['summary']
    report += f"📊 **Summary:**\n"
    report += f"   Total rows: {summary['total_rows']:,}\n"
    report += f"   Error rows: {summary['total_error_rows']:,} ({summary['error_percentage']:.1f}%)\n\n"
    
    # Detailed breakdown by error type
    report += f"🔍 **Error Breakdown:**\n"
    for error_type, error_info in summary['errors_by_type'].items():
        if error_info['count'] > 0:
            report += f"   • {error_type.replace('_', ' ').title()}: {error_info['count']} rows ({error_info['percentage']:.1f}%)\n"
    
    report += "\n"
    
    # Individual error reports
    for error_type, error_report in results['reports'].items():
        if results['masks'][error_type].sum() > 0:
            report += f"{error_report}\n\n"
    
    # Recommendations
    if summary['total_error_rows'] > 0:
        report += f"💡 **Overall Recommendations:**\n"
        
        # Priority-based recommendations
        error_percentage = summary['error_percentage']
        if error_percentage > 25:
            report += f"   🔴 **CRITICAL**: {error_percentage:.1f}% of data has issues - immediate attention required\n"
            report += f"   • Use 'Auto-Clean' feature to automatically fix common issues\n"
            report += f"   • Review data source quality and collection processes\n"
            report += f"   • Consider re-exporting data with proper formatting\n"
        elif error_percentage > 10:
            report += f"   🟡 **MODERATE**: {error_percentage:.1f}% of data has issues - address before analysis\n"
            report += f"   • Use data quality dashboard to identify specific issues\n"
            report += f"   • Apply selective cleaning based on error types\n"
            report += f"   • Review most problematic rows manually\n"
        else:
            report += f"   🟢 **MINOR**: {error_percentage:.1f}% of data has issues - can proceed with caution\n"
            report += f"   • Monitor data quality over time\n"
            report += f"   • Apply light cleaning if needed\n"
        
        # Specific actions
        report += f"\n   **Immediate Actions:**\n"
        report += f"   • Open 'Data Quality Dashboard' → 'Auto-Clean' tab\n"
        report += f"   • Run automatic analysis and cleaning\n"
        report += f"   • Review cleaning results and re-analyze\n"
        report += f"   • Use ReAct agent for custom cleaning commands\n"
        
        # Prevention recommendations
        report += f"\n   **Prevention for Future:**\n"
        report += f"   • Implement data validation at source\n"
        report += f"   • Standardize data export procedures\n"
        report += f"   • Use consistent delimiters and encoding\n"
        report += f"   • Test data uploads with small samples first\n"
    
    return results['masks']['combined'], report


def get_structural_error_details(df: pd.DataFrame, mask: pd.Series, max_examples: int = 5) -> str:
    """
    Get detailed information about specific structural errors
    
    Args:
        df: Original DataFrame
        mask: Boolean mask indicating structural errors
        max_examples: Maximum number of examples to show
    
    Returns:
        Detailed error analysis string
    """
    if mask.sum() == 0:
        return "✅ No structural errors to analyze."
    
    error_rows = df[mask]
    
    report = f"📋 **STRUCTURAL ERROR DETAILS**\n\n"
    report += f"Analyzing {min(len(error_rows), max_examples)} of {len(error_rows)} error rows:\n\n"
    
    for i, (idx, row) in enumerate(error_rows.head(max_examples).iterrows()):
        report += f"**Row {idx}:**\n"
        
        # Basic info
        non_null_count = row.notna().sum()
        report += f"   Non-null cells: {non_null_count}/{len(row)}\n"
        
        # Show non-null values
        non_null_data = row.dropna()
        if len(non_null_data) > 0:
            report += f"   Values: {dict(non_null_data.head(3))}\n"
        
        # Analyze potential issues
        issues = []
        for col, value in non_null_data.items():
            value_str = str(value)
            if len(value_str) > 50:
                issues.append(f"Long text in '{col}' ({len(value_str)} chars)")
            if ',' in value_str and value_str.count(',') > 2:
                issues.append(f"Multiple commas in '{col}'")
        
        if issues:
            report += f"   Issues: {', '.join(issues)}\n"
        
        report += "\n"
    
    return report