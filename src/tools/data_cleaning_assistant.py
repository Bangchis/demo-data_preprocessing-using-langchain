"""
Data Cleaning Assistant for Data Preprocessing MVP
Provides automatic data cleaning suggestions and implementations.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
import re
from datetime import datetime

from src.tools.enhanced_structural_detection import detect_advanced_structural_errors


class DataCleaningAssistant:
    """Automated data cleaning assistant with intelligent suggestions"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_log = []
        self.suggestions = []
        
    def analyze_and_suggest(self) -> Dict[str, Any]:
        """Analyze data and generate cleaning suggestions"""
        suggestions = {
            'structural_issues': self._analyze_structural_issues(),
            'missing_data': self._analyze_missing_data(),
            'duplicates': self._analyze_duplicates(),
            'data_types': self._analyze_data_types(),
            'outliers': self._analyze_outliers(),
            'inconsistent_values': self._analyze_inconsistent_values()
        }
        
        return suggestions
    
    def _analyze_structural_issues(self) -> Dict[str, Any]:
        """Analyze structural issues in the data"""
        try:
            mask, report = detect_advanced_structural_errors(self.df)
            
            if mask.sum() == 0:
                return {'issues': False, 'description': 'No structural issues detected'}
            
            problem_rows = mask.sum()
            percentage = (problem_rows / len(self.df)) * 100
            
            return {
                'issues': True,
                'problem_rows': problem_rows,
                'percentage': percentage,
                'description': f'{problem_rows} rows ({percentage:.1f}%) have structural issues',
                'auto_fix_available': True,
                'fix_description': 'Remove or repair problematic rows',
                'mask': mask
            }
        except Exception as e:
            return {
                'issues': False,
                'description': f'Could not analyze structural issues: {str(e)}'
            }
    
    def _analyze_missing_data(self) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing_data = self.df.isnull().sum()
        total_cells = len(self.df)
        
        # Find columns with different levels of missing data
        high_missing = missing_data[missing_data > total_cells * 0.5]  # >50% missing
        medium_missing = missing_data[(missing_data > total_cells * 0.1) & (missing_data <= total_cells * 0.5)]  # 10-50% missing
        low_missing = missing_data[(missing_data > 0) & (missing_data <= total_cells * 0.1)]  # 0-10% missing
        
        suggestions = []
        
        if not high_missing.empty:
            suggestions.append({
                'level': 'high',
                'columns': list(high_missing.index),
                'action': 'consider_dropping',
                'description': f'{len(high_missing)} columns have >50% missing data - consider dropping'
            })
        
        if not medium_missing.empty:
            suggestions.append({
                'level': 'medium',
                'columns': list(medium_missing.index),
                'action': 'impute_or_drop',
                'description': f'{len(medium_missing)} columns have 10-50% missing data - consider imputation or dropping'
            })
        
        if not low_missing.empty:
            suggestions.append({
                'level': 'low',
                'columns': list(low_missing.index),
                'action': 'impute',
                'description': f'{len(low_missing)} columns have <10% missing data - good candidates for imputation'
            })
        
        return {
            'issues': len(suggestions) > 0,
            'total_missing_cells': missing_data.sum(),
            'suggestions': suggestions,
            'auto_fix_available': True
        }
    
    def _analyze_duplicates(self) -> Dict[str, Any]:
        """Analyze duplicate rows"""
        duplicates = self.df.duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count == 0:
            return {'issues': False, 'description': 'No duplicate rows found'}
        
        percentage = (duplicate_count / len(self.df)) * 100
        
        return {
            'issues': True,
            'duplicate_count': duplicate_count,
            'percentage': percentage,
            'description': f'{duplicate_count} duplicate rows ({percentage:.1f}%) found',
            'auto_fix_available': True,
            'fix_description': 'Remove duplicate rows keeping first occurrence'
        }
    
    def _analyze_data_types(self) -> Dict[str, Any]:
        """Analyze data types and suggest optimizations"""
        suggestions = []
        
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check if it could be numeric
                try:
                    numeric_converted = pd.to_numeric(self.df[col], errors='coerce')
                    if not numeric_converted.isna().all():
                        non_null_count = self.df[col].notna().sum()
                        if non_null_count > 0:
                            conversion_rate = numeric_converted.notna().sum() / non_null_count
                            if conversion_rate > 0.8:  # 80% can be converted
                                suggestions.append({
                                    'column': col,
                                    'current_type': 'object',
                                    'suggested_type': 'numeric',
                                    'conversion_rate': conversion_rate,
                                    'description': f'Column {col} can be converted to numeric ({conversion_rate:.1%} success rate)'
                                })
                except:
                    pass
                
                # Check if it could be datetime
                try:
                    datetime_converted = pd.to_datetime(self.df[col], errors='coerce')
                    if not datetime_converted.isna().all():
                        non_null_count = self.df[col].notna().sum()
                        if non_null_count > 0:
                            conversion_rate = datetime_converted.notna().sum() / non_null_count
                            if conversion_rate > 0.8:  # 80% can be converted
                                suggestions.append({
                                    'column': col,
                                    'current_type': 'object',
                                    'suggested_type': 'datetime',
                                    'conversion_rate': conversion_rate,
                                    'description': f'Column {col} can be converted to datetime ({conversion_rate:.1%} success rate)'
                                })
                except:
                    pass
        
        return {
            'issues': len(suggestions) > 0,
            'suggestions': suggestions,
            'auto_fix_available': True
        }
    
    def _analyze_outliers(self) -> Dict[str, Any]:
        """Analyze outliers in numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {'issues': False, 'description': 'No numeric columns to analyze for outliers'}
        
        outlier_info = []
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) < 4:  # Need at least 4 points for IQR
                continue
                
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(data)) * 100
                outlier_info.append({
                    'column': col,
                    'outlier_count': len(outliers),
                    'outlier_percentage': outlier_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'description': f'{len(outliers)} outliers ({outlier_percentage:.1f}%) in {col}'
                })
        
        return {
            'issues': len(outlier_info) > 0,
            'outlier_info': outlier_info,
            'auto_fix_available': True
        }
    
    def _analyze_inconsistent_values(self) -> Dict[str, Any]:
        """Analyze inconsistent values in text columns"""
        text_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(text_cols) == 0:
            return {'issues': False, 'description': 'No text columns to analyze'}
        
        inconsistencies = []
        
        for col in text_cols:
            unique_values = self.df[col].dropna().unique()
            
            if len(unique_values) < 2:
                continue
            
            # Check for case inconsistencies
            case_issues = []
            value_counts = self.df[col].value_counts()
            
            for value in unique_values:
                similar_values = [v for v in unique_values if v.lower() == value.lower() and v != value]
                if similar_values:
                    case_issues.append({
                        'main_value': value,
                        'similar_values': similar_values,
                        'total_count': sum(value_counts.get(v, 0) for v in [value] + similar_values)
                    })
            
            if case_issues:
                inconsistencies.append({
                    'column': col,
                    'type': 'case_inconsistency',
                    'issues': case_issues,
                    'description': f'Case inconsistencies found in {col}'
                })
            
            # Check for whitespace issues
            whitespace_issues = []
            for value in unique_values:
                if isinstance(value, str) and (value != value.strip() or '  ' in value):
                    whitespace_issues.append(value)
            
            if whitespace_issues:
                inconsistencies.append({
                    'column': col,
                    'type': 'whitespace_issue',
                    'issues': whitespace_issues,
                    'description': f'Whitespace issues found in {col}'
                })
        
        return {
            'issues': len(inconsistencies) > 0,
            'inconsistencies': inconsistencies,
            'auto_fix_available': True
        }
    
    def apply_automatic_fixes(self, fix_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply automatic fixes based on configuration"""
        df_cleaned = self.df.copy()
        
        # Apply structural fixes
        if fix_config.get('fix_structural', False):
            structural_analysis = self._analyze_structural_issues()
            if structural_analysis.get('issues', False):
                mask = structural_analysis['mask']
                df_cleaned = df_cleaned[~mask]
                self.cleaning_log.append(f"Removed {mask.sum()} rows with structural issues")
        
        # Apply duplicate removal
        if fix_config.get('remove_duplicates', False):
            original_len = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates()
            removed = original_len - len(df_cleaned)
            if removed > 0:
                self.cleaning_log.append(f"Removed {removed} duplicate rows")
        
        # Apply missing data fixes
        if fix_config.get('handle_missing', False):
            missing_config = fix_config.get('missing_config', {})
            
            # Drop columns with high missing rates
            if missing_config.get('drop_high_missing', False):
                threshold = missing_config.get('drop_threshold', 0.5)
                missing_rates = df_cleaned.isnull().sum() / len(df_cleaned)
                cols_to_drop = missing_rates[missing_rates > threshold].index
                if len(cols_to_drop) > 0:
                    df_cleaned = df_cleaned.drop(columns=cols_to_drop)
                    self.cleaning_log.append(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing data")
            
            # Impute missing values
            if missing_config.get('impute', False):
                impute_strategy = missing_config.get('impute_strategy', 'mean')
                
                for col in df_cleaned.columns:
                    if df_cleaned[col].isnull().sum() > 0:
                        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                            if impute_strategy == 'mean':
                                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                            elif impute_strategy == 'median':
                                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                            elif impute_strategy == 'mode':
                                df_cleaned[col].fillna(df_cleaned[col].mode().iloc[0], inplace=True)
                        else:
                            # For non-numeric, use mode
                            if not df_cleaned[col].mode().empty:
                                df_cleaned[col].fillna(df_cleaned[col].mode().iloc[0], inplace=True)
                
                self.cleaning_log.append(f"Imputed missing values using {impute_strategy} strategy")
        
        # Apply data type optimizations
        if fix_config.get('optimize_types', False):
            type_analysis = self._analyze_data_types()
            if type_analysis.get('issues', False):
                for suggestion in type_analysis['suggestions']:
                    col = suggestion['column']
                    target_type = suggestion['suggested_type']
                    
                    try:
                        if target_type == 'numeric':
                            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                        elif target_type == 'datetime':
                            df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                        
                        self.cleaning_log.append(f"Converted {col} to {target_type}")
                    except Exception as e:
                        self.cleaning_log.append(f"Failed to convert {col} to {target_type}: {str(e)}")
        
        # Apply outlier handling
        if fix_config.get('handle_outliers', False):
            outlier_config = fix_config.get('outlier_config', {})
            method = outlier_config.get('method', 'cap')  # 'cap' or 'remove'
            
            outlier_analysis = self._analyze_outliers()
            if outlier_analysis.get('issues', False):
                for outlier_info in outlier_analysis['outlier_info']:
                    col = outlier_info['column']
                    lower_bound = outlier_info['lower_bound']
                    upper_bound = outlier_info['upper_bound']
                    
                    if method == 'cap':
                        # Cap outliers to bounds
                        df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                        self.cleaning_log.append(f"Capped outliers in {col} to [{lower_bound:.2f}, {upper_bound:.2f}]")
                    elif method == 'remove':
                        # Remove outlier rows
                        outlier_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                        df_cleaned = df_cleaned[~outlier_mask]
                        self.cleaning_log.append(f"Removed {outlier_mask.sum()} outlier rows from {col}")
        
        # Apply consistency fixes
        if fix_config.get('fix_consistency', False):
            consistency_analysis = self._analyze_inconsistent_values()
            if consistency_analysis.get('issues', False):
                for inconsistency in consistency_analysis['inconsistencies']:
                    col = inconsistency['column']
                    issue_type = inconsistency['type']
                    
                    if issue_type == 'case_inconsistency':
                        # Standardize to most common case
                        for issue in inconsistency['issues']:
                            main_value = issue['main_value']
                            similar_values = issue['similar_values']
                            
                            for similar_val in similar_values:
                                df_cleaned[col] = df_cleaned[col].replace(similar_val, main_value)
                        
                        self.cleaning_log.append(f"Fixed case inconsistencies in {col}")
                    
                    elif issue_type == 'whitespace_issue':
                        # Strip whitespace and normalize multiple spaces
                        df_cleaned[col] = df_cleaned[col].str.strip().str.replace(r'\s+', ' ', regex=True)
                        self.cleaning_log.append(f"Fixed whitespace issues in {col}")
        
        return df_cleaned
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get summary of cleaning operations"""
        return {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'cleaning_log': self.cleaning_log,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'columns_removed': self.original_shape[1] - self.df.shape[1]
        }


def suggest_cleaning_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    """Suggest a complete cleaning pipeline for the dataset"""
    assistant = DataCleaningAssistant(df)
    analysis = assistant.analyze_and_suggest()
    
    # Generate recommended cleaning pipeline
    recommended_pipeline = []
    
    # 1. Structural issues (highest priority)
    if analysis['structural_issues'].get('issues', False):
        recommended_pipeline.append({
            'step': 'fix_structural',
            'priority': 1,
            'description': 'Fix structural issues in the data',
            'impact': analysis['structural_issues']['description']
        })
    
    # 2. Duplicates (high priority)
    if analysis['duplicates'].get('issues', False):
        recommended_pipeline.append({
            'step': 'remove_duplicates',
            'priority': 2,
            'description': 'Remove duplicate rows',
            'impact': analysis['duplicates']['description']
        })
    
    # 3. Data types (medium priority)
    if analysis['data_types'].get('issues', False):
        recommended_pipeline.append({
            'step': 'optimize_types',
            'priority': 3,
            'description': 'Optimize data types',
            'impact': f"Convert {len(analysis['data_types']['suggestions'])} columns to better types"
        })
    
    # 4. Missing data (medium priority)
    if analysis['missing_data'].get('issues', False):
        recommended_pipeline.append({
            'step': 'handle_missing',
            'priority': 4,
            'description': 'Handle missing data',
            'impact': f"Address {analysis['missing_data']['total_missing_cells']} missing cells"
        })
    
    # 5. Outliers (lower priority)
    if analysis['outliers'].get('issues', False):
        recommended_pipeline.append({
            'step': 'handle_outliers',
            'priority': 5,
            'description': 'Handle outliers',
            'impact': f"Address outliers in {len(analysis['outliers']['outlier_info'])} columns"
        })
    
    # 6. Consistency (lowest priority)
    if analysis['inconsistent_values'].get('issues', False):
        recommended_pipeline.append({
            'step': 'fix_consistency',
            'priority': 6,
            'description': 'Fix value inconsistencies',
            'impact': f"Fix inconsistencies in {len(analysis['inconsistent_values']['inconsistencies'])} columns"
        })
    
    return {
        'analysis': analysis,
        'recommended_pipeline': sorted(recommended_pipeline, key=lambda x: x['priority']),
        'assistant': assistant
    }