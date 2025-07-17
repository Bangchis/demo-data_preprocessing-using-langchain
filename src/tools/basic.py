import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import Optional, List
from langchain.tools import Tool

from .enhanced_structural_detection import detect_advanced_structural_errors, get_structural_error_details
# from src.ui.data_view_controller import get_current_dataframe_for_agent  # Simplified: removed complex view controller


def detect_structural_rows(df: pd.DataFrame,
                           text_len_thr: int = 100,
                           max_list: int = 5,
                           use_enhanced: bool = True,
                           id_columns: Optional[List[str]] = None):
    """
    Return (mask, report) for structural‑error rows.
    
    Args:
        df: DataFrame to analyze
        text_len_thr: Text length threshold for single cell detection
        max_list: Maximum examples to show in report
        use_enhanced: Whether to use enhanced detection (default: True)
        id_columns: List of critical ID columns for enhanced detection
    
    Returns:
        Tuple of (mask, report)
    """
    if use_enhanced:
        # Use enhanced detection system
        try:
            mask, report = detect_advanced_structural_errors(
                df=df,
                id_columns=id_columns,
                text_length_threshold=text_len_thr,
                delimiter_threshold=3,
                pattern_similarity_threshold=0.1,
                min_id_fill_rate=0.8
            )
            return mask, report
        except Exception as e:
            st.warning(f"Enhanced detection failed, falling back to basic detection: {e}")
            # Fall back to basic detection
            use_enhanced = False
    
    if not use_enhanced:
        # Original basic detection logic
        n_cols = df.shape[1]
        nn_cnt = df.notna().sum(axis=1)

        long_single = (nn_cnt == 1) & df.apply(
            lambda r: len(str(r.dropna().iloc[0])) > text_len_thr, axis=1
        )

        def with_delim(row):
            if nn_cnt[row.name] != 1: return False
            cell = str(row.dropna().iloc[0])
            return cell.count(",") >= n_cols-3 or cell.count(";") >= n_cols-3

        delim_single = df.apply(with_delim, axis=1)
        mask = long_single | delim_single

        if mask.sum() == 0:
            rep = "✅ Không phát hiện hàng lỗi cấu trúc."
        else:
            sample = df[mask].head(max_list).to_string(max_cols=5, max_rows=max_list)
            rep = (f"⚠️ Phát hiện {mask.sum():,} hàng nghi lỗi cấu trúc "
                   f"({mask.sum()/len(df)*100:.1f}% dữ liệu).\n"
                   f"Ví dụ:\n```\n{sample}\n```")
        return mask, rep


def quick_info(_: str) -> str:
    """Get quick DataFrame info and basic statistics with structural error detection"""
    if st.session_state.df is None:
        return "❌ No dataframe loaded. Please upload a file first."
    
    try:
        # Use df_original (real data) if available, fallback to df
        df_full = st.session_state.get("df_original", st.session_state.df)
        
        # Simplified: show data type info
        view_indicator = "📊 DATA"
        mask, struct_rep = detect_structural_rows(df_full)
        df = df_full[~mask]
        
        # Basic info
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            desc = df[numeric_cols].describe().round(2)
            desc_str = desc.to_string()
        else:
            desc_str = "No numeric columns found"
        
        result = f"{view_indicator} **DataFrame Info (clean, {len(df):,} rows):**\n```\n{info_str}\n```\n\n"
        result += f"📈 **Numeric Statistics:**\n```\n{desc_str}\n```\n\n"
        result += struct_rep
        
        return result
        
    except Exception as e:
        return f"❌ Error getting info: {str(e)}"
    


def full_info(_: str) -> str:
    """Comprehensive report (schema, missing, duplicates, outliers, structural rows)"""
    if st.session_state.df is None:
        return "❌ No dataframe loaded. Please upload a file first."
    
    try:
        # Use df_original (real data) if available, fallback to df
        df_full = st.session_state.get("df_original", st.session_state.df)
        view_indicator = "📊 DATA"
        mask, struct_rep = detect_structural_rows(df_full)
        df = df_full[~mask]
        
        # Schema information
        schema = pd.DataFrame({
            "dtype":  df.dtypes.astype(str),
            "%NA":    (df.isnull().mean()*100).round(1),
            "unique": df.nunique(dropna=False)
        })
        
        # Missing data analysis
        miss = (df.isnull().mean()*100).sort_values(ascending=False).head(10)
        
        # Duplicates
        dup = df.duplicated().sum()
        
        # Outliers analysis
        num_df = df.select_dtypes("number")
        outliers = {}
        for c in num_df.columns:
            Q1 = num_df[c].quantile(0.25)
            Q3 = num_df[c].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[c] = int(((num_df[c] < lower_bound) | (num_df[c] > upper_bound)).sum())
        
        out_top = pd.Series(outliers).sort_values(ascending=False).head(10)
        
        result = f"{view_indicator} **Comprehensive Report**\n\n"
        result += f"{struct_rep}\n\n"
        result += f"📑 **Schema (dtype / %NA / unique):**\n```\n{schema.to_string()}\n```\n\n"
        result += f"🕳️ **Top Missing (%):**\n```\n{miss.to_string() if miss.any() else 'No missing values'}\n```\n\n"
        result += f"📄 **Duplicates:** {dup:,} rows ({dup/len(df)*100:.1f}%)\n\n"
        result += f"🚩 **Outliers (IQR top 10):**\n```\n{out_top.to_string()}\n```"
        
        return result
        
    except Exception as e:
        return f"❌ Error generating full info: {str(e)}"


def missing_report(_: str) -> str:
    """Report missing data percentage by column"""
    if st.session_state.df is None:
        return "❌ No dataframe loaded. Please upload a file first."
    
    try:
        # Use original DataFrame with proper types if available
        df = st.session_state.get("df_original", st.session_state.df)
        
        # Calculate missing percentages
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        if len(missing_pct) == 0:
            return "✅ No missing data found in any column!"
        
        result = f"📊 **Missing Data Report (Top 10):**\n\n"
        
        for col, pct in missing_pct.head(10).items():
            result += f"• **{col}**: {pct}% ({int(pct/100 * len(df))} out of {len(df)} rows)\n"
        
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        overall_pct = (total_missing / total_cells * 100).round(2)
        
        result += f"\n📋 **Overall**: {overall_pct}% missing ({total_missing:,} out of {total_cells:,} cells)"
        
        return result
        
    except Exception as e:
        return f"❌ Error calculating missing data: {str(e)}"


def duplicate_check(_: str) -> str:
    """Check for duplicate rows"""
    if st.session_state.df is None:
        return "❌ No dataframe loaded. Please upload a file first."
    
    try:
        # Use original DataFrame with proper types if available
        df = st.session_state.get("df_original", st.session_state.df)
        
        # Check for duplicates
        duplicates = df.duplicated()
        num_duplicates = duplicates.sum()
        
        if num_duplicates == 0:
            return "✅ No duplicate rows found!"
        
        # Get some examples
        duplicate_rows = df[duplicates].head(3)
        
        result = f"⚠️ **Duplicate Rows Found:**\n\n"
        result += f"• Total duplicates: {num_duplicates} rows ({num_duplicates/len(df)*100:.1f}%)\n"
        result += f"• Unique rows: {len(df) - num_duplicates}\n\n"
        
        result += f"📋 **Example duplicate rows:**\n```\n{duplicate_rows.to_string()}\n```"
        
        return result
        
    except Exception as e:
        return f"❌ Error checking duplicates: {str(e)}"


def column_summary(_: str) -> str:
    """Summary of column types"""
    if st.session_state.df is None:
        return "❌ No dataframe loaded. Please upload a file first."
    
    try:
        # Use original DataFrame with proper types if available
        df = st.session_state.get("df_original", st.session_state.df)
        
        # Count by data type
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        boolean_cols = df.select_dtypes(include=['bool']).columns
        
        result = f"📊 **Column Summary:**\n\n"
        result += f"• **Total columns**: {len(df.columns)}\n"
        result += f"• **Numeric**: {len(numeric_cols)} columns\n"
        result += f"• **Categorical**: {len(categorical_cols)} columns\n"
        result += f"• **Datetime**: {len(datetime_cols)} columns\n"
        result += f"• **Boolean**: {len(boolean_cols)} columns\n\n"
        
        # List columns by type
        if len(numeric_cols) > 0:
            result += f"🔢 **Numeric columns**: {', '.join(numeric_cols[:10])}\n"
        if len(categorical_cols) > 0:
            result += f"📝 **Categorical columns**: {', '.join(categorical_cols[:10])}\n"
        if len(datetime_cols) > 0:
            result += f"📅 **Datetime columns**: {', '.join(datetime_cols)}\n"
        if len(boolean_cols) > 0:
            result += f"✅ **Boolean columns**: {', '.join(boolean_cols)}\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error getting column summary: {str(e)}"


def value_counts(column: str) -> str:
    """Get top 10 values for a specific column"""
    if st.session_state.df is None:
        return "❌ No dataframe loaded. Please upload a file first."
    
    try:
        # Use original DataFrame with proper types if available
        df = st.session_state.get("df_original", st.session_state.df)
        
        if column not in df.columns:
            available_cols = ", ".join(df.columns[:10])
            return f"❌ Column '{column}' not found. Available columns: {available_cols}"
        
        # Get value counts
        value_counts = df[column].value_counts(dropna=False).head(10)
        
        result = f"📊 **Value Counts for '{column}':**\n\n"
        
        for value, count in value_counts.items():
            percentage = (count / len(df) * 100).round(1)
            result += f"• **{value}**: {count} ({percentage}%)\n"
        
        # Additional info
        unique_count = df[column].nunique()
        total_count = len(df)
        
        result += f"\n📋 **Summary:**\n"
        result += f"• Total values: {total_count}\n"
        result += f"• Unique values: {unique_count}\n"
        result += f"• Most common: {value_counts.index[0]} ({value_counts.iloc[0]} times)\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error getting value counts: {str(e)}"


def strong_correlations(_: str) -> str:
    """Find strong correlations (>0.7) between numeric columns"""
    if st.session_state.df is None:
        return "❌ No dataframe loaded. Please upload a file first."
    
    try:
        # Use original DataFrame with proper types if available
        df = st.session_state.get("df_original", st.session_state.df)
        
        # Get only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return "❌ No numeric columns found for correlation analysis"
        
        if len(numeric_df.columns) < 2:
            return "❌ Need at least 2 numeric columns for correlation analysis"
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().abs()
        
        # Get upper triangle (avoid duplicates)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find strong correlations
        strong_corr = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                if upper_tri.loc[idx, col] > 0.7:
                    strong_corr.append((idx, col, upper_tri.loc[idx, col]))
        
        if not strong_corr:
            return "✅ No strong correlations (>0.7) found between numeric columns"
        
        # Sort by correlation strength
        strong_corr.sort(key=lambda x: x[2], reverse=True)
        
        result = f"📊 **Strong Correlations (>0.7):**\n\n"
        
        for col1, col2, corr_val in strong_corr[:10]:
            result += f"• **{col1}** ↔ **{col2}**: {corr_val:.3f}\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error calculating correlations: {str(e)}"


def outlier_check(column: str) -> str:
    """Check for outliers using IQR method"""
    if st.session_state.df is None:
        return "❌ No dataframe loaded. Please upload a file first."
    
    try:
        # Use original DataFrame with proper types if available
        df = st.session_state.get("df_original", st.session_state.df)
        
        if column not in df.columns:
            available_cols = ", ".join(df.select_dtypes(include=[np.number]).columns[:10])
            return f"❌ Column '{column}' not found. Available numeric columns: {available_cols}"
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"❌ Column '{column}' is not numeric. Cannot check for outliers."
        
        # Calculate IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find outliers
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        num_outliers = len(outliers)
        
        result = f"📊 **Outlier Analysis for '{column}' (IQR Method):**\n\n"
        result += f"• **Q1**: {Q1:.2f}\n"
        result += f"• **Q3**: {Q3:.2f}\n"
        result += f"• **IQR**: {IQR:.2f}\n"
        result += f"• **Lower bound**: {lower_bound:.2f}\n"
        result += f"• **Upper bound**: {upper_bound:.2f}\n\n"
        
        if num_outliers == 0:
            result += "✅ No outliers found!"
        else:
            percentage = (num_outliers / len(df) * 100).round(1)
            result += f"⚠️ **{num_outliers} outliers found ({percentage}% of data)**\n\n"
            
            # Show some examples
            result += f"📋 **Example outliers:**\n"
            sample_outliers = outliers[column].head(5)
            for val in sample_outliers:
                result += f"• {val:.2f}\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error checking outliers: {str(e)}"


def schema_report(_: str) -> str:
    """Complete schema report with data types and missing percentages"""
    if st.session_state.df is None:
        return "❌ No dataframe loaded. Please upload a file first."
    
    try:
        # Use original DataFrame with proper types if available
        df = st.session_state.get("df_original", st.session_state.df)
        
        # Create schema report
        schema_data = []
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_count = df[col].count()
            null_count = df[col].isnull().sum()
            null_percentage = (null_count / len(df) * 100).round(1)
            
            schema_data.append({
                'Column': col,
                'Data Type': dtype,
                'Non-Null Count': non_null_count,
                'Null Count': null_count,
                'Null %': null_percentage
            })
        
        # Convert to DataFrame for better formatting
        schema_df = pd.DataFrame(schema_data)
        
        result = f"📊 **Complete Schema Report:**\n\n"
        result += f"📋 **Dataset Info:**\n"
        result += f"• Total rows: {len(df):,}\n"
        result += f"• Total columns: {len(df.columns)}\n"
        result += f"• Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB\n\n"
        
        result += f"📝 **Column Details:**\n```\n{schema_df.to_string(index=False)}\n```"
        
        return result
        
    except Exception as e:
        return f"❌ Error generating schema report: {str(e)}"


def structural_error_analysis(id_columns_input: str = "") -> str:
    """
    Advanced structural error analysis with enhanced detection methods
    
    Args:
        id_columns_input: Comma-separated list of ID column names (optional)
    
    Returns:
        Comprehensive structural error analysis report
    """
    if st.session_state.df is None:
        return "❌ No dataframe loaded. Please upload a file first."
    
    try:
        # Use original DataFrame with proper types if available
        df = st.session_state.get("df_original", st.session_state.df)
        
        # Parse ID columns input
        id_columns = None
        if id_columns_input.strip():
            id_columns = [col.strip() for col in id_columns_input.split(",")]
            # Validate columns exist
            invalid_cols = [col for col in id_columns if col not in df.columns]
            if invalid_cols:
                return f"❌ Invalid ID columns: {invalid_cols}. Available columns: {list(df.columns)}"
        
        # Run enhanced structural error detection
        mask, report = detect_advanced_structural_errors(
            df=df,
            id_columns=id_columns,
            text_length_threshold=100,
            delimiter_threshold=3,
            pattern_similarity_threshold=0.7,
            min_id_fill_rate=0.8
        )
        
        # Add detailed analysis if errors found
        if mask.sum() > 0:
            details = get_structural_error_details(df, mask, max_examples=3)
            report += f"\n\n{details}"
        
        return report
        
    except Exception as e:
        return f"❌ Error in structural error analysis: {str(e)}"


def basic_structural_check(query: str = "") -> str:
    """
    Basic structural error check using original algorithm
    
    Args:
        query: Not used, for tool compatibility
    
    Returns:
        Basic structural error report
    """
    if st.session_state.df is None:
        return "❌ No dataframe loaded. Please upload a file first."
    
    try:
        # Use original DataFrame with proper types if available
        df = st.session_state.get("df_original", st.session_state.df)
        
        # Run basic structural error detection
        mask, report = detect_structural_rows(
            df=df,
            text_len_thr=100,
            max_list=5,
            use_enhanced=False,  # Force basic detection
            id_columns=None
        )
        
        return report
        
    except Exception as e:
        return f"❌ Error in basic structural check: {str(e)}"


# Create tools
QuickInfoTool = Tool(
    name="QuickInfo",
    func=quick_info,
    description="Get quick DataFrame info and basic statistics. No input required."
)

MissingReportTool = Tool(
    name="MissingReport", 
    func=missing_report,
    description="Report missing data percentage by column. No input required."
)

DuplicateCheckTool = Tool(
    name="DuplicateCheck",
    func=duplicate_check,
    description="Check for duplicate rows in the dataframe. No input required."
)

ColumnSummaryTool = Tool(
    name="ColumnSummary",
    func=column_summary,
    description="Get summary of column types (numeric, categorical, etc.). No input required."
)

ValueCountsTool = Tool(
    name="ValueCounts",
    func=value_counts,
    description="Get top 10 values for a specific column. Input: column name as string."
)

CorrelationTool = Tool(
    name="CorrelationMatrix",
    func=strong_correlations,
    description="Find strong correlations (>0.7) between numeric columns. No input required."
)

OutlierTool = Tool(
    name="OutlierCheck",
    func=outlier_check,
    description="Check for outliers using IQR method for a specific column. Input: column name as string."
)

SchemaReportTool = Tool(
    name="SchemaReport",
    func=schema_report,
    description="Generate complete schema report with data types and missing percentages. No input required."
)

FullInfoTool = Tool(
    name="FullInfo",
    func=full_info,
    description="Comprehensive report (schema, missing, duplicates, outliers, structural rows). No input required."
)

StructuralErrorAnalysisTool = Tool(
    name="StructuralErrorAnalysis",
    func=structural_error_analysis,
    description="Advanced structural error analysis with enhanced detection methods. Input: comma-separated list of ID column names (optional, e.g., 'id,user_id,code'). Detects missing ID columns, column collapse, inconsistent patterns, and more."
)

BasicStructuralCheckTool = Tool(
    name="BasicStructuralCheck",
    func=basic_structural_check,
    description="Basic structural error check using original algorithm. No input required. Use for comparison with enhanced analysis."
)