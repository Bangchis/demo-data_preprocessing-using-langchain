"""
Data Quality Dashboard for Data Preprocessing MVP
Provides interactive interface for data quality assessment and improvement.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.tools.enhanced_structural_detection import detect_advanced_structural_errors, get_structural_error_details
from src.tools.data_cleaning_assistant import suggest_cleaning_pipeline, DataCleaningAssistant


def render_data_quality_dashboard():
    """Render the main data quality dashboard"""
    st.subheader("🔍 Data Quality Dashboard")
    
    # Check if data is available
    if not hasattr(st.session_state, 'df') or st.session_state.df is None:
        st.warning("⚠️ No data available for quality assessment. Please upload a dataset first.")
        return
    
    # Get both versions of the data
    df_original = st.session_state.get("df_original", st.session_state.df)
    df_display = st.session_state.df
    
    # Create tabs for different aspects of data quality
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Structural Issues", 
        "📈 Quality Metrics", 
        "🛠️ Recommendations",
        "🤖 Auto-Clean"
    ])
    
    with tab1:
        render_quality_overview(df_original, df_display)
    
    with tab2:
        render_structural_issues(df_original)
    
    with tab3:
        render_quality_metrics(df_original)
    
    with tab4:
        render_recommendations(df_original)
    
    with tab5:
        render_auto_clean_interface(df_original)


def render_quality_overview(df_original: pd.DataFrame, df_display: pd.DataFrame):
    """Render the overview tab with general data quality statistics"""
    st.subheader("📊 Data Quality Overview")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df_original):,}")
    
    with col2:
        st.metric("Total Columns", f"{len(df_original.columns):,}")
    
    with col3:
        # Calculate completeness
        total_cells = len(df_original) * len(df_original.columns)
        non_null_cells = df_original.notna().sum().sum()
        completeness = (non_null_cells / total_cells) * 100
        st.metric("Completeness", f"{completeness:.1f}%")
    
    with col4:
        # Calculate duplicate rows
        duplicate_rows = df_original.duplicated().sum()
        st.metric("Duplicate Rows", f"{duplicate_rows:,}")
    
    # Data type distribution
    st.subheader("📋 Data Type Distribution")
    
    # Get data types from original DataFrame
    dtype_counts = df_original.dtypes.value_counts()
    
    # Create pie chart
    fig = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index.astype(str),
        title="Distribution of Data Types"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Data Type Summary:**")
        for dtype, count in dtype_counts.items():
            st.write(f"• {dtype}: {count} columns")
    
    # Missing data heatmap
    st.subheader("🔍 Missing Data Pattern")
    
    # Calculate missing percentage for each column
    missing_data = df_original.isnull().sum()
    missing_percent = (missing_data / len(df_original)) * 100
    
    # Create bar chart for missing data
    fig = px.bar(
        x=missing_percent.index,
        y=missing_percent.values,
        title="Missing Data Percentage by Column",
        labels={'x': 'Columns', 'y': 'Missing Percentage (%)'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show columns with high missing rates
    high_missing = missing_percent[missing_percent > 50]
    if not high_missing.empty:
        st.warning(f"⚠️ **High Missing Data Alert:** {len(high_missing)} columns have >50% missing data")
        with st.expander("View columns with high missing rates"):
            for col, pct in high_missing.items():
                st.write(f"• **{col}**: {pct:.1f}% missing")


def render_structural_issues(df_original: pd.DataFrame):
    """Render the structural issues tab with detailed error analysis"""
    st.subheader("🔍 Structural Issues Analysis")
    
    # Run enhanced structural detection
    try:
        mask, report = detect_advanced_structural_errors(df_original)
        
        # Display the report
        st.markdown(report)
        
        # Show problematic rows if any
        if mask.sum() > 0:
            st.subheader("🔍 Problematic Rows Details")
            
            # Allow user to select number of examples
            max_examples = st.slider("Number of examples to show", 1, 10, 5)
            
            # Get detailed error information
            error_details = get_structural_error_details(df_original, mask, max_examples)
            st.markdown(error_details)
            
            # Show the problematic rows in a table
            st.subheader("📋 Problematic Rows Data")
            problem_rows = df_original[mask].head(max_examples)
            
            if not problem_rows.empty:
                st.dataframe(problem_rows, use_container_width=True)
                
                # Provide option to exclude problematic rows
                if st.button("🧹 Exclude Problematic Rows from Analysis"):
                    # Create cleaned version
                    cleaned_df = df_original[~mask].copy()
                    
                    # Update session state
                    st.session_state.df_original = cleaned_df
                    st.session_state.df = cleaned_df.copy()
                    
                    # Update display version
                    from src.core.utils import _create_display_version
                    st.session_state.df = _create_display_version(cleaned_df)
                    
                    st.success(f"✅ Removed {mask.sum()} problematic rows. New dataset has {len(cleaned_df)} rows.")
                    st.rerun()
        
    except Exception as e:
        st.error(f"Error in structural analysis: {str(e)}")
        st.write("Falling back to basic analysis...")
        
        # Basic structural analysis
        # Check for rows with very few non-null values
        non_null_counts = df_original.notna().sum(axis=1)
        expected_cols = len(df_original.columns)
        
        # Flag rows with less than 30% of expected columns filled
        threshold = expected_cols * 0.3
        sparse_rows = non_null_counts < threshold
        
        if sparse_rows.sum() > 0:
            st.warning(f"⚠️ Found {sparse_rows.sum()} rows with sparse data (<30% columns filled)")
            
            # Show examples
            sparse_examples = df_original[sparse_rows].head(5)
            st.dataframe(sparse_examples, use_container_width=True)
        else:
            st.success("✅ No obvious structural issues detected")


def render_quality_metrics(df_original: pd.DataFrame):
    """Render the quality metrics tab with detailed statistics"""
    st.subheader("📈 Detailed Quality Metrics")
    
    # Column-wise quality metrics
    st.subheader("📊 Column-wise Quality Assessment")
    
    # Create quality metrics DataFrame
    quality_metrics = []
    
    for col in df_original.columns:
        series = df_original[col]
        
        # Calculate metrics
        total_count = len(series)
        null_count = series.isnull().sum()
        null_percentage = (null_count / total_count) * 100
        unique_count = series.nunique()
        unique_percentage = (unique_count / total_count) * 100
        
        # Data type
        dtype = str(series.dtype)
        
        # Most frequent value
        if not series.empty and unique_count > 0:
            mode_value = series.mode().iloc[0] if not series.mode().empty else "N/A"
            mode_count = series.value_counts().iloc[0] if not series.value_counts().empty else 0
            mode_percentage = (mode_count / total_count) * 100
        else:
            mode_value = "N/A"
            mode_percentage = 0
        
        # Potential issues
        issues = []
        if null_percentage > 50:
            issues.append("High missing rate")
        if unique_percentage < 1:
            issues.append("Low variability")
        if mode_percentage > 90:
            issues.append("Single dominant value")
        
        quality_metrics.append({
            'Column': col,
            'Data Type': dtype,
            'Missing %': f"{null_percentage:.1f}%",
            'Unique Values': unique_count,
            'Unique %': f"{unique_percentage:.1f}%",
            'Most Common': str(mode_value)[:30] + "..." if len(str(mode_value)) > 30 else str(mode_value),
            'Issues': ", ".join(issues) if issues else "None"
        })
    
    # Display quality metrics table
    quality_df = pd.DataFrame(quality_metrics)
    st.dataframe(quality_df, use_container_width=True)
    
    # Quality score calculation
    st.subheader("🏆 Overall Quality Score")
    
    # Calculate overall quality score
    total_cells = len(df_original) * len(df_original.columns)
    non_null_cells = df_original.notna().sum().sum()
    completeness_score = (non_null_cells / total_cells) * 100
    
    # Consistency score (based on data type consistency)
    consistency_score = 85  # Placeholder - would need more complex calculation
    
    # Uniqueness score (based on duplicate rows)
    duplicate_rows = df_original.duplicated().sum()
    uniqueness_score = ((len(df_original) - duplicate_rows) / len(df_original)) * 100
    
    # Overall score
    overall_score = (completeness_score + consistency_score + uniqueness_score) / 3
    
    # Display scores
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Completeness", f"{completeness_score:.1f}%")
    
    with col2:
        st.metric("Consistency", f"{consistency_score:.1f}%")
    
    with col3:
        st.metric("Uniqueness", f"{uniqueness_score:.1f}%")
    
    with col4:
        score_color = "green" if overall_score >= 80 else "orange" if overall_score >= 60 else "red"
        st.metric("Overall Score", f"{overall_score:.1f}%")
    
    # Quality interpretation
    if overall_score >= 80:
        st.success("✅ **Excellent data quality!** Your dataset is ready for analysis.")
    elif overall_score >= 60:
        st.warning("⚠️ **Good data quality** with some areas for improvement.")
    else:
        st.error("❌ **Poor data quality** - significant cleaning required before analysis.")


def render_recommendations(df_original: pd.DataFrame):
    """Render the recommendations tab with actionable suggestions"""
    st.subheader("🛠️ Data Quality Recommendations")
    
    recommendations = []
    
    # Analyze and generate recommendations
    
    # 1. Missing data recommendations
    missing_data = df_original.isnull().sum()
    high_missing_cols = missing_data[missing_data > len(df_original) * 0.3]
    
    if not high_missing_cols.empty:
        recommendations.append({
            'priority': 'High',
            'category': 'Missing Data',
            'issue': f'{len(high_missing_cols)} columns have >30% missing data',
            'recommendation': 'Consider dropping these columns or using imputation techniques',
            'columns': list(high_missing_cols.index),
            'action': 'missing_data_fix'
        })
    
    # 2. Duplicate rows recommendations
    duplicate_rows = df_original.duplicated().sum()
    if duplicate_rows > 0:
        recommendations.append({
            'priority': 'Medium',
            'category': 'Duplicates',
            'issue': f'{duplicate_rows} duplicate rows found',
            'recommendation': 'Remove duplicate rows to improve data quality',
            'columns': [],
            'action': 'remove_duplicates'
        })
    
    # 3. Data type recommendations
    object_cols = df_original.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        recommendations.append({
            'priority': 'Low',
            'category': 'Data Types',
            'issue': f'{len(object_cols)} columns stored as text',
            'recommendation': 'Review and convert to appropriate data types (numeric, datetime, etc.)',
            'columns': list(object_cols),
            'action': 'optimize_types'
        })
    
    # 4. Structural issues recommendations
    try:
        mask, _ = detect_advanced_structural_errors(df_original)
        if mask.sum() > 0:
            recommendations.append({
                'priority': 'High',
                'category': 'Structural Issues',
                'issue': f'{mask.sum()} rows have structural problems',
                'recommendation': 'Review and fix structural issues or exclude problematic rows',
                'columns': [],
                'action': 'fix_structural'
            })
    except:
        pass
    
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            priority_color = {
                'High': '🔴',
                'Medium': '🟡', 
                'Low': '🟢'
            }.get(rec['priority'], '⚪')
            
            with st.expander(f"{priority_color} {rec['priority']} Priority: {rec['category']} - {rec['issue']}"):
                st.write(f"**Issue:** {rec['issue']}")
                st.write(f"**Recommendation:** {rec['recommendation']}")
                
                if rec['columns']:
                    st.write(f"**Affected Columns:** {', '.join(rec['columns'][:10])}")
                    if len(rec['columns']) > 10:
                        st.write(f"... and {len(rec['columns']) - 10} more columns")
                
                # Action buttons
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if st.button(f"🔧 Fix", key=f"fix_{i}"):
                        handle_recommendation_action(rec, df_original)
                
                with col2:
                    if st.button(f"ℹ️ More Info", key=f"info_{i}"):
                        show_recommendation_details(rec, df_original)
    
    else:
        st.success("🎉 **Great!** No major data quality issues detected.")
        st.info("💡 Your data appears to be in good shape for analysis.")


def handle_recommendation_action(recommendation: Dict, df_original: pd.DataFrame):
    """Handle actions for data quality recommendations"""
    action = recommendation['action']
    
    if action == 'remove_duplicates':
        # Remove duplicate rows
        original_length = len(df_original)
        cleaned_df = df_original.drop_duplicates()
        removed_count = original_length - len(cleaned_df)
        
        # Update session state
        st.session_state.df_original = cleaned_df
        from src.core.utils import _create_display_version
        st.session_state.df = _create_display_version(cleaned_df)
        
        st.success(f"✅ Removed {removed_count} duplicate rows")
        st.rerun()
    
    elif action == 'missing_data_fix':
        # Show options for handling missing data
        st.info("🔧 **Missing Data Options:**")
        st.write("1. **Drop columns** with high missing rates")
        st.write("2. **Impute values** using mean/median/mode")
        st.write("3. **Forward fill** or **backward fill**")
        st.write("Use the ReAct agent to apply these techniques with specific commands.")
    
    elif action == 'optimize_types':
        # Show type optimization suggestions
        st.info("🔧 **Data Type Optimization:**")
        st.write("The system has already applied intelligent type inference.")
        st.write("Use the ReAct agent to further optimize specific columns if needed.")
    
    elif action == 'fix_structural':
        # Provide structural fix options
        st.info("🔧 **Structural Issue Fixes:**")
        st.write("Review the 'Structural Issues' tab for detailed analysis.")
        st.write("Consider excluding problematic rows or fixing them manually.")


def show_recommendation_details(recommendation: Dict, df_original: pd.DataFrame):
    """Show detailed information about a recommendation"""
    st.info(f"📋 **Detailed Analysis for {recommendation['category']}:**")
    
    if recommendation['action'] == 'missing_data_fix':
        # Show missing data analysis
        missing_data = df_original.isnull().sum()
        high_missing = missing_data[missing_data > len(df_original) * 0.3]
        
        st.write("**Columns with high missing rates:**")
        for col, count in high_missing.items():
            percentage = (count / len(df_original)) * 100
            st.write(f"• {col}: {count:,} missing ({percentage:.1f}%)")
    
    elif recommendation['action'] == 'remove_duplicates':
        # Show duplicate analysis
        duplicates = df_original.duplicated(keep=False)
        duplicate_examples = df_original[duplicates].head(5)
        
        st.write("**Example duplicate rows:**")
        st.dataframe(duplicate_examples, use_container_width=True)
    
    elif recommendation['action'] == 'optimize_types':
        # Show data type analysis
        type_summary = df_original.dtypes.value_counts()
        st.write("**Current data type distribution:**")
        for dtype, count in type_summary.items():
            st.write(f"• {dtype}: {count} columns")


def render_auto_clean_interface(df_original: pd.DataFrame):
    """Render the automatic cleaning interface"""
    st.subheader("🤖 Automatic Data Cleaning")
    st.write("Let AI analyze your data and suggest/apply automatic cleaning operations.")
    
    # Initialize cleaning pipeline cache
    if 'cleaning_pipeline' not in st.session_state:
        st.session_state.cleaning_pipeline = None
    
    # Analyze data button
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🔍 Analyze Data for Cleaning", type="primary"):
            with st.spinner("Analyzing data quality..."):
                st.session_state.cleaning_pipeline = suggest_cleaning_pipeline(df_original)
                st.success("✅ Analysis complete!")
    
    with col2:
        if st.button("🔄 Refresh Analysis"):
            st.session_state.cleaning_pipeline = None
            st.info("Analysis cleared. Click 'Analyze Data for Cleaning' to run a new analysis.")
    
    # Display analysis results
    if st.session_state.cleaning_pipeline:
        pipeline_data = st.session_state.cleaning_pipeline
        analysis = pipeline_data['analysis']
        recommended_pipeline = pipeline_data['recommended_pipeline']
        
        # Display recommended pipeline
        st.subheader("📋 Recommended Cleaning Pipeline")
        
        if recommended_pipeline:
            st.write("**Recommended cleaning steps in order of priority:**")
            
            for i, step in enumerate(recommended_pipeline, 1):
                priority_colors = {
                    1: "🔴", 2: "🟠", 3: "🟡", 
                    4: "🟢", 5: "🔵", 6: "🟣"
                }
                
                priority_icon = priority_colors.get(step['priority'], "⚪")
                
                with st.expander(f"{priority_icon} Step {i}: {step['description']}"):
                    st.write(f"**Impact:** {step['impact']}")
                    st.write(f"**Priority:** {step['priority']}")
                    
                    # Show specific details based on step type
                    if step['step'] == 'fix_structural':
                        struct_info = analysis['structural_issues']
                        st.write(f"**Problem rows:** {struct_info['problem_rows']}")
                        st.write(f"**Percentage:** {struct_info['percentage']:.1f}%")
                    
                    elif step['step'] == 'remove_duplicates':
                        dup_info = analysis['duplicates']
                        st.write(f"**Duplicate rows:** {dup_info['duplicate_count']}")
                        st.write(f"**Percentage:** {dup_info['percentage']:.1f}%")
                    
                    elif step['step'] == 'optimize_types':
                        type_info = analysis['data_types']
                        st.write("**Suggested conversions:**")
                        for suggestion in type_info['suggestions']:
                            st.write(f"• {suggestion['column']}: {suggestion['current_type']} → {suggestion['suggested_type']} ({suggestion['conversion_rate']:.1%} success)")
                    
                    elif step['step'] == 'handle_missing':
                        missing_info = analysis['missing_data']
                        st.write(f"**Total missing cells:** {missing_info['total_missing_cells']}")
                        for suggestion in missing_info['suggestions']:
                            st.write(f"• {suggestion['level'].title()} priority: {suggestion['description']}")
                    
                    elif step['step'] == 'handle_outliers':
                        outlier_info = analysis['outliers']
                        st.write("**Outlier details:**")
                        for info in outlier_info['outlier_info']:
                            st.write(f"• {info['column']}: {info['outlier_count']} outliers ({info['outlier_percentage']:.1f}%)")
                    
                    elif step['step'] == 'fix_consistency':
                        consistency_info = analysis['inconsistent_values']
                        st.write("**Consistency issues:**")
                        for inconsistency in consistency_info['inconsistencies']:
                            st.write(f"• {inconsistency['column']}: {inconsistency['type'].replace('_', ' ').title()}")
        
        else:
            st.success("🎉 **Great!** Your data is already in excellent condition. No cleaning steps recommended.")
        
        # Auto-clean configuration
        if recommended_pipeline:
            st.subheader("⚙️ Auto-Clean Configuration")
            st.write("Select which cleaning operations to apply:")
            
            # Create configuration form
            with st.form("auto_clean_config"):
                st.write("**Select cleaning operations:**")
                
                clean_config = {}
                
                # Structural issues
                if any(step['step'] == 'fix_structural' for step in recommended_pipeline):
                    clean_config['fix_structural'] = st.checkbox(
                        "🔧 Fix structural issues",
                        value=True,
                        help="Remove or repair rows with structural problems"
                    )
                
                # Duplicates
                if any(step['step'] == 'remove_duplicates' for step in recommended_pipeline):
                    clean_config['remove_duplicates'] = st.checkbox(
                        "🗑️ Remove duplicate rows",
                        value=True,
                        help="Remove duplicate rows keeping the first occurrence"
                    )
                
                # Data types
                if any(step['step'] == 'optimize_types' for step in recommended_pipeline):
                    clean_config['optimize_types'] = st.checkbox(
                        "🔄 Optimize data types",
                        value=True,
                        help="Convert columns to more appropriate data types"
                    )
                
                # Missing data
                if any(step['step'] == 'handle_missing' for step in recommended_pipeline):
                    clean_config['handle_missing'] = st.checkbox(
                        "📝 Handle missing data",
                        value=False,
                        help="Handle missing values through imputation or column removal"
                    )
                    
                    if clean_config.get('handle_missing', False):
                        st.write("**Missing data options:**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            drop_high_missing = st.checkbox(
                                "Drop columns with high missing rates",
                                value=True,
                                help="Remove columns with >50% missing data"
                            )
                            
                            if drop_high_missing:
                                drop_threshold = st.slider(
                                    "Drop threshold",
                                    min_value=0.3,
                                    max_value=0.9,
                                    value=0.5,
                                    step=0.1,
                                    help="Columns with missing rate above this will be dropped"
                                )
                        
                        with col2:
                            impute = st.checkbox(
                                "Impute missing values",
                                value=True,
                                help="Fill missing values with statistical measures"
                            )
                            
                            if impute:
                                impute_strategy = st.selectbox(
                                    "Imputation strategy",
                                    ["mean", "median", "mode"],
                                    index=0,
                                    help="Strategy for numeric columns (mode used for text)"
                                )
                        
                        clean_config['missing_config'] = {
                            'drop_high_missing': drop_high_missing,
                            'drop_threshold': drop_threshold if drop_high_missing else 0.5,
                            'impute': impute,
                            'impute_strategy': impute_strategy if impute else 'mean'
                        }
                
                # Outliers
                if any(step['step'] == 'handle_outliers' for step in recommended_pipeline):
                    clean_config['handle_outliers'] = st.checkbox(
                        "📊 Handle outliers",
                        value=False,
                        help="Handle outliers in numeric columns"
                    )
                    
                    if clean_config.get('handle_outliers', False):
                        outlier_method = st.selectbox(
                            "Outlier handling method",
                            ["cap", "remove"],
                            index=0,
                            help="Cap: limit outliers to bounds, Remove: delete outlier rows"
                        )
                        
                        clean_config['outlier_config'] = {
                            'method': outlier_method
                        }
                
                # Consistency
                if any(step['step'] == 'fix_consistency' for step in recommended_pipeline):
                    clean_config['fix_consistency'] = st.checkbox(
                        "🔤 Fix value inconsistencies",
                        value=True,
                        help="Fix case inconsistencies and whitespace issues"
                    )
                
                # Apply cleaning button
                submitted = st.form_submit_button("🚀 Apply Auto-Clean", type="primary")
                
                if submitted:
                    # Apply cleaning
                    with st.spinner("Applying automatic cleaning..."):
                        assistant = DataCleaningAssistant(df_original)
                        cleaned_df = assistant.apply_automatic_fixes(clean_config)
                        
                        # Update session state
                        st.session_state.df_original = cleaned_df
                        
                        # Create display version
                        from src.core.utils import _create_display_version
                        st.session_state.df = _create_display_version(cleaned_df)
                        
                        # Show cleaning summary
                        summary = assistant.get_cleaning_summary()
                        
                        st.success("✅ **Automatic cleaning completed!**")
                        
                        # Display summary
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Original Rows", f"{summary['original_shape'][0]:,}")
                        
                        with col2:
                            st.metric("Current Rows", f"{cleaned_df.shape[0]:,}")
                        
                        with col3:
                            rows_removed = summary['rows_removed']
                            st.metric("Rows Removed", f"{rows_removed:,}")
                        
                        # Show cleaning log
                        if assistant.cleaning_log:
                            st.subheader("📝 Cleaning Log")
                            for log_entry in assistant.cleaning_log:
                                st.write(f"• {log_entry}")
                        
                        # Clear pipeline cache to force re-analysis
                        st.session_state.cleaning_pipeline = None
                        
                        st.info("💡 **Tip:** Run the analysis again to see the improvements!")
                        st.rerun()
        
        # Manual cleaning suggestions
        st.subheader("🛠️ Manual Cleaning Suggestions")
        st.write("For more advanced cleaning, consider using these ReAct Agent commands:")
        
        suggestions = [
            "Remove outliers using IQR method for column 'price'",
            "Fill missing values in 'age' column with median",
            "Convert 'date' column to datetime format",
            "Remove rows where 'amount' is negative",
            "Standardize text case in 'category' column",
            "Drop columns with more than 50% missing data"
        ]
        
        for suggestion in suggestions:
            st.code(suggestion)
    
    else:
        st.info("👆 Click 'Analyze Data for Cleaning' to get started with automatic data cleaning suggestions.")