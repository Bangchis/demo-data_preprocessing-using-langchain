"""
Data View Controller for Data Preprocessing MVP
Provides interface for switching between raw/processed/display data views
and shows transparent processing pipeline information.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple


def render_data_view_selector():
    """Render data view selector in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Data View Control")
        
        # Check if raw data is available
        has_raw_data = (hasattr(st.session_state, 'df_raw') and 
                       st.session_state.df_raw is not None)
        
        if has_raw_data:
            # View mode selector
            view_options = {
                "raw": "📄 Raw Data (as uploaded)",
                "processed": "🔧 Processed Data (type-inferred)",
                "display": "🖥️ Display Data (UI-friendly)"
            }
            
            current_mode = st.session_state.get("data_view_mode", "processed")
            
            selected_mode = st.selectbox(
                "Select Data View:",
                options=list(view_options.keys()),
                format_func=lambda x: view_options[x],
                index=list(view_options.keys()).index(current_mode),
                help="Choose which version of data to display and analyze"
            )
            
            if selected_mode != current_mode:
                st.session_state.data_view_mode = selected_mode
                _update_current_dataframe(selected_mode)
                st.rerun()
            
            # Show current view info
            _show_current_view_info(selected_mode)
            
            # Processing pipeline button
            if st.button("🔍 View Processing Pipeline"):
                st.session_state.show_processing_pipeline = True
        else:
            st.info("📄 Upload a file to access data view controls")


def _update_current_dataframe(view_mode: str):
    """Update current DataFrame based on selected view mode"""
    if view_mode == "raw" and st.session_state.df_raw is not None:
        st.session_state.df = st.session_state.df_raw.copy()
    elif view_mode == "processed" and st.session_state.df_original is not None:
        st.session_state.df = st.session_state.df_original.copy()
    elif view_mode == "display" and st.session_state.df_display is not None:
        st.session_state.df = st.session_state.df_display.copy()


def _show_current_view_info(view_mode: str):
    """Show information about current data view"""
    if view_mode == "raw":
        st.info("📄 **Raw Data**: Exactly as uploaded, no processing applied")
    elif view_mode == "processed":
        st.info("🔧 **Processed Data**: Type-inferred, cleaned, analysis-ready")
    elif view_mode == "display":
        st.info("🖥️ **Display Data**: UI-friendly format for visualization")


def render_processing_pipeline():
    """Render processing pipeline information"""
    if not st.session_state.get('show_processing_pipeline', False):
        return
    
    st.subheader("🔄 Data Processing Pipeline")
    st.write("This shows all automatic processing steps applied to your data:")
    
    # Close button
    if st.button("❌ Close Pipeline View"):
        st.session_state.show_processing_pipeline = False
        st.rerun()
    
    # Get processing pipeline
    pipeline = st.session_state.get('processing_pipeline', [])
    
    if not pipeline:
        st.info("📭 No processing pipeline available. Upload a file to see processing steps.")
        return
    
    # Display pipeline steps
    for i, step in enumerate(pipeline, 1):
        with st.expander(f"Step {i}: {step['description']}", expanded=True):
            st.write(f"**Step Type:** {step['step']}")
            st.write(f"**Changes:** {step['changes']}")
            
            # Add step-specific information
            if step['step'] == 'file_load':
                st.write("**Impact:** This is the original data as loaded from your file")
            elif step['step'] == 'type_inference':
                st.write("**Impact:** Columns were automatically converted to appropriate data types")
                st.write("**Note:** This may hide issues present in the raw data")
            elif step['step'] == 'display_conversion':
                st.write("**Impact:** Data was converted to strings for UI compatibility")
                st.write("**Note:** This version is used for display only")
    
    # Show data comparison
    st.subheader("📊 Data Comparison")
    
    if (st.session_state.df_raw is not None and 
        st.session_state.df_original is not None):
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Raw Data Shape", f"{st.session_state.df_raw.shape[0]} × {st.session_state.df_raw.shape[1]}")
            st.write("**Data Types:**")
            raw_types = st.session_state.df_raw.dtypes.value_counts()
            for dtype, count in raw_types.items():
                st.write(f"• {dtype}: {count}")
        
        with col2:
            st.metric("Processed Data Shape", f"{st.session_state.df_original.shape[0]} × {st.session_state.df_original.shape[1]}")
            st.write("**Data Types:**")
            orig_types = st.session_state.df_original.dtypes.value_counts()
            for dtype, count in orig_types.items():
                st.write(f"• {dtype}: {count}")
        
        with col3:
            st.metric("Display Data Shape", f"{st.session_state.df_display.shape[0]} × {st.session_state.df_display.shape[1]}")
            st.write("**Data Types:**")
            display_types = st.session_state.df_display.dtypes.value_counts()
            for dtype, count in display_types.items():
                st.write(f"• {dtype}: {count}")
    
    # Processing impact analysis
    st.subheader("⚡ Processing Impact Analysis")
    
    if (st.session_state.df_raw is not None and 
        st.session_state.df_original is not None):
        
        # Calculate type conversion impact
        type_changes = []
        for col in st.session_state.df_raw.columns:
            raw_dtype = str(st.session_state.df_raw[col].dtype)
            orig_dtype = str(st.session_state.df_original[col].dtype)
            if raw_dtype != orig_dtype:
                type_changes.append({
                    'column': col,
                    'raw_type': raw_dtype,
                    'processed_type': orig_dtype,
                    'impact': _analyze_type_change_impact(
                        st.session_state.df_raw[col], 
                        st.session_state.df_original[col]
                    )
                })
        
        if type_changes:
            st.write("**Type Conversion Impact:**")
            
            for change in type_changes:
                with st.expander(f"Column: {change['column']} ({change['raw_type']} → {change['processed_type']})"):
                    st.write(f"**Impact:** {change['impact']}")
                    
                    # Show sample values
                    raw_sample = st.session_state.df_raw[change['column']].dropna().head(3)
                    processed_sample = st.session_state.df_original[change['column']].dropna().head(3)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Raw Values:**")
                        for val in raw_sample:
                            st.code(f"'{val}' ({type(val).__name__})")
                    
                    with col2:
                        st.write("**Processed Values:**")
                        for val in processed_sample:
                            st.code(f"'{val}' ({type(val).__name__})")
        else:
            st.info("✅ No type conversions were applied - data types remained unchanged")


def _analyze_type_change_impact(raw_series: pd.Series, processed_series: pd.Series) -> str:
    """Analyze the impact of type conversion"""
    raw_null_count = raw_series.isnull().sum()
    processed_null_count = processed_series.isnull().sum()
    
    if processed_null_count > raw_null_count:
        lost_values = processed_null_count - raw_null_count
        return f"⚠️ {lost_values} values were lost during conversion (became NaN)"
    elif processed_null_count < raw_null_count:
        gained_values = raw_null_count - processed_null_count
        return f"✅ {gained_values} values were recovered during conversion"
    else:
        return "✅ No data loss during conversion"


def render_agent_data_source_warning():
    """Render warning about which data source the agent is using"""
    if not hasattr(st.session_state, 'df_raw') or st.session_state.df_raw is None:
        return
    
    current_mode = st.session_state.get("data_view_mode", "processed")
    
    if current_mode == "raw":
        st.warning("⚠️ **Agent Note:** You're working with RAW data. The agent will see the exact file as uploaded, which may contain structural issues.")
    elif current_mode == "processed":
        st.info("ℹ️ **Agent Note:** You're working with PROCESSED data. The agent will see type-inferred, cleaned data. Use 'Raw Data' view to see original issues.")
    elif current_mode == "display":
        st.warning("⚠️ **Agent Note:** You're working with DISPLAY data. This is optimized for UI display and may not be suitable for analysis.")


def get_current_dataframe_for_agent() -> pd.DataFrame:
    """Get the appropriate DataFrame for agent operations based on current view mode"""
    view_mode = st.session_state.get("data_view_mode", "processed")
    
    if view_mode == "raw" and st.session_state.get("df_raw") is not None:
        return st.session_state.df_raw
    elif view_mode == "processed" and st.session_state.get("df_original") is not None:
        return st.session_state.df_original
    elif view_mode == "display" and st.session_state.get("df_display") is not None:
        return st.session_state.df_display
    else:
        # Fallback to legacy behavior
        return st.session_state.get("df", pd.DataFrame())


def render_data_view_comparison():
    """Render a comparison view of all data versions"""
    if not st.session_state.get('show_data_comparison', False):
        return
    
    st.subheader("📊 Data Version Comparison")
    
    # Close button
    if st.button("❌ Close Comparison View"):
        st.session_state.show_data_comparison = False
        st.rerun()
    
    # Check if all versions are available
    has_all_versions = (
        st.session_state.get("df_raw") is not None and
        st.session_state.get("df_original") is not None and
        st.session_state.get("df_display") is not None
    )
    
    if not has_all_versions:
        st.warning("⚠️ Not all data versions are available. Upload a file to see comparison.")
        return
    
    # Create tabs for different comparisons
    tab1, tab2, tab3 = st.tabs(["📋 Basic Info", "🔍 Sample Data", "📊 Column Details"])
    
    with tab1:
        # Basic information comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("### 📄 Raw Data")
            st.write(f"**Shape:** {st.session_state.df_raw.shape}")
            st.write(f"**Memory Usage:** {st.session_state.df_raw.memory_usage(deep=True).sum() / 1024:.1f} KB")
            st.write("**Data Types:**")
            for dtype, count in st.session_state.df_raw.dtypes.value_counts().items():
                st.write(f"• {dtype}: {count}")
        
        with col2:
            st.write("### 🔧 Processed Data")
            st.write(f"**Shape:** {st.session_state.df_original.shape}")
            st.write(f"**Memory Usage:** {st.session_state.df_original.memory_usage(deep=True).sum() / 1024:.1f} KB")
            st.write("**Data Types:**")
            for dtype, count in st.session_state.df_original.dtypes.value_counts().items():
                st.write(f"• {dtype}: {count}")
        
        with col3:
            st.write("### 🖥️ Display Data")
            st.write(f"**Shape:** {st.session_state.df_display.shape}")
            st.write(f"**Memory Usage:** {st.session_state.df_display.memory_usage(deep=True).sum() / 1024:.1f} KB")
            st.write("**Data Types:**")
            for dtype, count in st.session_state.df_display.dtypes.value_counts().items():
                st.write(f"• {dtype}: {count}")
    
    with tab2:
        # Sample data comparison
        st.write("### Sample Data Comparison (First 5 rows)")
        
        # Column selector
        selected_columns = st.multiselect(
            "Select columns to compare:",
            options=st.session_state.df_raw.columns,
            default=st.session_state.df_raw.columns[:3].tolist(),
            help="Select columns to see how they differ across versions"
        )
        
        if selected_columns:
            for col in selected_columns:
                st.write(f"**Column: {col}**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("📄 Raw:")
                    st.dataframe(st.session_state.df_raw[col].head(5), use_container_width=True)
                
                with col2:
                    st.write("🔧 Processed:")
                    st.dataframe(st.session_state.df_original[col].head(5), use_container_width=True)
                
                with col3:
                    st.write("🖥️ Display:")
                    st.dataframe(st.session_state.df_display[col].head(5), use_container_width=True)
                
                st.markdown("---")
    
    with tab3:
        # Detailed column information
        st.write("### Column-by-Column Analysis")
        
        for col in st.session_state.df_raw.columns:
            with st.expander(f"Column: {col}"):
                # Type information
                raw_dtype = str(st.session_state.df_raw[col].dtype)
                processed_dtype = str(st.session_state.df_original[col].dtype)
                display_dtype = str(st.session_state.df_display[col].dtype)
                
                st.write(f"**Types:** {raw_dtype} → {processed_dtype} → {display_dtype}")
                
                # Missing values
                raw_missing = st.session_state.df_raw[col].isnull().sum()
                processed_missing = st.session_state.df_original[col].isnull().sum()
                display_missing = st.session_state.df_display[col].isnull().sum()
                
                st.write(f"**Missing Values:** {raw_missing} → {processed_missing} → {display_missing}")
                
                # Unique values
                raw_unique = st.session_state.df_raw[col].nunique()
                processed_unique = st.session_state.df_original[col].nunique()
                display_unique = st.session_state.df_display[col].nunique()
                
                st.write(f"**Unique Values:** {raw_unique} → {processed_unique} → {display_unique}")
                
                # Flag significant changes
                if processed_missing > raw_missing:
                    st.warning(f"⚠️ Processing increased missing values by {processed_missing - raw_missing}")
                if processed_unique != raw_unique:
                    st.info(f"ℹ️ Processing changed unique value count from {raw_unique} to {processed_unique}")