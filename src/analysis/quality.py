import streamlit as st
import pandas as pd
from src.core.utils import clean_dataframe_for_display

def render_data_types_section(df):
    """Render data types and missing values section."""
    with st.expander("📋 Data Types & Info"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            try:
                # Convert dtypes to a more displayable format
                dtypes_df = df.dtypes.to_frame('Data Type').reset_index()
                dtypes_df.columns = ['Column', 'Data Type']
                clean_dtypes_df = clean_dataframe_for_display(dtypes_df)
                st.dataframe(clean_dtypes_df, use_container_width=True)
            except:
                st.text(str(df.dtypes))
        
        with col2:
            st.write("**Missing Values:**")
            try:
                # Convert missing values to a more displayable format
                missing_df = df.isnull().sum().to_frame('Missing Count').reset_index()
                missing_df.columns = ['Column', 'Missing Count']
                clean_missing_df = clean_dataframe_for_display(missing_df)
                st.dataframe(clean_missing_df, use_container_width=True)
            except:
                st.text(str(df.isnull().sum()))
        
        # Additional summary statistics similar to STATA's summarize
        st.write("**Summary Statistics (STATA-style):**")
        try:
            # Get numeric columns only
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary_stats = []
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        stats = {
                            'Variable': col,
                            'Obs': len(col_data),
                            'Mean': col_data.mean(),
                            'Std. Dev.': col_data.std(),
                            'Min': col_data.min(),
                            'Max': col_data.max(),
                            '25%': col_data.quantile(0.25),
                            '50%': col_data.quantile(0.50),
                            '75%': col_data.quantile(0.75)
                        }
                        summary_stats.append(stats)
                
                if summary_stats:
                    summary_df = pd.DataFrame(summary_stats)
                    clean_summary_df = clean_dataframe_for_display(summary_df)
                    st.dataframe(clean_summary_df, use_container_width=True)
                else:
                    st.write("No numeric columns found for summary statistics.")
            else:
                st.write("No numeric columns found for summary statistics.")
        except Exception as e:
            st.warning(f"Error calculating summary statistics: {str(e)}")

def render_storage_types_check(df):
    """Render storage types check section."""
    st.write("### 1. Correct Storage Types")
    st.write("Below are the storage types for each variable. Review to ensure numeric variables are stored as numeric types.")
    try:
        # Convert dtypes to a more displayable format
        dtypes_df = df.dtypes.to_frame('Data Type').reset_index()
        dtypes_df.columns = ['Column', 'Data Type']
        clean_dtypes_df = clean_dataframe_for_display(dtypes_df)
        st.dataframe(clean_dtypes_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Error displaying dtypes: {str(e)}")
    st.write("---")

def render_descriptive_stats_check(df):
    """Render descriptive statistics check section."""
    st.write("### 2. Descriptive Statistics (describe)")
    try:
        describe_df = df.describe(include='all')
        clean_describe_df = clean_dataframe_for_display(describe_df)
        st.dataframe(clean_describe_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Error displaying describe: {str(e)}")
    st.write("---")

def render_duplicates_check(df):
    """Render duplicates check section."""
    st.write("### 3. No Unwanted Duplicates")
    st.write("Select key columns to check for duplicate rows:")
    key_cols = st.multiselect(
        "Key columns for uniqueness check:",
        options=list(df.columns),
        default=[]
    )
    if key_cols:
        dupes = df.duplicated(subset=key_cols, keep=False)
        n_dupes = dupes.sum()
        if n_dupes > 0:
            st.warning(f"Found {n_dupes} duplicate rows based on selected keys.")
            dupes_df = df.loc[dupes, key_cols + [c for c in df.columns if c not in key_cols][:3]].head(20)
            clean_dupes_df = clean_dataframe_for_display(dupes_df)
            st.dataframe(clean_dupes_df, use_container_width=True)
        else:
            st.success("No duplicates found based on selected keys.")
    else:
        st.info("Select columns to check for duplicates.")
    st.write("---")

def render_plausible_values_check(df):
    """Render plausible values check section."""
    st.write("### 4. Plausible Values (Min/Max)")
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            minmax = df[numeric_cols].agg(['min', 'max']).T
            clean_minmax = clean_dataframe_for_display(minmax)
            st.dataframe(clean_minmax, use_container_width=True)
            st.info("Review min/max for suspicious values (e.g., negative ages, out-of-range scores).")
        else:
            st.write("No numeric columns found.")
    except Exception as e:
        st.warning(f"Error displaying min/max: {str(e)}")
    st.write("---")

def render_categories_check(df):
    """Render sensible categories check section."""
    st.write("### 5. Sensible Categories (Frequency Tables)")
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            st.write(f"**{col}**")
            try:
                freq_df = df[col].value_counts(dropna=False).rename('count').to_frame()
                clean_freq_df = clean_dataframe_for_display(freq_df)
                st.dataframe(clean_freq_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Error displaying frequency table for {col}: {str(e)}")
    else:
        st.write("No categorical columns found.")
    st.write("---")

def render_logical_consistency_check():
    """Render logical consistency check section."""
    st.write("### 6. Logical Consistency (User-defined Rules)")
    st.info("Logical consistency checks (e.g., year_of_death >= year_of_birth) must be defined by the user. Please review your data and use custom code or rules as needed.")

def render_data_quality_checks(df):
    """Render the complete data quality checks section."""
    with st.expander("🛡️ Data Quality Checks"):
        render_storage_types_check(df)
        render_descriptive_stats_check(df)
        render_duplicates_check(df)
        render_plausible_values_check(df)
        render_categories_check(df)
        render_logical_consistency_check()