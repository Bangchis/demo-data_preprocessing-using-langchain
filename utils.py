import pandas as pd
import numpy as np
import streamlit as st
import pickle
import os
from typing import Dict, Optional

def init_session_state():
    """Initialize Streamlit session state"""
    if "dfs" not in st.session_state:
        st.session_state.dfs = {}
    if "df" not in st.session_state:
        st.session_state.df = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load uploaded CSV/XLSX file into DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format")
        
        # Apply comprehensive cleaning for Arrow compatibility
        df_clean = df.copy()
        
        # Handle each column type appropriately
        for col in df_clean.columns:
            try:
                # For object/string columns, convert to string
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].astype(str)
                
                # For datetime columns, convert to string
                elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].astype(str)
                
                # For boolean columns, convert to string
                elif df_clean[col].dtype == 'bool':
                    df_clean[col] = df_clean[col].astype(str)
                
                # For numeric columns, try to keep as numeric but handle any issues
                elif pd.api.types.is_numeric_dtype(df_clean[col]):
                    # Check if there are any problematic values
                    try:
                        pd.to_numeric(df_clean[col], errors='raise')
                    except (ValueError, TypeError):
                        # If numeric conversion fails, convert to string
                        df_clean[col] = df_clean[col].astype(str)
                
                # For any other type, convert to string
                else:
                    df_clean[col] = df_clean[col].astype(str)
                    
            except Exception as e:
                # If any conversion fails, convert to string as fallback
                df_clean[col] = df_clean[col].astype(str)
        
        # Final safety check - ensure all columns are string type and handle any problematic values
        for col in df_clean.columns:
            try:
                # Replace any problematic values with empty string
                df_clean[col] = df_clean[col].fillna('')
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], '')
                df_clean[col] = df_clean[col].astype(str)
                # Replace any remaining problematic strings
                df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaN', 'NULL', 'null'], '')
            except Exception as e:
                # If even string conversion fails, replace with placeholder
                df_clean[col] = pd.Series([''] * len(df_clean), dtype=str)
        
        return df_clean
    except Exception as e:
        st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
        return None

def save_dataframes_to_pickle(dfs: Dict, df: Optional[pd.DataFrame], filepath: str):
    """Save dataframes to pickle file for Docker container"""
    data = {
        'dfs': dfs,
        'df': df
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_dataframes_from_pickle(filepath: str) -> tuple:
    """Load dataframes from pickle file"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data.get('dfs', {}), data.get('df', None)
    except:
        return {}, None

def apply_changes():
    """Save current state to history"""
    if st.session_state.df is not None:
        st.session_state.history.append(st.session_state.df.copy())
        st.success("✅ Changes applied! You can undo if needed.")

def undo_changes():
    """Restore previous state from history"""
    if st.session_state.history:
        st.session_state.df = st.session_state.history.pop()
        st.success("↩️ Changes undone successfully!")
    else:
        st.warning("No changes to undo!")
        
def clean_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame for Arrow compatibility"""
    df_clean = df.copy()
    
    for col in df_clean.columns:
        # Fix object columns with mixed types
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).replace('nan', '').replace('None', '')
        
        # Fix float columns that might have precision issues
        elif df_clean[col].dtype in ['float64', 'float32']:
            # Round to avoid precision errors
            df_clean[col] = df_clean[col].round(6)
            # Convert NaN to None for better Arrow handling
            df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)
    
    return df_clean