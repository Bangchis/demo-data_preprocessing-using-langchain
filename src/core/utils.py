import pandas as pd
import numpy as np
import streamlit as st
import pickle
import os
import io
import csv
import chardet
from typing import Dict, Optional, Tuple, List

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
    
    # Initialize triple DataFrame support (raw, processed, display)
    if "dfs_raw" not in st.session_state:
        st.session_state.dfs_raw = {}  # Raw data as loaded from file
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = None
    if "dfs_original" not in st.session_state:
        st.session_state.dfs_original = {}  # Processed data with proper types
    if "df_original" not in st.session_state:
        st.session_state.df_original = None
    if "dfs_display" not in st.session_state:
        st.session_state.dfs_display = {}  # Display-friendly data
    if "df_display" not in st.session_state:
        st.session_state.df_display = None
    if "history_original" not in st.session_state:
        st.session_state.history_original = []
    
    # Initialize processing pipeline tracking
    if "processing_pipeline" not in st.session_state:
        st.session_state.processing_pipeline = []
    if "data_view_mode" not in st.session_state:
        st.session_state.data_view_mode = "processed"  # raw, processed, display

def load_uploaded_file_preserve_types(uploaded_file) -> tuple:
    """Load uploaded CSV/XLSX file preserving original data types with enhanced parsing"""
    try:
        # Clear processing pipeline for new file
        st.session_state.processing_pipeline = []
        
        # Load the file based on extension (RAW DATA)
        if uploaded_file.name.endswith('.csv'):
            df_raw = _load_csv_with_enhanced_parsing(uploaded_file)
            st.session_state.processing_pipeline.append({
                'step': 'file_load',
                'description': 'Loaded CSV with enhanced parsing',
                'changes': f'Shape: {df_raw.shape}, Columns: {list(df_raw.columns)[:5]}...'
            })
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df_raw = pd.read_excel(uploaded_file)
            st.session_state.processing_pipeline.append({
                'step': 'file_load',
                'description': 'Loaded Excel file',
                'changes': f'Shape: {df_raw.shape}, Columns: {list(df_raw.columns)[:5]}...'
            })
        else:
            raise ValueError("Unsupported file format")
        
        # Apply intelligent type inference and Excel-specific handling (PROCESSED DATA)
        df_original = _apply_intelligent_type_inference(df_raw.copy())
        
        # Track processing changes
        type_changes = []
        for col in df_original.columns:
            if str(df_raw[col].dtype) != str(df_original[col].dtype):
                type_changes.append(f"{col}: {df_raw[col].dtype} → {df_original[col].dtype}")
        
        if type_changes:
            st.session_state.processing_pipeline.append({
                'step': 'type_inference',
                'description': 'Applied intelligent type inference',
                'changes': f'Changed types for {len(type_changes)} columns: {type_changes[:3]}...'
            })
        
        # Create display version for Streamlit compatibility (DISPLAY DATA)
        df_display = _create_display_version(df_original)
        
        st.session_state.processing_pipeline.append({
            'step': 'display_conversion',
            'description': 'Created display-friendly version',
            'changes': 'Converted datetime/boolean to strings for UI compatibility'
        })
        
        return df_raw, df_original, df_display
        
    except Exception as e:
        st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
        return None, None, None


def _detect_csv_parameters(uploaded_file) -> Dict:
    """Detect CSV parameters (delimiter, quotechar, encoding) for better parsing"""
    # Reset file pointer
    uploaded_file.seek(0)
    
    # Read raw bytes for encoding detection
    raw_data = uploaded_file.read()
    uploaded_file.seek(0)
    
    # Detect encoding
    encoding_result = chardet.detect(raw_data)
    encoding = encoding_result.get('encoding', 'utf-8')
    confidence = encoding_result.get('confidence', 0.0)
    
    # If confidence is low, try common encodings
    if confidence < 0.7:
        for test_encoding in ['utf-8', 'latin-1', 'cp1252', 'utf-16']:
            try:
                raw_data.decode(test_encoding)
                encoding = test_encoding
                break
            except UnicodeDecodeError:
                continue
    
    # Read first few lines as text for dialect detection
    try:
        text_data = raw_data.decode(encoding)
        lines = text_data.split('\n')[:10]  # Use first 10 lines for detection
        sample_text = '\n'.join(lines)
    except UnicodeDecodeError:
        # Fallback to utf-8 with error handling
        text_data = raw_data.decode('utf-8', errors='ignore')
        lines = text_data.split('\n')[:10]
        sample_text = '\n'.join(lines)
        encoding = 'utf-8'
    
    # Detect delimiter and quote character
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text, delimiters=',;\t|:')
        delimiter = dialect.delimiter
        quotechar = dialect.quotechar
        
        # Validate delimiter by checking column consistency
        test_lines = [line for line in lines if line.strip()]
        if len(test_lines) >= 2:
            col_counts = []
            for line in test_lines[:5]:  # Check first 5 lines
                reader = csv.reader([line], delimiter=delimiter, quotechar=quotechar)
                try:
                    row = next(reader)
                    col_counts.append(len(row))
                except:
                    col_counts.append(0)
            
            # If column counts are inconsistent, try other delimiters
            if len(set(col_counts)) > 2:  # More than 2 different column counts
                for test_delimiter in [',', ';', '\t', '|']:
                    if test_delimiter == delimiter:
                        continue
                    test_col_counts = []
                    for line in test_lines[:5]:
                        reader = csv.reader([line], delimiter=test_delimiter, quotechar=quotechar)
                        try:
                            row = next(reader)
                            test_col_counts.append(len(row))
                        except:
                            test_col_counts.append(0)
                    
                    # If this delimiter gives more consistent results, use it
                    if len(set(test_col_counts)) <= len(set(col_counts)):
                        delimiter = test_delimiter
                        col_counts = test_col_counts
                        break
        
    except Exception as e:
        # Fallback to comma delimiter
        delimiter = ','
        quotechar = '"'
    
    return {
        'delimiter': delimiter,
        'quotechar': quotechar,
        'encoding': encoding,
        'confidence': confidence
    }


def _load_csv_with_enhanced_parsing(uploaded_file) -> pd.DataFrame:
    """Load CSV with enhanced parsing to handle structural issues"""
    # Detect CSV parameters
    params = _detect_csv_parameters(uploaded_file)
    
    # Reset file pointer
    uploaded_file.seek(0)
    
    # Try loading with detected parameters
    try:
        df = pd.read_csv(
            uploaded_file,
            delimiter=params['delimiter'],
            quotechar=params['quotechar'],
            encoding=params['encoding'],
            skipinitialspace=True,  # Skip whitespace after delimiter
            keep_default_na=True,   # Keep default NaN values
            na_values=['', 'N/A', 'NA', 'NULL', 'null', 'NaN', 'nan'],  # Extended NaN values
            dtype=str,              # Load as strings first, then infer types
            engine='python'         # Use Python engine for better error handling
        )
        
        # Log parsing details
        st.info(f"📊 CSV loaded with: delimiter='{params['delimiter']}', encoding={params['encoding']} (confidence: {params['confidence']:.1f})")
        
        return df
        
    except Exception as e:
        st.warning(f"Enhanced parsing failed: {e}")
        # Fallback to basic parsing
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, dtype=str, engine='python')
            st.info("📊 CSV loaded with basic parsing (fallback)")
            return df
        except Exception as e2:
            st.error(f"Both enhanced and basic parsing failed: {e2}")
            raise e2


def _apply_intelligent_type_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Apply intelligent type inference to preserve data types"""
    df_typed = df.copy()
    
    for col in df_typed.columns:
        try:
            # Skip if already a good numeric type
            if pd.api.types.is_numeric_dtype(df_typed[col]) and not df_typed[col].dtype == 'object':
                continue
            
            # Skip if already a good datetime type
            if pd.api.types.is_datetime64_any_dtype(df_typed[col]):
                continue
            
            # Try to infer better types for object columns
            if df_typed[col].dtype == 'object':
                # First, handle string 'nan' values by converting to proper NaN
                df_typed[col] = df_typed[col].replace(['nan', 'None', 'NaN', 'NULL', 'null', 'NaT'], np.nan)
                
                # Try numeric conversion first
                try:
                    numeric_converted = pd.to_numeric(df_typed[col], errors='coerce')
                    if not numeric_converted.isna().all():
                        # Count non-NaN values for percentage calculation
                        total_non_nan = len(df_typed) - df_typed[col].isna().sum()
                        if total_non_nan > 0:
                            numeric_ratio = numeric_converted.notna().sum() / total_non_nan
                            # If more than 70% of non-NaN values are numeric, treat as numeric
                            if numeric_ratio > 0.7:
                                df_typed[col] = numeric_converted
                                continue
                except:
                    pass
                
                # Try datetime conversion
                try:
                    datetime_converted = pd.to_datetime(df_typed[col], errors='coerce')
                    if not datetime_converted.isna().all():
                        # Count non-NaN values for percentage calculation
                        total_non_nan = len(df_typed) - df_typed[col].isna().sum()
                        if total_non_nan > 0:
                            datetime_ratio = datetime_converted.notna().sum() / total_non_nan
                            # If more than 70% of non-NaN values are valid dates, treat as datetime
                            if datetime_ratio > 0.7:
                                df_typed[col] = datetime_converted
                                continue
                except:
                    pass
                
                # Try boolean conversion for Excel TRUE/FALSE
                try:
                    unique_values = df_typed[col].dropna().unique()
                    bool_values = {'TRUE', 'FALSE', 'True', 'False', 'true', 'false', '1', '0'}
                    if len(unique_values) <= 10 and all(str(v) in bool_values for v in unique_values):
                        df_typed[col] = df_typed[col].map({
                            'TRUE': True, 'True': True, 'true': True, '1': True,
                            'FALSE': False, 'False': False, 'false': False, '0': False
                        })
                        continue
                except:
                    pass
                
                # For remaining object columns, clean up string 'nan' values
                df_typed[col] = df_typed[col].replace(['nan', 'None', 'NaN', 'NULL', 'null', 'NaT'], np.nan)
                
        except Exception as e:
            # If conversion fails, keep original but clean string 'nan'
            try:
                if df_typed[col].dtype == 'object':
                    df_typed[col] = df_typed[col].replace(['nan', 'None', 'NaN', 'NULL', 'null', 'NaT'], np.nan)
            except:
                pass
    
    return df_typed


def _create_display_version(df_original: pd.DataFrame) -> pd.DataFrame:
    """Create a display-friendly version of the DataFrame with PyArrow safety checks"""
    df_display = df_original.copy()
    
    for col in df_display.columns:
        try:
            # Convert datetime to string for display
            if pd.api.types.is_datetime64_any_dtype(df_display[col]):
                df_display[col] = df_display[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                df_display[col] = df_display[col].replace('NaT', '')
            
            # Convert boolean to string for display
            elif df_display[col].dtype == 'bool':
                df_display[col] = df_display[col].astype(str)
                df_display[col] = df_display[col].replace(['nan', 'None', 'NaN', 'NULL', 'null'], '')
            
            # Handle numeric columns - keep as numeric but clean problematic values
            elif pd.api.types.is_numeric_dtype(df_display[col]):
                df_display[col] = df_display[col].replace([np.inf, -np.inf], np.nan)
                
                # Safety check: ensure no string 'nan' values in numeric columns
                if df_display[col].astype(str).str.contains('nan').any():
                    # Convert any remaining string 'nan' to proper NaN
                    df_display[col] = pd.to_numeric(df_display[col], errors='coerce')
            
            # Handle object columns
            elif df_display[col].dtype == 'object':
                df_display[col] = df_display[col].astype(str)
                df_display[col] = df_display[col].replace(['nan', 'None', 'NaN', 'NULL', 'null', 'NaT'], '')
            
        except Exception as e:
            # If conversion fails, convert to string as fallback
            try:
                df_display[col] = df_display[col].astype(str)
                df_display[col] = df_display[col].replace(['nan', 'None', 'NaN', 'NULL', 'null'], '')
            except:
                # Last resort: create empty string series
                df_display[col] = pd.Series([''] * len(df_display), dtype=str)
    
    # Final PyArrow safety check
    df_display = _ensure_pyarrow_compatibility(df_display)
    
    return df_display


def _ensure_pyarrow_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame is fully compatible with PyArrow serialization"""
    df_safe = df.copy()
    
    for col in df_safe.columns:
        try:
            # Check if column has mixed types or problematic values
            if df_safe[col].dtype == 'object':
                # Convert to string and clean all problematic values
                df_safe[col] = df_safe[col].astype(str)
                df_safe[col] = df_safe[col].replace(['nan', 'None', 'NaN', 'NULL', 'null', 'NaT', 'inf', '-inf'], '')
                df_safe[col] = df_safe[col].fillna('')
            
            elif pd.api.types.is_numeric_dtype(df_safe[col]):
                # For numeric columns, ensure no inf values and no string 'nan'
                df_safe[col] = df_safe[col].replace([np.inf, -np.inf], np.nan)
                
                # Double check for string 'nan' values that somehow persisted
                col_str = df_safe[col].astype(str)
                if col_str.str.contains('nan').any():
                    # If there are string 'nan' values, convert entire column to string
                    df_safe[col] = col_str.replace(['nan', 'None', 'NaN', 'NULL', 'null'], '')
                    
        except Exception as e:
            # If anything fails, convert to clean string
            try:
                df_safe[col] = df_safe[col].astype(str)
                df_safe[col] = df_safe[col].replace(['nan', 'None', 'NaN', 'NULL', 'null', 'inf', '-inf'], '')
                df_safe[col] = df_safe[col].fillna('')
            except:
                # Absolute last resort
                df_safe[col] = pd.Series([''] * len(df_safe), dtype=str)
    
    return df_safe


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load uploaded CSV/XLSX file into DataFrame (legacy function for backward compatibility)"""
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
    """Clean DataFrame to be compatible with Streamlit's PyArrow serialization."""
    if df is None:
        return None
    
    df_clean = df.copy()
    
    # Pre-processing: Handle string 'nan' values before type conversion
    for col in df_clean.columns:
        try:
            # First, handle all string 'nan' variations regardless of column type
            if df_clean[col].dtype == 'object':
                # Convert to string first to ensure consistency
                df_clean[col] = df_clean[col].astype(str)
                # Replace all string 'nan' variations with actual NaN
                df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaN', 'NULL', 'null', 'NaT'], np.nan)
                
                # Try to detect if this should be a numeric column
                # by checking if most non-nan values are numeric
                non_nan_values = df_clean[col].dropna()
                if len(non_nan_values) > 0:
                    try:
                        numeric_converted = pd.to_numeric(non_nan_values, errors='coerce')
                        numeric_ratio = numeric_converted.notna().sum() / len(non_nan_values)
                        
                        # If more than 70% of non-nan values are numeric, treat as numeric
                        if numeric_ratio > 0.7:
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    except:
                        pass
            
        except Exception as e:
            pass
    
    # Handle each column type appropriately
    for col in df_clean.columns:
        try:
            # For object/string columns, convert to string and clean
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaN', 'NULL', 'null', 'NaT'], '')
            
            # For datetime columns, convert to string
            elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaN', 'NULL', 'null', 'NaT'], '')
            
            # For boolean columns, convert to string
            elif df_clean[col].dtype == 'bool':
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaN', 'NULL', 'null'], '')
            
            # For numeric columns, handle carefully
            elif pd.api.types.is_numeric_dtype(df_clean[col]):
                # Replace inf values with NaN first
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                
                # Check for any remaining problematic values
                try:
                    # Test if the column can be safely converted to Arrow
                    test_series = df_clean[col].copy()
                    pd.to_numeric(test_series, errors='raise')
                except (ValueError, TypeError):
                    # If numeric conversion fails, convert to string
                    df_clean[col] = df_clean[col].astype(str)
                    df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaN', 'NULL', 'null', 'inf', '-inf'], '')
            
            # For any other type, convert to string
            else:
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaN', 'NULL', 'null'], '')
                
        except Exception as e:
            # If any conversion fails, convert to string as fallback
            try:
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaN', 'NULL', 'null'], '')
            except:
                df_clean[col] = pd.Series([''] * len(df_clean), dtype=str)
    
    # Final safety check - ensure PyArrow compatibility
    for col in df_clean.columns:
        try:
            # For string columns, ensure no problematic values
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('')
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaN', 'NULL', 'null', 'NaT'], '')
            
            # For numeric columns, ensure no inf values
            elif pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                
                # Final check: if there are still string 'nan' values, convert to string
                if df_clean[col].astype(str).str.contains('nan').any():
                    df_clean[col] = df_clean[col].astype(str)
                    df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaN', 'NULL', 'null'], '')
                    
        except Exception as e:
            # Last resort: convert to string
            try:
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'None', 'NaN', 'NULL', 'null'], '')
            except:
                df_clean[col] = pd.Series([''] * len(df_clean), dtype=str)
    
    return df_clean


def sync_dataframe_versions(updated_df: pd.DataFrame):
    """
    Synchronize all DataFrame versions after code execution
    
    Args:
        updated_df (pd.DataFrame): The updated DataFrame from code execution
            This should be the "real data" from df_original after CodeRunner execution
    """
    if updated_df is None:
        return
    
    try:
        # Since CodeRunner now works with df_original, the updated_df IS the real data
        # Update df_original first (this is the source of truth)
        st.session_state.df_original = updated_df.copy()
        
        # Update main DataFrame for legacy compatibility
        st.session_state.df = updated_df.copy()
        
        # Update dfs dictionary
        if "dfs" not in st.session_state:
            st.session_state.dfs = {}
        st.session_state.dfs["df"] = updated_df.copy()
        
        # Update original dfs dictionary
        if "dfs_original" not in st.session_state:
            st.session_state.dfs_original = {}
        st.session_state.dfs_original["df"] = updated_df.copy()
        
        # Create new display version for UI compatibility
        df_display_updated = _create_display_version(updated_df)
        st.session_state.df_display = df_display_updated.copy()
        
        # Update display dfs dictionary
        if "dfs_display" not in st.session_state:
            st.session_state.dfs_display = {}
        st.session_state.dfs_display["df"] = df_display_updated.copy()
        
    except Exception as e:
        # Fallback: at least ensure main df is updated
        st.session_state.df = updated_df.copy()
        st.session_state.df_original = updated_df.copy()
        st.session_state.df_display = updated_df.copy()


def get_current_dataframe_for_display() -> pd.DataFrame:
    """
    Get the current DataFrame for display, prioritizing the most recently updated version
    
    Returns:
        pd.DataFrame: The current DataFrame for display
    """
    # Try to get the display version first
    if hasattr(st.session_state, 'df_display') and st.session_state.df_display is not None:
        return clean_dataframe_for_display(st.session_state.df_display)
    
    # Fallback to main df
    elif hasattr(st.session_state, 'df') and st.session_state.df is not None:
        return clean_dataframe_for_display(st.session_state.df)
    
    # Last resort: return None
    else:
        return None


def refresh_data_preview():
    """
    Force refresh of data preview by ensuring all DataFrame versions are synced
    """
    if st.session_state.df is not None:
        # Sync all versions with current main df
        sync_dataframe_versions(st.session_state.df)
        
        # Force Streamlit to rerun and refresh the display
        st.rerun()