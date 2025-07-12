import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from utils import clean_dataframe_for_display
from analysis_agent import create_analysis_agent

def get_ai_data_type_suggestion(df):
    """Get AI suggestion for data type based on heuristics."""
    suggestion = ""
    # Heuristic: Panel if two+ columns look like time, and one looks like id; Time-Series if one time col; else Cross-Sectional
    year_pattern = re.compile(r"^(19[7-9][0-9]|20[0-2][0-9]|2025)$")
    time_like_cols = [col for col in df.columns if year_pattern.match(str(col))]
    if not time_like_cols:
        alt_time_pattern = re.compile(r"(\d{4}|Q[1-4]_\d{4}|\d{4}Q[1-4])")
        time_like_cols = [col for col in df.columns if alt_time_pattern.search(str(col))]
    n_time = len(time_like_cols)
    # Try to guess id col: non-numeric, many unique values
    id_candidates = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() > 1 and df[col].nunique() < len(df)//2]
    if n_time > 1 and id_candidates:
        suggestion = f"Panel Data (multiple time-like columns: {time_like_cols}, possible ID: {id_candidates[0]})"
    elif n_time == 1:
        suggestion = f"Time-Series (time variable: {time_like_cols[0]})"
    else:
        suggestion = "Cross-Sectional (no clear time or panel structure detected)"
    return suggestion

def render_data_declaration_section():
    """Render the data declaration section."""
    st.subheader("📑 Data Declaration")
    col_decl, col_ai = st.columns([4,1])
    with col_decl:
        declaration = st.radio(
            "Declare your data type:",
            ["None", "Panel Data", "Time-Series", "Cross-Sectional"],
            index=0,
            horizontal=True,
            help="Choose the structure that best describes your data."
        )
    with col_ai:
        ask_ai = st.button("Ask AI", help="Let AI suggest the data type. This is only a recommendation and may be based on assumptions.")

    if 'ai_data_type_suggestion' not in st.session_state:
        st.session_state.ai_data_type_suggestion = None

    if ask_ai:
        suggestion = get_ai_data_type_suggestion(st.session_state.df)
        st.session_state.ai_data_type_suggestion = suggestion

    if st.session_state.ai_data_type_suggestion:
        st.info(f"**AI Suggestion:** {st.session_state.ai_data_type_suggestion}\n\n*This is only a recommendation and based on assumptions from the AI. Please review and do not use immediately without checking.*")

    return declaration

def detect_data_format(df):
    """Detect if data is in wide or long format for panel data."""
    # Pattern 1: Pure year columns (2020, 2021, etc.)
    year_pattern = re.compile(r"^(19[7-9][0-9]|20[0-2][0-9]|2025)$")
    pure_time_cols = [col for col in df.columns if year_pattern.match(str(col))]
    
    # Pattern 2: Variable_Time format (GDP_2020, Population_2021, etc.)
    var_time_pattern = re.compile(r"^(.+)_(\d{4}|Q[1-4]_\d{4}|\d{4}Q[1-4])$")
    var_time_cols = []
    var_time_mapping = {}
    
    for col in df.columns:
        match = var_time_pattern.match(str(col))
        if match:
            var_name, time_part = match.groups()
            var_time_cols.append(col)
            if var_name not in var_time_mapping:
                var_time_mapping[var_name] = []
            var_time_mapping[var_name].append((col, time_part))
    
    # Pattern 3: Time_Variable format (2020_GDP, 2021_Population, etc.)
    time_var_pattern = re.compile(r"^(\d{4}|Q[1-4]_\d{4}|\d{4}Q[1-4])_(.+)$")
    time_var_cols = []
    time_var_mapping = {}
    
    for col in df.columns:
        match = time_var_pattern.match(str(col))
        if match:
            time_part, var_name = match.groups()
            time_var_cols.append(col)
            if var_name not in time_var_mapping:
                time_var_mapping[var_name] = []
            time_var_mapping[var_name].append((col, time_part))
    
    # Determine if it's wide format and what type
    is_wide_format = False
    wide_format_type = None
    time_like_cols = []
    variable_mapping = {}
    
    if len(pure_time_cols) > 1:
        is_wide_format = True
        wide_format_type = "pure_time"
        time_like_cols = pure_time_cols
    elif len(var_time_cols) > 1 and len(var_time_mapping) >= 1:
        is_wide_format = True
        wide_format_type = "var_time"
        time_like_cols = var_time_cols
        variable_mapping = var_time_mapping
    elif len(time_var_cols) > 1 and len(time_var_mapping) >= 1:
        is_wide_format = True
        wide_format_type = "time_var"
        time_like_cols = time_var_cols
        variable_mapping = time_var_mapping
    
    return is_wide_format, time_like_cols, wide_format_type, variable_mapping

def transform_wide_to_long(df, panel_id, time_var, wide_format_type, time_like_cols, variable_mapping):
    """Transform wide format data to long format properly."""
    
    if wide_format_type == "pure_time":
        # Case: ID | 2020 | 2021 | 2022 | other_cols
        # Transform to: ID | year | value | other_cols
        id_vars = [panel_id] + [c for c in df.columns if c not in time_like_cols and c != panel_id]
        id_vars = list(dict.fromkeys(id_vars))
        
        long_df = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=time_like_cols,
            var_name=time_var,
            value_name="value"
        )
        
    elif wide_format_type == "var_time":
        # Case: ID | GDP_2020 | GDP_2021 | Pop_2020 | Pop_2021
        # Transform to: ID | year | GDP | Pop
        
        # Use pandas wide_to_long for proper transformation
        try:
            # First, prepare the dataframe for wide_to_long
            df_copy = df.copy()
            
            # Get non-variable columns (ID and other static columns)
            all_var_cols = set()
            for var_cols_list in variable_mapping.values():
                all_var_cols.update([col for col, _ in var_cols_list])
            
            id_vars = [c for c in df.columns if c not in all_var_cols]
            
            # Rename columns to match wide_to_long requirements (remove underscores between var and time)
            # GDP_2020 -> GDP2020, Population_2021 -> Population2021
            rename_mapping = {}
            stub_names = list(variable_mapping.keys())
            
            for col in all_var_cols:
                for var_name in stub_names:
                    if col.startswith(var_name + '_'):
                        time_part = col[len(var_name + '_'):]
                        new_name = var_name + time_part
                        rename_mapping[col] = new_name
                        break
            
            df_copy = df_copy.rename(columns=rename_mapping)
            
            # Create a row index for wide_to_long
            df_copy = df_copy.reset_index(drop=True)
            df_copy['row_id'] = df_copy.index
            
            # Use wide_to_long
            long_df = pd.wide_to_long(
                df_copy,
                stubnames=stub_names,
                i=['row_id'] + id_vars,
                j=time_var,
                sep='',
                suffix=r'\d+'
            ).reset_index()
            
            # Clean up - remove the temporary row_id
            long_df = long_df.drop(columns=['row_id'])
            
        except Exception:
            # Fallback to manual method if wide_to_long fails
            long_df = _manual_var_time_transform(df, panel_id, time_var, variable_mapping)
            
    elif wide_format_type == "time_var":
        # Case: ID | 2020_GDP | 2021_GDP | 2020_Pop | 2021_Pop
        # Transform to: ID | year | GDP | Pop
        long_df = _manual_time_var_transform(df, panel_id, time_var, variable_mapping)
    
    else:
        raise ValueError(f"Unknown wide format type: {wide_format_type}")
    
    return long_df

def _manual_var_time_transform(df, panel_id, time_var, variable_mapping):
    """Manual transformation for var_time format when wide_to_long fails."""
    # Get non-variable columns (ID and other static columns)
    all_var_cols = set()
    for var_cols_list in variable_mapping.values():
        all_var_cols.update([col for col, _ in var_cols_list])
    
    id_vars = [c for c in df.columns if c not in all_var_cols]
    
    # Extract all unique time periods
    all_times = set()
    for var_cols_list in variable_mapping.values():
        for _, time_part in var_cols_list:
            all_times.add(time_part)
    
    # Create long format by processing each row individually
    long_rows = []
    
    for idx, row in df.iterrows():
        # For each time period, create a new row
        for time_period in sorted(all_times):
            new_row = {}
            
            # Copy ID variables
            for id_col in id_vars:
                new_row[id_col] = row[id_col]
            
            # Add time variable
            new_row[time_var] = time_period
            
            # Add each variable for this time period
            for var_name, var_cols_list in variable_mapping.items():
                value = np.nan  # Default value
                for col, time_part in var_cols_list:
                    if time_part == time_period:
                        value = row[col]
                        break
                new_row[var_name] = value
            
            long_rows.append(new_row)
    
    return pd.DataFrame(long_rows)

def _manual_time_var_transform(df, panel_id, time_var, variable_mapping):
    """Manual transformation for time_var format."""
    # Get non-variable columns (ID and other static columns)
    all_var_cols = set()
    for var_cols_list in variable_mapping.values():
        all_var_cols.update([col for col, _ in var_cols_list])
    
    id_vars = [c for c in df.columns if c not in all_var_cols]
    
    # Extract all unique time periods
    all_times = set()
    for var_cols_list in variable_mapping.values():
        for _, time_part in var_cols_list:
            all_times.add(time_part)
    
    # Create long format by processing each row individually
    long_rows = []
    
    for idx, row in df.iterrows():
        # For each time period, create a new row
        for time_period in sorted(all_times):
            new_row = {}
            
            # Copy ID variables
            for id_col in id_vars:
                new_row[id_col] = row[id_col]
            
            # Add time variable
            new_row[time_var] = time_period
            
            # Add each variable for this time period
            for var_name, var_cols_list in variable_mapping.items():
                value = np.nan  # Default value
                for col, time_part in var_cols_list:
                    if time_part == time_period:
                        value = row[col]
                        break
                new_row[var_name] = value
            
            long_rows.append(new_row)
    
    return pd.DataFrame(long_rows)

def handle_panel_data_declaration(model_choice):
    """Handle panel data declaration and transformation."""
    if 'panel_transform_message' not in st.session_state:
        st.session_state.panel_transform_message = ""

    st.info("Panel data: In long format, each panel ID is repeated for each time period, and the time variable increases. Each (panel ID, time) pair is a row.")
    
    # Detect if data is in wide or long format
    df = st.session_state.df.copy()
    is_wide_format, time_like_cols, wide_format_type, variable_mapping = detect_data_format(df)
    
    if is_wide_format:
        st.info("📊 **Wide format detected**: Data appears to be in wide format with time-like columns. You'll need to specify a name for the new time variable.")
    else:
        st.info("📋 **Long format detected**: Data appears to already be in long format. Please select the existing time variable.")
    
    with st.form("panel_data_form", clear_on_submit=False):
        panel_id = st.selectbox(
            "Select the Panel ID Variable (e.g., country_name):",
            options=list(st.session_state.df.columns),
            key="panel_id_var"
        )
        
        if is_wide_format:
            time_var = st.text_input(
                "Enter the name for the new Time Variable (e.g., year):",
                value="year",
                key="panel_time_var"
            )
        else:
            time_var = st.selectbox(
                "Select the existing Time Variable:",
                options=list(st.session_state.df.columns),
                key="panel_time_var"
            )
        
        submit_panel = st.form_submit_button("Declare Panel Data")

    if submit_panel:
        df = st.session_state.df.copy()
        
        if is_wide_format:
            # Data is in wide format - reshape to long using proper transformation
            try:
                long_df = transform_wide_to_long(df, panel_id, time_var, wide_format_type, time_like_cols, variable_mapping)
                
                # Sort by panel_id and time_var for true long format
                try:
                    long_df[time_var] = pd.to_numeric(long_df[time_var], errors='ignore')
                except Exception:
                    pass
                long_df = long_df.sort_values([panel_id, time_var])
                
                # Apply the transformation directly to the root dataset
                st.session_state.df = long_df.copy()
                
                # Create informative message based on transformation type
                var_info = ""
                if wide_format_type == "var_time":
                    var_info = f" Detected {len(variable_mapping)} variables: {', '.join(variable_mapping.keys())}"
                elif wide_format_type == "time_var":
                    var_info = f" Detected {len(variable_mapping)} variables: {', '.join(variable_mapping.keys())}"
                
                st.session_state.panel_transform_message = f"Data was in wide format ({wide_format_type}). Reshaped to long format using Panel ID '{panel_id}' and Time Variable '{time_var}'.{var_info} The transformation has been applied to the main dataset."
                
                # Update analysis agent with the transformed dataset
                if "analysis_agent" in st.session_state:
                    st.session_state.analysis_agent = create_analysis_agent(
                        api_key=os.getenv("OPENAI_API_KEY"),
                        model_name=model_choice,
                        data_type="Panel Data",
                        df=st.session_state.df,
                        panelvar=panel_id,
                        timevar=time_var
                    )
            except Exception as e:
                st.session_state.panel_transform_message = f"Error during reshaping: {str(e)}"
        else:
            # Data is already in long format - just sort it
            if time_var in df.columns:
                st.session_state.df = df.sort_values([panel_id, time_var]).copy()
                st.session_state.panel_transform_message = f"Data is already in long format. Sorted by Panel ID '{panel_id}' and Time Variable '{time_var}' for clarity. The main dataset has been updated."
            else:
                st.session_state.panel_transform_message = f"Error: Selected time variable '{time_var}' not found in the dataset."
            
            # Update analysis agent
            if "analysis_agent" in st.session_state:
                st.session_state.analysis_agent = create_analysis_agent(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name=model_choice,
                    data_type="Panel Data",
                    df=st.session_state.df,
                    panelvar=panel_id,
                    timevar=time_var
                )

    if st.session_state.panel_transform_message:
        st.success(st.session_state.panel_transform_message)
        clean_df = clean_dataframe_for_display(st.session_state.df.head(20))
        st.dataframe(clean_df, use_container_width=True)

def handle_time_series_declaration(model_choice):
    """Handle time-series data declaration."""
    st.info("Time-series data requires a Time variable.")
    with st.form("ts_data_form", clear_on_submit=False):
        ts_time_var = st.selectbox(
            "Select the Time Variable:",
            options=list(st.session_state.df.columns),
            key="ts_time_var"
        )
        submit_ts = st.form_submit_button("Declare Time-Series Data")
    if submit_ts:
        st.success(f"Time-Series declared with Time Variable: {ts_time_var}")
        clean_df = clean_dataframe_for_display(st.session_state.df.head(20))
        st.dataframe(clean_df, use_container_width=True)
        
        # Update analysis agent
        if "analysis_agent" in st.session_state:
            st.session_state.analysis_agent = create_analysis_agent(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=model_choice,
                data_type="Time-Series",
                df=st.session_state.df,
                timevar=ts_time_var
            )

def handle_cross_sectional_declaration(model_choice):
    """Handle cross-sectional data declaration."""
    st.info("Cross-sectional data: each row is a unique observation at a single point in time. Please select the variable that identifies the groups or clusters (panelvar).")
    with st.form("cs_data_form", clear_on_submit=False):
        cs_panelvar = st.selectbox(
            "Select the Group/Cluster Variable (panelvar):",
            options=list(st.session_state.df.columns),
            key="cs_panelvar"
        )
        submit_cs = st.form_submit_button("Declare Cross-Sectional Data")
    if submit_cs:
        df = st.session_state.df.copy()
        n_groups = df[cs_panelvar].nunique()
        group_sizes = df[cs_panelvar].value_counts().head(10)
        st.success(f"Cross-sectional data declared with {n_groups} unique groups/clusters based on '{cs_panelvar}'.")
        st.write("Sample of group sizes:")
        clean_group_sizes = clean_dataframe_for_display(group_sizes.rename('count').to_frame())
        st.dataframe(clean_group_sizes, use_container_width=True)
        clean_df = clean_dataframe_for_display(df.head(20))
        st.dataframe(clean_df, use_container_width=True)
        
        # Update analysis agent
        if "analysis_agent" in st.session_state:
            st.session_state.analysis_agent = create_analysis_agent(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=model_choice,
                data_type="Cross-Sectional",
                df=df
            )