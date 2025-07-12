import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from utils import (
    init_session_state, 
    load_uploaded_file, 
    apply_changes, 
    undo_changes,
    clean_dataframe_for_display
)

def clean_dataframe_for_streamlit(df):
    """Clean DataFrame to be compatible with Streamlit's PyArrow serialization."""
    if df is None:
        return None
    
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
from agent_manager import AgentManager
from analysis_agent import create_analysis_agent

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Data Preprocessing Chat",
    page_icon="🤖",
    layout="wide"
)

def get_analysis_toolbar(data_type):
    """Return toolbar buttons based on data type."""
    if data_type == "Panel Data":
        return {
            "📊 Panel Description": "xtdescribe",
            "📈 Panel Summary": "xtsum", 
            "📋 Panel Tabulation": "xttab",
            "📉 Panel Line Plot": "xtline",
            "📊 Panel Regression": "xtreg",
            "📊 Panel Logit": "xtlogit",
            "📊 Panel Probit": "xtprobit",
            "📊 Panel Poisson": "xtpoisson",
            "📊 Unit Root Test": "xtunitroot"
        }
    elif data_type == "Time-Series":
        return {
            "⏰ Declare Time Series": "tsset",
            "📊 ARIMA Model": "arima",
            "📊 Newey-West": "newey",
            "📊 Unit Root Test": "dfuller",
            "📊 Autocorrelations": "corrgram",
            "📊 VAR Model": "var_model",
            "📊 Granger Causality": "vargranger",
            "📉 Time Series Plot": "tsline"
        }
    elif data_type == "Cross-Sectional":
        return {
            "📊 Summary Statistics": "summarize",
            "📋 Frequency Table": "tabulate",
            "📊 Correlation": "correlate",
            "📊 Linear Regression": "regress",
            "📊 Logistic Regression": "logit",
            "📊 Probit Regression": "probit",
            "📊 T-Test": "ttest",
            "📊 Chi-Square Test": "chi2",
            "📊 Histogram": "histogram",
            "📊 Scatter Plot": "scatter"
        }
    else:
        return {}

def prompt_user_for_inputs(function_name, df, analysis_tool):
    """Prompt user for required inputs based on the function being called."""
    
    # Helper function to detect numeric columns in cleaned DataFrame
    def get_numeric_columns(df):
        """Detect which columns can be converted to numeric."""
        numeric_cols = []
        for col in df.columns:
            try:
                # Skip if column is empty or all NaN
                if df[col].isna().all() or (df[col] == '').all():
                    continue
                
                # Try to convert to numeric, ignoring empty strings
                test_values = df[col].replace('', np.nan).dropna()
                if len(test_values) > 0:
                    pd.to_numeric(test_values, errors='raise')
                    numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
        return numeric_cols
    
    # Get numeric columns
    numeric_cols = get_numeric_columns(df)
    
    # Debug info (only show if no numeric columns found)
    if len(numeric_cols) == 0:
        st.warning(f"No numeric columns detected. All columns are: {list(df.columns)}")
        st.info("This might happen if the data cleaning process converted all columns to strings. The functions will still work but may show limited options.")
        
        # Fallback: try to detect numeric columns more leniently
        st.write("Attempting fallback detection...")
        fallback_numeric_cols = []
        for col in df.columns:
            try:
                # Try to convert a sample of values
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Try to convert to float
                    pd.to_numeric(sample_values.astype(str).str.replace(',', ''), errors='raise')
                    fallback_numeric_cols.append(col)
            except:
                continue
        
        if fallback_numeric_cols:
            st.success(f"Fallback detection found {len(fallback_numeric_cols)} potentially numeric columns: {fallback_numeric_cols}")
            numeric_cols = fallback_numeric_cols
        else:
            st.error("No numeric columns could be detected even with fallback method.")
            return None
    
    # Panel Data functions
    if function_name == "xtsum":
        with st.form(f"form_{function_name}"):
            st.write("**Panel Summary Statistics**")
            st.write("Select variables to analyze:")
            varlist = st.multiselect(
                "Variables:",
                options=numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols  # Default to first 3 numeric
            )
            submitted = st.form_submit_button("Run Analysis")
            if submitted:
                if varlist:
                    query = f"show summary statistics for {', '.join(varlist)}"
                    return analysis_tool._run(query)
                else:
                    st.error("Please select at least one variable.")
        return None
    
    elif function_name == "xtline":
        with st.form(f"form_{function_name}"):
            st.write("**Panel Line Plot**")
            yvar = st.selectbox(
                "Select variable to plot:",
                options=numeric_cols,
                key="xtline_yvar"
            )
            n_panels = st.slider(
                "Maximum number of panels to show:",
                min_value=1,
                max_value=20,
                value=10,
                key="xtline_n_panels"
            )
            submitted = st.form_submit_button("Create Plot")
            if submitted:
                query = f"create panel line plot for {yvar} with max {n_panels} panels"
                return analysis_tool._run(query)
        return None
    
    elif function_name in ["xtreg", "xtlogit", "xtprobit", "xtpoisson"]:
        with st.form(f"form_{function_name}"):
            st.write(f"**Panel {function_name.replace('xt', '').title()} Regression**")
            
            if len(numeric_cols) == 0:
                st.error("No numeric columns found. Please check your data.")
                return None
            
            depvar = st.selectbox(
                "Dependent variable:",
                options=numeric_cols,
                key=f"{function_name}_depvar"
            )
            
            # Filter out the selected dependent variable from independent variables
            available_indepvars = [col for col in numeric_cols if col != depvar]
            
            # Debug: Show available options
            st.write(f"Debug - Available independent variables: {available_indepvars}")
            st.write(f"Debug - Selected dependent variable: {depvar}")
            
            indepvars = st.multiselect(
                "Independent variables:",
                options=available_indepvars,
                default=available_indepvars[:1] if available_indepvars else [],  # Default to first available variable
                key=f"{function_name}_indepvars"
            )
            
            submitted = st.form_submit_button("Run Regression")
            if submitted:
                # Debug: Print the values to understand what's happening
                st.write(f"Debug - indepvars: {indepvars}")
                st.write(f"Debug - type of indepvars: {type(indepvars)}")
                st.write(f"Debug - len of indepvars: {len(indepvars) if indepvars else 0}")
                
                # Store in session state for debugging
                st.session_state[f"{function_name}_last_indepvars"] = indepvars
                st.session_state[f"{function_name}_last_depvar"] = depvar
                
                if indepvars and len(indepvars) > 0:
                    query = f"run panel {function_name.replace('xt', '')} regression with {depvar} as dependent and {', '.join(indepvars)} as independent"
                    return analysis_tool._run(query)
                else:
                    st.error("Please select at least one independent variable.")
                    # Show what was stored in session state
                    st.write(f"Session state - last indepvars: {st.session_state.get(f'{function_name}_last_indepvars', 'None')}")
                    st.write(f"Session state - last depvar: {st.session_state.get(f'{function_name}_last_depvar', 'None')}")
        return None
    
    elif function_name == "xtunitroot":
        with st.form(f"form_{function_name}"):
            st.write("**Panel Unit Root Test**")
            var = st.selectbox(
                "Select variable to test:",
                options=numeric_cols,
                key="xtunitroot_var"
            )
            submitted = st.form_submit_button("Run Test")
            if submitted:
                query = f"run panel unit root test on {var}"
                return analysis_tool._run(query)
        return None
    
    # Time Series functions
    elif function_name == "arima":
        with st.form(f"form_{function_name}"):
            st.write("**ARIMA Model**")
            var = st.selectbox(
                "Select variable:",
                options=numeric_cols,
                key="arima_var"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.number_input("AR order (p):", min_value=0, max_value=5, value=1, key="arima_p")
            with col2:
                d = st.number_input("Difference order (d):", min_value=0, max_value=3, value=1, key="arima_d")
            with col3:
                q = st.number_input("MA order (q):", min_value=0, max_value=5, value=1, key="arima_q")
            submitted = st.form_submit_button("Fit ARIMA")
            if submitted:
                query = f"fit ARIMA({p},{d},{q}) model for {var}"
                return analysis_tool._run(query)
        return None
    
    elif function_name in ["dfuller", "corrgram", "tsline"]:
        with st.form(f"form_{function_name}"):
            st.write(f"**{function_name.upper()} Analysis**")
            var = st.selectbox(
                "Select variable:",
                options=numeric_cols,
                key=f"{function_name}_var"
            )
            submitted = st.form_submit_button("Run Analysis")
            if submitted:
                if function_name == "dfuller":
                    query = f"run unit root test on {var}"
                elif function_name == "corrgram":
                    query = f"show autocorrelations for {var}"
                elif function_name == "tsline":
                    query = f"create time series plot for {var}"
                return analysis_tool._run(query)
        return None
    
    elif function_name in ["newey", "var_model"]:
        with st.form(f"form_{function_name}"):
            st.write(f"**{function_name.replace('_', ' ').title()} Analysis**")
            if function_name == "newey":
                depvar = st.selectbox(
                    "Dependent variable:",
                    options=numeric_cols,
                    key="newey_depvar"
                )
                indepvars = st.multiselect(
                    "Independent variables:",
                    options=[col for col in numeric_cols if col != depvar],
                    default=[col for col in numeric_cols if col != depvar][:1] if [col for col in numeric_cols if col != depvar] else [],  # Default to first available variable
                    key="newey_indepvars"
                )
                submitted = st.form_submit_button("Run Newey-West")
                if submitted:
                    # Debug: Print the values to understand what's happening
                    st.write(f"Debug - indepvars: {indepvars}")
                    st.write(f"Debug - type of indepvars: {type(indepvars)}")
                    st.write(f"Debug - len of indepvars: {len(indepvars) if indepvars else 0}")
                    
                    if indepvars and len(indepvars) > 0:
                        query = f"run Newey-West regression with {depvar} as dependent and {', '.join(indepvars)} as independent"
                        return analysis_tool._run(query)
                    else:
                        st.error("Please select at least one independent variable.")
            else:  # var_model
                vars = st.multiselect(
                    "Select variables for VAR:",
                    options=numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
                    key="var_vars"
                )
                submitted = st.form_submit_button("Fit VAR")
                if submitted:
                    if len(vars) >= 2:
                        query = f"fit VAR model with {', '.join(vars)}"
                        return analysis_tool._run(query)
                    else:
                        st.error("Please select at least 2 variables for VAR.")
        return None
    
    # Cross-Sectional functions
    elif function_name == "summarize":
        with st.form(f"form_{function_name}"):
            st.write("**Summary Statistics**")
            varlist = st.multiselect(
                "Select variables:",
                options=numeric_cols,
                default=numeric_cols,
                key="summarize_vars"
            )
            submitted = st.form_submit_button("Show Summary")
            if submitted:
                if varlist:
                    query = f"show summary statistics for {', '.join(varlist)}"
                    return analysis_tool._run(query)
                else:
                    st.error("Please select at least one variable.")
        return None
    
    elif function_name == "tabulate":
        with st.form(f"form_{function_name}"):
            st.write("**Frequency Table**")
            var1 = st.selectbox(
                "First variable:",
                options=df.columns.tolist(),
                key="tabulate_var1"
            )
            var2 = st.selectbox(
                "Second variable (optional):",
                options=["None"] + [col for col in df.columns.tolist() if col != var1],
                key="tabulate_var2"
            )
            submitted = st.form_submit_button("Create Table")
            if submitted:
                if var2 == "None":
                    query = f"create frequency table for {var1}"
                else:
                    query = f"create frequency table for {var1} and {var2}"
                return analysis_tool._run(query)
        return None
    
    elif function_name == "correlate":
        with st.form(f"form_{function_name}"):
            st.write("**Correlation Matrix**")
            varlist = st.multiselect(
                "Select variables:",
                options=numeric_cols,
                default=numeric_cols,
                key="correlate_vars"
            )
            submitted = st.form_submit_button("Show Correlation")
            if submitted:
                if varlist:
                    query = f"show correlation matrix for {', '.join(varlist)}"
                    return analysis_tool._run(query)
                else:
                    st.error("Please select at least one variable.")
        return None
    
    elif function_name in ["regress", "logit", "probit"]:
        with st.form(f"form_{function_name}"):
            st.write(f"**{function_name.title()} Regression**")
            depvar = st.selectbox(
                "Dependent variable:",
                options=numeric_cols,
                key=f"{function_name}_depvar"
            )
            indepvars = st.multiselect(
                "Independent variables:",
                options=[col for col in numeric_cols if col != depvar],
                default=[col for col in numeric_cols if col != depvar][:1] if [col for col in numeric_cols if col != depvar] else [],  # Default to first available variable
                key=f"{function_name}_indepvars"
            )
            submitted = st.form_submit_button(f"Run {function_name.title()}")
            if submitted:
                # Debug: Print the values to understand what's happening
                st.write(f"Debug - indepvars: {indepvars}")
                st.write(f"Debug - type of indepvars: {type(indepvars)}")
                st.write(f"Debug - len of indepvars: {len(indepvars) if indepvars else 0}")
                
                if indepvars and len(indepvars) > 0:
                    query = f"run {function_name} regression with {depvar} as dependent and {', '.join(indepvars)} as independent"
                    return analysis_tool._run(query)
                else:
                    st.error("Please select at least one independent variable.")
        return None
    
    elif function_name in ["ttest", "chi2"]:
        with st.form(f"form_{function_name}"):
            st.write(f"**{function_name.upper()} Test**")
            if function_name == "ttest":
                var = st.selectbox(
                    "Variable to test:",
                    options=numeric_cols,
                    key="ttest_var"
                )
                by = st.selectbox(
                    "Grouping variable:",
                    options=["None"] + [col for col in df.columns.tolist() if col != var],
                    key="ttest_by"
                )
                submitted = st.form_submit_button("Run T-Test")
                if submitted:
                    if by == "None":
                        query = f"run t-test for {var}"
                    else:
                        query = f"run t-test for {var} by {by}"
                    return analysis_tool._run(query)
            else:  # chi2
                var1 = st.selectbox(
                    "First variable:",
                    options=df.columns.tolist(),
                    key="chi2_var1"
                )
                var2 = st.selectbox(
                    "Second variable:",
                    options=[col for col in df.columns.tolist() if col != var1],
                    key="chi2_var2"
                )
                submitted = st.form_submit_button("Run Chi-Square Test")
                if submitted:
                    query = f"run chi-square test for {var1} and {var2}"
                    return analysis_tool._run(query)
        return None
    
    elif function_name in ["histogram", "scatter"]:
        with st.form(f"form_{function_name}"):
            st.write(f"**{function_name.title()} Plot**")
            if function_name == "histogram":
                var = st.selectbox(
                    "Variable to plot:",
                    options=numeric_cols,
                    key="histogram_var"
                )
                submitted = st.form_submit_button("Create Histogram")
                if submitted:
                    query = f"create histogram for {var}"
                    return analysis_tool._run(query)
            else:  # scatter
                var1 = st.selectbox(
                    "X-axis variable:",
                    options=numeric_cols,
                    key="scatter_var1"
                )
                var2 = st.selectbox(
                    "Y-axis variable:",
                    options=numeric_cols,
                    key="scatter_var2"
                )
                by = st.selectbox(
                    "Grouping variable (optional):",
                    options=["None"] + [col for col in df.columns.tolist() if col not in [var1, var2]],
                    key="scatter_by"
                )
                submitted = st.form_submit_button("Create Scatter Plot")
                if submitted:
                    if by == "None":
                        query = f"create scatter plot for {var1} vs {var2}"
                    else:
                        query = f"create scatter plot for {var1} vs {var2} by {by}"
                    return analysis_tool._run(query)
        return None
    
    # Default case - simple functions that don't need additional inputs
    else:
        # Map function names to descriptive queries
        query_mapping = {
            "xtdescribe": "describe panel structure",
            "xttab": "tabulate panel data",
            "tsset": "declare time series",
            "vargranger": "run Granger causality test"
        }
        query = query_mapping.get(function_name, function_name)
        return analysis_tool._run(query)

def main():
    st.title("🤖 Data Preprocessing Chat MVP")
    st.markdown("Upload your CSV/XLSX files and chat with AI to preprocess your data!")
    
    # Initialize session state
    init_session_state()
    
    # Initialize session state for analysis console
    if "console_log" not in st.session_state:
        st.session_state.console_log = []
    
    # Model selection FIRST (before using it)
    model_choice = st.sidebar.selectbox(
        "🤖 Choose AI Model:",
        ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "o1-preview"],
        index=0,
        help="Higher quality models = better results but more expensive"
    )
    
    # Initialize agent manager
    if "agent_manager" not in st.session_state or st.session_state.get("current_model") != model_choice:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ Please set OPENAI_API_KEY in your .env file")
            st.stop()
        st.session_state.agent_manager = AgentManager(api_key, model_choice)
        st.session_state.current_model = model_choice
    
    # Initialize analysis agent
    if "analysis_agent" not in st.session_state:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.session_state.analysis_agent = create_analysis_agent(
                api_key=api_key,
                model_name=model_choice,
                data_type="Cross-Sectional",
                df=st.session_state.df
            )
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 File Upload")
        
        uploaded_files = st.file_uploader(
            "Choose CSV/XLSX files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.dfs:
                    df = load_uploaded_file(uploaded_file)
                    if df is not None:
                        st.session_state.dfs[uploaded_file.name] = df
                        st.success(f"✅ Loaded {uploaded_file.name}")
                        
                        # Set as main df if first file
                        if st.session_state.df is None:
                            st.session_state.df = df.copy()
        
        # Display loaded files
        if st.session_state.dfs:
            st.subheader("📊 Loaded Files")
            for filename in st.session_state.dfs.keys():
                rows, cols = st.session_state.dfs[filename].shape
                st.write(f"• **{filename}**: {rows} rows, {cols} cols")
        
        # Control buttons
        st.subheader("🎛️ Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Apply", disabled=st.session_state.df is None):
                apply_changes()
        
        with col2:
            if st.button("↩️ Undo", disabled=len(st.session_state.history) == 0):
                undo_changes()
        
        # Download button
        if st.session_state.df is not None:
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
    
    # Main content area
    if st.session_state.df is not None:
        # --- Data Declaration Section ---
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
            df = st.session_state.df
            suggestion = ""
            # Heuristic: Panel if two+ columns look like time, and one looks like id; Time-Series if one time col; else Cross-Sectional
            import re
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
            st.session_state.ai_data_type_suggestion = suggestion

        if st.session_state.ai_data_type_suggestion:
            st.info(f"**AI Suggestion:** {st.session_state.ai_data_type_suggestion}\n\n*This is only a recommendation and based on assumptions from the AI. Please review and do not use immediately without checking.*")

        # For storing transformation message
        if 'panel_transform_message' not in st.session_state:
            st.session_state.panel_transform_message = ""

        # --- Panel Data Declaration ---
        if declaration == "Panel Data":
            st.info("Panel data: In long format, each panel ID is repeated for each time period, and the time variable increases. Each (panel ID, time) pair is a row.")
            
            # Detect if data is in wide or long format
            df = st.session_state.df.copy()
            import re
            year_pattern = re.compile(r"^(19[7-9][0-9]|20[0-2][0-9]|2025)$")
            time_like_cols = [col for col in df.columns if year_pattern.match(str(col))]
            if not time_like_cols:
                alt_time_pattern = re.compile(r"(\d{4}|Q[1-4]_\d{4}|\d{4}Q[1-4])")
                time_like_cols = [col for col in df.columns if alt_time_pattern.search(str(col))]
            
            is_wide_format = len(time_like_cols) > 0
            
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
                    # Data is in wide format - reshape to long
                    try:
                        id_vars = [panel_id] + [c for c in df.columns if c not in time_like_cols and c != panel_id]
                        id_vars = list(dict.fromkeys(id_vars))
                        long_df = pd.melt(
                            df,
                            id_vars=id_vars,
                            value_vars=time_like_cols,
                            var_name=time_var,
                            value_name="value"
                        )
                        # Sort by panel_id and time_var for true long format
                        try:
                            long_df[time_var] = pd.to_numeric(long_df[time_var], errors='ignore')
                        except Exception:
                            pass
                        long_df = long_df.sort_values([panel_id, time_var])
                        
                        # Apply the transformation directly to the root dataset
                        st.session_state.df = long_df.copy()
                        st.session_state.panel_transform_message = f"Data was in wide format. Reshaped to long format using Panel ID '{panel_id}' and Time Variable '{time_var}'. Sorted for clarity. The transformation has been applied to the main dataset."
                        
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
                clean_df = clean_dataframe_for_streamlit(st.session_state.df.head(20))
                st.dataframe(clean_df, use_container_width=True)

        # --- Time-Series Declaration ---
        elif declaration == "Time-Series":
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
                clean_df = clean_dataframe_for_streamlit(st.session_state.df.head(20))
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

        # --- Cross-Sectional Declaration ---
        elif declaration == "Cross-Sectional":
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
                clean_group_sizes = clean_dataframe_for_streamlit(group_sizes.rename('count').to_frame())
                st.dataframe(clean_group_sizes, use_container_width=True)
                clean_df = clean_dataframe_for_streamlit(df.head(20))
                st.dataframe(clean_df, use_container_width=True)
                
                # Update analysis agent
                if "analysis_agent" in st.session_state:
                    st.session_state.analysis_agent = create_analysis_agent(
                        api_key=os.getenv("OPENAI_API_KEY"),
                        model_name=model_choice,
                        data_type="Cross-Sectional",
                        df=df
                    )

        # --- Analysis Toolbar ---
        if declaration != "None":
            st.subheader("🔧 Analysis Toolbar")
            st.write(f"Available {declaration} analysis functions:")
            
            toolbar_buttons = get_analysis_toolbar(declaration)

            # Initialize session state for active function
            if 'active_analysis_function' not in st.session_state:
                st.session_state.active_analysis_function = None
            if 'analysis_result' not in st.session_state:
                st.session_state.analysis_result = None

            # If a function is active, show its input form
            if st.session_state.active_analysis_function:
                function_name = st.session_state.active_analysis_function
                analysis_tool = st.session_state.analysis_agent.tools[0]
                result = prompt_user_for_inputs(function_name, st.session_state.df, analysis_tool)
                if result is not None:
                    # Log the function call and result to the console
                    call_str = f"{analysis_tool.name}._run('{function_name}')"
                    st.session_state.console_log.append({"call": call_str, "result": result})
                    st.session_state.analysis_result = result
                    st.session_state.active_analysis_function = None
                    st.rerun()
                # Show result if available
                if st.session_state.analysis_result:
                    # Check if result contains HTML image tag
                    if "<img" in st.session_state.analysis_result:
                        text_parts = st.session_state.analysis_result.split("<img")
                        if text_parts[0].strip():
                            st.write(text_parts[0].strip())
                        img_start = st.session_state.analysis_result.find("<img")
                        img_end = st.session_state.analysis_result.find(">", img_start) + 1
                        img_html = st.session_state.analysis_result[img_start:img_end]
                        st.markdown(img_html, unsafe_allow_html=True)
                        if len(text_parts) > 1:
                            after_img = text_parts[1].split(">", 1)
                            if len(after_img) > 1 and after_img[1].strip():
                                st.write(after_img[1].strip())
                    else:
                        # Check if the result is a DataFrame and clean it if needed
                        if isinstance(st.session_state.analysis_result, pd.DataFrame):
                            clean_result = clean_dataframe_for_streamlit(st.session_state.analysis_result)
                            st.dataframe(clean_result, use_container_width=True)
                        else:
                            st.write(st.session_state.analysis_result)
                    st.session_state.analysis_result = None
            else:
                # Show toolbar buttons if no function is active
                cols = st.columns(3)
                for i, (button_text, function_name) in enumerate(toolbar_buttons.items()):
                    col_idx = i % 3
                    with cols[col_idx]:
                        if st.button(button_text, key=f"toolbar_{function_name}"):
                            st.session_state.active_analysis_function = function_name
                            st.rerun()

        # --- Analysis Console Section ---
        st.subheader("🖥️ Analysis Console")
        if st.button("Clear Console"):
            st.session_state.console_log = []
        for entry in reversed(st.session_state.console_log):
            st.code(entry["call"], language="python")
            if isinstance(entry["result"], pd.DataFrame):
                clean_result = clean_dataframe_for_streamlit(entry["result"])
                st.dataframe(clean_result, use_container_width=True)
            else:
                st.write(entry["result"])
            st.markdown("---")
        
        # Chat interface
        st.subheader("💬 Chat with your data")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("What would you like to do with your data?"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Process with appropriate agent
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    # Use analysis agent if data type is declared, otherwise use general agent
                    if declaration != "None" and "analysis_agent" in st.session_state:
                        try:
                            response = st.session_state.analysis_agent.run(prompt)
                            # Log the function call and result to the console
                            st.session_state.console_log.append({"call": f"analysis_agent.run('{prompt}')", "result": response})
                            # Check if response contains HTML image tag
                            if "<img" in response:
                                # Extract the text content before the image
                                text_parts = response.split("<img")
                                if text_parts[0].strip():
                                    st.write(text_parts[0].strip())
                                
                                # Extract and display the image
                                img_start = response.find("<img")
                                img_end = response.find(">", img_start) + 1
                                img_html = response[img_start:img_end]
                                
                                # Display the image using st.markdown with unsafe_allow_html
                                st.markdown(img_html, unsafe_allow_html=True)
                                
                                # Display any text after the image
                                if len(text_parts) > 1:
                                    after_img = text_parts[1].split(">", 1)
                                    if len(after_img) > 1 and after_img[1].strip():
                                        st.write(after_img[1].strip())
                            else:
                                st.write(response)
                        except Exception as e:
                            st.error(f"Analysis error: {str(e)}")
                            # Fallback to general agent
                            response = st.session_state.agent_manager.process_query(prompt, st.session_state.df)
                            st.write(response)
                    else:
                        # Use general agent for data preprocessing
                        response = st.session_state.agent_manager.process_query(prompt, st.session_state.df)
                        st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Data preview
        st.subheader("📊 Current Data Preview")
        
        # Display basic info
        rows, cols = st.session_state.df.shape
        st.write(f"**Shape**: {rows} rows × {cols} columns")
        
        # Display data
        try:
            clean_df = clean_dataframe_for_streamlit(st.session_state.df.head(20))
            st.dataframe(clean_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Display issue with dataframe: {str(e)}")
            # Fallback: show as text
            st.text(str(st.session_state.df.head(20)))
        
        # Display data types
        with st.expander("📋 Data Types & Info"):
            col1, col2 = st.columns(2)
            
            with col1:
                            st.write("**Data Types:**")
            try:
                # Convert dtypes to a more displayable format
                dtypes_df = st.session_state.df.dtypes.to_frame('Data Type').reset_index()
                dtypes_df.columns = ['Column', 'Data Type']
                clean_dtypes_df = clean_dataframe_for_streamlit(dtypes_df)
                st.dataframe(clean_dtypes_df, use_container_width=True)
            except:
                st.text(str(st.session_state.df.dtypes))
            
            with col2:
                            st.write("**Missing Values:**")
            try:
                # Convert missing values to a more displayable format
                missing_df = st.session_state.df.isnull().sum().to_frame('Missing Count').reset_index()
                missing_df.columns = ['Column', 'Missing Count']
                clean_missing_df = clean_dataframe_for_streamlit(missing_df)
                st.dataframe(clean_missing_df, use_container_width=True)
            except:
                st.text(str(st.session_state.df.isnull().sum()))
            
            # Additional summary statistics similar to STATA's summarize
            st.write("**Summary Statistics (STATA-style):**")
            try:
                # Get numeric columns only
                numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    summary_stats = []
                    for col in numeric_cols:
                        col_data = st.session_state.df[col].dropna()
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
                        clean_summary_df = clean_dataframe_for_streamlit(summary_df)
                        st.dataframe(clean_summary_df, use_container_width=True)
                    else:
                        st.write("No numeric columns found for summary statistics.")
                else:
                    st.write("No numeric columns found for summary statistics.")
            except Exception as e:
                st.warning(f"Error calculating summary statistics: {str(e)}")
        
        # Data Quality Checks section
        with st.expander("🛡️ Data Quality Checks"):
            st.write("### 1. Correct Storage Types")
            st.write("Below are the storage types for each variable. Review to ensure numeric variables are stored as numeric types.")
            try:
                # Convert dtypes to a more displayable format
                dtypes_df = st.session_state.df.dtypes.to_frame('Data Type').reset_index()
                dtypes_df.columns = ['Column', 'Data Type']
                clean_dtypes_df = clean_dataframe_for_streamlit(dtypes_df)
                st.dataframe(clean_dtypes_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Error displaying dtypes: {str(e)}")
            st.write("---")

            st.write("### 2. Descriptive Statistics (describe)")
            try:
                describe_df = st.session_state.df.describe(include='all')
                clean_describe_df = clean_dataframe_for_streamlit(describe_df)
                st.dataframe(clean_describe_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Error displaying describe: {str(e)}")
            st.write("---")

            st.write("### 3. No Unwanted Duplicates")
            st.write("Select key columns to check for duplicate rows:")
            key_cols = st.multiselect(
                "Key columns for uniqueness check:",
                options=list(st.session_state.df.columns),
                default=[]
            )
            if key_cols:
                dupes = st.session_state.df.duplicated(subset=key_cols, keep=False)
                n_dupes = dupes.sum()
                if n_dupes > 0:
                    st.warning(f"Found {n_dupes} duplicate rows based on selected keys.")
                    dupes_df = st.session_state.df.loc[dupes, key_cols + [c for c in st.session_state.df.columns if c not in key_cols][:3]].head(20)
                    clean_dupes_df = clean_dataframe_for_streamlit(dupes_df)
                    st.dataframe(clean_dupes_df, use_container_width=True)
                else:
                    st.success("No duplicates found based on selected keys.")
            else:
                st.info("Select columns to check for duplicates.")
            st.write("---")

            st.write("### 4. Plausible Values (Min/Max)")
            try:
                numeric_cols = st.session_state.df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    minmax = st.session_state.df[numeric_cols].agg(['min', 'max']).T
                    clean_minmax = clean_dataframe_for_streamlit(minmax)
                    st.dataframe(clean_minmax, use_container_width=True)
                    st.info("Review min/max for suspicious values (e.g., negative ages, out-of-range scores).")
                else:
                    st.write("No numeric columns found.")
            except Exception as e:
                st.warning(f"Error displaying min/max: {str(e)}")
            st.write("---")

            st.write("### 5. Sensible Categories (Frequency Tables)")
            cat_cols = st.session_state.df.select_dtypes(include=['object', 'category', 'bool']).columns
            if len(cat_cols) > 0:
                for col in cat_cols:
                    st.write(f"**{col}**")
                    try:
                        freq_df = st.session_state.df[col].value_counts(dropna=False).rename('count').to_frame()
                        clean_freq_df = clean_dataframe_for_streamlit(freq_df)
                        st.dataframe(clean_freq_df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Error displaying frequency table for {col}: {str(e)}")
            else:
                st.write("No categorical columns found.")
            st.write("---")

            st.write("### 6. Logical Consistency (User-defined Rules)")
            st.info("Logical consistency checks (e.g., year_of_death >= year_of_birth) must be defined by the user. Please review your data and use custom code or rules as needed.")
    
    else:
        st.info("👆 Please upload at least one CSV or XLSX file to get started!")
        
        # Show example commands
        st.subheader("💡 Example Commands")
        examples = [
            "Show me basic statistics of the data",
            "Remove rows with missing values",
            "Fill missing values in column 'age' with the mean",
            "Merge all uploaded files on column 'id'",
            "Create a new column 'total' by adding column1 and column2",
            "Remove outliers from column 'price' using IQR method",
            "Group by 'category' and calculate mean of 'value'",
            "Convert 'date' column to datetime format"
        ]
        
        for example in examples:
            st.code(example)

if __name__ == "__main__":
    main()