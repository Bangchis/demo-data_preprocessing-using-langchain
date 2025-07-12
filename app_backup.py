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
    
    # Panel Data functions
    if function_name == "xtsum":
        st.write("**Panel Summary Statistics**")
        st.write("Select variables to analyze:")
        varlist = st.multiselect(
            "Variables:",
            options=df.select_dtypes(include=[np.number]).columns.tolist(),
            default=df.select_dtypes(include=[np.number]).columns.tolist()[:3]  # Default to first 3 numeric
        )
        if st.button("Run Analysis", key="run_xtsum"):
            if varlist:
                query = f"show summary statistics for {', '.join(varlist)}"
                return analysis_tool._run(query)
            else:
                st.error("Please select at least one variable.")
                return None
        return None
    
    elif function_name == "xtline":
        st.write("**Panel Line Plot**")
        yvar = st.selectbox(
            "Select variable to plot:",
            options=df.select_dtypes(include=[np.number]).columns.tolist(),
            key="xtline_yvar"
        )
        n_panels = st.slider(
            "Maximum number of panels to show:",
            min_value=1,
            max_value=20,
            value=10,
            key="xtline_n_panels"
        )
        if st.button("Create Plot", key="run_xtline"):
            query = f"create panel line plot for {yvar} with max {n_panels} panels"
            return analysis_tool._run(query)
        return None
    
    elif function_name in ["xtreg", "xtlogit", "xtprobit", "xtpoisson"]:
        st.write(f"**Panel {function_name.replace('xt', '').title()} Regression**")
        depvar = st.selectbox(
            "Dependent variable:",
            options=df.select_dtypes(include=[np.number]).columns.tolist(),
            key=f"{function_name}_depvar"
        )
        indepvars = st.multiselect(
            "Independent variables:",
            options=[col for col in df.columns if col != depvar],
            key=f"{function_name}_indepvars"
        )
        if st.button("Run Regression", key=f"run_{function_name}"):
            if indepvars:
                query = f"run panel {function_name.replace('xt', '')} regression with {depvar} as dependent and {', '.join(indepvars)} as independent"
                return analysis_tool._run(query)
            else:
                st.error("Please select at least one independent variable.")
                return None
        return None
    
    elif function_name == "xtunitroot":
        st.write("**Panel Unit Root Test**")
        var = st.selectbox(
            "Select variable to test:",
            options=df.select_dtypes(include=[np.number]).columns.tolist(),
            key="xtunitroot_var"
        )
        if st.button("Run Test", key="run_xtunitroot"):
            query = f"run panel unit root test on {var}"
            return analysis_tool._run(query)
        return None
    
    # Time Series functions
    elif function_name == "arima":
        st.write("**ARIMA Model**")
        var = st.selectbox(
            "Select variable:",
            options=df.select_dtypes(include=[np.number]).columns.tolist(),
            key="arima_var"
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("AR order (p):", min_value=0, max_value=5, value=1, key="arima_p")
        with col2:
            d = st.number_input("Difference order (d):", min_value=0, max_value=3, value=1, key="arima_d")
        with col3:
            q = st.number_input("MA order (q):", min_value=0, max_value=5, value=1, key="arima_q")
        if st.button("Fit ARIMA", key="run_arima"):
            query = f"fit ARIMA({p},{d},{q}) model for {var}"
            return analysis_tool._run(query)
        return None
    
    elif function_name in ["dfuller", "corrgram", "tsline"]:
        st.write(f"**{function_name.upper()} Analysis**")
        var = st.selectbox(
            "Select variable:",
            options=df.select_dtypes(include=[np.number]).columns.tolist(),
            key=f"{function_name}_var"
        )
        if st.button("Run Analysis", key=f"run_{function_name}"):
            if function_name == "dfuller":
                query = f"run unit root test on {var}"
            elif function_name == "corrgram":
                query = f"show autocorrelations for {var}"
            elif function_name == "tsline":
                query = f"create time series plot for {var}"
            return analysis_tool._run(query)
        return None
    
    elif function_name in ["newey", "var_model"]:
        st.write(f"**{function_name.replace('_', ' ').title()} Analysis**")
        if function_name == "newey":
            depvar = st.selectbox(
                "Dependent variable:",
                options=df.select_dtypes(include=[np.number]).columns.tolist(),
                key="newey_depvar"
            )
            indepvars = st.multiselect(
                "Independent variables:",
                options=[col for col in df.select_dtypes(include=[np.number]).columns.tolist() if col != depvar],
                key="newey_indepvars"
            )
            if st.button("Run Newey-West", key="run_newey"):
                if indepvars:
                    query = f"run Newey-West regression with {depvar} as dependent and {', '.join(indepvars)} as independent"
                    return analysis_tool._run(query)
                else:
                    st.error("Please select at least one independent variable.")
                    return None
        else:  # var_model
            vars = st.multiselect(
                "Select variables for VAR:",
                options=df.select_dtypes(include=[np.number]).columns.tolist(),
                default=df.select_dtypes(include=[np.number]).columns.tolist()[:3],
                key="var_vars"
            )
            if st.button("Fit VAR", key="run_var"):
                if len(vars) >= 2:
                    query = f"fit VAR model with {', '.join(vars)}"
                    return analysis_tool._run(query)
                else:
                    st.error("Please select at least 2 variables for VAR.")
                    return None
        return None
    
    # Cross-Sectional functions
    elif function_name == "summarize":
        st.write("**Summary Statistics**")
        varlist = st.multiselect(
            "Select variables:",
            options=df.select_dtypes(include=[np.number]).columns.tolist(),
            default=df.select_dtypes(include=[np.number]).columns.tolist(),
            key="summarize_vars"
        )
        if st.button("Show Summary", key="run_summarize"):
            if varlist:
                query = f"show summary statistics for {', '.join(varlist)}"
                return analysis_tool._run(query)
            else:
                st.error("Please select at least one variable.")
                return None
        return None
    
    elif function_name == "tabulate":
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
        if st.button("Create Table", key="run_tabulate"):
            if var2 == "None":
                query = f"create frequency table for {var1}"
            else:
                query = f"create frequency table for {var1} and {var2}"
            return analysis_tool._run(query)
        return None
    
    elif function_name == "correlate":
        st.write("**Correlation Matrix**")
        varlist = st.multiselect(
            "Select variables:",
            options=df.select_dtypes(include=[np.number]).columns.tolist(),
            default=df.select_dtypes(include=[np.number]).columns.tolist(),
            key="correlate_vars"
        )
        if st.button("Show Correlation", key="run_correlate"):
            if varlist:
                query = f"show correlation matrix for {', '.join(varlist)}"
                return analysis_tool._run(query)
            else:
                st.error("Please select at least one variable.")
                return None
        return None
    
    elif function_name in ["regress", "logit", "probit"]:
        st.write(f"**{function_name.title()} Regression**")
        depvar = st.selectbox(
            "Dependent variable:",
            options=df.select_dtypes(include=[np.number]).columns.tolist(),
            key=f"{function_name}_depvar"
        )
        indepvars = st.multiselect(
            "Independent variables:",
            options=[col for col in df.columns.tolist() if col != depvar],
            key=f"{function_name}_indepvars"
        )
        if st.button(f"Run {function_name.title()}", key=f"run_{function_name}"):
            if indepvars:
                query = f"run {function_name} regression with {depvar} as dependent and {', '.join(indepvars)} as independent"
                return analysis_tool._run(query)
            else:
                st.error("Please select at least one independent variable.")
                return None
        return None
    
    elif function_name in ["ttest", "chi2"]:
        st.write(f"**{function_name.upper()} Test**")
        if function_name == "ttest":
            var = st.selectbox(
                "Variable to test:",
                options=df.select_dtypes(include=[np.number]).columns.tolist(),
                key="ttest_var"
            )
            by = st.selectbox(
                "Grouping variable:",
                options=["None"] + [col for col in df.columns.tolist() if col != var],
                key="ttest_by"
            )
            if st.button("Run T-Test", key="run_ttest"):
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
            if st.button("Run Chi-Square Test", key="run_chi2"):
                query = f"run chi-square test for {var1} and {var2}"
                return analysis_tool._run(query)
        return None
    
    elif function_name in ["histogram", "scatter"]:
        st.write(f"**{function_name.title()} Plot**")
        if function_name == "histogram":
            var = st.selectbox(
                "Variable to plot:",
                options=df.select_dtypes(include=[np.number]).columns.tolist(),
                key="histogram_var"
            )
            if st.button("Create Histogram", key="run_histogram"):
                query = f"create histogram for {var}"
                return analysis_tool._run(query)
        else:  # scatter
            var1 = st.selectbox(
                "X-axis variable:",
                options=df.select_dtypes(include=[np.number]).columns.tolist(),
                key="scatter_var1"
            )
            var2 = st.selectbox(
                "Y-axis variable:",
                options=df.select_dtypes(include=[np.number]).columns.tolist(),
                key="scatter_var2"
            )
            by = st.selectbox(
                "Grouping variable (optional):",
                options=["None"] + [col for col in df.columns.tolist() if col not in [var1, var2]],
                key="scatter_by"
            )
            if st.button("Create Scatter Plot", key="run_scatter"):
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
            with st.form("panel_data_form", clear_on_submit=False):
                panel_id = st.selectbox(
                    "Select the Panel ID Variable (e.g., country_name):",
                    options=list(st.session_state.df.columns),
                    key="panel_id_var"
                )
                time_var = st.text_input(
                    "Enter the name for the new Time Variable (e.g., year):",
                    value="year",
                    key="panel_time_var"
                )
                submit_panel = st.form_submit_button("Declare Panel Data")

            if submit_panel:
                df = st.session_state.df.copy()
                import re
                year_pattern = re.compile(r"^(19[7-9][0-9]|20[0-2][0-9]|2025)$")
                time_like_cols = [col for col in df.columns if year_pattern.match(str(col))]
                if not time_like_cols:
                    alt_time_pattern = re.compile(r"(\d{4}|Q[1-4]_\d{4}|\d{4}Q[1-4])")
                    time_like_cols = [col for col in df.columns if alt_time_pattern.search(str(col))]

                if time_like_cols:
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
                        st.session_state.panel_transform_message = "Data appears to already be in long format. Sorted for clarity. The main dataset has been updated."
                    else:
                        st.session_state.panel_transform_message = "Data appears to already be in long format."
                    
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
                st.dataframe(st.session_state.df.head(20), use_container_width=True)

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
                st.dataframe(st.session_state.df.head(20), use_container_width=True)
                
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
                st.dataframe(group_sizes.rename('count').to_frame(), use_container_width=True)
                st.dataframe(df.head(20), use_container_width=True)
                
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
            
            # Create columns for buttons (3 per row)
            cols = st.columns(3)
            for i, (button_text, function_name) in enumerate(toolbar_buttons.items()):
                col_idx = i % 3
                with cols[col_idx]:
                    if st.button(button_text, key=f"toolbar_{function_name}"):
                        # Execute the analysis function with user prompts
                        if "analysis_agent" in st.session_state:
                            try:
                                # Get the analysis tool from the agent
                                analysis_tool = st.session_state.analysis_agent.tools[0]
                                
                                # Prompt user for inputs and get result
                                result = prompt_user_for_inputs(function_name, st.session_state.df, analysis_tool)
                                
                                # Display result if available
                                if result is not None:
                                    # Check if result contains HTML image tag
                                    if "<img" in result:
                                        # Extract the text content before the image
                                        text_parts = result.split("<img")
                                        if text_parts[0].strip():
                                            st.write(text_parts[0].strip())
                                        
                                        # Extract and display the image
                                        img_start = result.find("<img")
                                        img_end = result.find(">", img_start) + 1
                                        img_html = result[img_start:img_end]
                                        
                                        # Display the image using st.markdown with unsafe_allow_html
                                        st.markdown(img_html, unsafe_allow_html=True)
                                        
                                        # Display any text after the image
                                        if len(text_parts) > 1:
                                            after_img = text_parts[1].split(">", 1)
                                            if len(after_img) > 1 and after_img[1].strip():
                                                st.write(after_img[1].strip())
                                    else:
                                        st.write(result)
                            except Exception as e:
                                st.error(f"Error executing {function_name}: {str(e)}")
                        else:
                            st.error("Analysis agent not initialized. Please check your API key.")

        # --- End Data Declaration Section ---

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
            st.dataframe(st.session_state.df.head(20), use_container_width=True)
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
                    st.write(st.session_state.df.dtypes)
                except:
                    st.text(str(st.session_state.df.dtypes))
            
            with col2:
                st.write("**Missing Values:**")
                try:
                    st.write(st.session_state.df.isnull().sum())
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
                        st.dataframe(summary_df, use_container_width=True)
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
                st.write(st.session_state.df.dtypes)
            except Exception as e:
                st.warning(f"Error displaying dtypes: {str(e)}")
            st.write("---")

            st.write("### 2. Descriptive Statistics (describe)")
            try:
                st.dataframe(st.session_state.df.describe(include='all'), use_container_width=True)
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
                    st.dataframe(st.session_state.df.loc[dupes, key_cols + [c for c in st.session_state.df.columns if c not in key_cols][:3]].head(20), use_container_width=True)
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
                    st.dataframe(minmax, use_container_width=True)
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
                        st.dataframe(st.session_state.df[col].value_counts(dropna=False).rename('count').to_frame(), use_container_width=True)
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