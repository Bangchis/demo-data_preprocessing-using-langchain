import streamlit as st
import pandas as pd
import numpy as np
from src.core.utils import clean_dataframe_for_display

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

def handle_numeric_detection_fallback(df):
    """Handle fallback detection for numeric columns."""
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
        return fallback_numeric_cols
    else:
        st.error("No numeric columns could be detected even with fallback method.")
        return []

def render_analysis_result(result):
    """Render analysis result with proper formatting."""
    if result is None:
        return
        
    # Check if result contains HTML image tag
    if "<img" in result:
        text_parts = result.split("<img")
        if text_parts[0].strip():
            st.write(text_parts[0].strip())
        img_start = result.find("<img")
        img_end = result.find(">", img_start) + 1
        img_html = result[img_start:img_end]
        st.markdown(img_html, unsafe_allow_html=True)
        if len(text_parts) > 1:
            after_img = text_parts[1].split(">", 1)
            if len(after_img) > 1 and after_img[1].strip():
                st.write(after_img[1].strip())
    else:
        # Check if the result is a DataFrame and clean it if needed
        if isinstance(result, pd.DataFrame):
            clean_result = clean_dataframe_for_display(result)
            st.dataframe(clean_result, use_container_width=True)
        else:
            st.write(result)

def render_analysis_toolbar(declaration, toolbar_buttons):
    """Render the analysis toolbar with buttons."""
    st.subheader("🔧 Analysis Toolbar")
    st.write(f"Available {declaration} analysis functions:")
    
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
            render_analysis_result(st.session_state.analysis_result)
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

def render_analysis_console():
    """Render the analysis console section."""
    st.subheader("🖥️ Analysis Console")
    if st.button("Clear Console"):
        st.session_state.console_log = []
    for entry in reversed(st.session_state.console_log):
        st.code(entry["call"], language="python")
        if isinstance(entry["result"], pd.DataFrame):
            clean_result = clean_dataframe_for_display(entry["result"])
            st.dataframe(clean_result, use_container_width=True)
        else:
            st.write(entry["result"])
        st.markdown("---")