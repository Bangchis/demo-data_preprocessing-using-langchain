import streamlit as st
import pandas as pd
import numpy as np
from .ui import get_numeric_columns, handle_numeric_detection_fallback

def prompt_user_for_inputs(function_name, df, analysis_tool):
    """Prompt user for required inputs based on the function being called."""
    
    # Get numeric columns
    numeric_cols = get_numeric_columns(df)
    
    # Debug info (only show if no numeric columns found)
    if len(numeric_cols) == 0:
        numeric_cols = handle_numeric_detection_fallback(df)
        if not numeric_cols:
            return None
    
    # Route to appropriate input form
    if function_name in ["xtsum"]:
        return panel_summary_form(function_name, df, analysis_tool, numeric_cols)
    elif function_name in ["xtline"]:
        return panel_line_plot_form(function_name, df, analysis_tool, numeric_cols)
    elif function_name in ["xtreg", "xtlogit", "xtprobit", "xtpoisson"]:
        return panel_regression_form(function_name, df, analysis_tool, numeric_cols)
    elif function_name in ["xtunitroot"]:
        return panel_unit_root_form(function_name, df, analysis_tool, numeric_cols)
    elif function_name in ["arima"]:
        return time_series_arima_form(function_name, df, analysis_tool, numeric_cols)
    elif function_name in ["dfuller", "corrgram", "tsline"]:
        return time_series_single_var_form(function_name, df, analysis_tool, numeric_cols)
    elif function_name in ["newey", "var_model"]:
        return time_series_multi_var_form(function_name, df, analysis_tool, numeric_cols)
    elif function_name in ["summarize"]:
        return cross_sectional_summary_form(function_name, df, analysis_tool, numeric_cols)
    elif function_name in ["tabulate"]:
        return cross_sectional_tabulate_form(function_name, df, analysis_tool, numeric_cols)
    elif function_name in ["correlate"]:
        return cross_sectional_correlate_form(function_name, df, analysis_tool, numeric_cols)
    elif function_name in ["regress", "logit", "probit"]:
        return cross_sectional_regression_form(function_name, df, analysis_tool, numeric_cols)
    elif function_name in ["ttest", "chi2"]:
        return cross_sectional_test_form(function_name, df, analysis_tool, numeric_cols)
    elif function_name in ["histogram", "scatter"]:
        return cross_sectional_plot_form(function_name, df, analysis_tool, numeric_cols)
    else:
        return simple_function_form(function_name, analysis_tool)

def panel_summary_form(function_name, df, analysis_tool, numeric_cols):
    """Panel summary statistics form."""
    with st.form(f"form_{function_name}"):
        st.write("**Panel Summary Statistics**")
        st.write("Select variables to analyze:")
        varlist = st.multiselect(
            "Variables:",
            options=numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )
        submitted = st.form_submit_button("Run Analysis")
        if submitted:
            if varlist:
                query = f"show summary statistics for {', '.join(varlist)}"
                return analysis_tool._run(query)
            else:
                st.error("Please select at least one variable.")
    return None

def panel_line_plot_form(function_name, df, analysis_tool, numeric_cols):
    """Panel line plot form."""
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

def panel_regression_form(function_name, df, analysis_tool, numeric_cols):
    """Panel regression forms (xtreg, xtlogit, xtprobit, xtpoisson)."""
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
        
        available_indepvars = [col for col in numeric_cols if col != depvar]
        indepvars = st.multiselect(
            "Independent variables:",
            options=available_indepvars,
            default=available_indepvars[:1] if available_indepvars else [],
            key=f"{function_name}_indepvars"
        )
        
        submitted = st.form_submit_button("Run Regression")
        if submitted:
            if indepvars and len(indepvars) > 0:
                query = f"run panel {function_name.replace('xt', '')} regression with {depvar} as dependent and {', '.join(indepvars)} as independent"
                return analysis_tool._run(query)
            else:
                st.error("Please select at least one independent variable.")
    return None

def panel_unit_root_form(function_name, df, analysis_tool, numeric_cols):
    """Panel unit root test form."""
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

def time_series_arima_form(function_name, df, analysis_tool, numeric_cols):
    """ARIMA model form."""
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

def time_series_single_var_form(function_name, df, analysis_tool, numeric_cols):
    """Single variable time series forms (dfuller, corrgram, tsline)."""
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

def time_series_multi_var_form(function_name, df, analysis_tool, numeric_cols):
    """Multi-variable time series forms (newey, var_model)."""
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
                default=[col for col in numeric_cols if col != depvar][:1] if [col for col in numeric_cols if col != depvar] else [],
                key="newey_indepvars"
            )
            submitted = st.form_submit_button("Run Newey-West")
            if submitted:
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

def cross_sectional_summary_form(function_name, df, analysis_tool, numeric_cols):
    """Cross-sectional summary statistics form."""
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

def cross_sectional_tabulate_form(function_name, df, analysis_tool, numeric_cols):
    """Cross-sectional frequency table form."""
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

def cross_sectional_correlate_form(function_name, df, analysis_tool, numeric_cols):
    """Cross-sectional correlation form."""
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

def cross_sectional_regression_form(function_name, df, analysis_tool, numeric_cols):
    """Cross-sectional regression forms (regress, logit, probit)."""
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
            default=[col for col in numeric_cols if col != depvar][:1] if [col for col in numeric_cols if col != depvar] else [],
            key=f"{function_name}_indepvars"
        )
        submitted = st.form_submit_button(f"Run {function_name.title()}")
        if submitted:
            if indepvars and len(indepvars) > 0:
                query = f"run {function_name} regression with {depvar} as dependent and {', '.join(indepvars)} as independent"
                return analysis_tool._run(query)
            else:
                st.error("Please select at least one independent variable.")
    return None

def cross_sectional_test_form(function_name, df, analysis_tool, numeric_cols):
    """Cross-sectional test forms (ttest, chi2)."""
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

def cross_sectional_plot_form(function_name, df, analysis_tool, numeric_cols):
    """Cross-sectional plot forms (histogram, scatter)."""
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

def simple_function_form(function_name, analysis_tool):
    """Simple functions that don't need additional inputs."""
    query_mapping = {
        "xtdescribe": "describe panel structure",
        "xttab": "tabulate panel data",
        "tsset": "declare time series",
        "vargranger": "run Granger causality test"
    }
    query = query_mapping.get(function_name, function_name)
    return analysis_tool._run(query)