# analysis_agent.py
#
# This file implements a specialized LangChain agent for handling statistical analysis
# functions from panel_utils, time_series_utils, and cross_sectional_utils.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import AgentOutputParser
import re
import json
from typing import List, Union, Dict, Any, ClassVar, Optional
import io
import base64

# Import our utility modules
import panel_utils
import time_series_utils
import cross_sectional_utils

class AnalysisPromptTemplate(StringPromptTemplate):
    """Custom prompt template for the analysis agent."""
    
    template: ClassVar[str] = """You are a specialized statistical analysis agent. Your job is to help users perform statistical analysis on their data.

Available data types and their corresponding analysis functions:

PANEL DATA ANALYSIS:
- xtdescribe(panelvar, timevar): Describe panel structure
- xtsum(panelvar, timevar, varlist): Panel summary statistics  
- xttab(panelvar, timevar): Tabulate panel data
- xtline(panelvar, timevar, yvar): Panel line plots
- xtreg(depvar, exogvars, panelvar, timevar, fe=True): Panel regression
- xtlogit(depvar, exogvars, panelvar): Panel logit
- xtprobit(depvar, exogvars, panelvar): Panel probit
- xtpoisson(depvar, exogvars, panelvar): Panel Poisson
- xtunitroot(panelvar, timevar, var): Panel unit root tests

TIME SERIES ANALYSIS:
- tsset(timevar, panelvar): Declare time series
- tsfill(): Fill time gaps
- arima(endog, order): ARIMA models
- newey(endog, exog, maxlags): Newey-West regression
- dfuller(series): Unit root test
- corrgram(series, lags): Autocorrelations
- var_model(df, maxlags): VAR models
- vargranger(results, maxlags): Granger causality
- tsline(series): Time series plots

CROSS-SECTIONAL ANALYSIS:
- summarize(varlist, detail): Summary statistics
- tabulate(var1, var2): Frequency tables
- correlate(varlist): Correlation matrix
- regress(depvar, indepvars, robust): Linear regression
- logit(depvar, indepvars): Logistic regression
- probit(depvar, indepvars): Probit regression
- ttest(var, by): T-tests
- chi2(var1, var2): Chi-square test
- histogram(var): Histograms
- scatter(var1, var2, by): Scatter plots

Current data type: {data_type}
Available variables: {variables}

User request: {input}

Think step by step:
1. Determine what type of analysis the user wants
2. Identify the appropriate function(s) to use
3. Extract the required parameters from the user's request
4. Execute the analysis and return results

{agent_scratchpad}"""

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

class AnalysisTool(BaseTool):
    """Custom tool for executing statistical analysis functions."""
    
    name: str
    description: str
    data_type: str
    df: Optional[pd.DataFrame] = None
    panelvar: Optional[str] = None
    timevar: Optional[str] = None
    
    def _run(self, query: str) -> str:
        """Execute the analysis based on the query."""
        try:
            # Check if data is available
            if self.df is None:
                return "No data available. Please upload a dataset first."
            
            # Parse the query to extract function and parameters
            result = self._execute_analysis(query)
            return result
        except Exception as e:
            return f"Error executing analysis: {str(e)}"
    
    def _execute_analysis(self, query: str) -> str:
        """Execute the appropriate analysis function based on user request."""
        query_lower = query.lower()
        
        # Panel Data Analysis
        if self.data_type == "Panel Data":
            return self._execute_panel_analysis(query)
        
        # Time Series Analysis  
        elif self.data_type == "Time-Series":
            return self._execute_timeseries_analysis(query)
        
        # Cross-Sectional Analysis
        elif self.data_type == "Cross-Sectional":
            return self._execute_crosssectional_analysis(query)
        
        else:
            return "Please declare your data type first."
    
    def _execute_panel_analysis(self, query: str) -> str:
        """Execute panel data analysis functions based on user request."""
        query_lower = query.lower()
        
        # Use the stored panel and time variables
        panelvar = self.panelvar
        timevar = self.timevar
        
        if not panelvar or not timevar:
            return "Please specify panel and time variables first."
        
        # Function mapping based on user intent
        if any(word in query_lower for word in ['describe', 'structure', 'pattern', 'overview']):
            print(f"Executing: panel_utils.xtdescribe(df, '{panelvar}', '{timevar}')")
            result = panel_utils.xtdescribe(self.df, panelvar, timevar)
            return f"**Function executed:** `panel_utils.xtdescribe(df, '{panelvar}', '{timevar}')`\n\n**Result:**\n{result}"
        
        elif any(word in query_lower for word in ['summary', 'statistics', 'stats', 'summarize']):
            varlist = self._extract_variables(query)
            print(f"Executing: panel_utils.xtsum(df, '{panelvar}', '{timevar}', {varlist})")
            result = panel_utils.xtsum(self.df, panelvar, timevar, varlist)
            return f"**Function executed:** `panel_utils.xtsum(df, '{panelvar}', '{timevar}', {varlist})`\n\n**Result:**\n{result.to_string()}"
        
        elif any(word in query_lower for word in ['tabulate', 'table', 'frequency', 'count']):
            print(f"Executing: panel_utils.xttab(df, '{panelvar}', '{timevar}')")
            result = panel_utils.xttab(self.df, panelvar, timevar)
            return f"**Function executed:** `panel_utils.xttab(df, '{panelvar}', '{timevar}')`\n\n**Result:**\n{result.to_string()}"
        
        elif any(word in query_lower for word in ['plot', 'graph', 'line', 'visualize', 'chart']):
            yvar = self._extract_variables(query)[0] if self._extract_variables(query) else self.df.select_dtypes(include=[np.number]).columns[0]
            print(f"Executing: panel_utils.xtline(df, '{panelvar}', '{timevar}', '{yvar}')")
            fig = panel_utils.xtline(self.df, panelvar, timevar, yvar)
            return f"**Function executed:** `panel_utils.xtline(df, '{panelvar}', '{timevar}', '{yvar}')`\n\n**Result:**\n{self._fig_to_html(fig)}"
        
        elif any(word in query_lower for word in ['regression', 'regress', 'linear', 'model', 'xtreg']):
            depvar = self._extract_dependent_var(query)
            indepvars = self._extract_independent_vars(query)
            if depvar and indepvars:
                print(f"Executing: panel_utils.xtreg(df, '{depvar}', {indepvars}, '{panelvar}', '{timevar}')")
                result = panel_utils.xtreg(self.df, depvar, indepvars, panelvar, timevar)
                return f"**Function executed:** `panel_utils.xtreg(df, '{depvar}', {indepvars}, '{panelvar}', '{timevar}')`\n\n**Result:**\n{result}"
            else:
                return "Please specify dependent and independent variables for regression."
        
        elif any(word in query_lower for word in ['logit', 'logistic']):
            depvar = self._extract_dependent_var(query)
            indepvars = self._extract_independent_vars(query)
            if depvar and indepvars:
                print(f"Executing: panel_utils.xtlogit(df, '{depvar}', {indepvars}, '{panelvar}')")
                result = panel_utils.xtlogit(self.df, depvar, indepvars, panelvar)
                return f"**Function executed:** `panel_utils.xtlogit(df, '{depvar}', {indepvars}, '{panelvar}')`\n\n**Result:**\n{result}"
            else:
                return "Please specify dependent and independent variables for logit regression."
        
        elif any(word in query_lower for word in ['probit']):
            depvar = self._extract_dependent_var(query)
            indepvars = self._extract_independent_vars(query)
            if depvar and indepvars:
                print(f"Executing: panel_utils.xtprobit(df, '{depvar}', {indepvars}, '{panelvar}')")
                result = panel_utils.xtprobit(self.df, depvar, indepvars, panelvar)
                return f"**Function executed:** `panel_utils.xtprobit(df, '{depvar}', {indepvars}, '{panelvar}')`\n\n**Result:**\n{result}"
            else:
                return "Please specify dependent and independent variables for probit regression."
        
        elif any(word in query_lower for word in ['poisson', 'count']):
            depvar = self._extract_dependent_var(query)
            indepvars = self._extract_independent_vars(query)
            if depvar and indepvars:
                print(f"Executing: panel_utils.xtpoisson(df, '{depvar}', {indepvars}, '{panelvar}')")
                result = panel_utils.xtpoisson(self.df, depvar, indepvars, panelvar)
                return f"**Function executed:** `panel_utils.xtpoisson(df, '{depvar}', {indepvars}, '{panelvar}')`\n\n**Result:**\n{result}"
            else:
                return "Please specify dependent and independent variables for Poisson regression."
        
        elif any(word in query_lower for word in ['unit root', 'stationarity', 'xtunitroot']):
            var = self._extract_variables(query)[0] if self._extract_variables(query) else self.df.select_dtypes(include=[np.number]).columns[0]
            print(f"Executing: panel_utils.xtunitroot(df, '{panelvar}', '{timevar}', '{var}')")
            result = panel_utils.xtunitroot(self.df, panelvar, timevar, var)
            return f"**Function executed:** `panel_utils.xtunitroot(df, '{panelvar}', '{timevar}', '{var}')`\n\n**Result:**\n{result}"
        
        else:
            return "Available panel analysis functions: describe, summary, tabulate, plot, regression, logit, probit, poisson, unit root test. Please be more specific about what you'd like to analyze."
    
    def _execute_timeseries_analysis(self, query: str) -> str:
        """Execute time series analysis functions based on user request."""
        query_lower = query.lower()
        
        # Use the stored time variable
        timevar = self.timevar
        
        if not timevar:
            return "Please specify time variable first."
        
        # Function mapping based on user intent
        if any(word in query_lower for word in ['set', 'declare', 'tsset']):
            print(f"Executing: time_series_utils.tsset(df, '{timevar}')")
            result = time_series_utils.tsset(self.df, timevar)
            return f"**Function executed:** `time_series_utils.tsset(df, '{timevar}')`\n\n**Result:**\nTime series data declared with time variable: {timevar}"
        
        elif any(word in query_lower for word in ['arima', 'autoregressive', 'moving average']):
            var = self._extract_variables(query)[0] if self._extract_variables(query) else self.df.select_dtypes(include=[np.number]).columns[0]
            order = self._extract_arima_order(query) or (1, 1, 1)
            print(f"Executing: time_series_utils.arima(df['{var}'], {order})")
            result = time_series_utils.arima(self.df[var], order)
            return f"**Function executed:** `time_series_utils.arima(df['{var}'], {order})`\n\n**Result:**\n{result}"
        
        elif any(word in query_lower for word in ['unit root', 'stationarity', 'dfuller', 'adf']):
            var = self._extract_variables(query)[0] if self._extract_variables(query) else self.df.select_dtypes(include=[np.number]).columns[0]
            print(f"Executing: time_series_utils.dfuller(df['{var}'])")
            time_series_utils.dfuller(self.df[var])
            return f"**Function executed:** `time_series_utils.dfuller(df['{var}'])`\n\n**Result:**\nUnit root test completed. Check console output."
        
        elif any(word in query_lower for word in ['correlation', 'autocorrelation', 'corrgram', 'acf']):
            var = self._extract_variables(query)[0] if self._extract_variables(query) else self.df.select_dtypes(include=[np.number]).columns[0]
            print(f"Executing: time_series_utils.corrgram(df['{var}'])")
            time_series_utils.corrgram(self.df[var])
            return f"**Function executed:** `time_series_utils.corrgram(df['{var}'])`\n\n**Result:**\nCorrelogram generated. Check plot."
        
        elif any(word in query_lower for word in ['var', 'vector autoregression', 'multivariate']):
            # Select numeric variables for VAR
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                print(f"Executing: time_series_utils.var_model(df[{numeric_cols}])")
                result = time_series_utils.var_model(self.df[numeric_cols])
                return f"**Function executed:** `time_series_utils.var_model(df[{numeric_cols}])`\n\n**Result:**\n{result}"
            else:
                return "Need at least 2 numeric variables for VAR analysis."
        
        elif any(word in query_lower for word in ['granger', 'causality', 'causal']):
            return "Please run VAR model first, then use granger causality tests."
        
        elif any(word in query_lower for word in ['plot', 'graph', 'line', 'visualize', 'chart', 'tsline']):
            var = self._extract_variables(query)[0] if self._extract_variables(query) else self.df.select_dtypes(include=[np.number]).columns[0]
            print(f"Executing: time_series_utils.tsline(df['{var}'])")
            time_series_utils.tsline(self.df[var])
            return f"**Function executed:** `time_series_utils.tsline(df['{var}'])`\n\n**Result:**\nTime series plot generated. Check plot."
        
        elif any(word in query_lower for word in ['newey', 'west', 'hac', 'heteroskedasticity']):
            depvar = self._extract_dependent_var(query)
            indepvars = self._extract_independent_vars(query)
            if depvar and indepvars:
                print(f"Executing: time_series_utils.newey(df['{depvar}'], df[{indepvars}])")
                result = time_series_utils.newey(self.df[depvar], self.df[indepvars])
                return f"**Function executed:** `time_series_utils.newey(df['{depvar}'], df[{indepvars}])`\n\n**Result:**\n{result}"
            else:
                return "Please specify dependent and independent variables for Newey-West regression."
        
        elif any(word in query_lower for word in ['filter', 'smooth', 'trend', 'cycle']):
            var = self._extract_variables(query)[0] if self._extract_variables(query) else self.df.select_dtypes(include=[np.number]).columns[0]
            if 'hodrick' in query_lower or 'prescott' in query_lower:
                print(f"Executing: time_series_utils.tsfilter_hp(df['{var}'])")
                cycle, trend = time_series_utils.tsfilter_hp(self.df[var])
                return f"**Function executed:** `time_series_utils.tsfilter_hp(df['{var}'])`\n\n**Result:**\nHodrick-Prescott filter applied to {var}. Trend and cycle components extracted."
            else:
                # Default to moving average smoothing
                print(f"Executing: time_series_utils.tssmooth_ma(df['{var}'], window=5)")
                smoothed = time_series_utils.tssmooth_ma(self.df[var], window=5)
                return f"**Function executed:** `time_series_utils.tssmooth_ma(df['{var}'], window=5)`\n\n**Result:**\nMoving average smoothing applied to {var} with window=5."
        
        else:
            return "Available time series functions: declare, arima, unit root test, autocorrelation, VAR, granger causality, plot, Newey-West regression, filtering. Please be more specific about what you'd like to analyze."
    
    def _execute_crosssectional_analysis(self, query: str) -> str:
        """Execute cross-sectional analysis functions based on user request."""
        query_lower = query.lower()
        
        # Function mapping based on user intent
        if any(word in query_lower for word in ['summary', 'statistics', 'stats', 'summarize', 'describe']):
            varlist = self._extract_variables(query)
            print(f"Executing: cross_sectional_utils.summarize(df, {varlist})")
            result = cross_sectional_utils.summarize(self.df, varlist)
            return f"**Function executed:** `cross_sectional_utils.summarize(df, {varlist})`\n\n**Result:**\n{result.to_string()}"
        
        elif any(word in query_lower for word in ['table', 'tabulate', 'frequency', 'count', 'crosstab']):
            vars = self._extract_variables(query)
            if len(vars) >= 1:
                var1 = vars[0]
                var2 = vars[1] if len(vars) > 1 else None
                print(f"Executing: cross_sectional_utils.tabulate(df, '{var1}', {var2})")
                result = cross_sectional_utils.tabulate(self.df, var1, var2)
                return f"**Function executed:** `cross_sectional_utils.tabulate(df, '{var1}', {var2})`\n\n**Result:**\n{result.to_string()}"
            else:
                return "Please specify at least one variable for tabulation."
        
        elif any(word in query_lower for word in ['correlation', 'correlate', 'corr']):
            varlist = self._extract_variables(query)
            print(f"Executing: cross_sectional_utils.correlate(df, {varlist})")
            result = cross_sectional_utils.correlate(self.df, varlist)
            return f"**Function executed:** `cross_sectional_utils.correlate(df, {varlist})`\n\n**Result:**\n{result.to_string()}"
        
        elif any(word in query_lower for word in ['regression', 'regress', 'linear', 'model', 'ols']):
            depvar = self._extract_dependent_var(query)
            indepvars = self._extract_independent_vars(query)
            if depvar and indepvars:
                print(f"Executing: cross_sectional_utils.regress(df, '{depvar}', {indepvars})")
                result = cross_sectional_utils.regress(self.df, depvar, indepvars)
                return f"**Function executed:** `cross_sectional_utils.regress(df, '{depvar}', {indepvars})`\n\n**Result:**\n{result['summary']}"
            else:
                return "Please specify dependent and independent variables for regression."
        
        elif any(word in query_lower for word in ['logit', 'logistic', 'binary']):
            depvar = self._extract_dependent_var(query)
            indepvars = self._extract_independent_vars(query)
            if depvar and indepvars:
                print(f"Executing: cross_sectional_utils.logit(df, '{depvar}', {indepvars})")
                result = cross_sectional_utils.logit(self.df, depvar, indepvars)
                return f"**Function executed:** `cross_sectional_utils.logit(df, '{depvar}', {indepvars})`\n\n**Result:**\n{result['summary']}"
            else:
                return "Please specify dependent and independent variables for logit regression."
        
        elif any(word in query_lower for word in ['probit']):
            depvar = self._extract_dependent_var(query)
            indepvars = self._extract_independent_vars(query)
            if depvar and indepvars:
                print(f"Executing: cross_sectional_utils.probit(df, '{depvar}', {indepvars})")
                result = cross_sectional_utils.probit(self.df, depvar, indepvars)
                return f"**Function executed:** `cross_sectional_utils.probit(df, '{depvar}', {indepvars})`\n\n**Result:**\n{result['summary']}"
            else:
                return "Please specify dependent and independent variables for probit regression."
        
        elif any(word in query_lower for word in ['t-test', 'ttest', 't test', 'difference', 'compare']):
            var = self._extract_variables(query)[0] if self._extract_variables(query) else self.df.select_dtypes(include=[np.number]).columns[0]
            by = self._extract_grouping_var(query)
            print(f"Executing: cross_sectional_utils.ttest(df, '{var}', {by})")
            result = cross_sectional_utils.ttest(self.df, var, by)
            return f"**Function executed:** `cross_sectional_utils.ttest(df, '{var}', {by})`\n\n**Result:**\n{result}"
        
        elif any(word in query_lower for word in ['chi-square', 'chi2', 'chi square', 'independence']):
            vars = self._extract_variables(query)
            if len(vars) >= 2:
                print(f"Executing: cross_sectional_utils.chi2(df, '{vars[0]}', '{vars[1]}')")
                result = cross_sectional_utils.chi2(self.df, vars[0], vars[1])
                return f"**Function executed:** `cross_sectional_utils.chi2(df, '{vars[0]}', '{vars[1]}')`\n\n**Result:**\n{result}"
            else:
                return "Please specify two variables for chi-square test."
        
        elif any(word in query_lower for word in ['histogram', 'distribution', 'density']):
            var = self._extract_variables(query)[0] if self._extract_variables(query) else self.df.select_dtypes(include=[np.number]).columns[0]
            print(f"Executing: cross_sectional_utils.histogram(df, '{var}')")
            fig = cross_sectional_utils.histogram(self.df, var)
            return f"**Function executed:** `cross_sectional_utils.histogram(df, '{var}')`\n\n**Result:**\n{self._fig_to_html(fig)}"
        
        elif any(word in query_lower for word in ['scatter', 'plot', 'graph', 'relationship']):
            vars = self._extract_variables(query)
            if len(vars) >= 2:
                var1, var2 = vars[0], vars[1]
                by = self._extract_grouping_var(query)
                print(f"Executing: cross_sectional_utils.scatter(df, '{var1}', '{var2}', {by})")
                fig = cross_sectional_utils.scatter(self.df, var1, var2, by)
                return f"**Function executed:** `cross_sectional_utils.scatter(df, '{var1}', '{var2}', {by})`\n\n**Result:**\n{self._fig_to_html(fig)}"
            else:
                return "Please specify two variables for scatter plot."
        
        elif any(word in query_lower for word in ['boxplot', 'box plot', 'box']):
            var = self._extract_variables(query)[0] if self._extract_variables(query) else self.df.select_dtypes(include=[np.number]).columns[0]
            by = self._extract_grouping_var(query)
            print(f"Executing: cross_sectional_utils.boxplot(df, '{var}', {by})")
            fig = cross_sectional_utils.boxplot(self.df, var, by)
            return f"**Function executed:** `cross_sectional_utils.boxplot(df, '{var}', {by})`\n\n**Result:**\n{self._fig_to_html(fig)}"
        
        elif any(word in query_lower for word in ['bar', 'barplot', 'bar chart']):
            var = self._extract_variables(query)[0] if self._extract_variables(query) else self.df.select_dtypes(include=[np.number]).columns[0]
            by = self._extract_grouping_var(query)
            print(f"Executing: cross_sectional_utils.bar(df, '{var}', {by})")
            fig = cross_sectional_utils.bar(self.df, var, by)
            return f"**Function executed:** `cross_sectional_utils.bar(df, '{var}', {by})`\n\n**Result:**\n{self._fig_to_html(fig)}"
        
        elif any(word in query_lower for word in ['poisson', 'count', 'negative binomial']):
            depvar = self._extract_dependent_var(query)
            indepvars = self._extract_independent_vars(query)
            if depvar and indepvars:
                if 'negative' in query_lower:
                    print(f"Executing: cross_sectional_utils.nbreg(df, '{depvar}', {indepvars})")
                    result = cross_sectional_utils.nbreg(self.df, depvar, indepvars)
                    return f"**Function executed:** `cross_sectional_utils.nbreg(df, '{depvar}', {indepvars})`\n\n**Result:**\n{result['summary']}"
                else:
                    print(f"Executing: cross_sectional_utils.poisson(df, '{depvar}', {indepvars})")
                    result = cross_sectional_utils.poisson(self.df, depvar, indepvars)
                    return f"**Function executed:** `cross_sectional_utils.poisson(df, '{depvar}', {indepvars})`\n\n**Result:**\n{result['summary']}"
            else:
                return "Please specify dependent and independent variables for count regression."
        
        else:
            return "Available cross-sectional functions: summary, tabulate, correlation, regression, logit, probit, t-test, chi-square, histogram, scatter, boxplot, bar chart, poisson. Please be more specific about what you'd like to analyze."
    
    def _extract_variables(self, query: str) -> List[str]:
        """Extract variable names from query."""
        # Check if data is available
        if self.df is None:
            return []
        
        # Simple extraction - look for words that match column names
        variables = []
        for col in self.df.columns:
            if col.lower() in query.lower():
                variables.append(col)
        return variables
    
    def _extract_dependent_var(self, query: str) -> str:
        """Extract dependent variable from query."""
        # Check if data is available
        if self.df is None:
            return None
        
        # Look for patterns like "regress y on x" or "y ~ x"
        patterns = [
            r'regress\s+(\w+)\s+on',
            r'(\w+)\s*~\s*\w+',
            r'dependent.*?(\w+)',
            r'(\w+)\s*=\s*\w+'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                var = match.group(1)
                if var in self.df.columns:
                    return var
        
        # Fallback to first numeric column
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        return numeric_cols[0] if len(numeric_cols) > 0 else None
    
    def _extract_independent_vars(self, query: str) -> List[str]:
        """Extract independent variables from query."""
        # Check if data is available
        if self.df is None:
            return []
        
        # Extract all variables mentioned except the dependent variable
        depvar = self._extract_dependent_var(query)
        all_vars = self._extract_variables(query)
        
        if depvar and depvar in all_vars:
            all_vars.remove(depvar)
        
        return all_vars if all_vars else self.df.select_dtypes(include=[np.number]).columns.tolist()[1:3]
    
    def _extract_grouping_var(self, query: str) -> str:
        """Extract grouping variable from query."""
        # Check if data is available
        if self.df is None:
            return None
        
        # Look for patterns like "by group" or "grouped by"
        patterns = [
            r'by\s+(\w+)',
            r'grouped\s+by\s+(\w+)',
            r'(\w+)\s+group'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                var = match.group(1)
                if var in self.df.columns:
                    return var
        
        return None
    
    def _extract_arima_order(self, query: str) -> tuple:
        """Extract ARIMA order from query."""
        # Look for patterns like "ARIMA(1,1,1)" or "order 1 1 1"
        patterns = [
            r'ARIMA\((\d+),(\d+),(\d+)\)',
            r'order\s+(\d+)\s+(\d+)\s+(\d+)',
            r'(\d+)\s+(\d+)\s+(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
        return None
    
    def _fig_to_html(self, fig) -> str:
        """Convert matplotlib figure to HTML string."""
        # Save figure to bytes
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format='png', bbox_inches='tight')
        img_bytes.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(img_bytes.read()).decode()
        
        # Create HTML
        html = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; height:auto;">'
        
        # Close figure to free memory
        plt.close(fig)
        
        return html
    


class AnalysisOutputParser(AgentOutputParser):
    """Custom output parser for the analysis agent."""
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse the output of the LLM."""
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text,
            )
        
        # Parse action and input
        action_match = re.search(r"Action: (.*?)\n", text, re.DOTALL)
        input_match = re.search(r"Action Input: (.*?)(?:\n|$)", text, re.DOTALL)
        
        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            return AgentAction(tool=action, tool_input=action_input, log=text)
        
        return AgentFinish(
            return_values={"output": text},
            log=text,
        )

class AnalysisAgent:
    """Specialized agent for statistical analysis."""
    
    def __init__(self, llm, data_type="Cross-Sectional", df=None, **kwargs):
        self.llm = llm
        self.data_type = data_type
        self.df = df
        
        # Set additional parameters based on data type
        if data_type == "Panel Data":
            self.panelvar = kwargs.get('panelvar')
            self.timevar = kwargs.get('timevar')
        elif data_type == "Time-Series":
            self.timevar = kwargs.get('timevar')
        
        # Create tools
        self.tools = [
            AnalysisTool(
                name="statistical_analysis",
                description="Execute statistical analysis functions based on data type",
                data_type=data_type,
                df=df,
                panelvar=kwargs.get('panelvar') if data_type == "Panel Data" else None,
                timevar=kwargs.get('timevar') if data_type in ["Panel Data", "Time-Series"] else None
            )
        ]
        
        # Create prompt template
        self.prompt = AnalysisPromptTemplate(
            input_variables=["input", "agent_scratchpad", "data_type", "variables"]
        )
        
        # Create LLMChain
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )
        
        # Create output parser
        self.output_parser = AnalysisOutputParser()
        
        # Create agent
        self.agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=["statistical_analysis"]
        )
        
        # Create executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )
    
    def run(self, query: str) -> str:
        """Run the analysis agent."""
        # The agent executor handles prompt formatting internally
        # We just need to pass the query and let the agent handle the rest
        result = self.agent_executor.run(query)
        
        return result

def create_analysis_agent(api_key: str, model_name: str = "gpt-4o", **kwargs) -> AnalysisAgent:
    """Create an analysis agent with the specified configuration."""
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=api_key,
        model_name=model_name
    )
    
    return AnalysisAgent(llm=llm, **kwargs) 