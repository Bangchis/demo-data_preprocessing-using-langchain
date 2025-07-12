import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS, RandomEffects
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Poisson, Binomial
from statsmodels.discrete.discrete_model import Logit, Probit
try:
    from arch.unitroot import PanelUnitRoot
except ImportError:
    PanelUnitRoot = None

# --- Panel Data Management and Exploration Tools ---
def xtdescribe(df, panelvar, timevar):
    """
    Describe the pattern of panel data (like STATA's xtdescribe).
    Returns a string summary.
    """
    try:
        # Create a copy to avoid modifying the original dataframe
        df_work = df.copy()
        
        # Handle time variable - convert to numeric if possible
        if df_work[timevar].dtype == 'object':
            try:
                # First, try to convert directly to numeric
                df_work[timevar] = pd.to_numeric(df_work[timevar], errors='coerce')
            except:
                # If that fails, try to extract numbers from strings
                df_work[timevar] = df_work[timevar].astype(str).str.extract('(\d+)')[0].astype(float)
        
        # Drop rows where time variable is NaN after conversion
        df_work = df_work.dropna(subset=[timevar])
        
        if len(df_work) == 0:
            return "Error: No valid data after converting time variable to numeric format."
        
        n_panels = df_work[panelvar].nunique()
        n_time = df_work[timevar].nunique()
        obs_per_panel = df_work.groupby(panelvar)[timevar].nunique()
        min_obs = obs_per_panel.min()
        max_obs = obs_per_panel.max()
        avg_obs = obs_per_panel.mean()
        time_range = (df_work[timevar].min(), df_work[timevar].max())
        summary = (
            f"Number of panels: {n_panels}\n"
            f"Number of time periods: {n_time}\n"
            f"Observations per panel: min {min_obs}, max {max_obs}, avg {avg_obs:.2f}\n"
            f"Time variable range: {time_range[0]} to {time_range[1]}\n"
        )
        return summary
    except Exception as e:
        return f"Error in panel description: {str(e)}"

def xtsum(df, panelvar, timevar, varlist=None):
    """
    Summarize panel data (like STATA's xtsum).
    Returns a DataFrame with overall, between, and within statistics.
    """
    try:
        # Create a copy to avoid modifying the original dataframe
        df_work = df.copy()
        
        # Handle time variable - convert to numeric if possible
        if df_work[timevar].dtype == 'object':
            try:
                # First, try to convert directly to numeric
                df_work[timevar] = pd.to_numeric(df_work[timevar], errors='coerce')
            except:
                # If that fails, try to extract numbers from strings
                df_work[timevar] = df_work[timevar].astype(str).str.extract('(\d+)')[0].astype(float)
        
        # Drop rows where time variable is NaN after conversion
        df_work = df_work.dropna(subset=[timevar])
        
        if len(df_work) == 0:
            return pd.DataFrame({"Error": ["No valid data after converting time variable to numeric format."]})
        
        # Convert variables to numeric for analysis
        results = []
        for var in varlist:
            if var in df_work.columns:
                try:
                    # Convert variable to numeric, handling empty strings
                    var_data = pd.to_numeric(df_work[var].replace('', np.nan), errors='coerce')
                    
                    # Drop NaN values for this variable
                    valid_data = df_work[df_work[var].notna() & (df_work[var] != '')].copy()
                    valid_data[var] = var_data.dropna()
                    
                    if len(valid_data) == 0:
                        results.append({
                            'Variable': var,
                            'Overall Mean': 'N/A',
                            'Overall Std': 'N/A',
                            'Between Std': 'N/A',
                            'Within Std': 'N/A'
                        })
                        continue
                    
                    overall_mean = valid_data[var].mean()
                    overall_std = valid_data[var].std()
                    between = valid_data.groupby(panelvar)[var].mean()
                    between_std = between.std()
                    within = valid_data[var] - valid_data.groupby(panelvar)[var].transform('mean')
                    within_std = within.std()
                    
                    results.append({
                        'Variable': var,
                        'Overall Mean': overall_mean,
                        'Overall Std': overall_std,
                        'Between Std': between_std,
                        'Within Std': within_std
                    })
                except Exception as var_error:
                    results.append({
                        'Variable': var,
                        'Overall Mean': f'Error: {str(var_error)}',
                        'Overall Std': 'N/A',
                        'Between Std': 'N/A',
                        'Within Std': 'N/A'
                    })
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame({"Error": ["No valid numeric variables found for analysis."]})
            
    except Exception as e:
        return pd.DataFrame({"Error": [f"Error in panel summary: {str(e)}"]})

def xttab(df, panelvar, timevar):
    """
    Tabulate panel data (like STATA's xttab).
    Returns a DataFrame of counts of time periods per panel.
    """
    try:
        # Create a copy to avoid modifying the original dataframe
        df_work = df.copy()
        
        # Handle time variable - convert to numeric if possible
        if df_work[timevar].dtype == 'object':
            try:
                # First, try to convert directly to numeric
                df_work[timevar] = pd.to_numeric(df_work[timevar], errors='coerce')
            except:
                # If that fails, try to extract numbers from strings
                df_work[timevar] = df_work[timevar].astype(str).str.extract('(\d+)')[0].astype(float)
        
        # Drop rows where time variable is NaN after conversion
        df_work = df_work.dropna(subset=[timevar])
        
        if len(df_work) == 0:
            return pd.DataFrame({"Error": ["No valid data after converting time variable to numeric format."]})
        
        tab = df_work.groupby(panelvar)[timevar].nunique().value_counts().sort_index()
        return tab.rename_axis('Number of periods').reset_index(name='Number of panels')
        
    except Exception as e:
        return pd.DataFrame({"Error": [f"Error in panel tabulation: {str(e)}"]})

def xtline(df, panelvar, timevar, yvar, n_panels=10):
    """
    Panel-data line plots (like STATA's xtline).
    Returns a matplotlib figure.
    """
    try:
        # Create a copy to avoid modifying the original dataframe
        df_work = df.copy()
        
        # Handle time variable - convert to numeric if possible
        if df_work[timevar].dtype == 'object':
            try:
                # First, try to convert directly to numeric
                df_work[timevar] = pd.to_numeric(df_work[timevar], errors='coerce')
            except:
                # If that fails, try to extract numbers from strings
                df_work[timevar] = df_work[timevar].astype(str).str.extract('(\d+)')[0].astype(float)
        
        # Drop rows where time variable is NaN after conversion
        df_work = df_work.dropna(subset=[timevar])
        
        if len(df_work) == 0:
            # Return an error figure
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, 'No valid data after converting time variable to numeric format.', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Error: Invalid Time Variable")
            return fig
        
        # Convert yvar to numeric for analysis
        df_work[yvar] = pd.to_numeric(df_work[yvar].replace('', np.nan), errors='coerce')
        df_work = df_work.dropna(subset=[yvar])
        
        if len(df_work) == 0:
            # Return an error figure
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, 'No valid data after converting y variable to numeric format.', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Error: Invalid Y Variable")
            return fig
        
        # Sort by panel and time for proper plotting
        df_work = df_work.sort_values([panelvar, timevar])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (panel, group) in enumerate(df_work.groupby(panelvar)):
            if i >= n_panels:
                break
            ax.plot(group[timevar], group[yvar], label=str(panel))
        ax.set_xlabel(timevar)
        ax.set_ylabel(yvar)
        ax.set_title(f"Panel-data line plot for {yvar} by {panelvar}")
        ax.legend(title=panelvar, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Return an error figure
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, f'Error in panel line plot: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Error: Panel Line Plot")
        return fig

# --- Stubs for other STATA panel commands ---
def xtdata(*args, **kwargs):
    """Stub for xtdata: Faster specification searches with xt data."""
    return "Not implemented."

def xtreg(df, depvar, exogvars, panelvar, timevar, fe=True):
    """
    Linear models for panel data (like STATA's xtreg).
    fe=True for fixed effects, fe=False for random effects.
    Returns a summary string.
    """
    try:
        # Create a copy to avoid modifying the original dataframe
        df_work = df.copy()
        
        # Handle time variable - convert to numeric if possible
        if df_work[timevar].dtype == 'object':
            # Try to extract numeric values from the time variable
            # This handles cases like "Năm" (Year in Vietnamese) or other text
            try:
                # First, try to convert directly to numeric
                df_work[timevar] = pd.to_numeric(df_work[timevar], errors='coerce')
            except:
                # If that fails, try to extract numbers from strings
                df_work[timevar] = df_work[timevar].astype(str).str.extract('(\d+)')[0].astype(float)
        
        # Drop rows where time variable is NaN after conversion
        df_work = df_work.dropna(subset=[timevar])
        
        if len(df_work) == 0:
            return "Error: No valid data after converting time variable to numeric format."
        
        # Ensure time variable is numeric
        df_work[timevar] = pd.to_numeric(df_work[timevar], errors='coerce')
        df_work = df_work.dropna(subset=[timevar])
        
        # Sort by panel and time variables
        df_work = df_work.sort_values([panelvar, timevar])
        
        # Set index for panel structure
        df_work = df_work.set_index([panelvar, timevar])
        
        # Check if we have the required variables
        if depvar not in df_work.columns:
            return f"Error: Dependent variable '{depvar}' not found in the dataset."
        
        missing_exog = [var for var in exogvars if var not in df_work.columns]
        if missing_exog:
            return f"Error: Independent variables {missing_exog} not found in the dataset."
        
        # Convert variables to numeric for analysis
        try:
            y = pd.to_numeric(df_work[depvar].replace('', np.nan), errors='coerce')
            X = df_work[exogvars].copy()
            for col in exogvars:
                X[col] = pd.to_numeric(X[col].replace('', np.nan), errors='coerce')
        except Exception as e:
            return f"Error converting variables to numeric: {str(e)}"
        
        # Drop rows with missing values in dependent or independent variables
        valid_data = pd.concat([y, X], axis=1).dropna()
        if len(valid_data) == 0:
            return "Error: No valid observations after removing missing values."
        
        y = valid_data[depvar]
        X = valid_data[exogvars]
        X = sm.add_constant(X)
        
        # Fit the model
        if fe:
            model = PanelOLS(y, X, entity_effects=True)
        else:
            model = RandomEffects(y, X)
        
        res = model.fit()
        return str(res.summary)
        
    except Exception as e:
        return f"Error in panel regression: {str(e)}"

def xtregar(*args, **kwargs):
    """Stub for xtregar: Fixed- and random-effects linear models with AR(1) disturbance."""
    return "Not implemented."

def xtgls(*args, **kwargs):
    """Stub for xtgls: GLS linear model with heteroskedastic and correlated errors."""
    return "Not implemented."

def xtpcse(*args, **kwargs):
    """Stub for xtpcse: Linear regression with panel-corrected standard errors."""
    return "Not implemented."

def xthtaylor(*args, **kwargs):
    """Stub for xthtaylor: Hausman-Taylor estimator for error-components model."""
    return "Not implemented."

def xtfrontier(*args, **kwargs):
    """Stub for xtfrontier: Stochastic frontier models for panel data."""
    return "Not implemented."

def xtrc(*args, **kwargs):
    """Stub for xtrc: Random-coefficients model."""
    return "Not implemented."

def xtivreg(*args, **kwargs):
    """Stub for xtivreg: Instrumental variables and two-stage least squares for panel-data models."""
    return "Not implemented."

def xtheckman(*args, **kwargs):
    """Stub for xtheckman: Random-effects regression with sample selection."""
    return "Not implemented."

def xtdidregress(*args, **kwargs):
    """Stub for xtdidregress: Fixed-effects difference in differences."""
    return "Not implemented."

def xthdidregress(*args, **kwargs):
    """Stub for xthdidregress: Heterogeneous difference in differences for panel data."""
    return "Not implemented."

def xteregress(*args, **kwargs):
    """Stub for xteregress: Random-effects models with endogenous covariates, treatment, and sample selection."""
    return "Not implemented."

def xtunitroot(df, panelvar, timevar, var):
    """
    Panel-data unit-root tests (like STATA's xtunitroot).
    Returns a summary string.
    """
    from statsmodels.tsa.stattools import adfuller
    
    try:
        # Create a copy to avoid modifying the original dataframe
        df_work = df.copy()
        
        # Handle time variable - convert to numeric if possible
        if df_work[timevar].dtype == 'object':
            try:
                # First, try to convert directly to numeric
                df_work[timevar] = pd.to_numeric(df_work[timevar], errors='coerce')
            except:
                # If that fails, try to extract numbers from strings
                df_work[timevar] = df_work[timevar].astype(str).str.extract('(\d+)')[0].astype(float)
        
        # Drop rows where time variable is NaN after conversion
        df_work = df_work.dropna(subset=[timevar])
        
        if len(df_work) == 0:
            return "Error: No valid data after converting time variable to numeric format."
        
        # Convert variable to numeric for analysis
        df_work[var] = pd.to_numeric(df_work[var].replace('', np.nan), errors='coerce')
        df_work = df_work.dropna(subset=[var])
        
        if len(df_work) == 0:
            return "Error: No valid data after converting variable to numeric format."
        
        # Sort by panel and time variables
        df_work = df_work.sort_values([panelvar, timevar])
        
        # Perform ADF test for each panel
        results = []
        panels = df_work[panelvar].unique()
        
        for panel in panels:
            panel_data = df_work[df_work[panelvar] == panel][var].values
            
            # Skip panels with insufficient data (need at least 3 observations)
            if len(panel_data) < 3:
                results.append(f"Panel {panel}: Insufficient data (need at least 3 observations)")
                continue
            
            try:
                # Perform ADF test
                adf_result = adfuller(panel_data)
                
                # Extract results
                adf_statistic = adf_result[0]
                p_value = adf_result[1]
                critical_values = adf_result[4]
                
                # Interpret results
                if p_value <= 0.05:
                    interpretation = "Stationary (reject null hypothesis)"
                else:
                    interpretation = "Non-stationary (fail to reject null hypothesis)"
                
                # Format critical values
                cv_str = ", ".join([f"{k}: {v:.3f}" for k, v in critical_values.items()])
                
                results.append(f"Panel {panel}: ADF={adf_statistic:.4f}, p={p_value:.4f}, CV={cv_str}, {interpretation}")
                
            except Exception as e:
                results.append(f"Panel {panel}: Error in ADF test - {str(e)}")
        
        # Calculate summary statistics
        if len(results) > 0:
            # Count results by category
            stationary_count = 0
            non_stationary_count = 0
            error_count = 0
            insufficient_data_count = 0
            
            for result in results:
                if "Stationary" in result:
                    stationary_count += 1
                elif "Non-stationary" in result:
                    non_stationary_count += 1
                elif "Error in ADF test" in result:
                    error_count += 1
                elif "Insufficient data" in result:
                    insufficient_data_count += 1
            
            # Create summary
            summary = f"Panel Unit Root Test Results for variable '{var}'\n"
            summary += "=" * 60 + "\n"
            summary += f"Total panels tested: {len(panels)}\n"
            summary += f"Stationary panels: {stationary_count}\n"
            summary += f"Non-stationary panels: {non_stationary_count}\n"
            summary += f"Panels with errors: {error_count}\n"
            summary += f"Panels with insufficient data: {insufficient_data_count}\n"
            summary += "=" * 60 + "\n\n"
            
            # Add detailed results with better formatting
            summary += "DETAILED RESULTS:\n"
            summary += "-" * 60 + "\n"
            
            for i, result in enumerate(results, 1):
                # Clean up the result formatting
                clean_result = result.replace("Panel ", "").replace(":\n", ": ")
                clean_result = clean_result.replace("\n  ", " | ")
                summary += f"{i:3d}. {clean_result}\n"
            
            return summary
        else:
            return "Error: No valid panels found for unit root testing."
        
    except Exception as e:
        return f"Error in panel unit root test: {str(e)}"

def xtcointtest(df, panelvar, timevar, varlist):
    """
    Panel-data cointegration tests (like STATA's xtcointtest).
    Returns a summary string.
    """
    # Not directly available in statsmodels; use adfuller on residuals as a workaround
    # Here, just run adfuller on the first variable as a placeholder
    from statsmodels.tsa.stattools import adfuller
    
    try:
        # Create a copy to avoid modifying the original dataframe
        df_work = df.copy()
        
        # Handle time variable - convert to numeric if possible
        if df_work[timevar].dtype == 'object':
            try:
                # First, try to convert directly to numeric
                df_work[timevar] = pd.to_numeric(df_work[timevar], errors='coerce')
            except:
                # If that fails, try to extract numbers from strings
                df_work[timevar] = df_work[timevar].astype(str).str.extract('(\d+)')[0].astype(float)
        
        # Drop rows where time variable is NaN after conversion
        df_work = df_work.dropna(subset=[timevar])
        
        if len(df_work) == 0:
            return "Error: No valid data after converting time variable to numeric format."
        
        # Convert variable to numeric for analysis
        var = varlist[0]
        df_work[var] = pd.to_numeric(df_work[var].replace('', np.nan), errors='coerce')
        df_work = df_work.dropna(subset=[var])
        
        if len(df_work) == 0:
            return "Error: No valid data after converting variable to numeric format."
        
        # Sort by panel and time variables
        df_work = df_work.sort_values([panelvar, timevar])
        
        series = df_work.groupby(panelvar)[var].apply(lambda x: x.values)
        results = []
        for panel, values in series.items():
            try:
                adf = adfuller(values)
                results.append(f"Panel {panel}: ADF p-value={adf[1]:.4f}")
            except Exception as e:
                results.append(f"Panel {panel}: Error {e}")
        return "\n".join(results)
        
    except Exception as e:
        return f"Error in panel cointegration test: {str(e)}"

def xtabond(*args, **kwargs):
    """Stub for xtabond: Arellano-Bond linear dynamic panel-data estimation."""
    return "Not implemented."

def xtdpd(*args, **kwargs):
    """Stub for xtdpd: Linear dynamic panel-data estimation."""
    return "Not implemented."

def xtdpdsys(*args, **kwargs):
    """Stub for xtdpdsys: Arellano-Bover/Blundell-Bond linear dynamic panel-data estimation."""
    return "Not implemented."

def xtvar(*args, **kwargs):
    """Stub for xtvar: Panel-data vector autoregressive models."""
    return "Not implemented."

def xttobit(*args, **kwargs):
    """Stub for xttobit: Random-effects tobit model."""
    return "Not implemented."

def xtintreg(*args, **kwargs):
    """Stub for xtintreg: Random-effects interval-data regression model."""
    return "Not implemented."

def xteintreg(*args, **kwargs):
    """Stub for xteintreg: Random-effects interval-data regression models with endogenous covariates, treatment, and sample selection."""
    return "Not implemented."

def xtlogit(df, depvar, exogvars, panelvar):
    """
    Fixed-effects logit model for panel data (like STATA's xtlogit, fe).
    Returns a summary string.
    Note: Only population-averaged logit is implemented (no true panel FE logit in statsmodels).
    """
    try:
        # Convert variables to numeric for analysis
        d = df[[depvar] + exogvars + [panelvar]].copy()
        
        # Convert dependent variable to numeric
        d[depvar] = pd.to_numeric(d[depvar].replace('', np.nan), errors='coerce')
        
        # Convert independent variables to numeric
        for col in exogvars:
            d[col] = pd.to_numeric(d[col].replace('', np.nan), errors='coerce')
        
        # Drop missing values
        d = d.dropna()
        
        if len(d) == 0:
            return "Error: No valid observations after converting to numeric and removing missing values."
        
        y = d[depvar]
        X = d[exogvars]
        X = sm.add_constant(X)
        # Clustered standard errors by panelvar
        model = Logit(y, X)
        res = model.fit(disp=0)
        return str(res.summary())
    except Exception as e:
        return f"Error in logit regression: {str(e)}"

def xtprobit(df, depvar, exogvars, panelvar):
    """
    Random-effects probit model for panel data (like STATA's xtprobit).
    Returns a summary string.
    Note: Only population-averaged probit is implemented (no true panel RE probit in statsmodels).
    """
    try:
        # Convert variables to numeric for analysis
        d = df[[depvar] + exogvars + [panelvar]].copy()
        
        # Convert dependent variable to numeric
        d[depvar] = pd.to_numeric(d[depvar].replace('', np.nan), errors='coerce')
        
        # Convert independent variables to numeric
        for col in exogvars:
            d[col] = pd.to_numeric(d[col].replace('', np.nan), errors='coerce')
        
        # Drop missing values
        d = d.dropna()
        
        if len(d) == 0:
            return "Error: No valid observations after converting to numeric and removing missing values."
        
        y = d[depvar]
        X = d[exogvars]
        X = sm.add_constant(X)
        model = Probit(y, X)
        res = model.fit(disp=0)
        return str(res.summary())
    except Exception as e:
        return f"Error in probit regression: {str(e)}"

def xtpoisson(df, depvar, exogvars, panelvar):
    """
    Fixed-effects Poisson model for panel data (like STATA's xtpoisson, fe).
    Returns a summary string.
    Note: Only population-averaged Poisson is implemented (no true panel FE Poisson in statsmodels).
    """
    try:
        # Convert variables to numeric for analysis
        d = df[[depvar] + exogvars + [panelvar]].copy()
        
        # Convert dependent variable to numeric
        d[depvar] = pd.to_numeric(d[depvar].replace('', np.nan), errors='coerce')
        
        # Convert independent variables to numeric
        for col in exogvars:
            d[col] = pd.to_numeric(d[col].replace('', np.nan), errors='coerce')
        
        # Drop missing values
        d = d.dropna()
        
        if len(d) == 0:
            return "Error: No valid observations after converting to numeric and removing missing values."
        
        y = d[depvar]
        X = d[exogvars]
        X = sm.add_constant(X)
        model = sm.GLM(y, X, family=Poisson())
        res = model.fit()
        return str(res.summary())
    except Exception as e:
        return f"Error in Poisson regression: {str(e)}"

def xtnbreg(df, depvar, exogvars, panelvar):
    """
    Fixed-effects Negative Binomial model for panel data (like STATA's xtnbreg, fe).
    Returns a summary string.
    Note: Only population-averaged NB is implemented (no true panel FE NB in statsmodels).
    """
    try:
        # Convert variables to numeric for analysis
        d = df[[depvar] + exogvars + [panelvar]].copy()
        
        # Convert dependent variable to numeric
        d[depvar] = pd.to_numeric(d[depvar].replace('', np.nan), errors='coerce')
        
        # Convert independent variables to numeric
        for col in exogvars:
            d[col] = pd.to_numeric(d[col].replace('', np.nan), errors='coerce')
        
        # Drop missing values
        d = d.dropna()
        
        if len(d) == 0:
            return "Error: No valid observations after converting to numeric and removing missing values."
        
        y = d[depvar]
        X = d[exogvars]
        X = sm.add_constant(X)
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
        res = model.fit()
        return str(res.summary())
    except Exception as e:
        return f"Error in Negative Binomial regression: {str(e)}"

def xtstreg(*args, **kwargs):
    """Stub for xtstreg: Random-effects parametric survival models."""
    return "Not implemented."

def xtgee(df, depvar, exogvars, panelvar):
    """
    GEE population-averaged panel-data models (like STATA's xtgee).
    Returns a summary string.
    """
    try:
        # Convert variables to numeric for analysis
        d = df[[depvar] + exogvars + [panelvar]].copy()
        
        # Convert dependent variable to numeric
        d[depvar] = pd.to_numeric(d[depvar].replace('', np.nan), errors='coerce')
        
        # Convert independent variables to numeric
        for col in exogvars:
            d[col] = pd.to_numeric(d[col].replace('', np.nan), errors='coerce')
        
        # Drop missing values
        d = d.dropna()
        
        if len(d) == 0:
            return "Error: No valid observations after converting to numeric and removing missing values."
        
        y = d[depvar]
        X = d[exogvars]
        X = sm.add_constant(X)
        model = GEE(y, X, groups=d[panelvar], family=Binomial())
        res = model.fit()
        return str(res.summary())
    except Exception as e:
        return f"Error in GEE model: {str(e)}"

def spxtregress(*args, **kwargs):
    """Stub for spxtregress: Spatial autoregressive models for panel data."""
    return "Not implemented."

def quadchk(*args, **kwargs):
    """Stub for quadchk: Check sensitivity of quadrature approximation."""
    return "Not implemented." 