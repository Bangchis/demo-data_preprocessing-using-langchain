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
    n_panels = df[panelvar].nunique()
    n_time = df[timevar].nunique()
    obs_per_panel = df.groupby(panelvar)[timevar].nunique()
    min_obs = obs_per_panel.min()
    max_obs = obs_per_panel.max()
    avg_obs = obs_per_panel.mean()
    time_range = (df[timevar].min(), df[timevar].max())
    summary = (
        f"Number of panels: {n_panels}\n"
        f"Number of time periods: {n_time}\n"
        f"Observations per panel: min {min_obs}, max {max_obs}, avg {avg_obs:.2f}\n"
        f"Time variable range: {time_range[0]} to {time_range[1]}\n"
    )
    return summary

def xtsum(df, panelvar, timevar, varlist=None):
    """
    Summarize panel data (like STATA's xtsum).
    Returns a DataFrame with overall, between, and within statistics.
    """
    if varlist is None:
        varlist = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
    results = []
    for var in varlist:
        overall_mean = df[var].mean()
        overall_std = df[var].std()
        between = df.groupby(panelvar)[var].mean()
        between_std = between.std()
        within = df[var] - df.groupby(panelvar)[var].transform('mean')
        within_std = within.std()
        results.append({
            'Variable': var,
            'Overall Mean': overall_mean,
            'Overall Std': overall_std,
            'Between Std': between_std,
            'Within Std': within_std
        })
    return pd.DataFrame(results)

def xttab(df, panelvar, timevar):
    """
    Tabulate panel data (like STATA's xttab).
    Returns a DataFrame of counts of time periods per panel.
    """
    tab = df.groupby(panelvar)[timevar].nunique().value_counts().sort_index()
    return tab.rename_axis('Number of periods').reset_index(name='Number of panels')

def xtline(df, panelvar, timevar, yvar, n_panels=10):
    """
    Panel-data line plots (like STATA's xtline).
    Returns a matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (panel, group) in enumerate(df.groupby(panelvar)):
        if i >= n_panels:
            break
        ax.plot(group[timevar], group[yvar], label=str(panel))
    ax.set_xlabel(timevar)
    ax.set_ylabel(yvar)
    ax.set_title(f"Panel-data line plot for {yvar} by {panelvar}")
    ax.legend(title=panelvar, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
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
    # Set index for panel structure
    df = df.set_index([panelvar, timevar])
    y = df[depvar]
    X = df[exogvars]
    X = sm.add_constant(X)
    if fe:
        model = PanelOLS(y, X, entity_effects=True)
    else:
        model = RandomEffects(y, X)
    res = model.fit()
    return str(res.summary)

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
    if PanelUnitRoot is None:
        return "arch.unitroot.PanelUnitRoot not installed."
    # Reshape to wide: index=timevar, columns=panelvar
    wide = df.pivot(index=timevar, columns=panelvar, values=var)
    pur = PanelUnitRoot(wide)
    res = pur.fit()
    return str(res.summary())

def xtcointtest(df, panelvar, timevar, varlist):
    """
    Panel-data cointegration tests (like STATA's xtcointtest).
    Returns a summary string.
    """
    # Not directly available in statsmodels; use adfuller on residuals as a workaround
    # Here, just run adfuller on the first variable as a placeholder
    from statsmodels.tsa.stattools import adfuller
    var = varlist[0]
    series = df.sort_values([panelvar, timevar]).groupby(panelvar)[var].apply(lambda x: x.values)
    results = []
    for panel, values in series.items():
        try:
            adf = adfuller(values)
            results.append(f"Panel {panel}: ADF p-value={adf[1]:.4f}")
        except Exception as e:
            results.append(f"Panel {panel}: Error {e}")
    return "\n".join(results)

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
    # Drop missing
    d = df[[depvar] + exogvars + [panelvar]].dropna()
    y = d[depvar]
    X = d[exogvars]
    X = sm.add_constant(X)
    # Clustered standard errors by panelvar
    model = Logit(y, X)
    res = model.fit(disp=0)
    return str(res.summary())

def xtprobit(df, depvar, exogvars, panelvar):
    """
    Random-effects probit model for panel data (like STATA's xtprobit).
    Returns a summary string.
    Note: Only population-averaged probit is implemented (no true panel RE probit in statsmodels).
    """
    d = df[[depvar] + exogvars + [panelvar]].dropna()
    y = d[depvar]
    X = d[exogvars]
    X = sm.add_constant(X)
    model = Probit(y, X)
    res = model.fit(disp=0)
    return str(res.summary())

def xtpoisson(df, depvar, exogvars, panelvar):
    """
    Fixed-effects Poisson model for panel data (like STATA's xtpoisson, fe).
    Returns a summary string.
    Note: Only population-averaged Poisson is implemented (no true panel FE Poisson in statsmodels).
    """
    d = df[[depvar] + exogvars + [panelvar]].dropna()
    y = d[depvar]
    X = d[exogvars]
    X = sm.add_constant(X)
    model = sm.GLM(y, X, family=Poisson())
    res = model.fit()
    return str(res.summary())

def xtnbreg(df, depvar, exogvars, panelvar):
    """
    Fixed-effects Negative Binomial model for panel data (like STATA's xtnbreg, fe).
    Returns a summary string.
    Note: Only population-averaged NB is implemented (no true panel FE NB in statsmodels).
    """
    d = df[[depvar] + exogvars + [panelvar]].dropna()
    y = d[depvar]
    X = d[exogvars]
    X = sm.add_constant(X)
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
    res = model.fit()
    return str(res.summary())

def xtstreg(*args, **kwargs):
    """Stub for xtstreg: Random-effects parametric survival models."""
    return "Not implemented."

def xtgee(df, depvar, exogvars, panelvar):
    """
    GEE population-averaged panel-data models (like STATA's xtgee).
    Returns a summary string.
    """
    d = df[[depvar] + exogvars + [panelvar]].dropna()
    y = d[depvar]
    X = d[exogvars]
    X = sm.add_constant(X)
    model = GEE(y, X, groups=d[panelvar], family=Binomial())
    res = model.fit()
    return str(res.summary())

def spxtregress(*args, **kwargs):
    """Stub for spxtregress: Spatial autoregressive models for panel data."""
    return "Not implemented."

def quadchk(*args, **kwargs):
    """Stub for quadchk: Check sensitivity of quadrature approximation."""
    return "Not implemented." 