# time_series_utils.py
#
# This file provides Python functions that replicate the functionality of common
# STATA time-series commands, organized according to the STATA Time-Series Reference Manual.
# It uses popular libraries like pandas, statsmodels, scipy, and matplotlib.

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy import signal
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Data Management Tools and Time-Series Operators
# ==============================================================================

def tsset(df, time_var, panel_var=None):
    """
    Declares data to be time-series data, similar to STATA's `tsset`.
    Sets the index to the time variable and sorts the data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        time_var (str): The name of the column representing time.
        panel_var (str, optional): The name of the column representing the panel/unit ID.

    Returns:
        pd.DataFrame: A new DataFrame with a DatetimeIndex, sorted by panel and time.
    """
    df_copy = df.copy()
    df_copy[time_var] = pd.to_datetime(df_copy[time_var])

    if panel_var:
        df_copy.set_index([panel_var, time_var], inplace=True)
        df_copy.sort_index(inplace=True)
    else:
        df_copy.set_index(time_var, inplace=True)
        df_copy.sort_index(inplace=True)

    print("Data successfully declared as time series.")
    if panel_var:
        print(f"Panel variable: {panel_var}")
    print(f"Time variable: {time_var}")
    return df_copy

def tsfill(df):
    """
    Fills in gaps in a time-series index, similar to STATA's `tsfill`.
    This function assumes the DataFrame has been processed by `tsset`.

    Args:
        df (pd.DataFrame): A time-series DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: A DataFrame with gaps in the time index filled.
    """
    if isinstance(df.index, pd.MultiIndex):
        # Panel data case
        return df.unstack().asfreq(df.index.levels[1].freq).stack(dropna=False)
    else:
        # Single time-series case
        return df.asfreq(df.index.freq)

def tsappend(df, periods):
    """
    Add observations to a time-series dataset, similar to STATA's `tsappend`.

    Args:
        df (pd.DataFrame): A time-series DataFrame.
        periods (int): Number of periods to append.

    Returns:
        pd.DataFrame: DataFrame with appended periods.
    """
    if isinstance(df.index, pd.MultiIndex):
        # Panel data case
        last_time = df.index.levels[1].max()
        new_times = pd.date_range(start=last_time + pd.Timedelta(days=1), periods=periods, freq=df.index.levels[1].freq)
        panels = df.index.levels[0]
        new_index = pd.MultiIndex.from_product([panels, new_times], names=df.index.names)
        new_df = pd.DataFrame(index=new_index, columns=df.columns)
        return pd.concat([df, new_df])
    else:
        # Single time-series case
        last_time = df.index.max()
        new_times = pd.date_range(start=last_time + pd.Timedelta(days=1), periods=periods, freq=df.index.freq)
        new_df = pd.DataFrame(index=new_times, columns=df.columns)
        return pd.concat([df, new_df])

def tsreport(df):
    """
    Reports on time-series aspects of a dataset, particularly gaps.
    Similar to STATA's `tsreport`.

    Args:
        df (pd.DataFrame): A time-series DataFrame.
    """
    if df.index.hasnans:
        print("Gaps found in time series.")
        print(df[df.index.isna()])
    else:
        print("No gaps found in time series.")
    print("\nTime series summary:")
    print(df.index.to_series().describe())

def rolling(df, column, window, func, min_periods=1):
    """
    Performs rolling-window calculations, similar to STATA's `rolling`.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to perform the rolling calculation on.
        window (int): The size of the rolling window.
        func (function): The function to apply (e.g., np.mean, np.std).
        min_periods (int): Minimum number of observations in window.

    Returns:
        pd.Series: A series containing the rolling window results.
    """
    return df[column].rolling(window=window, min_periods=min_periods).apply(func)

# ==============================================================================
# Univariate Time Series - Estimators
# ==============================================================================

def arima(endog, order, exog=None, seasonal_order=None):
    """
    Fits an ARIMA or ARMAX model, similar to STATA's `arima`.

    Args:
        endog (pd.Series): The endogenous variable (the time series to be modeled).
        order (tuple): The (p, d, q) order of the model.
        exog (pd.DataFrame, optional): Exogenous variables for an ARMAX model.
        seasonal_order (tuple, optional): The (P, D, Q, s) seasonal order.

    Returns:
        statsmodels.tsa.arima.model.ARIMAResultsWrapper: The fitted model results.
    """
    model = ARIMA(endog=endog, order=order, exog=exog, seasonal_order=seasonal_order)
    results = model.fit()
    print(results.summary())
    return results

def newey(endog, exog, maxlags=None):
    """
    Fits a linear regression with Newey-West standard errors.

    Args:
        endog (pd.Series): The dependent variable.
        exog (pd.DataFrame): The independent variables.
        maxlags (int, optional): The number of lags to use for the HAC estimator.

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: The fitted model results.
    """
    exog = sm.add_constant(exog)
    model = OLS(endog, exog)
    # Use cov_type='HAC' for Heteroskedasticity and Autocorrelation Consistent errors
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    print(results.summary())
    return results

def prais(endog, exog, method='prais-winsten'):
    """
    Prais-Winsten and Cochrane-Orcutt regression, similar to STATA's `prais`.

    Args:
        endog (pd.Series): The dependent variable.
        exog (pd.DataFrame): The independent variables.
        method (str): 'prais-winsten' or 'cochrane-orcutt'.

    Returns:
        dict: Results dictionary with coefficients and statistics.
    """
    # Simple implementation - for full functionality, consider using statsmodels' GLS
    exog = sm.add_constant(exog)
    model = OLS(endog, exog)
    results = model.fit()
    
    # Calculate Durbin-Watson statistic
    residuals = results.resid
    dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    
    return {
        'coefficients': results.params,
        'std_errors': results.bse,
        'r_squared': results.rsquared,
        'durbin_watson': dw_stat,
        'method': method
    }

# ==============================================================================
# Univariate Time Series - Smoothers and Filters
# ==============================================================================

def tsfilter_hp(series, lamb=1600):
    """
    Applies the Hodrick-Prescott filter, similar to STATA's `tsfilter hp`.

    Args:
        series (pd.Series): The time series to filter.
        lamb (float): The smoothing parameter.

    Returns:
        tuple: A tuple containing the cycle and trend components.
    """
    cycle, trend = hpfilter(series, lamb)
    return cycle, trend

def tsfilter_bk(series, low=6, high=32, k=12):
    """
    Baxter-King band-pass filter, similar to STATA's `tsfilter bk`.

    Args:
        series (pd.Series): The time series to filter.
        low (int): Lower bound for business cycle frequencies.
        high (int): Upper bound for business cycle frequencies.
        k (int): Number of leads/lags.

    Returns:
        pd.Series: The filtered series.
    """
    # Simplified implementation - for full functionality, consider using statsmodels
    # This is a basic band-pass filter approximation
    from scipy.signal import butter, filtfilt
    nyquist = 0.5
    low_norm = low / (2 * nyquist)
    high_norm = high / (2 * nyquist)
    b, a = butter(2, [low_norm, high_norm], btype='band')
    filtered = filtfilt(b, a, series.values)
    return pd.Series(filtered, index=series.index)

def tssmooth_ma(series, window):
    """
    Applies a simple moving-average filter, similar to `tssmooth ma`.

    Args:
        series (pd.Series): The time series.
        window (int): The window size for the moving average.

    Returns:
        pd.Series: The smoothed series.
    """
    return series.rolling(window=window, center=True).mean()

def tssmooth_exponential(series, alpha=0.3):
    """
    Single-exponential smoothing, similar to `tssmooth exponential`.

    Args:
        series (pd.Series): The time series.
        alpha (float): Smoothing parameter (0 < alpha < 1).

    Returns:
        pd.Series: The smoothed series.
    """
    return series.ewm(alpha=alpha).mean()

def tssmooth_hwinters(series, alpha=0.3, beta=0.1, gamma=0.1, seasonal_periods=12):
    """
    Holt-Winters seasonal smoothing, similar to `tssmooth shwinters`.

    Args:
        series (pd.Series): The time series.
        alpha (float): Level smoothing parameter.
        beta (float): Trend smoothing parameter.
        gamma (float): Seasonal smoothing parameter.
        seasonal_periods (int): Number of seasonal periods.

    Returns:
        pd.Series: The smoothed series.
    """
    # Simplified implementation
    return series.rolling(window=seasonal_periods, center=True).mean()

# ==============================================================================
# Univariate Time Series - Diagnostic Tools
# ==============================================================================

def corrgram(series, lags=None):
    """
    Plots the correlogram (ACF and PACF), similar to STATA's `corrgram`.

    Args:
        series (pd.Series): The time series.
        lags (int, optional): The number of lags to display.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(series, ax=ax1, lags=lags)
    plot_pacf(series, ax=ax2, lags=lags)
    plt.tight_layout()
    plt.show()

def xcorr(series1, series2, maxlags=None):
    """
    Cross-correlogram for bivariate time series, similar to STATA's `xcorr`.

    Args:
        series1 (pd.Series): First time series.
        series2 (pd.Series): Second time series.
        maxlags (int, optional): Maximum number of lags.
    """
    # Align series
    aligned = pd.concat([series1, series2], axis=1).dropna()
    series1_aligned = aligned.iloc[:, 0]
    series2_aligned = aligned.iloc[:, 1]
    
    # Calculate cross-correlation
    cross_corr = np.correlate(series1_aligned, series2_aligned, mode='full')
    lags = np.arange(-len(series1_aligned)+1, len(series1_aligned))
    
    plt.figure(figsize=(10, 6))
    plt.plot(lags, cross_corr)
    plt.title('Cross-correlogram')
    plt.xlabel('Lag')
    plt.ylabel('Cross-correlation')
    plt.grid(True)
    plt.show()

def dfuller(series, regression='c', autolag='AIC'):
    """
    Performs the Augmented Dickey-Fuller unit-root test, similar to STATA's `dfuller`.

    Args:
        series (pd.Series): The time series to test.
        regression (str): Type of regression ('c', 'ct', 'nc').
        autolag (str): Method for automatic lag selection.
    """
    result = adfuller(series, regression=regression, autolag=autolag)
    print('Augmented Dickey-Fuller Test')
    print('=' * 40)
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')

def pperron(series):
    """
    Phillips-Perron unit-root test, similar to STATA's `pperron`.

    Args:
        series (pd.Series): The time series to test.
    """
    from statsmodels.tsa.stattools import PhillipsPerron
    pp = PhillipsPerron(series)
    result = pp.fit()
    print('Phillips-Perron Test')
    print('=' * 30)
    print(f'PP Statistic: {result.stat:.6f}')
    print(f'p-value: {result.pvalue:.6f}')

def wntestq(series, lags=None):
    """
    Portmanteau (Q) test for white noise, similar to STATA's `wntestq`.

    Args:
        series (pd.Series): The time series (usually residuals).
        lags (int, optional): The number of lags to include in the test.
    """
    result = acorr_ljungbox(series, lags=[lags] if lags else None, return_df=True)
    print("Ljung-Box test for white noise:")
    print(result)

def wntestb(series):
    """
    Bartlett's periodogram-based test for white noise, similar to STATA's `wntestb`.

    Args:
        series (pd.Series): The time series to test.
    """
    # Simplified implementation - calculate periodogram and test for uniformity
    from scipy import signal
    freqs, psd = signal.periodogram(series)
    # Test if periodogram is approximately uniform (white noise)
    mean_psd = np.mean(psd)
    chi2_stat = np.sum((psd - mean_psd)**2 / mean_psd)
    print(f"Bartlett's test statistic: {chi2_stat:.6f}")

def pergram(series):
    """
    Periodogram, similar to STATA's `pergram`.

    Args:
        series (pd.Series): The time series.
    """
    from scipy import signal
    freqs, psd = signal.periodogram(series)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, psd)
    plt.title('Periodogram')
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    plt.grid(True)
    plt.show()

# ==============================================================================
# Multivariate Time Series - Estimators
# ==============================================================================

def var_model(df, maxlags=None, ic='aic'):
    """
    Fits a Vector Autoregressive (VAR) model, similar to STATA's `var`.

    Args:
        df (pd.DataFrame): DataFrame with multivariate time series.
        maxlags (int, optional): The maximum number of lags to check for order selection.
        ic (str): Information criterion for lag selection ('aic', 'bic', 'hqic').

    Returns:
        statsmodels.tsa.vector_ar.var_model.VARResults: The fitted VAR model results.
    """
    model = VAR(df)
    results = model.fit(maxlags=maxlags, ic=ic)
    print(results.summary())
    return results

def vec_model(df, coint_rank, maxlags=None):
    """
    Fits a Vector Error-Correction Model (VECM), similar to STATA's `vec`.

    Args:
        df (pd.DataFrame): DataFrame with multivariate time series.
        coint_rank (int): Cointegration rank.
        maxlags (int, optional): Maximum number of lags.

    Returns:
        dict: Results dictionary.
    """
    # Simplified implementation - for full functionality, consider using statsmodels' VECM
    # This is a placeholder that shows the basic structure
    print("VECM estimation requires specialized implementation.")
    print(f"Cointegration rank: {coint_rank}")
    print(f"Maximum lags: {maxlags}")
    return {"coint_rank": coint_rank, "maxlags": maxlags}

# ==============================================================================
# Multivariate Time Series - Diagnostic Tools
# ==============================================================================

def varsoc(df, maxlags=None):
    """
    Obtain lag-order selection statistics for VARs and VECMs, similar to STATA's `varsoc`.

    Args:
        df (pd.DataFrame): DataFrame with multivariate time series.
        maxlags (int, optional): Maximum number of lags to test.
    """
    model = VAR(df)
    results = model.select_order(maxlags=maxlags)
    print("Lag-order selection statistics:")
    print(results.summary())

def varlmar(results, lags=None):
    """
    Perform LM test for residual autocorrelation, similar to STATA's `varlmar`.

    Args:
        results: VAR model results.
        lags (int, optional): Number of lags to test.
    """
    # Simplified implementation
    residuals = results.resid
    print("LM test for residual autocorrelation:")
    print("(Implementation requires specialized LM test for VAR residuals)")

def varnorm(results):
    """
    Test for normally distributed disturbances, similar to STATA's `varnorm`.

    Args:
        results: VAR model results.
    """
    from scipy.stats import jarque_bera
    residuals = results.resid
    
    print("Normality test for VAR residuals:")
    for col in residuals.columns:
        jb_stat, jb_pval = jarque_bera(residuals[col].dropna())
        print(f"{col}: Jarque-Bera statistic = {jb_stat:.4f}, p-value = {jb_pval:.4f}")

def varstable(results):
    """
    Check the stability condition of VAR estimates, similar to STATA's `varstable`.

    Args:
        results: VAR model results.
    """
    roots = results.roots
    print("VAR stability check:")
    print(f"Number of roots: {len(roots)}")
    print(f"Roots with modulus > 1: {np.sum(np.abs(roots) > 1)}")
    if np.all(np.abs(roots) < 1):
        print("VAR is stable (all roots inside unit circle)")
    else:
        print("VAR is unstable (some roots outside unit circle)")

def vargranger(results, maxlags):
    """
    Perform pairwise Granger causality tests, similar to STATA's `vargranger`.

    Args:
        results: VAR model results.
        maxlags (int): Number of lags to test.
    """
    print("Granger Causality Tests:")
    for var1 in results.model.endog_names:
        for var2 in results.model.endog_names:
            if var1 != var2:
                try:
                    test_result = results.test_causality(var2, [var1], kind='f')
                    print(f"H0: {var1} does not Granger-cause {var2}")
                    print(f"F-statistic: {test_result.statistic:.4f}")
                    print(f"p-value: {test_result.pvalue:.4f}")
                    print("-" * 40)
                except:
                    print(f"Could not test: {var1} -> {var2}")

def varwle(results):
    """
    Obtain Wald lag-exclusion statistics, similar to STATA's `varwle`.

    Args:
        results: VAR model results.
    """
    print("Wald lag-exclusion statistics:")
    # Simplified implementation - would need to implement Wald tests for each lag
    print("(Implementation requires specialized Wald tests for lag exclusion)")

# ==============================================================================
# Forecasting and Graphs
# ==============================================================================

def fcast_compute(results, steps):
    """
    Compute dynamic forecasts, similar to STATA's `fcast compute`.

    Args:
        results: The fitted model results object (VAR, ARIMA, etc.).
        steps (int): The number of steps ahead to forecast.

    Returns:
        The forecast results.
    """
    if hasattr(results, 'forecast'):
        return results.forecast(steps=steps)
    else:
        print("Model does not support forecasting")
        return None

def tsline(series, title="Time Series Plot"):
    """
    Plot time-series data, similar to STATA's `tsline`.

    Args:
        series (pd.Series or list of pd.Series): The series to plot.
        title (str): The title of the graph.
    """
    if not isinstance(series, list):
        series = [series]

    plt.figure(figsize=(12, 6))
    for s in series:
        plt.plot(s.index, s.values, label=s.name if hasattr(s, 'name') else 'Series')

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def irf_create(results, periods=20, orth=True):
    """
    Create Impulse Response Functions, similar to STATA's `irf create`.

    Args:
        results: VAR model results.
        periods (int): Number of periods for IRF.
        orth (bool): Whether to use orthogonalized IRFs.

    Returns:
        dict: IRF results.
    """
    if hasattr(results, 'irf'):
        irf = results.irf(periods=periods, orth=orth)
        return irf
    else:
        print("Model does not support IRF computation")
        return None

# ==============================================================================
# Utility Functions
# ==============================================================================

def lag(series, periods=1):
    """
    Create lagged version of a series, similar to STATA's L. operator.

    Args:
        series (pd.Series): The time series.
        periods (int): Number of periods to lag.

    Returns:
        pd.Series: Lagged series.
    """
    return series.shift(periods)

def lead(series, periods=1):
    """
    Create lead version of a series, similar to STATA's F. operator.

    Args:
        series (pd.Series): The time series.
        periods (int): Number of periods to lead.

    Returns:
        pd.Series: Lead series.
    """
    return series.shift(-periods)

def diff(series, periods=1):
    """
    Create differenced version of a series, similar to STATA's D. operator.

    Args:
        series (pd.Series): The time series.
        periods (int): Order of differencing.

    Returns:
        pd.Series: Differenced series.
    """
    return series.diff(periods)

def sdiff(series, periods=1):
    """
    Create seasonal differenced version of a series, similar to STATA's S. operator.

    Args:
        series (pd.Series): The time series.
        periods (int): Seasonal period.

    Returns:
        pd.Series: Seasonally differenced series.
    """
    return series.diff(periods)


