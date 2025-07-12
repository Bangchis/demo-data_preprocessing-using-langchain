# cross_sectional_utils.py
#
# This file provides Python functions that replicate the functionality of common
# STATA cross-sectional analysis commands. It uses popular libraries like pandas,
# statsmodels, scipy, and matplotlib.

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Descriptive Statistics
# ==============================================================================

def summarize(df, varlist=None, detail=False):
    """
    Generate summary statistics, similar to STATA's summarize.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        varlist (list): List of variables to summarize
        detail (bool): If True, include detailed statistics
        
    Returns:
        pd.DataFrame: Summary statistics
    """
    if varlist is None:
        varlist = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if detail:
        return df[varlist].describe(include='all')
    else:
        summary = df[varlist].agg(['count', 'mean', 'std', 'min', 'max']).T
        summary.columns = ['Obs', 'Mean', 'Std. Dev.', 'Min', 'Max']
        return summary

def tabulate(df, var1, var2=None, row=False, col=False, cell=False):
    """
    Create frequency tables, similar to STATA's tabulate.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        var1 (str): First variable
        var2 (str): Second variable (optional)
        row (bool): Show row percentages
        col (bool): Show column percentages
        cell (bool): Show cell percentages
        
    Returns:
        pd.DataFrame: Frequency table
    """
    if var2 is None:
        # One-way table
        freq = df[var1].value_counts().sort_index()
        total = freq.sum()
        pct = (freq / total * 100).round(2)
        
        result = pd.DataFrame({
            'Freq.': freq,
            'Percent': pct,
            'Cum.': pct.cumsum()
        })
        result.loc['Total'] = [total, 100.0, 100.0]
        return result
    else:
        # Two-way table
        crosstab = pd.crosstab(df[var1], df[var2], margins=True)
        if row:
            crosstab_pct = pd.crosstab(df[var1], df[var2], normalize='index', margins=True) * 100
            return crosstab_pct.round(2)
        elif col:
            crosstab_pct = pd.crosstab(df[var1], df[var2], normalize='columns', margins=True) * 100
            return crosstab_pct.round(2)
        elif cell:
            crosstab_pct = pd.crosstab(df[var1], df[var2], normalize='all', margins=True) * 100
            return crosstab_pct.round(2)
        else:
            return crosstab

def correlate(df, varlist=None, method='pearson'):
    """
    Calculate correlation matrix, similar to STATA's correlate.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        varlist (list): List of variables
        method (str): Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    if varlist is None:
        varlist = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return df[varlist].corr(method=method)

# ==============================================================================
# Regression Analysis
# ==============================================================================

def regress(df, depvar, indepvars, robust=False, cluster=None):
    """
    Linear regression, similar to STATA's regress.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        depvar (str): Dependent variable
        indepvars (list): Independent variables
        robust (bool): Use robust standard errors
        cluster (str): Clustering variable
        
    Returns:
        dict: Regression results
    """
    # Prepare data
    y = df[depvar]
    X = df[indepvars]
    X = sm.add_constant(X)
    
    # Fit model
    if robust:
        model = sm.OLS(y, X)
        results = model.fit(cov_type='HC1')
    elif cluster:
        model = sm.OLS(y, X)
        results = model.fit(cov_type='cluster', cov_kwds={'groups': df[cluster]})
    else:
        model = sm.OLS(y, X)
        results = model.fit()
    
    return {
        'model': results,
        'summary': str(results.summary()),
        'coefficients': results.params,
        'std_errors': results.bse,
        't_values': results.tvalues,
        'p_values': results.pvalues,
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'f_statistic': results.fvalue,
        'f_pvalue': results.f_pvalue,
        'aic': results.aic,
        'bic': results.bic
    }

def logit(df, depvar, indepvars, robust=False):
    """
    Logistic regression, similar to STATA's logit.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        depvar (str): Dependent variable (binary)
        indepvars (list): Independent variables
        robust (bool): Use robust standard errors
        
    Returns:
        dict: Logistic regression results
    """
    from statsmodels.discrete.discrete_model import Logit
    
    # Prepare data
    y = df[depvar]
    X = df[indepvars]
    X = sm.add_constant(X)
    
    # Fit model
    model = Logit(y, X)
    if robust:
        results = model.fit_regularized(disp=0)
    else:
        results = model.fit(disp=0)
    
    return {
        'model': results,
        'summary': str(results.summary()),
        'coefficients': results.params,
        'std_errors': results.bse,
        'z_values': results.tvalues,
        'p_values': results.pvalues,
        'pseudo_r_squared': results.prsquared,
        'aic': results.aic,
        'bic': results.bic
    }

def probit(df, depvar, indepvars, robust=False):
    """
    Probit regression, similar to STATA's probit.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        depvar (str): Dependent variable (binary)
        indepvars (list): Independent variables
        robust (bool): Use robust standard errors
        
    Returns:
        dict: Probit regression results
    """
    from statsmodels.discrete.discrete_model import Probit
    
    # Prepare data
    y = df[depvar]
    X = df[indepvars]
    X = sm.add_constant(X)
    
    # Fit model
    model = Probit(y, X)
    if robust:
        results = model.fit_regularized(disp=0)
    else:
        results = model.fit(disp=0)
    
    return {
        'model': results,
        'summary': str(results.summary()),
        'coefficients': results.params,
        'std_errors': results.bse,
        'z_values': results.tvalues,
        'p_values': results.pvalues,
        'pseudo_r_squared': results.prsquared,
        'aic': results.aic,
        'bic': results.bic
    }

def poisson(df, depvar, indepvars, robust=False):
    """
    Poisson regression, similar to STATA's poisson.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        depvar (str): Dependent variable (count)
        indepvars (list): Independent variables
        robust (bool): Use robust standard errors
        
    Returns:
        dict: Poisson regression results
    """
    from statsmodels.discrete.discrete_model import Poisson
    
    # Prepare data
    y = df[depvar]
    X = df[indepvars]
    X = sm.add_constant(X)
    
    # Fit model
    model = Poisson(y, X)
    if robust:
        results = model.fit_regularized(disp=0)
    else:
        results = model.fit(disp=0)
    
    return {
        'model': results,
        'summary': str(results.summary()),
        'coefficients': results.params,
        'std_errors': results.bse,
        'z_values': results.tvalues,
        'p_values': results.pvalues,
        'pseudo_r_squared': results.prsquared,
        'aic': results.aic,
        'bic': results.bic
    }

def nbreg(df, depvar, indepvars, robust=False):
    """
    Negative Binomial regression, similar to STATA's nbreg.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        depvar (str): Dependent variable (count)
        indepvars (list): Independent variables
        robust (bool): Use robust standard errors
        
    Returns:
        dict: Negative Binomial regression results
    """
    from statsmodels.discrete.discrete_model import NegativeBinomialP
    
    # Prepare data
    y = df[depvar]
    X = df[indepvars]
    X = sm.add_constant(X)
    
    # Fit model
    model = NegativeBinomialP(y, X)
    if robust:
        results = model.fit_regularized(disp=0)
    else:
        results = model.fit(disp=0)
    
    return {
        'model': results,
        'summary': str(results.summary()),
        'coefficients': results.params,
        'std_errors': results.bse,
        'z_values': results.tvalues,
        'p_values': results.pvalues,
        'pseudo_r_squared': results.prsquared,
        'aic': results.aic,
        'bic': results.bic
    }

# ==============================================================================
# Hypothesis Testing
# ==============================================================================

def ttest(df, var, by=None, paired=False):
    """
    T-test, similar to STATA's ttest.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        var (str): Variable to test
        by (str): Grouping variable
        paired (bool): Paired t-test
        
    Returns:
        dict: T-test results
    """
    if by is None:
        # One-sample t-test against zero
        stat, pval = stats.ttest_1samp(df[var].dropna(), 0)
        return {
            'test_type': 'One-sample t-test',
            'statistic': stat,
            'p_value': pval,
            'mean': df[var].mean(),
            'std': df[var].std(),
            'n': len(df[var].dropna())
        }
    else:
        # Two-sample t-test
        groups = df.groupby(by)[var]
        group1 = groups.get_group(groups.groups[list(groups.groups.keys())[0]])
        group2 = groups.get_group(groups.groups[list(groups.groups.keys())[1]])
        
        if paired:
            stat, pval = stats.ttest_rel(group1, group2)
            test_type = 'Paired t-test'
        else:
            stat, pval = stats.ttest_ind(group1, group2)
            test_type = 'Two-sample t-test'
        
        return {
            'test_type': test_type,
            'statistic': stat,
            'p_value': pval,
            'group1_mean': group1.mean(),
            'group2_mean': group2.mean(),
            'group1_std': group1.std(),
            'group2_std': group2.std(),
            'n1': len(group1),
            'n2': len(group2)
        }

def ranksum(df, var, by):
    """
    Wilcoxon rank-sum test, similar to STATA's ranksum.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        var (str): Variable to test
        by (str): Grouping variable
        
    Returns:
        dict: Rank-sum test results
    """
    groups = df.groupby(by)[var]
    group1 = groups.get_group(groups.groups[list(groups.groups.keys())[0]])
    group2 = groups.get_group(groups.groups[list(groups.groups.keys())[1]])
    
    stat, pval = stats.ranksums(group1, group2)
    
    return {
        'test_type': 'Wilcoxon rank-sum test',
        'statistic': stat,
        'p_value': pval,
        'group1_median': group1.median(),
        'group2_median': group2.median(),
        'n1': len(group1),
        'n2': len(group2)
    }

def kwallis(df, var, by):
    """
    Kruskal-Wallis test, similar to STATA's kwallis.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        var (str): Variable to test
        by (str): Grouping variable
        
    Returns:
        dict: Kruskal-Wallis test results
    """
    groups = [group for name, group in df.groupby(by)[var]]
    stat, pval = stats.kruskal(*groups)
    
    return {
        'test_type': 'Kruskal-Wallis test',
        'statistic': stat,
        'p_value': pval,
        'n_groups': len(groups),
        'total_n': len(df[var].dropna())
    }

def chi2(df, var1, var2):
    """
    Chi-square test, similar to STATA's tabulate with chi2 option.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        var1 (str): First variable
        var2 (str): Second variable
        
    Returns:
        dict: Chi-square test results
    """
    contingency = pd.crosstab(df[var1], df[var2])
    chi2_stat, pval, dof, expected = stats.chi2_contingency(contingency)
    
    return {
        'test_type': 'Chi-square test',
        'statistic': chi2_stat,
        'p_value': pval,
        'degrees_of_freedom': dof,
        'expected_frequencies': expected
    }

# ==============================================================================
# Model Diagnostics
# ==============================================================================

def vif(df, indepvars):
    """
    Variance Inflation Factor, similar to STATA's estat vif.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        indepvars (list): Independent variables
        
    Returns:
        pd.DataFrame: VIF values
    """
    X = df[indepvars]
    X = sm.add_constant(X)
    
    vif_data = []
    for i in range(1, len(X.columns)):
        vif_val = variance_inflation_factor(X.values, i)
        vif_data.append({'Variable': X.columns[i], 'VIF': vif_val})
    
    return pd.DataFrame(vif_data)

def hettest(results, method='breusch-pagan'):
    """
    Heteroskedasticity test, similar to STATA's estat hettest.
    
    Args:
        results: Regression results object
        method (str): Test method ('breusch-pagan', 'white')
        
    Returns:
        dict: Heteroskedasticity test results
    """
    if method == 'breusch-pagan':
        stat, pval, f_stat, f_pval = het_breuschpagan(results.resid, results.model.exog)
        test_name = 'Breusch-Pagan test'
    elif method == 'white':
        stat, pval, f_stat, f_pval = het_white(results.resid, results.model.exog)
        test_name = 'White test'
    
    return {
        'test_type': test_name,
        'statistic': stat,
        'p_value': pval,
        'f_statistic': f_stat,
        'f_p_value': f_pval
    }

def ovtest(results):
    """
    Ramsey RESET test for omitted variables, similar to STATA's estat ovtest.
    
    Args:
        results: Regression results object
        
    Returns:
        dict: Omitted variables test results
    """
    from statsmodels.stats.diagnostic import reset_ramsey
    
    stat, pval, f_stat, f_pval = reset_ramsey(results, degree=3)
    
    return {
        'test_type': 'Ramsey RESET test',
        'statistic': stat,
        'p_value': pval,
        'f_statistic': f_stat,
        'f_p_value': f_pval
    }

# ==============================================================================
# Data Visualization
# ==============================================================================

def histogram(df, var, bins=30, density=False):
    """
    Create histogram, similar to STATA's histogram.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        var (str): Variable to plot
        bins (int): Number of bins
        density (bool): Plot density instead of counts
        
    Returns:
        matplotlib.figure.Figure: Histogram plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[var].dropna(), bins=bins, density=density, alpha=0.7, edgecolor='black')
    ax.set_xlabel(var)
    ax.set_ylabel('Density' if density else 'Frequency')
    ax.set_title(f'Histogram of {var}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def scatter(df, var1, var2, by=None):
    """
    Create scatter plot, similar to STATA's scatter.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        var1 (str): X-axis variable
        var2 (str): Y-axis variable
        by (str): Grouping variable (optional)
        
    Returns:
        matplotlib.figure.Figure: Scatter plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if by:
        for name, group in df.groupby(by):
            ax.scatter(group[var1], group[var2], label=name, alpha=0.7)
        ax.legend()
    else:
        ax.scatter(df[var1], df[var2], alpha=0.7)
    
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    ax.set_title(f'Scatter plot: {var1} vs {var2}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def boxplot(df, var, by=None):
    """
    Create box plot, similar to STATA's graph box.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        var (str): Variable to plot
        by (str): Grouping variable (optional)
        
    Returns:
        matplotlib.figure.Figure: Box plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if by:
        df.boxplot(column=var, by=by, ax=ax)
        ax.set_title(f'Box plot of {var} by {by}')
    else:
        df.boxplot(column=var, ax=ax)
        ax.set_title(f'Box plot of {var}')
    
    ax.set_ylabel(var)
    plt.tight_layout()
    return fig

def bar(df, var, by=None, stat='count'):
    """
    Create bar plot, similar to STATA's graph bar.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        var (str): Variable to plot
        by (str): Grouping variable (optional)
        stat (str): Statistic to plot ('count', 'mean', 'sum')
        
    Returns:
        matplotlib.figure.Figure: Bar plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if by:
        if stat == 'count':
            data = df.groupby(by)[var].count()
        elif stat == 'mean':
            data = df.groupby(by)[var].mean()
        elif stat == 'sum':
            data = df.groupby(by)[var].sum()
        
        data.plot(kind='bar', ax=ax)
        ax.set_title(f'{stat.title()} of {var} by {by}')
    else:
        if stat == 'count':
            data = df[var].value_counts()
        else:
            data = df[var].agg(stat)
        
        data.plot(kind='bar', ax=ax)
        ax.set_title(f'{stat.title()} of {var}')
    
    ax.set_ylabel(stat.title())
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# ==============================================================================
# Utility Functions
# ==============================================================================

def generate_sample_data(n=1000):
    """
    Generate sample cross-sectional data for testing.
    
    Args:
        n (int): Number of observations
        
    Returns:
        pd.DataFrame: Sample dataset
    """
    np.random.seed(42)
    
    # Generate variables
    age = np.random.normal(45, 15, n)
    income = 50000 + 1000 * age + np.random.normal(0, 10000, n)
    education = np.random.choice(['High School', 'College', 'Graduate'], n, p=[0.3, 0.5, 0.2])
    gender = np.random.choice(['Male', 'Female'], n)
    region = np.random.choice(['North', 'South', 'East', 'West'], n)
    
    # Create binary outcome
    prob = 1 / (1 + np.exp(-(-2 + 0.05 * age + 0.00001 * income)))
    outcome = np.random.binomial(1, prob, n)
    
    return pd.DataFrame({
        'age': age,
        'income': income,
        'education': education,
        'gender': gender,
        'region': region,
        'outcome': outcome
    })
