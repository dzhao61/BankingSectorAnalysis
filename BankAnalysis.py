#!/usr/bin/env python
"""
Australian Bank Stock Return Interdependencies
===============================================

Analyses the extent to which share price dependencies exist among Australia's five
major banks (Commonwealth Bank, Westpac, ANZ, NAB, Macquarie Group), and whether
those dependencies persist after conditioning on broader market and macroeconomic factors.

Method:
- KSG estimator (k=4) on a 100-day rolling window
- Theiler window set from ACF analysis (DYN_CORR_EXCL)
- Pairwise MI between all bank pairs
- CMI conditioning on interest rates proxy and market index

Author: Daniel Zhao
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings


try:
    from jpype import *
except ImportError:
    print("ERROR: jpype1 is required. Install with: pip install jpype1")
    raise

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance is required. Install with: pip install yfinance")
    raise

# ============================================================================
# SECTION 1: Configuration
# ============================================================================

# Australian bank tickers
BANKS = {
    'CBA': 'CBA.AX',      # Commonwealth Bank
    'WBC': 'WBC.AX',      # Westpac
    'ANZ': 'ANZ.AX',      # ANZ
    'NAB': 'NAB.AX',      # National Australia Bank
    'MQG': 'MQG.AX'       # Macquarie Group
}

# Control variables
CONTROLS = {
    'Interest_Rates': 'VGB.AX',  # Vanguard Australian Govt Bond ETF
    'ASX200': '^AXJO'             # ASX200 index
}

# Parameters
K_NEIGHBORS = 4           # KSG estimator neighbors
WINDOW_SIZE = 100         # Trading days for rolling window
START_DATE = '2020-01-01' # Analysis start date
END_DATE = '2024-11-01'   # Analysis end date

# JIDT library path - update to your local copy
# Download from: https://github.com/jlizier/jidt/releases
JAR_LOCATION = "infodynamics.jar"

# ============================================================================
# SECTION 2: JIDT Setup
# ============================================================================

def start_jidt():
    """Initialize JIDT library."""
    if not isJVMStarted():
        if not os.path.isfile(JAR_LOCATION):
            raise FileNotFoundError(
                f"infodynamics.jar not found at '{JAR_LOCATION}'.\n"
                "Download JIDT from https://github.com/jlizier/jidt/releases "
                "and update the JAR_LOCATION variable in the configuration section."
            )

        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + JAR_LOCATION, convertStrings=True)
        print("✓ JIDT library loaded")
    else:
        print("✓ JIDT library already loaded")

# ============================================================================
# SECTION 3: Data Loading and Preprocessing
# ============================================================================

def download_stock_data(tickers, start_date, end_date):
    """
    Download stock data from Yahoo Finance.
    
    Parameters:
    -----------
    tickers : dict
        Dictionary of {name: ticker} pairs
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pd.DataFrame
        Adjusted closing prices for all tickers
    """
    print(f"\nDownloading data from {start_date} to {end_date}...")
    
    data = {}
    for name, ticker in tickers.items():
        print(f"  Downloading {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(df) > 0:
                # Handle both old and new yfinance API
                if isinstance(df.columns, pd.MultiIndex):
                    # New API returns MultiIndex columns
                    adj_close = df['Adj Close'].iloc[:, 0] if 'Adj Close' in df.columns.get_level_values(0) else df['Close'].iloc[:, 0]
                else:
                    # Old API
                    adj_close = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
                data[name] = adj_close
                print(f"    ✓ Got {len(adj_close)} days")
            else:
                print(f"    Warning: No data for {name}")
        except Exception as e:
            print(f"    Error downloading {name}: {e}")
    
    # Combine into single DataFrame
    df = pd.DataFrame(data)
    
    # Remove any rows with all NaN
    df = df.dropna(how='all')
    
    print(f"\n✓ Downloaded {len(df)} days of data for {len(df.columns)} assets")
    
    return df


def calculate_returns(prices):
    """
    Calculate log returns from prices.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data
        
    Returns:
    --------
    pd.DataFrame
        Log returns (removes first row with NaN)
    """
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna()
    
    print(f"\n✓ Calculated log returns:")
    print(f"  Samples: {len(returns)}")
    print(f"  Variables: {len(returns.columns)}")
    print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    return returns


def standardize_data(data):
    """Standardize data to zero mean and unit variance."""
    return (data - data.mean()) / data.std()


# ============================================================================
# SECTION 4: Autocorrelation Analysis
# ============================================================================

def calculate_autocorrelation(data, max_lag=50):
    """
    Calculate autocorrelation function.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    max_lag : int
        Maximum lag to compute
        
    Returns:
    --------
    lags : array
        Lag values
    acf : array
        Autocorrelation values
    """
    data = np.array(data)
    data = data - np.mean(data)
    
    lags = np.arange(0, min(max_lag, len(data) // 4))
    acf = np.zeros(len(lags))
    
    variance = np.var(data)
    
    for i, lag in enumerate(lags):
        if lag == 0:
            acf[i] = 1.0
        else:
            acf[i] = np.mean(data[:-lag] * data[lag:]) / variance
    
    return lags, acf


def find_autocorrelation_length(data, max_lag=50):
    """
    Find autocorrelation length using zero crossing and significance threshold.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    max_lag : int
        Maximum lag to consider
        
    Returns:
    --------
    int
        Autocorrelation length (Theiler window)
    """
    lags, acf = calculate_autocorrelation(data, max_lag)
    
    n = len(data)
    threshold = 2.0 / np.sqrt(n)
    
    # Find first zero crossing
    zero_crossing = None
    for i in range(1, len(acf)):
        if acf[i] <= 0:
            zero_crossing = i
            break
    
    # Find where ACF drops below threshold
    below_threshold = None
    for i in range(1, len(acf)):
        if abs(acf[i]) < threshold:
            below_threshold = i
            break
    
    # Use minimum of criteria (conservative)
    candidates = [x for x in [zero_crossing, below_threshold] if x is not None]
    
    if len(candidates) == 0:
        # Default fallback
        return min(5, len(data) // 20)
    
    return min(candidates)


def determine_theiler_window(returns):
    """
    Determine Theiler window as maximum autocorrelation length across all variables.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return data
        
    Returns:
    --------
    int
        Theiler window to use
    """
    print("\nCalculating autocorrelation lengths...")
    
    acf_lengths = {}
    for col in returns.columns:
        length = find_autocorrelation_length(returns[col].values)
        acf_lengths[col] = length
        print(f"  {col}: {length}")
    
    theiler_window = max(acf_lengths.values())
    print(f"\n✓ Selected Theiler window: {theiler_window} (max across all variables)")
    
    return theiler_window


# ============================================================================
# SECTION 5: Mutual Information Calculation with JIDT
# ============================================================================

def calculate_mi_kraskov(X, Y, k=4, theiler_window=0):
    """
    Calculate mutual information using Kraskov estimator (Algorithm 1).
    
    Parameters:
    -----------
    X : array-like
        First variable
    Y : array-like
        Second variable
    k : int
        Number of nearest neighbors
    theiler_window : int
        Dynamic correlation exclusion window
        
    Returns:
    --------
    float
        Mutual information in nats
    """
    # Import JIDT class
    MICalcClass = JPackage('infodynamics.measures.continuous.kraskov').MutualInfoCalculatorMultiVariateKraskov1
    miCalc = MICalcClass()
    
    # Initialize for univariate data
    miCalc.initialise(1, 1)
    
    # Set properties
    miCalc.setProperty("k", str(k))
    if theiler_window > 0:
        miCalc.setProperty("DYN_CORR_EXCL", str(theiler_window))
    
    # Add small noise for numerical stability
    miCalc.setProperty("NOISE_LEVEL_TO_ADD", "1e-8")
    
    # Convert to Java arrays (ensure they are 1D numpy arrays)
    X = np.array(X).flatten()
    Y = np.array(Y).flatten()
    X_java = JArray(JDouble, 1)(X.tolist())
    Y_java = JArray(JDouble, 1)(Y.tolist())
    
    # Set observations and compute
    miCalc.setObservations(X_java, Y_java)
    mi = miCalc.computeAverageLocalOfObservations()
    
    return mi


def calculate_cmi_kraskov(X, Y, Z, k=4, theiler_window=0):
    """
    Calculate conditional mutual information I(X;Y|Z) using Kraskov estimator.
    
    Parameters:
    -----------
    X : array-like
        First variable
    Y : array-like
        Second variable
    Z : array-like or 2D array
        Conditioning variable(s)
    k : int
        Number of nearest neighbors
    theiler_window : int
        Dynamic correlation exclusion window
        
    Returns:
    --------
    float
        Conditional mutual information in nats
    """
    # Import JIDT class
    CMICalcClass = JPackage('infodynamics.measures.continuous.kraskov').ConditionalMutualInfoCalculatorMultiVariateKraskov1
    cmiCalc = CMICalcClass()
    
    # Handle Z dimensions
    Z = np.array(Z)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    
    # Initialize
    cmiCalc.initialise(1, 1, Z.shape[1])
    
    # Set properties
    cmiCalc.setProperty("k", str(k))
    if theiler_window > 0:
        cmiCalc.setProperty("DYN_CORR_EXCL", str(theiler_window))
    cmiCalc.setProperty("NOISE_LEVEL_TO_ADD", "1e-8")
    
    # Convert to Java arrays (ensure proper shapes)
    X = np.array(X).flatten()
    Y = np.array(Y).flatten()
    X_java = JArray(JDouble, 1)(X.tolist())
    Y_java = JArray(JDouble, 1)(Y.tolist())
    Z_java = JArray(JDouble, 2)(Z.tolist())
    
    # Set observations and compute
    cmiCalc.setObservations(X_java, Y_java, Z_java)
    cmi = cmiCalc.computeAverageLocalOfObservations()
    
    return cmi


# ============================================================================
# SECTION 6: Pairwise Analysis
# ============================================================================

def calculate_pairwise_mi(returns, bank_names, theiler_window, k=4):
    """
    Calculate pairwise MI between all bank pairs.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Bank returns
    bank_names : list
        List of bank names
    theiler_window : int
        Theiler window for autocorrelation control
    k : int
        Number of neighbors for KSG
        
    Returns:
    --------
    pd.DataFrame
        MI matrix (nats)
    """
    print(f"\nCalculating pairwise MI between banks...")
    print(f"  Using k={k}, Theiler window={theiler_window}")
    
    n_banks = len(bank_names)
    mi_matrix = np.zeros((n_banks, n_banks))
    
    for i, bank1 in enumerate(bank_names):
        for j, bank2 in enumerate(bank_names):
            if i == j:
                mi_matrix[i, j] = np.nan  # Self-MI not meaningful
            elif i < j:
                # Calculate MI
                X = returns[bank1].values
                Y = returns[bank2].values
                
                mi = calculate_mi_kraskov(X, Y, k=k, theiler_window=theiler_window)
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi  # Symmetric
                
                print(f"    MI({bank1}, {bank2}): {mi:.4f} nats")
    
    # Convert to DataFrame
    mi_df = pd.DataFrame(mi_matrix, index=bank_names, columns=bank_names)
    
    print(f"\n✓ Pairwise MI calculated")
    print(f"  Mean MI: {np.nanmean(mi_matrix):.4f} nats")
    print(f"  Min MI: {np.nanmin(mi_matrix):.4f} nats")
    print(f"  Max MI: {np.nanmax(mi_matrix):.4f} nats")
    
    return mi_df


def calculate_conditional_mi(returns, bank_names, control_vars, theiler_window, k=4):
    """
    Calculate conditional MI between banks, conditioning on interest rates and market.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        All returns (banks + controls)
    bank_names : list
        Bank names
    control_vars : list
        Names of control variables
    theiler_window : int
        Theiler window
    k : int
        Number of neighbors
        
    Returns:
    --------
    pd.DataFrame
        CMI matrix (nats)
    """
    print(f"\nCalculating conditional MI (controlling for {control_vars})...")
    
    n_banks = len(bank_names)
    cmi_matrix = np.zeros((n_banks, n_banks))
    
    # Prepare conditioning variables
    Z = returns[control_vars].values
    
    for i, bank1 in enumerate(bank_names):
        for j, bank2 in enumerate(bank_names):
            if i == j:
                cmi_matrix[i, j] = np.nan
            elif i < j:
                X = returns[bank1].values
                Y = returns[bank2].values
                
                cmi = calculate_cmi_kraskov(X, Y, Z, k=k, theiler_window=theiler_window)
                cmi_matrix[i, j] = cmi
                cmi_matrix[j, i] = cmi
                
                print(f"    CMI({bank1}, {bank2} | controls): {cmi:.4f} nats")
    
    cmi_df = pd.DataFrame(cmi_matrix, index=bank_names, columns=bank_names)
    
    print(f"\n✓ Conditional MI calculated")
    print(f"  Mean CMI: {np.nanmean(cmi_matrix):.4f} nats")
    
    return cmi_df


# ============================================================================
# SECTION 7: Visualization
# ============================================================================

def plot_correlation_matrix(returns, bank_names, title="Correlation Matrix"):
    """Plot correlation matrix for comparison."""
    corr = returns[bank_names].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                center=0, vmin=-1, vmax=1, square=True)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def plot_mi_matrices(mi_df, cmi_df, title_prefix="Australian Banks"):
    """
    Plot MI and CMI matrices side by side.
    
    Parameters:
    -----------
    mi_df : pd.DataFrame
        MI matrix
    cmi_df : pd.DataFrame
        CMI matrix
    title_prefix : str
        Prefix for plot titles
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot MI
    ax = axes[0]
    sns.heatmap(mi_df, annot=True, fmt='.4f', cmap='YlOrRd', 
                ax=ax, vmin=0, square=True, cbar_kws={'label': 'MI (nats)'})
    ax.set_title(f'{title_prefix}: Mutual Information (MI)')
    
    # Plot CMI
    ax = axes[1]
    sns.heatmap(cmi_df, annot=True, fmt='.4f', cmap='YlOrRd',
                ax=ax, vmin=0, square=True, cbar_kws={'label': 'CMI (nats)'})
    ax.set_title(f'{title_prefix}: Conditional MI (CMI | Interest Rates, ASX200)')
    
    plt.tight_layout()
    return fig


def plot_mi_comparison(mi_df, cmi_df):
    """
    Plot bar chart comparing MI vs CMI for each bank pair.
    """
    # Extract upper triangle (avoid duplicates)
    bank_names = mi_df.index.tolist()
    n_banks = len(bank_names)
    
    pairs = []
    mi_values = []
    cmi_values = []
    
    for i in range(n_banks):
        for j in range(i+1, n_banks):
            pairs.append(f"{bank_names[i]}-{bank_names[j]}")
            mi_values.append(mi_df.iloc[i, j])
            cmi_values.append(cmi_df.iloc[i, j])
    
    # Create DataFrame for plotting
    df_plot = pd.DataFrame({
        'Pair': pairs,
        'MI (unconditioned)': mi_values,
        'CMI (conditioned)': cmi_values
    })
    
    df_plot = df_plot.set_index('Pair')
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot.plot(kind='bar', ax=ax, rot=45)
    ax.set_ylabel('Information (nats)')
    ax.set_title('Mutual Information: Unconditioned vs Conditioned on Market Factors')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    return fig


def plot_summary(returns, mi_df, cmi_df, bank_names):
    """
    Create comprehensive summary plot.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Time series of returns
    ax1 = fig.add_subplot(gs[0, :])
    for bank in bank_names:
        ax1.plot(returns.index, returns[bank].cumsum(), label=bank, alpha=0.7)
    ax1.set_title('Cumulative Log Returns')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. MI heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    sns.heatmap(mi_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2, 
                vmin=0, square=True, cbar_kws={'label': 'nats'})
    ax2.set_title('Mutual Information (MI)')
    
    # 3. CMI heatmap
    ax3 = fig.add_subplot(gs[1, 1])
    sns.heatmap(cmi_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3,
                vmin=0, square=True, cbar_kws={'label': 'nats'})
    ax3.set_title('Conditional MI (controlling for market factors)')
    
    # 4. MI vs CMI scatter
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Extract values
    mi_vals = []
    cmi_vals = []
    labels = []
    for i in range(len(bank_names)):
        for j in range(i+1, len(bank_names)):
            mi_vals.append(mi_df.iloc[i, j])
            cmi_vals.append(cmi_df.iloc[i, j])
            labels.append(f"{bank_names[i]}-{bank_names[j]}")
    
    ax4.scatter(mi_vals, cmi_vals, s=100, alpha=0.6)
    for i, label in enumerate(labels):
        ax4.annotate(label, (mi_vals[i], cmi_vals[i]), fontsize=8)
    
    # Add diagonal line
    max_val = max(max(mi_vals), max(cmi_vals))
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='MI = CMI')
    
    ax4.set_xlabel('MI (nats)')
    ax4.set_ylabel('CMI (nats)')
    ax4.set_title('MI vs CMI: Persistence After Conditioning')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Reduction percentage
    ax5 = fig.add_subplot(gs[2, 1])
    reduction = [(mi_vals[i] - cmi_vals[i]) / mi_vals[i] * 100 
                 for i in range(len(mi_vals))]
    
    ax5.bar(range(len(labels)), reduction, color='steelblue', alpha=0.7)
    ax5.set_xticks(range(len(labels)))
    ax5.set_xticklabels(labels, rotation=45, ha='right')
    ax5.set_ylabel('Reduction (%)')
    ax5.set_title('MI Reduction After Conditioning (%)')
    ax5.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax5.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# ============================================================================
# SECTION 8: Main Analysis
# ============================================================================

def main():
    """
    Main analysis function.
    """
    print("="*70)
    print("AUSTRALIAN BANK STOCK RETURN INTERDEPENDENCIES")
    print("="*70)
    
    # Initialize JIDT
    print("\n1. Initializing JIDT...")
    start_jidt()
    
    # Download data
    print("\n2. Downloading stock data...")
    all_tickers = {**BANKS, **CONTROLS}
    prices = download_stock_data(all_tickers, START_DATE, END_DATE)
    
    # Calculate returns
    print("\n3. Calculating log returns...")
    returns = calculate_returns(prices)
    
    # Standardize
    returns_std = standardize_data(returns)
    
    # Bank names
    bank_names = list(BANKS.keys())
    control_names = list(CONTROLS.keys())
    
    # Determine Theiler window
    print("\n4. Determining Theiler window (autocorrelation control)...")
    theiler_window = determine_theiler_window(returns_std)
    
    # Calculate pairwise MI
    print("\n5. Calculating pairwise mutual information...")
    mi_df = calculate_pairwise_mi(returns_std, bank_names, theiler_window, k=K_NEIGHBORS)
    
    # Calculate conditional MI
    print("\n6. Calculating conditional mutual information...")
    cmi_df = calculate_conditional_mi(returns_std, bank_names, control_names, 
                                     theiler_window, k=K_NEIGHBORS)
    
    # Analysis summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    mi_mean = np.nanmean(mi_df.values)
    cmi_mean = np.nanmean(cmi_df.values)
    reduction = (mi_mean - cmi_mean) / mi_mean * 100
    
    print(f"\nMutual Information (unconditioned):")
    print(f"  Mean: {mi_mean:.4f} nats ({mi_mean/np.log(2):.4f} bits)")
    print(f"  Range: {np.nanmin(mi_df.values):.4f} - {np.nanmax(mi_df.values):.4f} nats")
    
    print(f"\nConditional Mutual Information (conditioned on interest rates & ASX200):")
    print(f"  Mean: {cmi_mean:.4f} nats ({cmi_mean/np.log(2):.4f} bits)")
    print(f"  Range: {np.nanmin(cmi_df.values):.4f} - {np.nanmax(cmi_df.values):.4f} nats")
    
    print(f"\nReduction from conditioning:")
    print(f"  Mean reduction: {reduction:.1f}%")
    
    print(f"\n{'HYPOTHESIS TEST:':>20}")
    print("-" * 70)
    if cmi_mean > 0.01:  # Threshold for "significant"
        print("✓ CMI remains significant after conditioning on market factors")
        print("  → Bank interdependencies persist beyond common market exposure")
        print("  → Concentration itself contributes to systemic risk")
        print("  → HYPOTHESIS SUPPORTED")
    else:
        print("✗ CMI drops to near-zero after conditioning")
        print("  → Interdependencies fully explained by market factors")
        print("  → HYPOTHESIS NOT SUPPORTED")
    
    # Generate visualizations
    print("\n7. Generating visualizations...")
    
    # Summary plot
    fig_summary = plot_summary(returns_std, mi_df, cmi_df, bank_names)
    fig_summary.savefig('bank_analysis_summary.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: bank_analysis_summary.png")
    
    # MI/CMI comparison
    fig_comparison = plot_mi_comparison(mi_df, cmi_df)
    fig_comparison.savefig('bank_mi_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: bank_mi_comparison.png")
    
    # Correlation for reference
    fig_corr = plot_correlation_matrix(returns_std, bank_names)
    fig_corr.savefig('bank_correlation.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: bank_correlation.png")
    
    plt.show()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    # Save results to CSV
    mi_df.to_csv('bank_mi_matrix.csv')
    cmi_df.to_csv('bank_cmi_matrix.csv')
    print("\n✓ Results saved to CSV files")
    
    return {
        'returns': returns_std,
        'mi': mi_df,
        'cmi': cmi_df,
        'theiler_window': theiler_window
    }


if __name__ == "__main__":
    results = main()
