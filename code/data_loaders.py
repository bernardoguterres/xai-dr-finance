"""
Financial Data Loaders for XAI Experiments
Utilities to load real financial datasets for testing XAI methods.

Author: Bernardo Guterres
Date: December 2025

Supported Datasets:
1. Fama-French Factors (market risk factors)
2. Yahoo Finance Stock Data
3. Lending Club Loan Data (requires download)
4. Synthetic with realistic distributions
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance not available. Install with: pip install yfinance")

try:
    import pandas_datareader as pdr
    PDR_AVAILABLE = True
except ImportError:
    PDR_AVAILABLE = False
    print("pandas_datareader not available. Install with: pip install pandas-datareader")

# ============================================================================
# SECTION 1: Fama-French Factor Data
# ============================================================================

def load_fama_french_factors(start_date='2010-01-01', end_date='2024-01-01'):
    """
    Load Fama-French 5 factors from Ken French's data library.
    
    Factors:
    - Mkt-RF: Market excess return
    - SMB: Small minus Big (size)
    - HML: High minus Low (value)
    - RMW: Robust minus Weak (profitability)
    - CMA: Conservative minus Aggressive (investment)
    
    Returns:
        DataFrame with daily factor returns
    """
    if not PDR_AVAILABLE:
        print("pandas_datareader required. Using synthetic Fama-French data.")
        return _generate_synthetic_ff_factors()
    
    try:
        # Fetch Fama-French 5 factors
        ff_factors = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3_daily', 
                                              start=start_date, end=end_date)[0]
        
        # Clean column names
        ff_factors.columns = ff_factors.columns.str.strip()
        
        # Convert to decimal returns
        ff_factors = ff_factors / 100
        
        print(f"Loaded Fama-French factors: {ff_factors.shape}")
        print(f"Date range: {ff_factors.index[0]} to {ff_factors.index[-1]}")
        print(f"Factors: {list(ff_factors.columns)}")
        
        return ff_factors
        
    except Exception as e:
        print(f"Error loading Fama-French data: {e}")
        print("Using synthetic data instead.")
        return _generate_synthetic_ff_factors()

def _generate_synthetic_ff_factors(n_days=2500):
    """Generate synthetic Fama-French style factors."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2014-01-01', periods=n_days, freq='B')
    
    # Generate correlated factor returns
    mean_returns = np.array([0.0003, 0.0001, 0.0001, 0.00005, 0.00005])
    volatilities = np.array([0.01, 0.005, 0.005, 0.003, 0.003])
    
    # Correlation matrix (realistic)
    corr = np.array([
        [1.0, 0.3, -0.2, 0.1, -0.1],
        [0.3, 1.0, -0.1, 0.2, 0.1],
        [-0.2, -0.1, 1.0, 0.3, 0.4],
        [0.1, 0.2, 0.3, 1.0, 0.2],
        [-0.1, 0.1, 0.4, 0.2, 1.0]
    ])
    
    cov = np.outer(volatilities, volatilities) * corr
    
    returns = np.random.multivariate_normal(mean_returns, cov, n_days)
    
    df = pd.DataFrame(
        returns,
        index=dates,
        columns=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    )
    
    return df

# ============================================================================
# SECTION 2: Stock Fundamentals from Yahoo Finance
# ============================================================================

def load_stock_fundamentals(tickers=None, include_price_data=True):
    """
    Load fundamental data for a list of stocks.
    
    Args:
        tickers: List of stock tickers (default: S&P 500 sample)
        include_price_data: Whether to include price-based metrics
    
    Returns:
        DataFrame with fundamental features
    """
    if not YFINANCE_AVAILABLE:
        print("yfinance required. Using synthetic stock data.")
        return _generate_synthetic_stock_fundamentals()
    
    if tickers is None:
        # Default: sample of large-cap stocks
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B',
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'NFLX',
            'ADBE', 'CRM', 'INTC', 'CSCO', 'PEP', 'ABT', 'KO', 'NKE', 'MRK',
            'PFE', 'TMO', 'COST', 'AVGO', 'ACN', 'DHR', 'TXN', 'NEE', 'LIN',
            'WMT', 'VZ', 'PM', 'RTX', 'HON', 'QCOM', 'UPS', 'BMY', 'SBUX'
        ]
    
    fundamentals = []
    
    print(f"Loading fundamentals for {len(tickers)} stocks...")
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            data = {
                'Ticker': ticker,
                'Sector': info.get('sector', 'Unknown'),
                'Industry': info.get('industry', 'Unknown'),
                
                # Valuation
                'PE_Ratio': info.get('trailingPE', np.nan),
                'Forward_PE': info.get('forwardPE', np.nan),
                'PB_Ratio': info.get('priceToBook', np.nan),
                'PS_Ratio': info.get('priceToSalesTrailing12Months', np.nan),
                'EV_EBITDA': info.get('enterpriseToEbitda', np.nan),
                
                # Profitability
                'ROE': info.get('returnOnEquity', np.nan),
                'ROA': info.get('returnOnAssets', np.nan),
                'Profit_Margin': info.get('profitMargins', np.nan),
                'Operating_Margin': info.get('operatingMargins', np.nan),
                'Gross_Margin': info.get('grossMargins', np.nan),
                
                # Growth
                'Revenue_Growth': info.get('revenueGrowth', np.nan),
                'Earnings_Growth': info.get('earningsGrowth', np.nan),
                
                # Financial Health
                'Debt_to_Equity': info.get('debtToEquity', np.nan),
                'Current_Ratio': info.get('currentRatio', np.nan),
                'Quick_Ratio': info.get('quickRatio', np.nan),
                
                # Dividends
                'Dividend_Yield': info.get('dividendYield', np.nan),
                'Payout_Ratio': info.get('payoutRatio', np.nan),
                
                # Size
                'Market_Cap': info.get('marketCap', np.nan),
                'Enterprise_Value': info.get('enterpriseValue', np.nan),
            }
            
            if include_price_data:
                # Get price history for momentum/volatility
                hist = stock.history(period='1y')
                if len(hist) > 0:
                    returns = hist['Close'].pct_change().dropna()
                    
                    data['Volatility_1Y'] = returns.std() * np.sqrt(252)
                    data['Return_1Y'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
                    data['Return_3M'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[-63]) - 1 if len(hist) > 63 else np.nan
                    data['Beta'] = info.get('beta', np.nan)
            
            fundamentals.append(data)
            
        except Exception as e:
            print(f"  Error loading {ticker}: {e}")
            continue
    
    df = pd.DataFrame(fundamentals)
    
    # Clean up
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    print(f"Loaded {len(df)} stocks with {len(df.columns)} features")
    
    return df

def _generate_synthetic_stock_fundamentals(n_stocks=200):
    """Generate synthetic stock fundamental data."""
    np.random.seed(42)
    
    sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Industrials', 
               'Energy', 'Materials', 'Utilities', 'Real Estate', 'Communications']
    
    data = {
        'Ticker': [f'STK{i:03d}' for i in range(n_stocks)],
        'Sector': np.random.choice(sectors, n_stocks),
        
        # Valuation (log-normal distributed)
        'PE_Ratio': np.exp(np.random.normal(2.8, 0.6, n_stocks)),  # Mean ~16
        'PB_Ratio': np.exp(np.random.normal(0.7, 0.8, n_stocks)),  # Mean ~2
        'PS_Ratio': np.exp(np.random.normal(0.5, 1.0, n_stocks)),
        'EV_EBITDA': np.exp(np.random.normal(2.3, 0.5, n_stocks)),
        'Dividend_Yield': np.abs(np.random.normal(0.02, 0.015, n_stocks)),
        
        # Profitability (bounded)
        'ROE': np.clip(np.random.normal(0.15, 0.12, n_stocks), -0.5, 0.8),
        'ROA': np.clip(np.random.normal(0.08, 0.06, n_stocks), -0.3, 0.4),
        'Profit_Margin': np.clip(np.random.normal(0.12, 0.1, n_stocks), -0.2, 0.5),
        'Operating_Margin': np.clip(np.random.normal(0.15, 0.1, n_stocks), -0.1, 0.5),
        'Gross_Margin': np.clip(np.random.normal(0.35, 0.15, n_stocks), 0.05, 0.9),
        
        # Growth
        'Revenue_Growth': np.random.normal(0.1, 0.15, n_stocks),
        'Earnings_Growth': np.random.normal(0.12, 0.25, n_stocks),
        
        # Financial Health
        'Debt_to_Equity': np.abs(np.random.normal(0.8, 0.6, n_stocks)),
        'Current_Ratio': np.abs(np.random.normal(1.5, 0.7, n_stocks)) + 0.5,
        
        # Risk/Return
        'Volatility_1Y': np.abs(np.random.normal(0.3, 0.15, n_stocks)),
        'Return_1Y': np.random.normal(0.1, 0.3, n_stocks),
        'Return_3M': np.random.normal(0.03, 0.15, n_stocks),
        'Beta': np.clip(np.random.normal(1.0, 0.4, n_stocks), 0.2, 2.5),
        
        # Size (log-normal)
        'Market_Cap': np.exp(np.random.normal(23, 2, n_stocks)),  # In dollars
    }
    
    df = pd.DataFrame(data)
    
    # Add sector-specific adjustments
    tech_mask = df['Sector'] == 'Technology'
    df.loc[tech_mask, 'PE_Ratio'] *= 1.5
    df.loc[tech_mask, 'Revenue_Growth'] += 0.1
    
    utility_mask = df['Sector'] == 'Utilities'
    df.loc[utility_mask, 'Dividend_Yield'] *= 2
    df.loc[utility_mask, 'Volatility_1Y'] *= 0.6
    
    return df

# ============================================================================
# SECTION 3: Lending Club Loan Data
# ============================================================================

def load_lending_club_data(filepath=None):
    """
    Load Lending Club loan data for credit risk analysis.
    
    The Lending Club dataset is popular for XAI research because:
    - Contains both approved and rejected loans
    - Rich feature set (financial, demographic, behavioral)
    - Ground truth outcomes (default/paid off)
    
    Download from: https://www.kaggle.com/datasets/wordsforthewise/lending-club
    
    Args:
        filepath: Path to downloaded CSV file
    
    Returns:
        DataFrame with loan features
    """
    if filepath is None:
        print("Lending Club data requires download from Kaggle.")
        print("URL: https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        print("Using synthetic credit data instead.")
        return _generate_synthetic_credit_data()
    
    try:
        # Load data
        df = pd.read_csv(filepath, low_memory=False)
        
        # Select relevant features for DR/XAI analysis
        feature_cols = [
            'loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment',
            'grade', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc',
            'verification_status', 'loan_status', 'purpose', 'dti',
            'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'open_acc',
            'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
            'initial_list_status', 'application_type', 'mort_acc',
            'pub_rec_bankruptcies'
        ]
        
        available_cols = [c for c in feature_cols if c in df.columns]
        df = df[available_cols].copy()
        
        # Clean target variable
        if 'loan_status' in df.columns:
            df['default'] = df['loan_status'].isin(['Charged Off', 'Default']).astype(int)
        
        # Convert interest rate
        if 'int_rate' in df.columns:
            df['int_rate'] = df['int_rate'].str.rstrip('%').astype(float) / 100
        
        # Handle employment length
        if 'emp_length' in df.columns:
            emp_map = {'< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3,
                      '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                      '8 years': 8, '9 years': 9, '10+ years': 10}
            df['emp_length_num'] = df['emp_length'].map(emp_map)
        
        print(f"Loaded Lending Club data: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"Error loading Lending Club data: {e}")
        return _generate_synthetic_credit_data()

def _generate_synthetic_credit_data(n_samples=5000):
    """Generate synthetic credit/loan data."""
    np.random.seed(42)
    
    # Generate correlated credit features
    # Good credit correlates: high income, low DTI, low delinquencies, long history
    
    credit_score_latent = np.random.normal(0, 1, n_samples)  # Underlying creditworthiness
    
    data = {
        'loan_amnt': np.exp(np.random.normal(9.5, 0.7, n_samples)),  # ~13k average
        'int_rate': np.clip(0.12 - 0.03 * credit_score_latent + 0.02 * np.random.randn(n_samples), 0.05, 0.30),
        'annual_inc': np.exp(10.8 + 0.3 * credit_score_latent + 0.5 * np.random.randn(n_samples)),
        'dti': np.clip(20 - 5 * credit_score_latent + 8 * np.random.randn(n_samples), 0, 50),
        'delinq_2yrs': np.clip(np.random.poisson(np.maximum(0.1, 1 - 0.5 * credit_score_latent), n_samples), 0, 10),
        'inq_last_6mths': np.clip(np.random.poisson(np.maximum(0.1, 1.5 - 0.3 * credit_score_latent), n_samples), 0, 8),
        'open_acc': np.clip(10 + 2 * credit_score_latent + 3 * np.random.randn(n_samples), 1, 40).astype(int),
        'revol_bal': np.exp(8 - 0.5 * credit_score_latent + 1.5 * np.random.randn(n_samples)),
        'revol_util': np.clip(50 - 15 * credit_score_latent + 20 * np.random.randn(n_samples), 0, 100),
        'total_acc': np.clip(20 + 5 * credit_score_latent + 5 * np.random.randn(n_samples), 1, 80).astype(int),
        'mort_acc': np.clip(np.random.poisson(np.maximum(0.1, 1 + 0.5 * credit_score_latent), n_samples), 0, 10),
        'pub_rec': np.clip(np.random.poisson(np.maximum(0.05, 0.3 - 0.2 * credit_score_latent), n_samples), 0, 5),
        'emp_length_num': np.clip(5 + 2 * credit_score_latent + 3 * np.random.randn(n_samples), 0, 10),
    }
    
    # Generate default probability based on features
    default_prob = 1 / (1 + np.exp(
        2 * credit_score_latent - 
        0.5 * (data['dti'] - 20) / 10 +
        0.3 * (data['annual_inc'] / 50000 - 1) -
        0.5 * data['delinq_2yrs']
    ))
    data['default'] = (np.random.random(n_samples) < default_prob).astype(int)
    
    # Categorical features
    grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    grade_probs = np.clip(0.5 + 0.1 * credit_score_latent, 0, 1)
    data['grade'] = [grades[min(6, max(0, int(6 * (1-p) + np.random.randint(-1, 2))))] 
                     for p in grade_probs]
    
    purposes = ['debt_consolidation', 'credit_card', 'home_improvement', 
                'major_purchase', 'small_business', 'car', 'medical', 'other']
    data['purpose'] = np.random.choice(purposes, n_samples, 
                                        p=[0.4, 0.2, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04])
    
    home_ownership = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
    data['home_ownership'] = np.random.choice(home_ownership, n_samples,
                                               p=[0.45, 0.40, 0.10, 0.05])
    
    return pd.DataFrame(data)

# ============================================================================
# SECTION 4: Data Preparation Utilities
# ============================================================================

def prepare_for_dr(df, exclude_cols=None, handle_categorical='onehot', 
                   scale=True, drop_na_thresh=0.5):
    """
    Prepare DataFrame for dimensionality reduction.
    
    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude (e.g., target, ID)
        handle_categorical: 'onehot', 'label', or 'drop'
        scale: Whether to standardize features
        drop_na_thresh: Drop columns with more than this fraction missing
    
    Returns:
        X: Prepared feature matrix
        feature_names: List of feature names
        metadata: Dict with preprocessing info
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    
    df = df.copy()
    
    # Exclude specified columns
    if exclude_cols:
        df = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    
    # Drop high-NA columns
    na_fractions = df.isna().mean()
    cols_to_drop = na_fractions[na_fractions > drop_na_thresh].index
    df = df.drop(columns=cols_to_drop)
    
    # Separate numeric and categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle categorical
    if handle_categorical == 'drop':
        df = df[numeric_cols]
        categorical_cols = []
    elif handle_categorical == 'label':
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    elif handle_categorical == 'onehot':
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Fill remaining NAs with median
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    feature_names = df.columns.tolist()
    X = df.values
    
    # Scale
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None
    
    metadata = {
        'original_shape': df.shape,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'dropped_cols': list(cols_to_drop),
        'scaler': scaler
    }
    
    return X, feature_names, metadata

def create_factor_labels(df):
    """
    Create categorical labels from continuous features for probing.
    """
    labels = {}
    
    # Valuation tier
    if 'PE_Ratio' in df.columns:
        labels['Value_Style'] = pd.cut(
            df['PE_Ratio'], 
            bins=[0, 15, 25, np.inf], 
            labels=['Value', 'Core', 'Growth']
        ).values
    
    # Quality tier
    if 'ROE' in df.columns:
        labels['Quality_Tier'] = pd.cut(
            df['ROE'],
            bins=[-np.inf, 0.1, 0.2, np.inf],
            labels=['Low', 'Medium', 'High']
        ).values
    
    # Size tier
    if 'Market_Cap' in df.columns:
        labels['Size'] = pd.cut(
            np.log(df['Market_Cap']),
            bins=3,
            labels=['Small', 'Mid', 'Large']
        ).values
    
    # Risk tier (volatility-based)
    if 'Volatility_1Y' in df.columns:
        labels['Risk_Level'] = pd.cut(
            df['Volatility_1Y'],
            bins=[0, 0.2, 0.35, np.inf],
            labels=['Low', 'Medium', 'High']
        ).values
    
    # Momentum
    if 'Return_3M' in df.columns:
        labels['Momentum'] = pd.cut(
            df['Return_3M'],
            bins=[-np.inf, -0.05, 0.05, np.inf],
            labels=['Negative', 'Neutral', 'Positive']
        ).values
    
    return labels

# ============================================================================
# MAIN: Demo Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FINANCIAL DATA LOADERS FOR XAI EXPERIMENTS")
    print("=" * 70)
    
    # Demo 1: Fama-French Factors
    print("\n1. Loading Fama-French Factors...")
    ff_data = load_fama_french_factors()
    print(f"   Shape: {ff_data.shape}")
    print(f"   Sample:\n{ff_data.head()}")
    
    # Demo 2: Stock Fundamentals
    print("\n2. Loading Stock Fundamentals...")
    stock_data = load_stock_fundamentals()
    print(f"   Shape: {stock_data.shape}")
    print(f"   Columns: {list(stock_data.columns)}")
    
    # Demo 3: Credit Data
    print("\n3. Loading Credit/Loan Data...")
    credit_data = load_lending_club_data()
    print(f"   Shape: {credit_data.shape}")
    print(f"   Default rate: {credit_data['default'].mean():.2%}")
    
    # Demo 4: Prepare for DR
    print("\n4. Preparing data for dimensionality reduction...")
    
    # Prepare stock data
    X_stocks, feature_names, metadata = prepare_for_dr(
        stock_data,
        exclude_cols=['Ticker', 'Sector', 'Industry'],
        handle_categorical='drop'
    )
    print(f"   Stock features shape: {X_stocks.shape}")
    print(f"   Features: {feature_names}")
    
    # Create factor labels
    factor_labels = create_factor_labels(stock_data)
    print(f"\n   Created labels: {list(factor_labels.keys())}")
    
    # Save sample data for testing
    print("\n5. Saving sample data...")
    
    stock_data.to_csv('/home/claude/sample_stock_data.csv', index=False)
    credit_data.to_csv('/home/claude/sample_credit_data.csv', index=False)
    ff_data.to_csv('/home/claude/sample_ff_factors.csv')
    
    print("\n" + "=" * 70)
    print("DATA LOADING COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    print("  • sample_stock_data.csv - Stock fundamentals")
    print("  • sample_credit_data.csv - Credit/loan data")
    print("  • sample_ff_factors.csv - Fama-French factors")
    
    print("\nTo use real data:")
    print("  1. Stock data: pip install yfinance")
    print("  2. FF factors: pip install pandas-datareader")
    print("  3. Lending Club: Download from Kaggle")
