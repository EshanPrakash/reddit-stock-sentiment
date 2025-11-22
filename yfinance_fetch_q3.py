# yfinance_fetch_q3.py: This file fetches Q3 2023 stock returns for a list of tickers
#                       obtained from aggregate_sentiment.py. It also fetches benchmark returns
#                       for SPY and sector ETFs to compute excess returns. It saves the results
#                       in both CSV and JSON formats for further analysis.
# Requires aggregate_sentiment.py to be run first

import yfinance as yf
import json
import pandas as pd
import os

# Creating a data directory if they don't exist for saving collected posts, keeping the output organized
os.makedirs('data', exist_ok=True)

# Define the date range for Q3 2023
start_date = "2023-07-01"
end_date = "2023-09-30"

# Stock tickers and their company names - 50 stocks across 5 sectors
# Used for mapping and sector classification later
STOCK_INFO = {
    # Technology (15 stocks)
    'AAPL': ['Apple', 'AAPL', '$AAPL'], 
    'MSFT': ['Microsoft', 'MSFT', '$MSFT'], 
    'GOOGL': ['Google', 'Alphabet', 'GOOGL', '$GOOGL'],
    'AMZN': ['Amazon', 'AMZN', '$AMZN'], 
    'META': ['Meta', 'Facebook', 'META', '$META'], 
    'NVDA': ['Nvidia', 'NVDA', '$NVDA'],
    'TSLA': ['Tesla', 'TSLA', '$TSLA'], 
    'AMD': ['AMD', '$AMD'], 
    'NFLX': ['Netflix', 'NFLX', '$NFLX'], 
    'INTC': ['Intel', 'INTC', '$INTC'],
    'CRM': ['Salesforce', 'CRM', '$CRM'], 
    'ORCL': ['Oracle', 'ORCL', '$ORCL'], 
    'ADBE': ['Adobe', 'ADBE', '$ADBE'],
    'CSCO': ['Cisco', 'CSCO', '$CSCO'], 
    'UBER': ['Uber', 'UBER', '$UBER'], 
    
    # Finance (10 stocks)
    'JPM': ['JPMorgan', 'JP Morgan', 'JPM', '$JPM'],
    'BAC': ['Bank of America', 'BofA', 'BAC', '$BAC'], 
    'WFC': ['Wells Fargo', 'WFC', '$WFC'], 
    'GS': ['Goldman Sachs', 'Goldman', '$GS'],
    'MS': ['Morgan Stanley', '$MS'], 
    'C': ['Citigroup', 'Citi', 'Citibank', '$C'], 
    'V': ['Visa', '$V'], 
    'MA': ['Mastercard', '$MA'],
    'AXP': ['American Express', 'Amex', 'AXP', '$AXP'], 
    'SCHW': ['Charles Schwab', 'Schwab', 'SCHW', '$SCHW'],
    
    # Healthcare (10 stocks)
    'JNJ': ['Johnson & Johnson', 'J&J', 'JNJ', '$JNJ'], 
    'UNH': ['UnitedHealth', 'United Health', 'UNH', '$UNH'],
    'PFE': ['Pfizer', 'PFE', '$PFE'], 
    'ABBV': ['AbbVie', 'ABBV', '$ABBV'], 
    'LLY': ['Eli Lilly', 'Lilly', 'LLY', '$LLY'],
    'MRK': ['Merck', 'MRK', '$MRK'], 
    'TMO': ['Thermo Fisher', 'TMO', '$TMO'], 
    'CVS': ['CVS', '$CVS'], 
    'AMGN': ['Amgen', 'AMGN', '$AMGN'],
    'BMY': ['Bristol Myers', 'Bristol-Myers Squibb', 'BMY', '$BMY'], 
    
    # Energy (10 stocks)
    'XOM': ['Exxon', 'ExxonMobil', 'Exxon Mobil', 'XOM', '$XOM'],
    'CVX': ['Chevron', 'CVX', '$CVX'], 
    'COP': ['ConocoPhillips', 'Conoco Phillips', 'COP', '$COP'],
    'SLB': ['Schlumberger', 'SLB', '$SLB'], 
    'EOG': ['EOG Resources', 'EOG', '$EOG'], 
    'OXY': ['Occidental', 'Occidental Petroleum', 'OXY', '$OXY'],
    'MPC': ['Marathon Petroleum', 'Marathon', 'MPC', '$MPC'], 
    'PSX': ['Phillips 66', 'PSX', '$PSX'], 
    'VLO': ['Valero', 'VLO', '$VLO'], 
    'HAL': ['Halliburton', 'HAL', '$HAL'], 
    
    # Aerospace/Defense (5 stocks)
    'LMT': ['Lockheed Martin', 'Lockheed', 'LMT', '$LMT'],
    'RTX': ['Raytheon', 'RTX', '$RTX'], 
    'BA': ['Boeing', '$BA'], 
    'NOC': ['Northrop Grumman', 'Northrop', 'NOC', '$NOC'],
    'GD': ['General Dynamics', '$GD']
}

# Sector ETF benchmarks
# Mapping of sectors to their corresponding ETFs, used for excess return calculations
SECTOR_ETF = {
    'Tech': 'XLK',
    'Finance': 'XLF',
    'Healthcare': 'XLV',
    'Energy': 'XLE',
    'Aerospace/Defense': 'ITA'
}

# Mapping tickers to sectors for later use, allowing for sector-based analysis
SECTOR_MAP = {}
for ticker in STOCK_INFO:
    if ticker in ['AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','AMD','NFLX','INTC','CRM','ORCL','ADBE','CSCO','UBER']:
        SECTOR_MAP[ticker] = 'Tech'
    elif ticker in ['JPM','BAC','WFC','GS','MS','C','V','MA','AXP','SCHW']:
        SECTOR_MAP[ticker] = 'Finance'
    elif ticker in ['JNJ','UNH','PFE','ABBV','LLY','MRK','TMO','CVS','AMGN','BMY']:
        SECTOR_MAP[ticker] = 'Healthcare'
    elif ticker in ['XOM','CVX','COP','SLB','EOG','OXY','MPC','PSX','VLO','HAL']:
        SECTOR_MAP[ticker] = 'Energy'
    elif ticker in ['LMT','RTX','BA','NOC','GD']:
        SECTOR_MAP[ticker] = 'Aerospace/Defense'

print("Fetching Q3 2023 stock returns and benchmarks...")
print("=" * 60)

# Step 1: Fetch the Q3 benchmark returns (SPY + sector ETFs)
print("\nFetching benchmark returns...")
print("-" * 60)

# benchmark_tickers includes SPY and all sector ETFs
benchmark_tickers = ['SPY'] + list(SECTOR_ETF.values())

# bench_data will hold the returns for benchmarks
bench_data = {} 

for t in benchmark_tickers:
    try:
        # auto_adjust=True to get adjusted close prices, makes OHLCV adjustments consistent across splits/dividends
        d = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if not d.empty:
            # Calculate return over the period: (last - first) / first
            close_np = d['Close'].to_numpy()
            first = close_np[0].item()
            last = close_np[-1].item()
            bench_data[t] = (last - first) / first
            print(f"  ✓ {t}: {bench_data[t]*100:+.2f}%")
    except Exception:
        print(f"  ✗ {t}: Failed to fetch")
        pass

print(f"\n✓ Fetched {len(bench_data)} benchmarks")

# Step 2: Load the Q2 sentiment data from aggregate_sentiment.py to get list of tickers
input_file = "data/q2_2023_sentiment_by_ticker.json"
print(f"\nLoading tickers from {input_file}...")
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    print(f"✓ Loaded {len(posts)} tickers\n")
except FileNotFoundError:
    print(f"✗ Error: {input_file} not found!")
    print("Run aggregate_sentiment.py first.")
    exit(1)

# Step 3: Fetch Q3 returns for each stock
print("=" * 60)
print("Fetching individual stock returns...")
print("-" * 60)

# Convert loaded posts to a pandas DataFrame for easier handling
df = pd.DataFrame(posts)

# Store results here
results = []

# Track failed fetches
failed_count = 0

for t in df['ticker']:
    try:
        # auto_adjust=True to get adjusted close prices, makes OHLCV adjustments consistent across splits/dividends
        d = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if d.empty:
            # No data fetched for this ticker, log and continue
            print(f"  ✗ {t}: No data")
            failed_count += 1
            continue

        # Calculate return over the period: (last - first) / first
        close_np = d['Close'].to_numpy()
        start_p = close_np[0].item()
        end_p = close_np[-1].item()
        pct = (end_p - start_p) / start_p

        # Get sector and corresponding ETF
        sector = SECTOR_MAP.get(t, None)
        sector_etf = SECTOR_ETF.get(sector, None)

        # Get benchmark returns
        spy = float(bench_data.get('SPY', 0.0))
        sec = float(bench_data.get(sector_etf, 0.0)) if sector_etf in bench_data else 0.0

        # Store results
        results.append({
            'ticker': t,
            'q3_return_pct': float(pct),
            'excess_vs_spy': float(pct - spy),
            'excess_vs_sector': float(pct - sec)
        })
        
        print(f"  ✓ {t}: {pct*100:+.2f}% (vs SPY: {(pct-spy)*100:+.2f}%)")
    # Catch any exceptions during fetching/processing
    except Exception:
        print(f"  ✗ {t}: Error fetching data")
        failed_count += 1
        pass

# Step 4: Save results to a CSV and JSON file for further statistical analysis
print("\n" + "=" * 60)
print(f"✓ Successfully fetched {len(results)} stocks")
if failed_count > 0:
    print(f"✗ Failed to fetch {failed_count} stocks")

# Save as CSV file
out_csv = "data/q3_2023_with_benchmarks.csv"
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"\n✓ Saved to {out_csv}")

# Save as JSON file
out_json = "data/q3_2023_with_benchmarks.json"
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved to {out_json}")

print("\n" + "=" * 60)
print("✓ Q3 2023 data collection complete!")
print("\nNext step: Correlate Q2 sentiment with Q3 returns")