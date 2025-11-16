# filter_posts.py

import json
import re
import os
import statistics

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Stock tickers and their company names - 50 stocks across 5 sectors
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
    'GD': ['General Dynamics', '$GD'],
}

def find_mentioned_tickers(text):
    """
    Find all tickers mentioned in the text (title + selftext)
    Returns a list of unique ticker symbols
    """
    if not text:
        return []
    
    text_lower = text.lower()
    mentioned = []
    
    for ticker, keywords in STOCK_INFO.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            
            if re.search(pattern, text_lower):
                mentioned.append(ticker)
                break
    
    return list(set(mentioned))

# Load the collected data
input_file = "data/reddit_posts_q2_2023_full.json"
print(f"Loading data from {input_file}...")

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        all_posts = json.load(f)
    print(f"✓ Loaded {len(all_posts):,} posts")
except FileNotFoundError:
    print(f"✗ Error: {input_file} not found!")
    print("Run collect_pullpush.py first to collect the data.")
    exit(1)

print("\nFiltering posts that mention our target stocks...")
print("=" * 60)

# Filter posts
filtered_posts = []
ticker_counts = {ticker: 0 for ticker in STOCK_INFO.keys()}

for post in all_posts:
    combined_text = f"{post.get('title', '')} {post.get('selftext', '')}"
    mentioned_tickers = find_mentioned_tickers(combined_text)
    
    if mentioned_tickers:
        post['mentioned_tickers'] = mentioned_tickers
        filtered_posts.append(post)
        
        for ticker in mentioned_tickers:
            ticker_counts[ticker] += 1

print(f"✓ Found {len(filtered_posts):,} posts mentioning our target stocks")
print(f"  ({len(filtered_posts)/len(all_posts)*100:.1f}% of total posts)")

# Calculate median threshold
post_counts = [count for count in ticker_counts.values() if count > 0]
if not post_counts:
    print("\n✗ No stocks mentioned! Check your data.")
    exit(1)

median_threshold = statistics.median(post_counts)

print(f"\nPost count distribution:")
print(f"  Median: {median_threshold:.0f}")
print(f"  Mean: {statistics.mean(post_counts):.1f}")
print(f"  Min: {min(post_counts)}")
print(f"  Max: {max(post_counts)}")

# Apply median threshold
print(f"\nApplying threshold: posts >= median ({median_threshold:.0f})...")

valid_tickers = {ticker for ticker, count in ticker_counts.items() if count >= median_threshold}
excluded_tickers = {ticker for ticker, count in ticker_counts.items() if 0 < count < median_threshold}

print(f"✓ {len(valid_tickers)} stocks meet threshold (>= {median_threshold:.0f} posts)")
print(f"✗ {len(excluded_tickers)} stocks excluded (< {median_threshold:.0f} posts)")

# Re-filter posts
original_count = len(filtered_posts)
filtered_posts = [
    post for post in filtered_posts 
    if any(ticker in valid_tickers for ticker in post['mentioned_tickers'])
]

# Update ticker lists
for post in filtered_posts:
    post['mentioned_tickers'] = [t for t in post['mentioned_tickers'] if t in valid_tickers]

# Recalculate counts
ticker_counts = {ticker: 0 for ticker in valid_tickers}
for post in filtered_posts:
    for ticker in post['mentioned_tickers']:
        ticker_counts[ticker] += 1

print(f"✓ {len(filtered_posts):,} posts remain after threshold ({original_count - len(filtered_posts)} removed)")

# Save filtered data
output_file = "data/reddit_posts_q2_2023_filtered.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_posts, f, indent=2, ensure_ascii=False)

print(f"✓ Filtered data saved to {output_file}")

# Print statistics
print("\n" + "=" * 60)
print("Posts mentioning each stock:")
print("=" * 60)

for ticker, count in sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        percentage = (count / len(filtered_posts)) * 100
        print(f"  {ticker}: {count:,} posts ({percentage:.1f}%)")

print("\n" + "=" * 60)
print("Posts by subreddit:")
print("=" * 60)

subreddit_counts = {}
for post in filtered_posts:
    sub = post['subreddit']
    subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1

for sub, count in sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(filtered_posts)) * 100
    print(f"  r/{sub}: {count:,} posts ({percentage:.1f}%)")

print("\n✓ Filtering complete!")