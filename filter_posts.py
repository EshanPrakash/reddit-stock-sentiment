# filter_posts.py: This file filters the collected Reddit posts to identify those that mention
#                  specific stock tickers and company names. It applies a median threshold to 
#                  focus on stocks with significant discussion volume, and saves the filtered 
#                  dataset for further analysis.
# Requires collect_pullpush.py to be run first.

import json
import re
import os
import statistics

# Creating a data directory if it doesn't exist for saving collected posts, keeping the output organized
os.makedirs('data', exist_ok=True)

# Stock tickers and their company names - 50 stocks across 5 sectors
# This dictionary maps stock tickers to a list of keywords (company names and ticker symbols)
# used to identify mentions within Reddit post titles and bodies.
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

# Function to find mentioned tickers in a given text.
# Given the text of a Reddit post (title + selftext (body)), this function searches for
# mentions of any stock tickers or company names defined in STOCK_INFO.
# It returns a list of unique ticker symbols that were mentioned, if any.
def find_mentioned_tickers(text):
    """
    Find all tickers mentioned in the text (title + selftext)
    Returns a list of unique ticker symbols
    """
    # Handle empty text
    if not text:
        return []
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    mentioned = []
    
    # Search for each keyword in the text
    for ticker, keywords in STOCK_INFO.items():
        for keyword in keywords:
            # Build regex pattern to match whole words only
            # This prevents partial matches (e.g., "Apple" matching "Pineapple")
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            
            # If a match is found, add the ticker to the mentioned list
            if re.search(pattern, text_lower):
                mentioned.append(ticker)
                break   # No need to check other keywords for this ticker, move to next ticker
    
    return list(set(mentioned)) # Return unique tickers only

# Load the collected Reddit posts from a JSON file produced by collect_pullpush.py
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

# 1st stage filtering: Identify and keep posts that mention any of our target stock tickers/companies
#                      Store filtered posts and count mentions per ticker

# Initialize structures for filtered posts and ticker counts
filtered_posts = []
ticker_counts = {ticker: 0 for ticker in STOCK_INFO.keys()}

# Process each post to find mentioned tickers
for post in all_posts:
    # Combine title and selftext for searching, as stock mentions can appear in either
    # If either field is missing, default to an empty string
    combined_text = f"{post.get('title', '')} {post.get('selftext', '')}"
    mentioned_tickers = find_mentioned_tickers(combined_text)
    
    # If any tickers were mentioned, add post to filtered list and update counts
    if mentioned_tickers:
        post['mentioned_tickers'] = mentioned_tickers
        filtered_posts.append(post)
        
        for ticker in mentioned_tickers:
            ticker_counts[ticker] += 1

print(f"✓ Found {len(filtered_posts):,} posts mentioning our target stocks")
print(f"  ({len(filtered_posts)/len(all_posts)*100:.1f}% of total posts)")

# 2nd stage filtering: Apply a median threshold to focus on stocks with significant discussion volume
#                      Stocks mentioned in fewer posts than the median will be excluded, helping to reduce noise.
#                      By focusing on stocks with higher mention counts, we can improve the quality of sentiment analysis.

# Calculate median of post counts per ticker, excluding tickers with zero mentions since they are not relevant
post_counts = [count for count in ticker_counts.values() if count > 0]
if not post_counts:
    print("\n✗ No stocks mentioned! Check your data.")
    exit(1)

# Calculate median threshold, mean, min, and max for reporting
median_threshold = statistics.median(post_counts)
print(f"\nPost count distribution:")
print(f"  Median: {median_threshold:.0f}")
print(f"  Mean: {statistics.mean(post_counts):.1f}")
print(f"  Min: {min(post_counts)}")
print(f"  Max: {max(post_counts)}")

# Apply median threshold
print(f"\nApplying threshold: posts >= median ({median_threshold:.0f})...")

# Determine valid/excluded tickers based on the median threshold
valid_tickers = {ticker for ticker, count in ticker_counts.items() if count >= median_threshold}
excluded_tickers = {ticker for ticker, count in ticker_counts.items() if 0 < count < median_threshold}

print(f"✓ {len(valid_tickers)} stocks meet threshold (>= {median_threshold:.0f} posts)")
print(f"✗ {len(excluded_tickers)} stocks excluded (< {median_threshold:.0f} posts)")

# Filter posts to keep only those that mention at least one valid ticker
original_count = len(filtered_posts)
filtered_posts = [
    post for post in filtered_posts 
    if any(ticker in valid_tickers for ticker in post['mentioned_tickers'])
]

# Update each post's mentioned_tickers to include only valid tickers
for post in filtered_posts:
    post['mentioned_tickers'] = [t for t in post['mentioned_tickers'] if t in valid_tickers]

# Recalculate ticker counts after filtering
ticker_counts = {ticker: 0 for ticker in valid_tickers}
for post in filtered_posts:
    for ticker in post['mentioned_tickers']:
        ticker_counts[ticker] += 1

print(f"✓ {len(filtered_posts):,} posts remain after threshold ({original_count - len(filtered_posts)} removed)")

# Save the filtered posts as a new JSON file to the data directory for further analysis, such as for sentiment scoring
output_file = "data/reddit_posts_q2_2023_filtered.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_posts, f, indent=2, ensure_ascii=False)

print(f"✓ Filtered data saved to {output_file}")

# Summary statistics of filtered data
print("\n" + "=" * 60)
print("Posts mentioning each stock:")
print("=" * 60)

# Print counts and percentages for each ticker
for ticker, count in sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        percentage = (count / len(filtered_posts)) * 100
        print(f"  {ticker}: {count:,} posts ({percentage:.1f}%)")

print("\n" + "=" * 60)
print("Posts by subreddit:")
print("=" * 60)

# Count posts per subreddit in the filtered dataset
subreddit_counts = {}
for post in filtered_posts:
    sub = post['subreddit']
    subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1

# Print counts and percentages for each subreddit
for sub, count in sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(filtered_posts)) * 100
    print(f"  r/{sub}: {count:,} posts ({percentage:.1f}%)")

print("\n✓ Filtering complete!")