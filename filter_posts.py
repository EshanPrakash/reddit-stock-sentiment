# filter_posts.py: filters collected reddit posts to identify those mentioning specific stock
#                  tickers and company names, applies median threshold to focus on stocks with
#                  significant discussion volume
# requires collect_pullpush.py to be run first

import json
import re
import os
import statistics

# create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# stock tickers and company names - 50 stocks across 5 sectors
# maps tickers to keywords used for identifying mentions in posts
STOCK_INFO = {
    # technology (15 stocks)
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
    
    # finance (10 stocks)
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
    
    # healthcare (10 stocks)
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
    
    # energy (10 stocks)
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
    
    # aerospace/defense (5 stocks)
    'LMT': ['Lockheed Martin', 'Lockheed', 'LMT', '$LMT'],
    'RTX': ['Raytheon', 'RTX', '$RTX'],
    'BA': ['Boeing', '$BA'],
    'NOC': ['Northrop Grumman', 'Northrop', 'NOC', '$NOC'],
    'GD': ['General Dynamics', '$GD'],
}

# finds mentioned tickers in post text, returns list of unique ticker symbols
def find_mentioned_tickers(text):
    # handle empty text
    if not text:
        return []
    
    text_lower = text.lower()
    mentioned = []

    for ticker, keywords in STOCK_INFO.items():
        for keyword in keywords:
            # match whole words only to prevent partial matches
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text_lower):
                mentioned.append(ticker)
                break

    return list(set(mentioned))

# load collected reddit posts
input_file = "data/reddit_posts_q2_2023_full.json"
print(f"Loading data from {input_file}...")

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        all_posts = json.load(f)
    print(f"Loaded {len(all_posts):,} posts")
except FileNotFoundError:
    print(f"Error: {input_file} not found!")
    print("Run collect_pullpush.py first to collect the data.")
    exit(1)

print("\nFiltering posts that mention our target stocks...")

# first stage filtering: identify posts mentioning target stocks
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

print(f"Found {len(filtered_posts):,} posts mentioning our target stocks")
print(f"  ({len(filtered_posts)/len(all_posts)*100:.1f}% of total posts)")

# second stage filtering: apply median threshold to focus on high-discussion stocks
post_counts = [count for count in ticker_counts.values() if count > 0]
if not post_counts:
    print("\nNo stocks mentioned! Check your data.")
    exit(1)

# calculate statistics
median_threshold = statistics.median(post_counts)
print(f"\nPost count distribution:")
print(f"  Median: {median_threshold:.0f}")
print(f"  Mean: {statistics.mean(post_counts):.1f}")
print(f"  Min: {min(post_counts)}")
print(f"  Max: {max(post_counts)}")

# apply median threshold
print(f"\nApplying threshold: posts >= median ({median_threshold:.0f})...")

valid_tickers = {ticker for ticker, count in ticker_counts.items() if count >= median_threshold}
excluded_tickers = {ticker for ticker, count in ticker_counts.items() if 0 < count < median_threshold}

print(f"{len(valid_tickers)} stocks meet threshold (>= {median_threshold:.0f} posts)")
print(f"{len(excluded_tickers)} stocks excluded (< {median_threshold:.0f} posts)")

# filter posts to keep only those mentioning valid tickers
original_count = len(filtered_posts)
filtered_posts = [
    post for post in filtered_posts 
    if any(ticker in valid_tickers for ticker in post['mentioned_tickers'])
]

# update each post's mentioned_tickers to include only valid tickers
for post in filtered_posts:
    post['mentioned_tickers'] = [t for t in post['mentioned_tickers'] if t in valid_tickers]

# recalculate ticker counts after filtering
ticker_counts = {ticker: 0 for ticker in valid_tickers}
for post in filtered_posts:
    for ticker in post['mentioned_tickers']:
        ticker_counts[ticker] += 1

print(f"{len(filtered_posts):,} posts remain after threshold ({original_count - len(filtered_posts)} removed)")

# save filtered posts
output_file = "data/reddit_posts_q2_2023_filtered.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_posts, f, indent=2, ensure_ascii=False)

print(f"Filtered data saved to {output_file}")

# summary statistics
print("\nPosts mentioning each stock:")

for ticker, count in sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        percentage = (count / len(filtered_posts)) * 100
        print(f"  {ticker}: {count:,} posts ({percentage:.1f}%)")

print("\nPosts by subreddit:")

subreddit_counts = {}
for post in filtered_posts:
    sub = post['subreddit']
    subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1

for sub, count in sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(filtered_posts)) * 100
    print(f"  r/{sub}: {count:,} posts ({percentage:.1f}%)")

print("\nFiltering complete!")