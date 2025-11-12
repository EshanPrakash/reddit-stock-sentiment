import json
import re

# Stock tickers and their company names
STOCK_INFO = {
    'AAPL': ['Apple', 'AAPL', '$AAPL'],
    'TSLA': ['Tesla', 'TSLA', '$TSLA'],
    'NVDA': ['Nvidia', 'NVDA', '$NVDA'],
    'MSFT': ['Microsoft', 'MSFT', '$MSFT'],
    'AMZN': ['Amazon', 'AMZN', '$AMZN'],
    'GOOGL': ['Google', 'Alphabet', 'GOOGL', '$GOOGL'],
    'META': ['Meta', 'Facebook', 'META', '$META'],
    'AMD': ['AMD', '$AMD'],  # "AMD" is already the company name
    'NFLX': ['Netflix', 'NFLX', '$NFLX'],
    'SPY': ['SPY', '$SPY', 'S&P 500', 'S&P']
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
            # Create regex pattern - word boundary for most, but flexible
            # Use word boundaries to avoid matching substrings
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            
            if re.search(pattern, text_lower):
                mentioned.append(ticker)
                break  # Found this ticker, move to next one
    
    return list(set(mentioned))  # Return unique tickers

# Load the collected data
input_file = "reddit_posts_q2_2023_full.json"
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
    # Combine title and selftext for searching
    combined_text = f"{post.get('title', '')} {post.get('selftext', '')}"
    
    # Find mentioned tickers
    mentioned_tickers = find_mentioned_tickers(combined_text)
    
    if mentioned_tickers:
        # Add the tickers to the post data
        post['mentioned_tickers'] = mentioned_tickers
        filtered_posts.append(post)
        
        # Count occurrences
        for ticker in mentioned_tickers:
            ticker_counts[ticker] += 1

print(f"✓ Found {len(filtered_posts):,} posts mentioning our target stocks")
print(f"  ({len(filtered_posts)/len(all_posts)*100:.1f}% of total posts)")

# Save filtered data
output_file = "reddit_posts_q2_2023_filtered.json"
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