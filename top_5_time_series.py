# top_5_time_series.py: This file generates daily price time-series plots for the five
#                       most-mentioned tickers on Reddit during Q2 2023. It uses yfinance
#                       to fetch price data and creates a multi-panel visualization.
#                       Also generates weekly sentiment time series for the same stocks.
# Requires filter_posts.py and sentiment_analysis.py to be run first

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import os

# Creating a data and images directory if they don't exist for saving collected posts, keeping the output organized
os.makedirs('images/time_series', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load the filtered Reddit posts from a JSON file produced by filter_posts.py
input_file = "./data/reddit_posts_q2_2023_filtered.json"
print(f"Loading filtered posts from {input_file}...")
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    print(f"✓ Loaded {len(posts):,} posts\n")
except FileNotFoundError:
    print(f"✗ Error: {input_file} not found!")
    print("Run filter_posts.py first to create the filtered dataset.")
    exit(1)

# Count mentions of each ticker across all posts to identify the top 5 most mentioned ones
mention_counts = {}

# Get top 5 most mentioned
for post in posts:
    for ticker in post.get("mentioned_tickers", []):
        mention_counts[ticker] = mention_counts.get(ticker, 0) + 1

# Sort tickers by mention count and selects the top 5
top5 = sorted(mention_counts, key=mention_counts.get, reverse=True)[:5]
print("Top 5 most mentioned stocks:", top5)

# Set the date range for Q2 and Q3 2023
start_date = "2023-04-01"
end_date = "2023-09-30"

# Fetch daily closing price data for the top 5 tickers using yfinance
print("Fetching daily closing price data from yfinance...")
data = yf.download(top5, start=start_date, end=end_date)
df_close = data["Close"]

# Save the fetched data to a CSV file for reference and further analysis
df_close.to_csv("./data/top5_price_history_q2_q3_2023.csv")
print("Saved → top5_price_history_q2_q3_2023.csv")

# Create multi-panel time-series plots for the top 5 tickers
# Each subplot shows daily closing prices with vertical lines 
# marking the start of Q2 and Q3 2023
print("Creating multi-panel time-series plots...")
fig, axes = plt.subplots(
    nrows=len(top5),
    ncols=1,
    figsize=(12, 18),
    sharex=True
)

plt.style.use("ggplot")

q2_start = datetime(2023, 4, 1)
q3_start = datetime(2023, 7, 1)

# Plot each ticker's closing prices in its own subplot
for i, ticker in enumerate(top5):
    ax = axes[i]
    
    ax.plot(df_close.index, df_close[ticker], label=ticker, linewidth=1.8)

    # Vertical markers for Q2 and Q3 starts
    ax.axvline(q2_start, color="black", linestyle="--", linewidth=1)
    ax.axvline(q3_start, color="black", linestyle="--", linewidth=1)

    ax.set_ylabel("Price (USD)", fontsize=10)
    ax.set_title(f"{ticker} Daily Closing Price (Q2 + Q3 2023)", fontsize=12)
    ax.legend()

# Shared x-label
plt.xlabel("Date", fontsize=12)

# Tight layout
plt.tight_layout()

# Save the figure as a PNG file to the images directory
output_path = "./images/time_series/top_5_time_series.png"
plt.savefig(output_path, dpi=300)
print(f"Saved PNG figure → {output_path}")

plt.show()

# Load sentiment data (need the sentiment scores per post)
sentiment_file = "./data/reddit_posts_q2_2023_with_sentiment.json"
print(f"\nLoading sentiment data from {sentiment_file}...")
try:
    with open(sentiment_file, 'r', encoding='utf-8') as f:
        sentiment_posts = json.load(f)
    print(f"Loaded {len(sentiment_posts):,} posts with sentiment\n")
except FileNotFoundError:
    print(f"Error: {sentiment_file} not found!")
    print("Run sentiment_analysis.py first.")
    exit(1)

# Build a DataFrame with date, ticker, and sentiment for top 5 stocks
sentiment_records = []
for post in sentiment_posts:
    post_date = datetime.fromtimestamp(post['created_utc'])
    compound = post['sentiment']['compound']
    for ticker in post.get('mentioned_tickers', []):
        if ticker in top5:
            sentiment_records.append({
                'date': post_date,
                'ticker': ticker,
                'compound': compound
            })

sentiment_df = pd.DataFrame(sentiment_records)
sentiment_df = sentiment_df.set_index('date')

# Aggregate by week and ticker (using W-MON for consistent week boundaries)
weekly_sentiment = sentiment_df.groupby('ticker').resample('W-MON')['compound'].mean().reset_index()
weekly_sentiment.columns = ['ticker', 'week', 'compound']
weekly_sentiment = weekly_sentiment.dropna()

print(f"Weekly sentiment data points: {len(weekly_sentiment)}")

# ============================================================================
# OVERLAY: PRICE CHANGE + SENTIMENT ON SAME GRAPH (DUAL Y-AXIS)
# ============================================================================

print("\n" + "=" * 60)
print("GENERATING OVERLAY: PRICE + SENTIMENT")
print("=" * 60)

# Resample price to weekly (W-MON) and compute percent change
price_weekly = df_close.resample('W-MON').last()
price_pct_change = price_weekly.pct_change() * 100  # Convert to percentage

# Create multi-panel overlay plot
fig, axes = plt.subplots(
    nrows=len(top5),
    ncols=1,
    figsize=(14, 20),
    sharex=True
)

plt.style.use("ggplot")

for i, ticker in enumerate(top5):
    ax1 = axes[i]

    # Get weekly sentiment for this ticker
    ticker_sentiment = weekly_sentiment[weekly_sentiment['ticker'] == ticker].sort_values('week')

    # Get weekly price percent change, aligned to sentiment weeks
    ticker_pct = price_pct_change[ticker].dropna()
    # Only keep weeks that exist in both datasets
    common_weeks = ticker_pct.index.intersection(ticker_sentiment['week'])
    ticker_pct = ticker_pct.loc[common_weeks]
    ticker_sentiment = ticker_sentiment[ticker_sentiment['week'].isin(common_weeks)]

    # Plot price percent change on left y-axis
    ax1.plot(ticker_pct.index, ticker_pct,
             color='steelblue', linewidth=2, marker='o', markersize=4, label='Price % Change')
    ax1.axhline(y=0, color='steelblue', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Weekly Price Change (%)", fontsize=10, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    # Set symmetric y-limits so 0 is centered
    price_max = max(abs(ticker_pct.min()), abs(ticker_pct.max())) * 1.1
    ax1.set_ylim(-price_max, price_max)

    # Create second y-axis for sentiment
    ax2 = ax1.twinx()

    # Plot sentiment on right y-axis
    ax2.plot(ticker_sentiment['week'], ticker_sentiment['compound'],
             color='darkorange', linewidth=2, marker='o', markersize=4, label='Sentiment')
    ax2.set_ylabel("Sentiment", fontsize=10, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color='darkorange', linestyle='--', linewidth=0.8, alpha=0.5)

    # Title and legend
    ax1.set_title(f"{ticker}: Weekly Price Change vs Weekly Sentiment (Q2 2023)", fontsize=12)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Shared x-label
plt.xlabel("Date", fontsize=12)

plt.tight_layout()

# Save the overlay figure
overlay_output_path = "./images/time_series/top_5_price_sentiment_overlay.png"
plt.savefig(overlay_output_path, dpi=300)
print(f"Saved PNG figure → {overlay_output_path}")

plt.show()