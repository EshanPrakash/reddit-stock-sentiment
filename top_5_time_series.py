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
os.makedirs('images', exist_ok=True)
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
output_path = "./images/top_5_time_series.png"
plt.savefig(output_path, dpi=300)
print(f"Saved PNG figure → {output_path}")

plt.show()

# ============================================================================
# WEEKLY SENTIMENT TIME SERIES FOR TOP 5 STOCKS
# ============================================================================

print("\n" + "=" * 60)
print("GENERATING WEEKLY SENTIMENT TIME SERIES")
print("=" * 60)

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

# Create multi-panel sentiment time series plot
print("Creating weekly sentiment time series plots...")
fig, axes = plt.subplots(
    nrows=len(top5),
    ncols=1,
    figsize=(12, 18),
    sharex=True
)

plt.style.use("ggplot")

# Color mapping for each ticker
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

for i, ticker in enumerate(top5):
    ax = axes[i]
    ticker_data = weekly_sentiment[weekly_sentiment['ticker'] == ticker].sort_values('week')

    ax.plot(ticker_data['week'], ticker_data['compound'],
            marker='o', linewidth=2, markersize=6, color=colors[i], label=ticker)
    ax.fill_between(ticker_data['week'], 0, ticker_data['compound'],
                    alpha=0.3, color=colors[i])

    # Reference line at 0 (neutral sentiment)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_ylabel("Avg Sentiment", fontsize=10)
    ax.set_title(f"{ticker} Weekly Average Sentiment (Q2 2023)", fontsize=12)
    ax.set_ylim(-1, 1)
    ax.legend(loc='upper right')

# Shared x-label
plt.xlabel("Week", fontsize=12)

plt.tight_layout()

# Save the sentiment time series figure
sentiment_output_path = "./images/top_5_sentiment_time_series.png"
plt.savefig(sentiment_output_path, dpi=300)
print(f"Saved PNG figure → {sentiment_output_path}")

plt.show()

# ============================================================================
# OVERLAY: PRICE + SENTIMENT ON SAME GRAPH (DUAL Y-AXIS)
# ============================================================================

print("\n" + "=" * 60)
print("GENERATING OVERLAY: PRICE + SENTIMENT")
print("=" * 60)

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

    # Get price data (Q2 only to match sentiment period)
    price_q2 = df_close[ticker].loc[df_close.index < '2023-07-01']

    # Plot price on left y-axis
    line1 = ax1.plot(price_q2.index, price_q2,
                     color='steelblue', linewidth=2, label='Price')
    ax1.set_ylabel("Price (USD)", fontsize=10, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Create second y-axis for sentiment
    ax2 = ax1.twinx()

    # Get weekly sentiment for this ticker
    ticker_sentiment = weekly_sentiment[weekly_sentiment['ticker'] == ticker].sort_values('week')

    # Plot sentiment on right y-axis
    line2 = ax2.plot(ticker_sentiment['week'], ticker_sentiment['compound'],
                     color='darkorange', linewidth=2, marker='o', markersize=5, label='Sentiment')
    ax2.fill_between(ticker_sentiment['week'], 0, ticker_sentiment['compound'],
                     alpha=0.2, color='darkorange')
    ax2.set_ylabel("Sentiment", fontsize=10, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color='darkorange', linestyle='--', linewidth=0.8, alpha=0.5)

    # Title and legend
    ax1.set_title(f"{ticker}: Price vs Weekly Sentiment (Q2 2023)", fontsize=12)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Shared x-label
plt.xlabel("Date", fontsize=12)

plt.tight_layout()

# Save the overlay figure
overlay_output_path = "./images/top_5_price_sentiment_overlay.png"
plt.savefig(overlay_output_path, dpi=300)
print(f"Saved PNG figure → {overlay_output_path}")

plt.show()

# ============================================================================
# Z-SCORE DIFFERENCE: NORMALIZED PRICE vs SENTIMENT
# ============================================================================

print("\n" + "=" * 60)
print("GENERATING Z-SCORE DIFFERENCE TIME SERIES")
print("=" * 60)

# Function to compute z-score
def zscore(series):
    return (series - series.mean()) / series.std()

# Create multi-panel z-score difference plot
fig, axes = plt.subplots(
    nrows=len(top5),
    ncols=1,
    figsize=(14, 20),
    sharex=True
)

plt.style.use("ggplot")

for i, ticker in enumerate(top5):
    ax = axes[i]

    # Get weekly price data (resample daily to weekly, using Monday as week start)
    price_q2 = df_close[ticker].loc[df_close.index < '2023-07-01']
    weekly_price = price_q2.resample('W-MON').mean()

    # Get weekly sentiment for this ticker (already aggregated with W-MON)
    ticker_sent_data = weekly_sentiment[weekly_sentiment['ticker'] == ticker].copy()
    weekly_sent_series = ticker_sent_data.set_index('week')['compound']

    # Align the two series by their common dates
    common_weeks = weekly_price.index.intersection(weekly_sent_series.index)
    if len(common_weeks) < 2:
        ax.text(0.5, 0.5, f"{ticker}: Insufficient data", ha='center', va='center', transform=ax.transAxes)
        continue

    price_aligned = weekly_price.loc[common_weeks]
    sentiment_aligned = weekly_sent_series.loc[common_weeks]

    # Compute z-scores
    price_z = zscore(price_aligned)
    sentiment_z = zscore(sentiment_aligned)

    # Z-score difference: sentiment - price
    # Positive = sentiment ahead of price, Negative = price ahead of sentiment
    z_diff = sentiment_z - price_z

    # Plot z-scores and difference
    ax.plot(common_weeks, price_z, color='steelblue', linewidth=2, label='Price (z)')
    ax.plot(common_weeks, sentiment_z, color='darkorange', linewidth=2, label='Sentiment (z)')
    ax.bar(common_weeks, z_diff, width=5, alpha=0.4, color='green', label='Difference (Sent - Price)')

    # Reference line at 0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_ylabel("Z-Score", fontsize=10)
    ax.set_title(f"{ticker}: Z-Score Comparison (Q2 2023)", fontsize=12)
    ax.legend(loc='upper left', fontsize=9)

# Shared x-label
plt.xlabel("Week", fontsize=12)

plt.tight_layout()

# Save the z-score figure
zscore_output_path = "./images/top_5_zscore_difference.png"
plt.savefig(zscore_output_path, dpi=300)
print(f"Saved PNG figure → {zscore_output_path}")

plt.show()

print("\n" + "=" * 60)
print("COMPLETE!")
print("=" * 60)