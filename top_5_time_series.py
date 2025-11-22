# top_5_time_series.py: This file generates daily price time-series plots for the five 
#                       most-mentioned tickers on Reddit during Q2 2023. It uses yfinance 
#                       to fetch price data and creates a multi-panel visualization.
# Requires filter_posts.py to be run first

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