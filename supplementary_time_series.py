# supplementary_time_series.py: This file generates daily price time-series plots for stocks
#                              beyond the top 5 most-mentioned on Reddit during Q2 2023.
#                              Uses a grid layout for compact visualization.
# Requires filter_posts.py to be run first

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import os

# Creating directories if they don't exist
os.makedirs('images/time_series', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load the filtered Reddit posts
input_file = "./data/reddit_posts_q2_2023_filtered.json"
print(f"Loading filtered posts from {input_file}...")
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    print(f"Loaded {len(posts):,} posts\n")
except FileNotFoundError:
    print(f"Error: {input_file} not found!")
    print("Run filter_posts.py first to create the filtered dataset.")
    exit(1)

# Count mentions of each ticker
mention_counts = {}
for post in posts:
    for ticker in post.get("mentioned_tickers", []):
        mention_counts[ticker] = mention_counts.get(ticker, 0) + 1

# Sort by mention count and separate top 5 from rest
sorted_tickers = sorted(mention_counts, key=mention_counts.get, reverse=True)
top5 = sorted_tickers[:5]
supplementary = sorted_tickers[5:]

print(f"Top 5 (excluded): {top5}")
print(f"Supplementary stocks ({len(supplementary)}): {supplementary}\n")

# Set date range for Q2 and Q3 2023
start_date = "2023-04-01"
end_date = "2023-09-30"

# Fetch daily closing price data for supplementary tickers
print("Fetching daily closing price data from yfinance...")
data = yf.download(supplementary, start=start_date, end=end_date, progress=False)
df_close = data["Close"]

# Save the fetched data to CSV
df_close.to_csv("./data/supplementary_price_history_q2_q3_2023.csv")
print("Saved -> supplementary_price_history_q2_q3_2023.csv\n")

# Create grid layout (3 rows x 6 columns for 18 stocks)
n_stocks = len(supplementary)
n_cols = 6
n_rows = (n_stocks + n_cols - 1) // n_cols  # Ceiling division

print(f"Creating {n_rows}x{n_cols} grid layout...")
fig, axes = plt.subplots(
    nrows=n_rows,
    ncols=n_cols,
    figsize=(20, 10),
    sharex=True
)

plt.style.use("ggplot")

q2_start = datetime(2023, 4, 1)
q3_start = datetime(2023, 7, 1)

# Flatten axes for easy iteration
axes_flat = axes.flatten()

# Plot each ticker
for i, ticker in enumerate(supplementary):
    ax = axes_flat[i]

    ax.plot(df_close.index, df_close[ticker], linewidth=1.2, color='steelblue')

    # Vertical markers for Q2 and Q3 starts
    ax.axvline(q2_start, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(q3_start, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_title(ticker, fontsize=10, fontweight='bold')
    ax.tick_params(axis='both', labelsize=7)

    # Only show y-label on leftmost column
    if i % n_cols == 0:
        ax.set_ylabel("Price", fontsize=8)

# Hide any unused subplots (if grid has empty cells)
for j in range(len(supplementary), len(axes_flat)):
    axes_flat[j].set_visible(False)

# Add shared x-label
fig.text(0.5, 0.02, 'Date', ha='center', fontsize=12)

# Main title
fig.suptitle('Supplementary Stocks: Daily Closing Prices (Q2 + Q3 2023)',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure
output_path = "./images/time_series/supplementary_time_series.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved PNG figure -> {output_path}")

plt.show()
