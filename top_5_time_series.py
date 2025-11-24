# top_5_time_series.py: generates price and sentiment time series plots for the five
#                       most-mentioned tickers on reddit during Q2 2023
# requires filter_posts.py and sentiment_analysis.py to be run first

import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import os

# create directories if they don't exist
os.makedirs('images/figures/time_series', exist_ok=True)
os.makedirs('data', exist_ok=True)

# color scheme for visualizations
COLORS = {
    'positive': '#2ecc71',
    'negative': '#e74c3c',
    'neutral': '#95a5a6',
    'returns': '#3498db',
    'sentiment': '#e67e22',
    'regression': '#2c3e50',
}

# load filtered reddit posts
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

# count mentions and get top 5
mention_counts = {}
for post in posts:
    for ticker in post.get("mentioned_tickers", []):
        mention_counts[ticker] = mention_counts.get(ticker, 0) + 1

# sort by mention count
top5 = sorted(mention_counts, key=mention_counts.get, reverse=True)[:5]
print("Top 5 most mentioned stocks:", top5)

# date range for Q2 and Q3 2023
start_date = "2023-04-01"
end_date = "2023-09-30"

# fetch daily closing price data
print("Fetching daily closing price data from yfinance...")
data = yf.download(top5, start=start_date, end=end_date)
df_close = data["Close"]

# save price data
df_close.to_csv("./data/top5_price_history_q2_q3_2023.csv")
print("Saved → top5_price_history_q2_q3_2023.csv")

# skip daily multi-panel price chart
print("Skipped generating daily multi-panel price chart (per request)")

# load sentiment data
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

# build dataframe with date, ticker, and sentiment
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

# aggregate by week and ticker
weekly_sentiment = sentiment_df.groupby('ticker').resample('W-MON')['compound'].mean().reset_index()
weekly_sentiment.columns = ['ticker', 'week', 'compound']
weekly_sentiment = weekly_sentiment.dropna()

print(f"Weekly sentiment data points: {len(weekly_sentiment)}")

# overlay: price change + sentiment

print("\nGenerating overlay: price + sentiment")

# resample price to weekly and compute percent change
price_weekly = df_close.resample('W-MON').last()
price_pct_change = price_weekly.pct_change() * 100

# create multi-panel overlay plot
fig, axes = plt.subplots(
    nrows=len(top5),
    ncols=1,
    figsize=(14, 20),
    sharex=True
)

plt.style.use("ggplot")

for i, ticker in enumerate(top5):
    ax1 = axes[i]

    # get weekly sentiment for this ticker
    ticker_sentiment = weekly_sentiment[weekly_sentiment['ticker'] == ticker].sort_values('week')

    # get weekly price percent change aligned to sentiment weeks
    ticker_pct = price_pct_change[ticker].dropna()
    common_weeks = ticker_pct.index.intersection(ticker_sentiment['week'])
    ticker_pct = ticker_pct.loc[common_weeks]
    ticker_sentiment = ticker_sentiment[ticker_sentiment['week'].isin(common_weeks)]

    # plot price percent change on left y-axis
    ax1.plot(ticker_pct.index, ticker_pct,
             color=COLORS['returns'], linewidth=2, marker='o', markersize=4, label='Price % Change')
    ax1.axhline(y=0, color=COLORS['returns'], linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Weekly Price Change (%)", fontsize=10, color=COLORS['returns'])
    ax1.tick_params(axis='y', labelcolor=COLORS['returns'])
    # set symmetric y-limits
    price_max = max(abs(ticker_pct.min()), abs(ticker_pct.max())) * 1.1
    ax1.set_ylim(-price_max, price_max)

    # create second y-axis for sentiment
    ax2 = ax1.twinx()

    # plot sentiment on right y-axis
    ax2.plot(ticker_sentiment['week'], ticker_sentiment['compound'],
             color=COLORS['regression'], linewidth=2, marker='o', markersize=4, label='Sentiment')
    ax2.set_ylabel("Sentiment", fontsize=10, color=COLORS['regression'])
    ax2.tick_params(axis='y', labelcolor=COLORS['regression'])
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color=COLORS['regression'], linestyle='--', linewidth=0.8, alpha=0.5)

    # title and legend
    ax1.set_title(f"{ticker}: Weekly Price Change vs Weekly Sentiment (Q2 2023)", fontsize=12)

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# shared x-label
plt.xlabel("Date", fontsize=12)

plt.tight_layout()

# save the overlay figure
overlay_output_path = "./images/figures/time_series/top_5_price_sentiment_overlay.png"
plt.savefig(overlay_output_path, dpi=300)
print(f"Saved PNG figure → {overlay_output_path}")

plt.show()

# overlay 2: excess returns vs SPY + sentiment

print("\nGenerating overlay: excess returns vs SPY + sentiment")

# fetch SPY benchmark data
print("Fetching SPY benchmark data...")
spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
# handle column index
if isinstance(spy_data.columns, pd.MultiIndex):
    spy_close = spy_data['Close']['SPY']
else:
    spy_close = spy_data['Close']

# resample SPY to weekly
spy_weekly = spy_close.resample('W-MON').last()
spy_pct_change = spy_weekly.pct_change() * 100

# create multi-panel overlay plot for excess returns vs SPY
fig, axes = plt.subplots(
    nrows=len(top5),
    ncols=1,
    figsize=(14, 20),
    sharex=True
)

plt.style.use("ggplot")

for i, ticker in enumerate(top5):
    ax1 = axes[i]

    # get weekly sentiment for this ticker
    ticker_sentiment = weekly_sentiment[weekly_sentiment['ticker'] == ticker].sort_values('week')

    # get weekly price percent change
    ticker_pct = price_pct_change[ticker].dropna()
    spy_pct = spy_pct_change.dropna()

    # calculate excess return vs SPY
    common_idx = ticker_pct.index.intersection(spy_pct.index)
    excess_vs_spy = ticker_pct.loc[common_idx] - spy_pct.loc[common_idx]

    # only keep weeks with sentiment data
    common_weeks = excess_vs_spy.index.intersection(ticker_sentiment['week'])
    excess_vs_spy = excess_vs_spy.loc[common_weeks]
    ticker_sentiment_filtered = ticker_sentiment[ticker_sentiment['week'].isin(common_weeks)]

    # plot excess return vs SPY
    ax1.plot(excess_vs_spy.index, excess_vs_spy,
             color=COLORS['positive'], linewidth=2, marker='o', markersize=4, label='Excess Return vs SPY')
    ax1.axhline(y=0, color=COLORS['positive'], linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Excess Return vs SPY (%)", fontsize=10, color=COLORS['positive'])
    ax1.tick_params(axis='y', labelcolor=COLORS['positive'])
    # set symmetric y-limits
    if len(excess_vs_spy) > 0:
        excess_max = max(abs(excess_vs_spy.min().item()), abs(excess_vs_spy.max().item())) * 1.1
        ax1.set_ylim(-excess_max, excess_max)

    # create second y-axis for sentiment
    ax2 = ax1.twinx()

    # plot sentiment on right y-axis
    ax2.plot(ticker_sentiment_filtered['week'], ticker_sentiment_filtered['compound'],
             color=COLORS['regression'], linewidth=2, marker='o', markersize=4, label='Sentiment')
    ax2.set_ylabel("Sentiment", fontsize=10, color=COLORS['regression'])
    ax2.tick_params(axis='y', labelcolor=COLORS['regression'])
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color=COLORS['regression'], linestyle='--', linewidth=0.8, alpha=0.5)

    # title and legend
    ax1.set_title(f"{ticker}: Weekly Excess Return vs SPY + Sentiment (Q2 2023)", fontsize=12)

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# shared x-label
plt.xlabel("Date", fontsize=12)

plt.tight_layout()

# save the overlay figure
excess_spy_output_path = "./images/figures/time_series/top_5_excess_spy_sentiment_overlay.png"
plt.savefig(excess_spy_output_path, dpi=300)
print(f"Saved PNG figure → {excess_spy_output_path}")

plt.show()

# overlay 3: excess returns vs sector + sentiment

print("\nGenerating overlay: excess returns vs sector + sentiment")

# sector ETF mappings
SECTOR_ETF = {
    'Tech': 'XLK',
    'Finance': 'XLF',
    'Healthcare': 'XLV',
    'Energy': 'XLE',
    'Aerospace/Defense': 'ITA'
}

# map tickers to sectors
SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'AMZN': 'Tech', 'META': 'Tech',
    'NVDA': 'Tech', 'TSLA': 'Tech', 'AMD': 'Tech', 'NFLX': 'Tech', 'INTC': 'Tech',
    'CRM': 'Tech', 'ORCL': 'Tech', 'ADBE': 'Tech', 'CSCO': 'Tech', 'UBER': 'Tech',
    'JPM': 'Finance', 'BAC': 'Finance', 'WFC': 'Finance', 'GS': 'Finance', 'MS': 'Finance',
    'C': 'Finance', 'V': 'Finance', 'MA': 'Finance', 'AXP': 'Finance', 'SCHW': 'Finance',
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
    'LLY': 'Healthcare', 'MRK': 'Healthcare', 'TMO': 'Healthcare', 'CVS': 'Healthcare',
    'AMGN': 'Healthcare', 'BMY': 'Healthcare',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy', 'EOG': 'Energy',
    'OXY': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy', 'HAL': 'Energy',
    'LMT': 'Aerospace/Defense', 'RTX': 'Aerospace/Defense', 'BA': 'Aerospace/Defense',
    'NOC': 'Aerospace/Defense', 'GD': 'Aerospace/Defense'
}

# fetch sector ETF data
sector_etfs_needed = set()
for ticker in top5:
    sector = SECTOR_MAP.get(ticker)
    if sector:
        sector_etfs_needed.add(SECTOR_ETF[sector])

print(f"Fetching sector ETF data: {sector_etfs_needed}")
sector_etf_data = {}
for etf in sector_etfs_needed:
    etf_data = yf.download(etf, start=start_date, end=end_date, progress=False)
    if not etf_data.empty:
        # handle column index
        if isinstance(etf_data.columns, pd.MultiIndex):
            etf_close = etf_data['Close'][etf]
        else:
            etf_close = etf_data['Close']
        sector_etf_data[etf] = etf_close.resample('W-MON').last().pct_change() * 100

# create multi-panel overlay plot for excess returns vs sector
fig, axes = plt.subplots(
    nrows=len(top5),
    ncols=1,
    figsize=(14, 20),
    sharex=True
)

plt.style.use("ggplot")

for i, ticker in enumerate(top5):
    ax1 = axes[i]

    # get weekly sentiment for this ticker
    ticker_sentiment = weekly_sentiment[weekly_sentiment['ticker'] == ticker].sort_values('week')

    # get weekly price percent change for stock
    ticker_pct = price_pct_change[ticker].dropna()

    # get sector ETF for this ticker
    sector = SECTOR_MAP.get(ticker)
    sector_etf = SECTOR_ETF.get(sector) if sector else None

    if sector_etf and sector_etf in sector_etf_data:
        sector_pct = sector_etf_data[sector_etf].dropna()

        # calculate excess return vs sector
        common_idx = ticker_pct.index.intersection(sector_pct.index)
        excess_vs_sector = ticker_pct.loc[common_idx] - sector_pct.loc[common_idx]

        # only keep weeks with sentiment data
        common_weeks = excess_vs_sector.index.intersection(ticker_sentiment['week'])
        excess_vs_sector = excess_vs_sector.loc[common_weeks]
        ticker_sentiment_filtered = ticker_sentiment[ticker_sentiment['week'].isin(common_weeks)]

        # plot excess return vs sector
        ax1.plot(excess_vs_sector.index, excess_vs_sector,
                 color=COLORS['negative'], linewidth=2, marker='o', markersize=4,
                 label=f'Excess Return vs {sector_etf}')
        ax1.axhline(y=0, color=COLORS['negative'], linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.set_ylabel(f"Excess Return vs {sector_etf} (%)", fontsize=10, color=COLORS['negative'])
        ax1.tick_params(axis='y', labelcolor=COLORS['negative'])
        # set symmetric y-limits
        if len(excess_vs_sector) > 0:
            excess_max = max(abs(excess_vs_sector.min().item()), abs(excess_vs_sector.max().item())) * 1.1
            ax1.set_ylim(-excess_max, excess_max)

        # create second y-axis for sentiment
        ax2 = ax1.twinx()

        # plot sentiment on right y-axis
        ax2.plot(ticker_sentiment_filtered['week'], ticker_sentiment_filtered['compound'],
                 color=COLORS['regression'], linewidth=2, marker='o', markersize=4, label='Sentiment')
        ax2.set_ylabel("Sentiment", fontsize=10, color=COLORS['regression'])
        ax2.tick_params(axis='y', labelcolor=COLORS['regression'])
        ax2.set_ylim(-1, 1)
        ax2.axhline(y=0, color=COLORS['regression'], linestyle='--', linewidth=0.8, alpha=0.5)

        # title and legend
        sector_label = f"{sector} ({sector_etf})" if sector else "Unknown"
        ax1.set_title(f"{ticker}: Weekly Excess Return vs {sector_label} + Sentiment (Q2 2023)", fontsize=12)

        # combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.set_title(f"{ticker}: No sector ETF data available", fontsize=12)

# shared x-label
plt.xlabel("Date", fontsize=12)

plt.tight_layout()

# save the overlay figure
excess_sector_output_path = "./images/figures/time_series/top_5_excess_sector_sentiment_overlay.png"
plt.savefig(excess_sector_output_path, dpi=300)
print(f"Saved PNG figure → {excess_sector_output_path}")

plt.show()

print("\nAll time series visualizations complete!")
print("\nGenerated files:")
print("  - images/figures/time_series/top_5_price_sentiment_overlay.png")
print("  - images/figures/time_series/top_5_excess_spy_sentiment_overlay.png")
print("  - images/figures/time_series/top_5_excess_sector_sentiment_overlay.png")