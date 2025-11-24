# time_series_analysis.py: weekly lagged regression analysis for top 5 tickers, tests whether
#                          sentiment predicts next week's returns and vice versa
# requires filter_posts.py and sentiment_analysis.py to be run first

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression
import yfinance as yf
import os

# create directories if they don't exist
os.makedirs('images/figures/time_series', exist_ok=True)
os.makedirs('images/diagnostics/time_series', exist_ok=True)
os.makedirs('data', exist_ok=True)

# plot styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# color scheme for visualizations
COLORS = {
    'positive': '#2ecc71',
    'negative': '#e74c3c',
    'neutral': '#95a5a6',
    'returns': '#3498db',
    'sentiment': '#e67e22',
    'regression': '#2c3e50',
}

print("Weekly lagged regression: top 5 tickers")
print("Testing week-to-week predictive relationships")

# load data

input_file = "./data/reddit_posts_q2_2023_filtered.json"
print(f"\nLoading filtered posts from {input_file}...")
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    print(f"Loaded {len(posts):,} posts")
except FileNotFoundError:
    print(f"Error: {input_file} not found!")
    exit(1)

# get top 5 most mentioned tickers
mention_counts = {}
for post in posts:
    for ticker in post.get("mentioned_tickers", []):
        mention_counts[ticker] = mention_counts.get(ticker, 0) + 1

top5 = sorted(mention_counts, key=mention_counts.get, reverse=True)[:5]
print(f"Top 5 tickers: {top5}")

# load sentiment data
sentiment_file = "./data/reddit_posts_q2_2023_with_sentiment.json"
print(f"Loading sentiment data from {sentiment_file}...")
try:
    with open(sentiment_file, 'r', encoding='utf-8') as f:
        sentiment_posts = json.load(f)
    print(f"Loaded {len(sentiment_posts):,} posts with sentiment")
except FileNotFoundError:
    print(f"Error: {sentiment_file} not found!")
    exit(1)

# build weekly sentiment data for top 5

print("\nBuilding weekly sentiment data")

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
weekly_sentiment.columns = ['ticker', 'week', 'sentiment']
weekly_sentiment = weekly_sentiment.dropna()

print(f"Weekly sentiment observations: {len(weekly_sentiment)}")
for ticker in top5:
    n = len(weekly_sentiment[weekly_sentiment['ticker'] == ticker])
    print(f"  {ticker}: {n} weeks")

# fetch weekly price data

print("\nFetching weekly price data")

start_date = "2023-04-01"
end_date = "2023-06-30"

# fetch daily data and resample to weekly
print("Fetching price data...")
price_data = yf.download(top5, start=start_date, end=end_date, progress=False)

# handle MultiIndex columns
if isinstance(price_data.columns, pd.MultiIndex):
    df_close = price_data['Close']
else:
    df_close = price_data[['Close']]
    df_close.columns = top5

# resample to weekly and compute percent change
price_weekly = df_close.resample('W-MON').last()
price_pct_change = price_weekly.pct_change() * 100

# fetch SPY for excess returns
print("Fetching SPY benchmark...")
spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
if isinstance(spy_data.columns, pd.MultiIndex):
    spy_close = spy_data['Close']['SPY']
else:
    spy_close = spy_data['Close']
spy_weekly = spy_close.resample('W-MON').last()
spy_pct_change = spy_weekly.pct_change() * 100

# fetch XLK sector ETF
print("Fetching XLK (Tech sector ETF)...")
xlk_data = yf.download('XLK', start=start_date, end=end_date, progress=False)
if isinstance(xlk_data.columns, pd.MultiIndex):
    xlk_close = xlk_data['Close']['XLK']
else:
    xlk_close = xlk_data['Close']
xlk_weekly = xlk_close.resample('W-MON').last()
xlk_pct_change = xlk_weekly.pct_change() * 100

print("Price data fetched")

# build panel dataset with lagged variables

print("\nBuilding panel dataset with lags")

panel_data = []

for ticker in top5:
    ticker_sent = weekly_sentiment[weekly_sentiment['ticker'] == ticker].copy()
    ticker_sent = ticker_sent.sort_values('week').reset_index(drop=True)

    ticker_pct = price_pct_change[ticker].dropna()
    spy_pct = spy_pct_change.dropna()
    xlk_pct = xlk_pct_change.dropna()

    for i, row in ticker_sent.iterrows():
        week = row['week']
        sentiment = row['sentiment']

        # get returns for this week
        if week in ticker_pct.index:
            raw_return = ticker_pct.loc[week]
            if isinstance(raw_return, pd.Series):
                raw_return = raw_return.iloc[0]

            spy_ret = spy_pct.loc[week] if week in spy_pct.index else np.nan
            if isinstance(spy_ret, pd.Series):
                spy_ret = spy_ret.iloc[0]

            xlk_ret = xlk_pct.loc[week] if week in xlk_pct.index else np.nan
            if isinstance(xlk_ret, pd.Series):
                xlk_ret = xlk_ret.iloc[0]

            excess_spy = raw_return - spy_ret if not np.isnan(spy_ret) else np.nan
            excess_sector = raw_return - xlk_ret if not np.isnan(xlk_ret) else np.nan
        else:
            raw_return = np.nan
            excess_spy = np.nan
            excess_sector = np.nan

        panel_data.append({
            'ticker': ticker,
            'week': week,
            'sentiment': sentiment,
            'return_raw': raw_return,
            'return_excess_spy': excess_spy,
            'return_excess_sector': excess_sector
        })

df = pd.DataFrame(panel_data)
df = df.sort_values(['ticker', 'week']).reset_index(drop=True)

# create lagged variables within each ticker
df['sentiment_lag1'] = df.groupby('ticker')['sentiment'].shift(1)
df['return_raw_lag1'] = df.groupby('ticker')['return_raw'].shift(1)
df['return_excess_spy_lag1'] = df.groupby('ticker')['return_excess_spy'].shift(1)
df['return_excess_sector_lag1'] = df.groupby('ticker')['return_excess_sector'].shift(1)

# drop rows with NaN
df_clean = df.dropna()

print(f"Panel dataset: {len(df_clean)} observations (5 tickers x ~12 weeks)")

# save the panel data
df_clean.to_csv('./data/weekly_panel_top5.csv', index=False)
print("Saved: data/weekly_panel_top5.csv")

# simple OLS regression

def run_regression(X, y, label):
    """simple OLS regression with standard t-test"""
    n = len(y)

    # pearson correlation
    pearson_r, pearson_p = stats.pearsonr(X, y)
    spearman_r, spearman_p = stats.spearmanr(X, y)

    # fit OLS
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)

    coef = model.coef_[0]
    intercept = model.intercept_
    y_pred = model.predict(X.reshape(-1, 1))
    residuals = y - y_pred

    # r-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - (ss_res / ss_tot)

    # standard error of coefficient
    mse = ss_res / (n - 2)
    x_ss = np.sum((X - X.mean())**2)
    se = np.sqrt(mse / x_ss)

    # t-stat and p-value
    t_stat = coef / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

    return {
        'label': label,
        'n': n,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'coefficient': coef,
        'intercept': intercept,
        'se': se,
        't_stat': t_stat,
        'p_value': p_value,
        'r_squared': r_squared
    }

print("\nLagged regressions: sentiment(t) -> returns(t+1)")
print("Does this week's sentiment predict next week's returns?")

results_sent_to_ret = []

return_cols = [
    ('return_raw', 'Raw Returns'),
    ('return_excess_spy', 'Excess vs SPY'),
    ('return_excess_sector', 'Excess vs Sector')
]

for return_col, return_label in return_cols:
    X = df_clean['sentiment_lag1'].values  # Last week's sentiment
    y = df_clean[return_col].values         # This week's returns

    result = run_regression(X, y, return_label)
    results_sent_to_ret.append(result)

    print(f"\n{return_label} (N = {result['n']}, df = {result['n']-2}):")
    print(f"  Pearson r = {result['pearson_r']:.4f}, p = {result['pearson_p']:.4f}")
    print(f"  Spearman r = {result['spearman_r']:.4f}, p = {result['spearman_p']:.4f}")
    print(f"  Coefficient = {result['coefficient']:.4f}, SE = {result['se']:.4f}")
    print(f"  t = {result['t_stat']:.4f}, p = {result['p_value']:.4f}")
    print(f"  R2 = {result['r_squared']:.4f}")

    if result['p_value'] < 0.05:
        print(f"  SIGNIFICANT at alpha = 0.05")
    else:
        print(f"  NOT significant at alpha = 0.05")

print("\nLagged regressions: returns(t) -> sentiment(t+1)")
print("Do this week's returns predict next week's sentiment?")

results_ret_to_sent = []

for return_col, return_label in return_cols:
    lag_col = return_col + '_lag1'
    X = df_clean[lag_col].values           # Last week's returns
    y = df_clean['sentiment'].values        # This week's sentiment

    result = run_regression(X, y, return_label)
    results_ret_to_sent.append(result)

    print(f"\n{return_label} (N = {result['n']}, df = {result['n']-2}):")
    print(f"  Pearson r = {result['pearson_r']:.4f}, p = {result['pearson_p']:.4f}")
    print(f"  Spearman r = {result['spearman_r']:.4f}, p = {result['spearman_p']:.4f}")
    print(f"  Coefficient = {result['coefficient']:.6f}, SE = {result['se']:.6f}")
    print(f"  t = {result['t_stat']:.4f}, p = {result['p_value']:.4f}")
    print(f"  R2 = {result['r_squared']:.4f}")

    if result['p_value'] < 0.05:
        print(f"  SIGNIFICANT at alpha = 0.05")
    else:
        print(f"  NOT significant at alpha = 0.05")

# save results

all_results = []
for r in results_sent_to_ret:
    all_results.append({
        'Direction': 'Sentiment(t) -> Returns(t+1)',
        'Return Type': r['label'],
        'N': r['n'],
        'Pearson_r': r['pearson_r'],
        'Pearson_p': r['pearson_p'],
        'Spearman_r': r['spearman_r'],
        'Spearman_p': r['spearman_p'],
        'Coefficient': r['coefficient'],
        'SE': r['se'],
        't_stat': r['t_stat'],
        'p_value': r['p_value'],
        'R2': r['r_squared']
    })
for r in results_ret_to_sent:
    all_results.append({
        'Direction': 'Returns(t) -> Sentiment(t+1)',
        'Return Type': r['label'],
        'N': r['n'],
        'Pearson_r': r['pearson_r'],
        'Pearson_p': r['pearson_p'],
        'Spearman_r': r['spearman_r'],
        'Spearman_p': r['spearman_p'],
        'Coefficient': r['coefficient'],
        'SE': r['se'],
        't_stat': r['t_stat'],
        'p_value': r['p_value'],
        'R2': r['r_squared']
    })

results_df = pd.DataFrame(all_results)
results_df.to_csv('./data/weekly_lagged_regression_results.csv', index=False)
print("\nSaved: data/weekly_lagged_regression_results.csv")

# visualizations

print("\nCreating visualizations")

# scatter plots: sentiment(t) -> returns(t+1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

model_colors = [COLORS['returns'], COLORS['positive'], COLORS['negative']]

for i, (return_col, return_label) in enumerate(return_cols):
    ax = axes[i]

    X = df_clean['sentiment_lag1'].values
    y = df_clean[return_col].values
    tickers = df_clean['ticker'].values

    # plot points
    ax.scatter(X, y, s=60, color=model_colors[i])

    # regression line
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, color=COLORS['regression'], linewidth=2, linestyle='--')

    result = results_sent_to_ret[i]
    ax.set_xlabel('Sentiment (week t)', fontsize=11)
    ax.set_ylabel(f'{return_label} % (week t+1)', fontsize=11)
    ax.set_title(f'Sentiment -> {return_label}\nr = {result["pearson_r"]:.3f}, p = {result["p_value"]:.3f}',
                 fontsize=11, fontweight='bold')
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color=COLORS['neutral'], linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./images/figures/time_series/weekly_sentiment_to_returns.png', dpi=300, bbox_inches='tight')
print("Saved: images/figures/time_series/weekly_sentiment_to_returns.png")
plt.close()

# scatter plots: returns(t) -> sentiment(t+1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

model_colors = [COLORS['returns'], COLORS['positive'], COLORS['negative']]

for i, (return_col, return_label) in enumerate(return_cols):
    ax = axes[i]

    lag_col = return_col + '_lag1'
    X = df_clean[lag_col].values
    y = df_clean['sentiment'].values

    # plot points
    ax.scatter(X, y, s=60, color=model_colors[i])

    # regression line
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, color=COLORS['regression'], linewidth=2, linestyle='--')

    result = results_ret_to_sent[i]
    ax.set_xlabel(f'{return_label} % (week t)', fontsize=11)
    ax.set_ylabel('Sentiment (week t+1)', fontsize=11)
    ax.set_title(f'{return_label} -> Sentiment\nr = {result["pearson_r"]:.3f}, p = {result["p_value"]:.3f}',
                 fontsize=11, fontweight='bold')
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color=COLORS['neutral'], linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./images/figures/time_series/weekly_returns_to_sentiment.png', dpi=300, bbox_inches='tight')
print("Saved: images/figures/time_series/weekly_returns_to_sentiment.png")
plt.close()

# p-value comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))

labels = ['Raw Returns', 'Excess vs SPY', 'Excess vs Sector']
x = np.arange(len(labels))
width = 0.35

sent_to_ret_p = [r['p_value'] for r in results_sent_to_ret]
ret_to_sent_p = [r['p_value'] for r in results_ret_to_sent]

# sentiment -> returns: solid bars
bars1 = ax.bar(x - width/2, sent_to_ret_p, width,
               color=model_colors, edgecolor='black')
# returns -> sentiment: hatched bars
bars2 = ax.bar(x + width/2, ret_to_sent_p, width,
               color=model_colors, edgecolor='black', hatch='//')

ax.axhline(y=0.05, color=COLORS['regression'], linestyle='--', linewidth=2, label='α = 0.05')

# custom legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='white', edgecolor='black', label='Sentiment → Returns'),
    Patch(facecolor='white', edgecolor='black', hatch='//', label='Returns → Sentiment'),
    Line2D([0], [0], color=COLORS['regression'], linestyle='--', linewidth=2, label='α = 0.05')
]

ax.set_xlabel('Return Type', fontsize=12)
ax.set_ylabel('P-Value', fontsize=12)
ax.set_title('Weekly Lagged Regression P-Values', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(handles=legend_elements)
ax.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}',
            ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('./images/diagnostics/time_series/weekly_pvalue_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: images/diagnostics/time_series/weekly_pvalue_comparison.png")
plt.close()

# summary

print("\nSummary of findings")

print("\n1. Sentiment(t) -> Returns(t+1)")
print("   Does this week's sentiment predict next week's returns?")
for r in results_sent_to_ret:
    sig = "SIGNIFICANT" if r['p_value'] < 0.05 else "NOT significant"
    print(f"   {r['label']}: r = {r['pearson_r']:.3f}, p = {r['p_value']:.3f} -> {sig}")

print("\n2. Returns(t) -> Sentiment(t+1)")
print("   Do this week's returns predict next week's sentiment?")
for r in results_ret_to_sent:
    sig = "SIGNIFICANT" if r['p_value'] < 0.05 else "NOT significant"
    print(f"   {r['label']}: r = {r['pearson_r']:.3f}, p = {r['p_value']:.3f} -> {sig}")

any_sig_sent = any(r['p_value'] < 0.05 for r in results_sent_to_ret)
any_sig_ret = any(r['p_value'] < 0.05 for r in results_ret_to_sent)

print("\nConclusion")

if any_sig_sent:
    print("\nWeekly sentiment DOES predict next week's returns")
else:
    print("\nWeekly sentiment does NOT predict next week's returns")

if any_sig_ret:
    print("Weekly returns DO predict next week's sentiment")
else:
    print("Weekly returns do NOT predict next week's sentiment")
