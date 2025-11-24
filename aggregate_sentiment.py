# aggregate_sentiment.py: aggregates sentiment scores by stock ticker for Q2 2023 reddit posts,
#                         computes statistics, and creates visualizations
# requires sentiment_analysis.py to be run first

import json
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os

# create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('images/figures/sentiment', exist_ok=True)

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

# colors for top 5 tickers in bar charts
TOP5_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#e67e22', '#9b59b6']

# custom colormap for sentiment gradients
SENTIMENT_CMAP = LinearSegmentedColormap.from_list(
    'sentiment',
    [COLORS['negative'], '#ffff00', COLORS['positive']]
)

# load reddit posts with sentiment scores
input_file = "data/reddit_posts_q2_2023_with_sentiment.json"
print(f"Loading sentiment data from {input_file}...")
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    print(f"Loaded {len(posts):,} posts with sentiment scores\n")
except FileNotFoundError:
    print(f"Error: {input_file} not found!")
    print("Run sentiment_analysis.py first.")
    exit(1)

print("Aggregating Q2 2023 sentiment by ticker...")

# aggregate sentiment scores and counts for each ticker
ticker_data = {}

for post in posts:
    for ticker in post.get('mentioned_tickers', []):
        if ticker not in ticker_data:
            ticker_data[ticker] = {
                'compound_scores': [],
                'post_count': 0,
                'positive_count': 0,
                'neutral_count': 0,
                'negative_count': 0
            }

        ticker_data[ticker]['compound_scores'].append(post['sentiment']['compound'])
        ticker_data[ticker]['post_count'] += 1
        label = post['sentiment_label']
        ticker_data[ticker][f'{label}_count'] += 1

# calculate statistics for each ticker
aggregated_results = []

for ticker in sorted(ticker_data.keys()):
    data = ticker_data[ticker]
    scores = data['compound_scores']
    
    result = {
        'ticker': ticker,
        'q2_2023_post_count': data['post_count'],
        'q2_2023_avg_sentiment': statistics.mean(scores),
        'q2_2023_median_sentiment': statistics.median(scores),
        'q2_2023_sentiment_stdev': statistics.stdev(scores) if len(scores) > 1 else 0,
        'positive_posts': data['positive_count'],
        'neutral_posts': data['neutral_count'],
        'negative_posts': data['negative_count'],
        'positive_ratio': data['positive_count'] / data['post_count']
    }
    aggregated_results.append(result)

# convert to dataframe and sort by sentiment
df = pd.DataFrame(aggregated_results)
df = df.sort_values('q2_2023_avg_sentiment', ascending=False)

print("\nQ2 2023 sentiment aggregation by ticker")
print(df.to_string(index=False))

# save results
csv_file = "data/q2_2023_sentiment_by_ticker.csv"
df.to_csv(csv_file, index=False)
print(f"\nSaved aggregated sentiment to {csv_file}")

json_file = "data/q2_2023_sentiment_by_ticker.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(aggregated_results, f, indent=2)
print(f"Saved aggregated sentiment to {json_file}")

# summary statistics
print("\nSummary")

print(f"\nTickers analyzed: {len(aggregated_results)}")
print(f"Total posts: {sum(r['q2_2023_post_count'] for r in aggregated_results)}")
print(f"\nMost positive ticker: {df.iloc[0]['ticker']} (avg: {df.iloc[0]['q2_2023_avg_sentiment']:.3f})")
print(f"Most negative ticker: {df.iloc[-1]['ticker']} (avg: {df.iloc[-1]['q2_2023_avg_sentiment']:.3f})")
print(f"\nMost discussed: {df.nlargest(1, 'q2_2023_post_count').iloc[0]['ticker']} ({df.nlargest(1, 'q2_2023_post_count').iloc[0]['q2_2023_post_count']} posts)")

print("\nNext step: Get Q3 2023 stock returns and correlate with Q2 sentiment!")

# create visualizations
print("\nCreating visualizations")

# bar chart of average sentiment by ticker
plt.figure(figsize=(12, 6))
colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in df['q2_2023_avg_sentiment']]
plt.barh(df['ticker'], df['q2_2023_avg_sentiment'], color=colors)
plt.xlabel('Average Sentiment Score', fontsize=12)
plt.ylabel('Stock Ticker', fontsize=12)
plt.title('Q2 2023 Reddit Sentiment by Stock Ticker', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig('images/figures/sentiment/sentiment_by_ticker.png', dpi=300, bbox_inches='tight')
print("Saved: images/figures/sentiment/sentiment_by_ticker.png")
plt.close()

# scatter plot: post count vs average sentiment
plt.figure(figsize=(10, 6))
plt.scatter(df['q2_2023_post_count'], df['q2_2023_avg_sentiment'],
            s=100, c=df['q2_2023_avg_sentiment'], cmap=SENTIMENT_CMAP)
for idx, row in df.iterrows():
    plt.annotate(row['ticker'], 
                (row['q2_2023_post_count'], row['q2_2023_avg_sentiment']),
                fontsize=9, ha='center')
plt.xlabel('Number of Posts', fontsize=12)
plt.ylabel('Average Sentiment Score', fontsize=12)
plt.title('Q2 2023 Sentiment vs Post Volume', fontsize=14, fontweight='bold')
plt.colorbar(label='Sentiment Score')
plt.tight_layout()
plt.savefig('images/figures/sentiment/sentiment_vs_volume.png', dpi=300, bbox_inches='tight')
print("Saved: images/figures/sentiment/sentiment_vs_volume.png")
plt.close()

# stacked bar chart of sentiment distribution
plt.figure(figsize=(12, 6))
sentiment_data = df.sort_values('q2_2023_post_count', ascending=False)[['ticker', 'positive_posts', 'neutral_posts', 'negative_posts']].set_index('ticker')
sentiment_data.plot(kind='barh', stacked=True,
                   color=[COLORS['positive'], COLORS['neutral'], COLORS['negative']],
                   figsize=(12, 6))
plt.xlabel('Number of Posts', fontsize=12)
plt.ylabel('Stock Ticker', fontsize=12)
plt.title('Q2 2023 Sentiment Distribution by Ticker', fontsize=14, fontweight='bold')
plt.legend(title='Sentiment', labels=['Positive', 'Neutral', 'Negative'])
plt.tight_layout()
plt.savefig('images/figures/sentiment/sentiment_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: images/figures/sentiment/sentiment_distribution.png")
plt.close()

# top 5 most discussed stocks
plt.figure(figsize=(10, 6))
top_5 = df.nlargest(5, 'q2_2023_post_count')
plt.bar(top_5['ticker'], top_5['q2_2023_post_count'], color=TOP5_COLORS)
plt.xlabel('Stock Ticker', fontsize=12)
plt.ylabel('Number of Posts', fontsize=12)
plt.title('Top 5 Most Discussed Stocks on Reddit (Q2 2023)', fontsize=14, fontweight='bold')
for i, row in enumerate(top_5.itertuples()):
    plt.text(i, row.q2_2023_post_count + 2, str(row.q2_2023_post_count), 
            ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('images/figures/sentiment/top_5_discussed.png', dpi=300, bbox_inches='tight')
print("Saved: images/figures/sentiment/top_5_discussed.png")
plt.close()

# sentiment heatmap
plt.figure(figsize=(10, 8))
heatmap_data = df[['ticker', 'q2_2023_avg_sentiment', 'positive_ratio']].set_index('ticker')
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap=SENTIMENT_CMAP,
            cbar_kws={'label': 'Score'})
plt.title('Q2 2023 Sentiment Metrics Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/figures/sentiment/sentiment_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: images/figures/sentiment/sentiment_heatmap.png")
plt.close()

print("\nAll visualizations created!")
print("\nGenerated files:")
print("  - images/figures/sentiment/sentiment_by_ticker.png")
print("  - images/figures/sentiment/sentiment_vs_volume.png")
print("  - images/figures/sentiment/sentiment_distribution.png")
print("  - images/figures/sentiment/top_5_discussed.png")
print("  - images/figures/sentiment/sentiment_heatmap.png")