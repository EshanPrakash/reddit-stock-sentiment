# aggregate_sentiment.py: This file aggregates sentiment scores by stock ticker for Q2 2023 Reddit 
#                         posts, computes statistics, and creates visualizations. It saves results 
#                         to the data/ and images/ directories for further analysis.
# Requires sentiment_analysis.py to be run first.

import json
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os

# Creating a data and images directory if they don't exist for saving collected posts, keeping the output organized
os.makedirs('data', exist_ok=True)
os.makedirs('images/figures/sentiment', exist_ok=True)

# Set style for seaborn and matplotlib styles for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# UNIFIED COLOR SCHEME
# ============================================================================
COLORS = {
    'positive': '#2ecc71',      # Green for positive sentiment
    'negative': '#e74c3c',      # Red for negative sentiment
    'neutral': '#95a5a6',       # Gray for neutral
    'returns': '#3498db',       # Blue for price/returns data
    'sentiment': '#e67e22',     # Orange for sentiment data
    'regression': '#2c3e50',    # Black for regression lines
}

# Distinct colors for top 5 tickers (used in bar charts where we want to distinguish them)
TOP5_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#e67e22', '#9b59b6']  # Red, Green, Blue, Orange, Purple

# Custom colormap using our exact red/yellow/green colors for gradients
# This ensures scatter plots and heatmaps use the same red/green as bar charts
SENTIMENT_CMAP = LinearSegmentedColormap.from_list(
    'sentiment',
    [COLORS['negative'], '#ffff00', COLORS['positive']]  # Red -> Pure Yellow -> Green
)

# Load the Reddit posts with sentiment scores from a JSON file produced by sentiment_analysis.py
input_file = "data/reddit_posts_q2_2023_with_sentiment.json"
print(f"Loading sentiment data from {input_file}...")
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    print(f"✓ Loaded {len(posts):,} posts with sentiment scores\n")
except FileNotFoundError:
    print(f"✗ Error: {input_file} not found!")
    print("Run sentiment_analysis.py first.")
    exit(1)

print("Aggregating Q2 2023 sentiment by ticker...")
print("=" * 60)

# Dictionary to hold sentiment data by ticker
ticker_data = {}

# Aggregate sentiment scores and counts for each ticker
# For each mentioned ticker, collect compound scores, total number of posts mentioning
# the ticker, and sentiment labels counts.
for post in posts:
    for ticker in post.get('mentioned_tickers', []):
        # Initialize data structure for ticker if not already present
        if ticker not in ticker_data:
            ticker_data[ticker] = {
                'compound_scores': [],
                'post_count': 0,
                'positive_count': 0,
                'neutral_count': 0,
                'negative_count': 0
            }
        
        # Append the VADER compound sentiment scores
        ticker_data[ticker]['compound_scores'].append(post['sentiment']['compound'])
        ticker_data[ticker]['post_count'] += 1
        
        # Count sentiment labels
        label = post['sentiment_label']
        ticker_data[ticker][f'{label}_count'] += 1

# Calculate statistics for each ticker
aggregated_results = []

for ticker in sorted(ticker_data.keys()):
    # Pull data for easier access
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
    # Append to results list
    aggregated_results.append(result)

# Convert the aggregated results to a pandas DataFrame
# Sort by average sentiment descending
df = pd.DataFrame(aggregated_results)
df = df.sort_values('q2_2023_avg_sentiment', ascending=False)

print("\nQ2 2023 SENTIMENT AGGREGATION BY TICKER")
print("=" * 60)
print(df.to_string(index=False))

# Save the aggregated results as a CSV and JSON file to the data directory for further analysis
csv_file = "data/q2_2023_sentiment_by_ticker.csv"
df.to_csv(csv_file, index=False)
print(f"\n✓ Saved aggregated sentiment to {csv_file}")

json_file = "data/q2_2023_sentiment_by_ticker.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(aggregated_results, f, indent=2)
print(f"✓ Saved aggregated sentiment to {json_file}")

# Print summary statistics
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nTickers analyzed: {len(aggregated_results)}")
print(f"Total posts: {sum(r['q2_2023_post_count'] for r in aggregated_results)}")

# Highlight most positive, most negative, and most discussed tickers
print(f"\nMost positive ticker: {df.iloc[0]['ticker']} (avg: {df.iloc[0]['q2_2023_avg_sentiment']:.3f})")
print(f"Most negative ticker: {df.iloc[-1]['ticker']} (avg: {df.iloc[-1]['q2_2023_avg_sentiment']:.3f})")
print(f"\nMost discussed: {df.nlargest(1, 'q2_2023_post_count').iloc[0]['ticker']} ({df.nlargest(1, 'q2_2023_post_count').iloc[0]['q2_2023_post_count']} posts)")

print("\n" + "=" * 60)
print("Next step: Get Q3 2023 stock returns and correlate with Q2 sentiment!")

# Create visualizations in the images directory
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# 1. Bar chart of average sentiment by ticker
# Horizontal bar chart for better readability with color coding
# Green for positive, red for negative sentiment
plt.figure(figsize=(12, 6))
colors = [COLORS['positive'] if x > 0 else COLORS['negative'] for x in df['q2_2023_avg_sentiment']]
plt.barh(df['ticker'], df['q2_2023_avg_sentiment'], color=colors)
plt.xlabel('Average Sentiment Score', fontsize=12)
plt.ylabel('Stock Ticker', fontsize=12)
plt.title('Q2 2023 Reddit Sentiment by Stock Ticker', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig('images/figures/sentiment/sentiment_by_ticker.png', dpi=300, bbox_inches='tight')
print("✓ Saved: images/figures/sentiment/sentiment_by_ticker.png")
plt.close()

# 2. Scatter plot: Post count vs Average Sentiment
# Size of points represents number of posts, color represents sentiment
# Positive sentiment in green, negative in red
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
print("✓ Saved: images/figures/sentiment/sentiment_vs_volume.png")
plt.close()

# 3. Stacked bar chart of sentiment distribution
# Shows positive, neutral, negative post counts per ticker
# Colors: Green (positive), Gray (neutral), Red (negative)
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
print("✓ Saved: images/figures/sentiment/sentiment_distribution.png")
plt.close()

# 4. Bar Chart with the Top 5 most discussed stocks
# Highlighting the most mentioned stocks in Q2 2023
# Use a consistent color palette for the bars
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
print("✓ Saved: images/figures/sentiment/top_5_discussed.png")
plt.close()

# 5. Color-Coded Heatmap-style visualization showing sentiment and volume
# Combining average sentiment and positive ratio for each ticker
plt.figure(figsize=(10, 8))
heatmap_data = df[['ticker', 'q2_2023_avg_sentiment', 'positive_ratio']].set_index('ticker')
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap=SENTIMENT_CMAP,
            cbar_kws={'label': 'Score'})
plt.title('Q2 2023 Sentiment Metrics Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/figures/sentiment/sentiment_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: images/figures/sentiment/sentiment_heatmap.png")
plt.close()

print("\n" + "=" * 60)
print("✓ All visualizations created!")
print("\nGenerated files:")
print("  • images/figures/sentiment/sentiment_by_ticker.png - Bar chart of sentiment scores")
print("  • images/figures/sentiment/sentiment_vs_volume.png - Scatter plot of sentiment vs discussion volume")
print("  • images/figures/sentiment/sentiment_distribution.png - Stacked bar of positive/neutral/negative")
print("  • images/figures/sentiment/top_5_discussed.png - Most discussed stocks")
print("  • images/figures/sentiment/sentiment_heatmap.png - Heatmap of sentiment metrics")