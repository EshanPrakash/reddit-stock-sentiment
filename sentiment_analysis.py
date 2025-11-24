# sentiment_analysis.py: performs sentiment analysis on filtered reddit posts using vader,
#                        calculates sentiment scores for each post and aggregates statistics
#                        by stock ticker, subreddit, and overall distribution
# requires filter_posts.py to be run first

import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import os

# create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# initialize vader sentiment analyzer - designed for social media text with slang and emojis
analyzer = SentimentIntensityAnalyzer()

# load filtered reddit posts
input_file = "data/reddit_posts_q2_2023_filtered.json"
print(f"Loading filtered posts from {input_file}...")
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    print(f"Loaded {len(posts):,} posts\n")
except FileNotFoundError:
    print(f"Error: {input_file} not found!")
    print("Run filter_posts.py first to create the filtered dataset.")
    exit(1)

print("Running VADER sentiment analysis...")

# compute sentiment scores for each post
for i, post in enumerate(posts, 1):
    text = f"{post.get('title', '')} {post.get('selftext', '')}"
    scores = analyzer.polarity_scores(text)

    post['sentiment'] = {
        'compound': scores['compound'],
        'positive': scores['pos'],
        'neutral': scores['neu'],
        'negative': scores['neg']
    }

    # classify sentiment based on compound score thresholds
    if scores['compound'] >= 0.05:
        post['sentiment_label'] = 'positive'
    elif scores['compound'] <= -0.05:
        post['sentiment_label'] = 'negative'
    else:
        post['sentiment_label'] = 'neutral'
    
    if i % 10 == 0:
        print(f"  Processed {i}/{len(posts)} posts...", end='\r')

print(f"  Processed {len(posts)}/{len(posts)} posts... Done!")

# save posts with sentiment scores
output_file = "data/reddit_posts_q2_2023_with_sentiment.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(posts, f, indent=2, ensure_ascii=False)

print(f"Sentiment analysis complete!")
print(f"Data saved to {output_file}\n")

# aggregate and print sentiment statistics
print("Sentiment analysis results")

# overall sentiment distribution
sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
for post in posts:
    sentiment_counts[post['sentiment_label']] += 1

print(f"\nOverall Sentiment Distribution ({len(posts)} posts):")
for label, count in sentiment_counts.items():
    percentage = (count / len(posts)) * 100
    print(f"  {label.capitalize()}: {count} posts ({percentage:.1f}%)")

# compound score statistics
compound_scores = [post['sentiment']['compound'] for post in posts]
print(f"\nCompound Score Statistics:")
print(f"  Mean:   {statistics.mean(compound_scores):.3f}")
print(f"  Median: {statistics.median(compound_scores):.3f}")
print(f"  Min:    {min(compound_scores):.3f}")
print(f"  Max:    {max(compound_scores):.3f}")

# sentiment by ticker
print("\nSentiment by stock")

ticker_sentiments = {}
for post in posts:
    for ticker in post.get('mentioned_tickers', []):
        if ticker not in ticker_sentiments:
            ticker_sentiments[ticker] = []
        ticker_sentiments[ticker].append(post['sentiment']['compound'])

for ticker in sorted(ticker_sentiments.keys()):
    scores = ticker_sentiments[ticker]
    avg_sentiment = statistics.mean(scores)

    if avg_sentiment >= 0.05:
        overall = "positive"
    elif avg_sentiment <= -0.05:
        overall = "negative"
    else:
        overall = "neutral"
    
    print(f"\n{ticker}:")
    print(f"  Posts: {len(scores)}")
    print(f"  Average compound: {avg_sentiment:.3f} ({overall})")
    print(f"  Median compound:  {statistics.median(scores):.3f}")

# sentiment by subreddit
print("\nSentiment by subreddit")

subreddit_sentiments = {}
for post in posts:
    sub = post['subreddit']
    if sub not in subreddit_sentiments:
        subreddit_sentiments[sub] = []
    subreddit_sentiments[sub].append(post['sentiment']['compound'])

for sub in sorted(subreddit_sentiments.keys()):
    scores = subreddit_sentiments[sub]
    avg_sentiment = statistics.mean(scores)
    
    print(f"\nr/{sub}:")
    print(f"  Posts: {len(scores)}")
    print(f"  Average compound: {avg_sentiment:.3f}")
    print(f"  Median compound:  {statistics.median(scores):.3f}")

print("\nAnalysis complete!")
print(f"\nNext steps:")
print(f"  1. Review the sentiment scores in {output_file}")
print(f"  2. Aggregate sentiment by ticker for Q2 2023")
print(f"  3. Compare with Q3 2023 stock price performance")