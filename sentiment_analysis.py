# sentiment_analysis.py: This file performs sentiment analysis on the filtered Reddit posts using the 
#                        VADER sentiment analysis tool. It calculates sentiment scores for each post 
#                        and aggregates sentiment statistics by stock ticker, subreddit, and overall 
#                        distribution. It outputs the results to a new JSON file for further analysis.
# Requires filter_posts.py to be run first.

import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics
import os

# Creating a data directory if it doesn't exist for saving collected posts, keeping the output organized
os.makedirs('data', exist_ok=True)

# Initialize VADER sentiment analyzer
# VADER (Valence Aware Dictionary and sEntiment Reasoner) 
# is a sentiment analysis tool which is designed to analyze 
# social media text and informal language. Unlike traditional 
# sentiment analysis methods it is best at detecting sentiment 
# in short pieces of text like tweets, product reviews or user 
# comments which contain slang, emojis and abbreviations. It 
# uses a pre-built lexicon of words associated with sentiment 
# values and applies specific rules to calculate sentiment scores.
analyzer = SentimentIntensityAnalyzer()

# Load the filtered Reddit posts from a JSON file produced by filter_posts.py
input_file = "data/reddit_posts_q2_2023_filtered.json"
print(f"Loading filtered posts from {input_file}...")
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    print(f"✓ Loaded {len(posts):,} posts\n")
except FileNotFoundError:
    print(f"✗ Error: {input_file} not found!")
    print("Run filter_posts.py first to create the filtered dataset.")
    exit(1)

print("Running VADER sentiment analysis...")
print("=" * 60)

# Sentiment analysis loop:
# For each post, compute sentiment scores 
# using VADER and store the outputs scores (compound, positive, neutral, negative) 
# and a sentiment label (positive, neutral, negative) based on the compound score.
# in the post dictionary for later analysis.
for i, post in enumerate(posts, 1):
    # Combine title and selftext for searching, as information can appear in either
    # If either field is missing, default to an empty string
    text = f"{post.get('title', '')} {post.get('selftext', '')}"
    
    # Get VADER sentiment scores for each combined text
    scores = analyzer.polarity_scores(text)
    
    # Add and save sentiment scores to the post dictionary
    post['sentiment'] = {
        'compound': scores['compound'],  # Overall sentiment (-1 to +1)
        'positive': scores['pos'],
        'neutral': scores['neu'],
        'negative': scores['neg']
    }
    
    # Classify sentiment based on compound score thresholds
    # Compound score is a normalized, weighted composite score
    # ranging from -1 (most extreme negative) to +1 (most extreme positive)
    # They are necessary to categorize the sentiment into discrete labels
    if scores['compound'] >= 0.05:
        post['sentiment_label'] = 'positive'
    elif scores['compound'] <= -0.05:
        post['sentiment_label'] = 'negative'
    else:
        post['sentiment_label'] = 'neutral'
    
    # Progress indicator for every 10 posts
    if i % 10 == 0:
        print(f"  Processed {i}/{len(posts)} posts...", end='\r')

print(f"  Processed {len(posts)}/{len(posts)} posts... Done!")

# Save posts with sentiment scores to the data directory for later analysis
output_file = "data/reddit_posts_q2_2023_with_sentiment.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(posts, f, indent=2, ensure_ascii=False)

print(f"✓ Sentiment analysis complete!")
print(f"✓ Data saved to {output_file}\n")

# Aggregate and print sentiment statistics
print("=" * 60)
print("SENTIMENT ANALYSIS RESULTS")
print("=" * 60)

# Overall sentiment distribution
sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
for post in posts:
    sentiment_counts[post['sentiment_label']] += 1

print(f"\nOverall Sentiment Distribution ({len(posts)} posts):")
for label, count in sentiment_counts.items():
    percentage = (count / len(posts)) * 100
    print(f"  {label.capitalize()}: {count} posts ({percentage:.1f}%)")

# Average compound scores distribution
compound_scores = [post['sentiment']['compound'] for post in posts]
print(f"\nCompound Score Statistics:")
print(f"  Mean:   {statistics.mean(compound_scores):.3f}")
print(f"  Median: {statistics.median(compound_scores):.3f}")
print(f"  Min:    {min(compound_scores):.3f}")
print(f"  Max:    {max(compound_scores):.3f}")

# Sentiment by ticker
print("\n" + "=" * 60)
print("SENTIMENT BY STOCK")
print("=" * 60)

# ticker_sentiments will hold lists of compound scores for each ticker
ticker_sentiments = {}

# Aggregate sentiment scores for each ticker
for post in posts:
    for ticker in post.get('mentioned_tickers', []):
        if ticker not in ticker_sentiments:
            ticker_sentiments[ticker] = []
        ticker_sentiments[ticker].append(post['sentiment']['compound'])

# Print average and median sentiment for each ticker
for ticker in sorted(ticker_sentiments.keys()):
    scores = ticker_sentiments[ticker]
    avg_sentiment = statistics.mean(scores)
    
    # Classify overall sentiment for this ticker
    if avg_sentiment >= 0.05:
        overall = "POSITIVE"
    elif avg_sentiment <= -0.05:
        overall = "NEGATIVE"
    else:
        overall = "NEUTRAL"
    
    print(f"\n{ticker}:")
    print(f"  Posts: {len(scores)}")
    print(f"  Average compound: {avg_sentiment:.3f} ({overall})")
    print(f"  Median compound:  {statistics.median(scores):.3f}")

# Sentiment by subreddit
print("\n" + "=" * 60)
print("SENTIMENT BY SUBREDDIT")
print("=" * 60)

# subreddit_sentiments will hold lists of compound scores for each subreddit
subreddit_sentiments = {}

# Aggregate sentiment scores for each subreddit
for post in posts:
    sub = post['subreddit']
    if sub not in subreddit_sentiments:
        subreddit_sentiments[sub] = []
    subreddit_sentiments[sub].append(post['sentiment']['compound'])

# Print average and median sentiment for each subreddit
for sub in sorted(subreddit_sentiments.keys()):
    scores = subreddit_sentiments[sub]
    avg_sentiment = statistics.mean(scores)
    
    print(f"\nr/{sub}:")
    print(f"  Posts: {len(scores)}")
    print(f"  Average compound: {avg_sentiment:.3f}")
    print(f"  Median compound:  {statistics.median(scores):.3f}")

print("\n" + "=" * 60)
print("✓ Analysis complete!")
print(f"\nNext steps:")
print(f"  1. Review the sentiment scores in {output_file}")
print(f"  2. Aggregate sentiment by ticker for Q2 2023")
print(f"  3. Compare with Q3 2023 stock price performance")