import json
import statistics

# Load the collected data
input_file = "reddit_posts_q2_2023_full.json"
print(f"Loading data from {input_file}...")

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        all_posts = json.load(f)
    print(f"✓ Loaded {len(all_posts):,} posts\n")
except FileNotFoundError:
    print(f"✗ Error: {input_file} not found!")
    exit(1)

print("=" * 60)
print("ENGAGEMENT STATISTICS BY SUBREDDIT")
print("=" * 60)

subreddits = ["wallstreetbets", "stocks", "StockMarket"]

for subreddit in subreddits:
    # Get posts from this subreddit
    sub_posts = [p for p in all_posts if p['subreddit'] == subreddit]
    
    if not sub_posts:
        print(f"\nr/{subreddit}: No posts found")
        continue
    
    # Extract scores and comment counts
    scores = [p['score'] for p in sub_posts]
    comments = [p['num_comments'] for p in sub_posts]
    
    print(f"\nr/{subreddit} ({len(sub_posts):,} posts)")
    print("-" * 60)
    
    # Score statistics
    print("UPVOTES:")
    print(f"  Mean:     {statistics.mean(scores):.1f}")
    print(f"  Median:   {statistics.median(scores):.1f}")
    print(f"  Min:      {min(scores)}")
    print(f"  Max:      {max(scores):,}")
    if len(scores) >= 2:
        print(f"  Std Dev:  {statistics.stdev(scores):.1f}")
    
    # Percentiles for scores
    sorted_scores = sorted(scores)
    p25_score = sorted_scores[len(sorted_scores) // 4]
    p75_score = sorted_scores[3 * len(sorted_scores) // 4]
    p90_score = sorted_scores[9 * len(sorted_scores) // 10]
    
    print(f"  25th %ile: {p25_score}")
    print(f"  75th %ile: {p75_score}")
    print(f"  90th %ile: {p90_score}")
    
    # Comment statistics
    print("\nCOMMENTS:")
    print(f"  Mean:     {statistics.mean(comments):.1f}")
    print(f"  Median:   {statistics.median(comments):.1f}")
    print(f"  Min:      {min(comments)}")
    print(f"  Max:      {max(comments):,}")
    if len(comments) >= 2:
        print(f"  Std Dev:  {statistics.stdev(comments):.1f}")
    
    # Percentiles for comments
    sorted_comments = sorted(comments)
    p25_comments = sorted_comments[len(sorted_comments) // 4]
    p75_comments = sorted_comments[3 * len(sorted_comments) // 4]
    p90_comments = sorted_comments[9 * len(sorted_comments) // 10]
    
    print(f"  25th %ile: {p25_comments}")
    print(f"  75th %ile: {p75_comments}")
    print(f"  90th %ile: {p90_comments}")

print("\n" + "=" * 60)
print("OVERALL STATISTICS (ALL SUBREDDITS)")
print("=" * 60)

all_scores = [p['score'] for p in all_posts]
all_comments = [p['num_comments'] for p in all_posts]

print(f"\nTotal posts: {len(all_posts):,}")
print(f"\nOverall median upvotes: {statistics.median(all_scores):.1f}")
print(f"Overall median comments: {statistics.median(all_comments):.1f}")

print("\n" + "=" * 60)
print("RECOMMENDED FILTERS:")
print("=" * 60)
print("\nTo filter by median:")
for subreddit in subreddits:
    sub_posts = [p for p in all_posts if p['subreddit'] == subreddit]
    if sub_posts:
        scores = [p['score'] for p in sub_posts]
        comments = [p['num_comments'] for p in sub_posts]
        med_score = statistics.median(scores)
        med_comments = statistics.median(comments)
        print(f"  r/{subreddit}: score >= {med_score:.0f} OR comments >= {med_comments:.0f}")

print("\nTo filter by 75th percentile (stricter):")
for subreddit in subreddits:
    sub_posts = [p for p in all_posts if p['subreddit'] == subreddit]
    if sub_posts:
        scores = [p['score'] for p in sub_posts]
        comments = [p['num_comments'] for p in sub_posts]
        sorted_scores = sorted(scores)
        sorted_comments = sorted(comments)
        p75_score = sorted_scores[3 * len(sorted_scores) // 4]
        p75_comments = sorted_comments[3 * len(sorted_comments) // 4]
        print(f"  r/{subreddit}: score >= {p75_score} OR comments >= {p75_comments}")