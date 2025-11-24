# collect_pullpush.py: collects reddit posts from specified subreddits during Q2 2023
#                      using the pullpush.io api, applying subreddit-specific median engagement filters
#                      and saves collected data in json format for further analysis

import requests
from datetime import datetime
import time
import json
import os

# create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Q2 2023 timeframe: april 1 to june 30
start_timestamp = int(datetime(2023, 4, 1).timestamp())
end_timestamp = int(datetime(2023, 6, 30).timestamp())

# pullpush.io api endpoint for historical reddit data
PULLPUSH_API = "https://api.pullpush.io/reddit/search/submission/"

# target subreddits - major finance-related communities
subreddits = ["wallstreetbets", "stocks", "StockMarket"]

# median engagement thresholds per subreddit to filter out low-engagement posts
MEDIAN_FILTERS = {
    'wallstreetbets': {'score': 27, 'comments': 21},
    'stocks': {'score': 11, 'comments': 25},
    'StockMarket': {'score': 20, 'comments': 15}
}


print("Collecting Reddit posts from Q2 2023 using Pullpush.io")
print(f"Timeframe: {datetime.fromtimestamp(start_timestamp).date()} to {datetime.fromtimestamp(end_timestamp).date()}")
print(f"Subreddits: {', '.join(subreddits)}")

# collection of all filtered posts
all_posts = []

# loop through each subreddit and collect posts
for subreddit in subreddits:
    print(f"\nCollecting from r/{subreddit}...")
    
    # paginate through results, moving backward in time
    current_before = end_timestamp
    total_collected = 0
    
    retries = 0
    max_retries = 3
    
    while True:
        params = {
            'subreddit': subreddit,
            'after': start_timestamp,
            'before': int(current_before),
            'size': 100,
            'sort': 'desc',
            'sort_type': 'created_utc'
        }
        
        try:
            response = requests.get(PULLPUSH_API, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            posts = data.get('data', [])
            
            # stop pagination if no more posts found
            if not posts:
                print(f"\n  No more posts found. Stopping pagination.")
                break
            
            retries = 0

            # filter posts by median engagement thresholds
            for post in posts:
                score = post.get('score', 0)
                num_comments = post.get('num_comments', 0)

                median_score = MEDIAN_FILTERS[subreddit]['score']
                median_comments = MEDIAN_FILTERS[subreddit]['comments']

                # must meet or exceed median for score or comments
                if score >= median_score or num_comments >= median_comments:
                    post_data = {
                        'subreddit': subreddit,
                        'id': post.get('id'),
                        'title': post.get('title', ''),
                        'selftext': post.get('selftext', ''),
                        'score': score,
                        'num_comments': num_comments,
                        'created_utc': post.get('created_utc'),
                        'created_date': datetime.fromtimestamp(post.get('created_utc')).isoformat(),
                        'author': post.get('author', '[deleted]'),
                        'url': post.get('url', '')
                    }
                    all_posts.append(post_data)

            total_collected += len(posts)
            print(f"  Collected {total_collected} posts so far...", end='\r')
            
            # move pagination cursor before oldest post
            current_before = posts[-1]['created_utc'] - 1

            # rate limiting to avoid 500 errors
            time.sleep(3)

            # safety limit per subreddit
            if total_collected >= 10000:
                print(f"\n  Reached 10k posts limit for r/{subreddit}")
                break
        
        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"\n  Error (attempt {retries}/{max_retries}): {e}")
            
            if retries >= max_retries:
                print(f"  Max retries reached. Moving on with {total_collected} posts.")
                break
            
            print(f"  Waiting 10 seconds before retry...")
            time.sleep(10)
            
        except Exception as e:
            print(f"\n  Unexpected error: {e}")
            break
    
    print(f"\n  Finished r/{subreddit}: {total_collected} posts")

    # pause between subreddits
    time.sleep(2)

print(f"\nTotal posts collected: {len(all_posts)}")

# save all posts to json
output_file = "data/reddit_posts_q2_2023_full.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_posts, f, indent=2, ensure_ascii=False)

print(f"Data saved to {output_file}")

# summary statistics
print(f"\nSummary by subreddit:")
for sub in subreddits:
    count = sum(1 for p in all_posts if p['subreddit'] == sub)
    print(f"  r/{sub}: {count:,} posts")

print(f"\nTotal size: {len(all_posts):,} posts")
print(f"Date range: {datetime.fromtimestamp(start_timestamp).date()} to {datetime.fromtimestamp(end_timestamp).date()}")