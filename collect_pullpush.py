# collect_pullpush.py: This file collects Reddit posts from specified subreddits during Q2 2023 
#                      using the Pullpush.io API, applying subreddit-specific median engagement filters.
#                      The collected data is saved in JSON format for further analysis.

import requests
from datetime import datetime
import time
import json
import os

# Creating a data directory if it doesn't exist for saving collected posts, keeping the output organized
os.makedirs('data', exist_ok=True)

# Defining the Q2 2023 timeframe
# From April 1, 2023 to June 30, 2023
start_timestamp = int(datetime(2023, 4, 1).timestamp())
end_timestamp = int(datetime(2023, 6, 30).timestamp())

# Pullpush.io API endpoint
# Instead of Pushshift, we are using Pullpush.io for this script
# as it is a free API for searching historical Reddit submissions.
PULLPUSH_API = "https://api.pullpush.io/reddit/search/submission/"

# The target subreddits to collect posts from
# These are major finance-related communities
# that are likely to discuss stocks and market trends.
subreddits = ["wallstreetbets", "stocks", "StockMarket"]

# Median engagement thresholds per subreddit
# These values are used to filter posts based on their score and number of comments.
# This removes low-engagement posts from the dataset, as only higher-engagement 
# posts are likely to be relevant.
MEDIAN_FILTERS = {
    'wallstreetbets': {'score': 27, 'comments': 21},
    'stocks': {'score': 11, 'comments': 25},
    'StockMarket': {'score': 20, 'comments': 15}
}


print("Collecting Reddit posts from Q2 2023 using Pullpush.io")
print("=" * 60)
print(f"Timeframe: {datetime.fromtimestamp(start_timestamp).date()} to {datetime.fromtimestamp(end_timestamp).date()}")
print(f"Subreddits: {', '.join(subreddits)}")
print("=" * 60)

# Collection of all filtered posts across our target subreddits
all_posts = []

# Loop through each subreddit and collect posts
for subreddit in subreddits:
    print(f"\n📊 Collecting from r/{subreddit}...")
    
    # We'll paginate through results
    # current_before starts at the end of our timeframe and moves backward in time
    current_before = end_timestamp
    total_collected = 0
    
    retries = 0
    max_retries = 3
    
    while True:
        params = {
            'subreddit': subreddit,
            'after': start_timestamp,       # Start of Q2 2023
            'before': int(current_before),  # End of Q2 2023 or current pagination point
            'size': 100,                    # Max posts per request
            'sort': 'desc',                 # Order by descending time
            'sort_type': 'created_utc'      # Sort by creation time, so newest posts first
        }
        
        try:
            # Make the API request to Pullpush.io
            response = requests.get(PULLPUSH_API, params=params, timeout=30)

            # Raise an error for bad responses
            response.raise_for_status()
            
            # Parse the JSON response
            data = response.json()

            # Extract posts from the response
            posts = data.get('data', [])
            
            # Stops pagination if no more posts are found
            if not posts:
                print(f"\n  No more posts found. Stopping pagination.")
                break
            
            # Reset retry counter on success
            retries = 0
            
            # Add posts to collection (with subreddit-specific median filters)
            for post in posts:
                score = post.get('score', 0)    # Upvotes
                num_comments = post.get('num_comments', 0)  # Number of comments
                
                # Get median thresholds for this subreddit
                median_score = MEDIAN_FILTERS[subreddit]['score']
                median_comments = MEDIAN_FILTERS[subreddit]['comments']
                
                # Filter: Must meet or exceed median for score OR comments
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
                    all_posts.append(post_data) # Add to overall collection
            
            # Update total collected count
            total_collected += len(posts)
            print(f"  Collected {total_collected} posts so far...", end='\r')
            
            # Update "before" timestamp for next page to be just before the oldest post we received
            current_before = posts[-1]['created_utc'] - 1
            
            # Rate limiting, wait longer to avoid 500 errors, since Pullpush.io has stricter limits
            time.sleep(3)
            
            # Safety limit to stop after collecting 10,000 posts per subreddit, preventing excessive data collection
            if total_collected >= 10000:
                print(f"\n  Reached 10k posts limit for r/{subreddit}")
                break
        
        # Handle request exceptions with retries
        except requests.exceptions.RequestException as e:
            retries += 1
            print(f"\n  Error (attempt {retries}/{max_retries}): {e}")
            
            # Check if max retries reached
            if retries >= max_retries:
                print(f"  Max retries reached. Moving on with {total_collected} posts.")
                break
            
            # Wait longer before retry
            print(f"  Waiting 10 seconds before retry...")
            time.sleep(10)
            
        # Catch-all for any other unexpected errors
        except Exception as e:
            print(f"\n  Unexpected error: {e}")
            break
    
    # Finished collecting for this subreddit
    print(f"\n  ✓ Finished r/{subreddit}: {total_collected} posts")

    # Short pause between subreddits to respect API limits
    time.sleep(2)

print(f"\n" + "=" * 60)
print(f"✓ Total posts collected: {len(all_posts)}")

# Save all posts as a JSON file to the data directory for later processing and analysis
output_file = "data/reddit_posts_q2_2023_full.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_posts, f, indent=2, ensure_ascii=False)

print(f"✓ Data saved to {output_file}")

# Summary statistics by subreddit
print(f"\nSummary by subreddit:")
for sub in subreddits:
    count = sum(1 for p in all_posts if p['subreddit'] == sub)
    print(f"  r/{sub}: {count:,} posts")

print(f"\nTotal size: {len(all_posts):,} posts")
print(f"Date range: {datetime.fromtimestamp(start_timestamp).date()} to {datetime.fromtimestamp(end_timestamp).date()}")