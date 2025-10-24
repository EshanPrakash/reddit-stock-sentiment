# reddit_client.py
import praw
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Test connection
print(f"Read only: {reddit.read_only}")
print("Connected successfully!")

# Example: Get hot posts from a subreddit
subreddit = reddit.subreddit("wallstreetbets")
for post in subreddit.hot(limit=5):
    print(f"\nTitle: {post.title}")
    print(f"Score: {post.score}")