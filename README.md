# Reddit Stock Sentiment - Setup Guide

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/EshanPrakash/reddit-stock-sentiment.git
cd reddit-stock-sentiment
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install praw python-dotenv
```

### 3. Get Reddit API Credentials

1. Go to https://www.reddit.com/prefs/apps
2. Scroll down and click **"create another app..."**
3. Fill out the form:
   - **name**: wsb-stock-sentiment (or whatever you want)
   - **type**: Select **"script"**
   - **description**: (leave blank)
   - **about url**: (leave blank)
   - **redirect uri**: `http://localhost:8080`
4. Click **"create app"**
5. You'll see your app. Note these values:
   - **client_id**: The string under "personal use script" (short random string)
   - **client_secret**: The string next to "secret" (longer random string)

### 4. Create Your .env File
```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and fill in your credentials:
```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=python:wsb-stock-sentiment:v1.0 (by u/your_reddit_username)
```

Replace `your_reddit_username` with your actual Reddit username.

### 5. Test Your Connection
```bash
python reddit_client.py
```

You should see:
```
Read only: True
Connected successfully!
```

Followed by some posts from r/wallstreetbets.

## Troubleshooting

- **"Invalid credentials"**: Double-check your client_id and client_secret in .env
- **Missing .env**: Make sure you created the .env file in the project root
- **Import errors**: Make sure your virtual environment is activated and dependencies are installed

## Important Notes

- **Never commit your .env file** - it contains your secrets!
- Each team member needs their own Reddit API credentials
- The .env file is already in .gitignore
