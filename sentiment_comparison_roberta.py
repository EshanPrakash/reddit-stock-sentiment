# sentiment_comparison_roberta.py: compares VADER sentiment scores with financial RoBERTa model
#                                  to validate sentiment analysis methodology
# requires sentiment_analysis.py to be run first

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('images/figures/comparison', exist_ok=True)

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

SENTIMENT_CMAP = LinearSegmentedColormap.from_list(
    'sentiment',
    [COLORS['negative'], '#ffff00', COLORS['positive']]
)

# check if comparison data already exists
comparison_csv = "data/vader_roberta_comparison.csv"

if os.path.exists(comparison_csv):
    print(f"Loading existing comparison data from {comparison_csv}...")
    df = pd.read_csv(comparison_csv)
    print(f"Loaded {len(df):,} comparisons\n")
else:
    print("Loading financial RoBERTa model...")
    print("Model: soleimanian/financial-roberta-large-sentiment")

    # load financial roberta model
    tokenizer = AutoTokenizer.from_pretrained("soleimanian/financial-roberta-large-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained(
        "soleimanian/financial-roberta-large-sentiment",
        use_safetensors=True
    )
    model.eval()

    # load existing VADER sentiment data
    input_file = "data/reddit_posts_q2_2023_with_sentiment.json"
    print(f"\nLoading VADER sentiment data from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            posts = json.load(f)
        print(f"Loaded {len(posts):,} posts with VADER sentiment scores\n")
    except FileNotFoundError:
        print(f"Error: {input_file} not found!")
        print("Run sentiment_analysis.py first.")
        exit(1)

    print("Running RoBERTa sentiment analysis for comparison...")

    # function to get roberta sentiment score
    def get_roberta_sentiment(text, max_length=512):
        """get sentiment score from financial RoBERTa model"""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # model outputs: [negative, neutral, positive]
        probs = probabilities[0].numpy()

        # convert to compound score similar to VADER (-1 to +1)
        compound = probs[2] - probs[0]  # positive - negative

        # classify based on highest probability
        label_idx = np.argmax(probs)
        labels = ['negative', 'neutral', 'positive']
        label = labels[label_idx]

        return {
            'compound': float(compound),
            'positive': float(probs[2]),
            'neutral': float(probs[1]),
            'negative': float(probs[0]),
            'label': label
        }

    # compare VADER and RoBERTa on each post
    comparison_data = []

    for i, post in enumerate(posts, 1):
        text = f"{post.get('title', '')} {post.get('selftext', '')}"

        # get roberta sentiment
        roberta = get_roberta_sentiment(text)

        # extract vader sentiment
        vader_compound = post['sentiment']['compound']
        vader_label = post['sentiment_label']

        comparison_data.append({
            'post_id': i,
            'vader_compound': vader_compound,
            'vader_label': vader_label,
            'roberta_compound': roberta['compound'],
            'roberta_label': roberta['label'],
            'text_length': len(text),
            'mentioned_tickers': post.get('mentioned_tickers', [])
        })

        if i % 10 == 0:
            print(f"  Processed {i}/{len(posts)} posts...", end='\r')

    print(f"  Processed {len(posts)}/{len(posts)} posts... Done!")

    # convert to dataframe
    df = pd.DataFrame(comparison_data)

    # save comparison data
    df.to_csv(comparison_csv, index=False)
    print(f"\nSaved comparison data to {comparison_csv}")

# calculate correlation statistics
pearson_r, pearson_p = stats.pearsonr(df['vader_compound'], df['roberta_compound'])
spearman_r, spearman_p = stats.spearmanr(df['vader_compound'], df['roberta_compound'])
mad = np.mean(np.abs(df['vader_compound'] - df['roberta_compound']))
rmse = np.sqrt(np.mean((df['vader_compound'] - df['roberta_compound'])**2))
agreement = (df['vader_label'] == df['roberta_label']).mean()

print("\nVADER vs RoBERTa sentiment comparison")
print(f"\nPearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4e}")
print(f"Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.4e}")
print(f"Mean absolute difference: {mad:.4f}")
print(f"Root mean squared error: {rmse:.4f}")
print(f"Sentiment label agreement: {agreement:.2%}")

# create visualization
print("\nCreating comparison visualization...")

# side-by-side distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(df['vader_compound'], bins=30, color=COLORS['negative'], edgecolor='black')
axes[0].axvline(df['vader_compound'].mean(), color='black', linestyle='--', linewidth=1, label=f'Mean: {df["vader_compound"].mean():.3f}')
axes[0].set_xlabel('Compound Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('VADER Sentiment Distribution', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(df['roberta_compound'], bins=30, color=COLORS['positive'], edgecolor='black')
axes[1].axvline(df['roberta_compound'].mean(), color='black', linestyle='--', linewidth=1, label=f'Mean: {df["roberta_compound"].mean():.3f}')
axes[1].set_xlabel('Compound Score', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('RoBERTa Sentiment Distribution', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/figures/comparison/distribution_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: images/figures/comparison/distribution_comparison.png")
plt.close()

print("\nVisualization created!")
print("\nGenerated files:")
print("  - data/vader_roberta_comparison.csv")
print("  - images/figures/comparison/distribution_comparison.png")
