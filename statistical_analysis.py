# statistical_analysis.py: This file performs statistical correlation and regression analysis
#                          between Reddit sentiment scores from Q2 2023 and stock returns in 
#                          Q3 2023. It generates visualizations and saves results to CSV and PNG files.
# Requires yfinance_fetch_q3.py to be run first


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import os

# Creating a data and images directory if they don't exist for saving collected posts, keeping the output organized
os.makedirs('data', exist_ok=True)
os.makedirs('images/figures/hypothesis', exist_ok=True)
os.makedirs('images/diagnostics/model', exist_ok=True)

# Set style for seaborn and matplotlib styles for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 70)
print("SENTIMENT-RETURN CORRELATION ANALYSIS")
print("=" * 70)

# Load the Q2 sentiment by ticker data from a JSON file produced by aggregate_sentiment.py
sentiment_file = "data/q2_2023_sentiment_by_ticker.json"
print(f"\nLoading Q2 sentiment data from {sentiment_file}...")
try:
    with open(sentiment_file, 'r', encoding='utf-8') as f:
        sentiment_data = json.load(f)
    print(f"✓ Loaded {len(sentiment_data)} tickers with sentiment scores")
except FileNotFoundError:
    print(f"✗ Error: {sentiment_file} not found!")
    print("Run aggregate_sentiment.py first.")
    exit(1)

# Load the Q3 returns data from a JSON file produced by yfinance_fetch_q3.py
returns_file = "data/q3_2023_with_benchmarks.json"
print(f"Loading Q3 returns data from {returns_file}...")
try:
    with open(returns_file, 'r', encoding='utf-8') as f:
        returns_data = json.load(f)
    print(f"✓ Loaded {len(returns_data)} tickers with Q3 returns\n")
except FileNotFoundError:
    print(f"✗ Error: {returns_file} not found!")
    print("Run yfinance_fetch_q3.py first.")
    exit(1)

# Merge the two datasets loaded above into a single pandas DataFrame on the 'ticker' field
sentiment_df = pd.DataFrame(sentiment_data)
returns_df = pd.DataFrame(returns_data)

merged_df = pd.merge(
    sentiment_df[['ticker', 'q2_2023_avg_sentiment', 'q2_2023_post_count']], 
    returns_df[['ticker', 'q3_return_pct', 'excess_vs_spy', 'excess_vs_sector']], 
    on='ticker'
)

print("=" * 70)
print(f"✓ Merged dataset: {len(merged_df)} tickers")
print("=" * 70)

# Save merged dataset to a CSV file for reference and further analysis in other scripts
merged_file = "data/merged_sentiment_returns.csv"
merged_df.to_csv(merged_file, index=False)
print(f"\n✓ Saved merged data to {merged_file}\n")

# Print the dataset preview
print("Dataset preview:")
print(merged_df.head(10).to_string(index=False))

# ========================================================================================#
# MODEL 1: RAW RETURNS
# Correlation and regression analysis between Q2 sentiment and Q3 raw returns/performance
# ========================================================================================#

print("\n" + "=" * 70)
print("MODEL 1: Q2 SENTIMENT vs Q3 RAW RETURNS")
print("=" * 70)

# Define X and y for Model 1
X1 = merged_df['q2_2023_avg_sentiment'].values
y1 = merged_df['q3_return_pct'].values

# Correlation tests
pearson_r1, pearson_p1 = stats.pearsonr(X1, y1)
spearman_r1, spearman_p1 = stats.spearmanr(X1, y1)

print(f"\nCorrelation Analysis:")
print(f"  Pearson r:  {pearson_r1:.4f} (p = {pearson_p1:.4f})")
print(f"  Spearman ρ: {spearman_r1:.4f} (p = {spearman_p1:.4f})")

# Linear regression
X1_reshaped = X1.reshape(-1, 1)
model1 = LinearRegression()
model1.fit(X1_reshaped, y1)

y1_pred = model1.predict(X1_reshaped)
r_squared1 = model1.score(X1_reshaped, y1)
coefficient1 = model1.coef_[0]
intercept1 = model1.intercept_

# Calculate standard error and confidence interval of regression residuals
residuals1 = y1 - y1_pred
mse1 = np.mean(residuals1**2)
se1 = np.sqrt(mse1 / (len(y1) - 2))

print(f"\nLinear Regression:")
print(f"  R² = {r_squared1:.4f}")
print(f"  Coefficient (β) = {coefficient1:.4f}")
print(f"  Intercept = {intercept1:.4f}")
print(f"  Standard Error = {se1:.4f}")
print(f"  Equation: Q3_Return = {intercept1:.2f} + {coefficient1:.2f} * Sentiment")

# ========================================================================================#
# MODEL 2: MARKET-ADJUSTED RETURNS (vs SPY)
# Assessing sentiment vs returns adjusted for market performance
# ========================================================================================#

print("\n" + "=" * 70)
print("MODEL 2: Q2 SENTIMENT vs Q3 EXCESS RETURNS (vs SPY)")
print("=" * 70)

# Define X and y for Model 2
X2 = merged_df['q2_2023_avg_sentiment'].values
y2 = merged_df['excess_vs_spy'].values

# Correlation tests
pearson_r2, pearson_p2 = stats.pearsonr(X2, y2)
spearman_r2, spearman_p2 = stats.spearmanr(X2, y2)

print(f"\nCorrelation Analysis:")
print(f"  Pearson r:  {pearson_r2:.4f} (p = {pearson_p2:.4f})")
print(f"  Spearman ρ: {spearman_r2:.4f} (p = {spearman_p2:.4f})")

# Linear regression
X2_reshaped = X2.reshape(-1, 1)
model2 = LinearRegression()
model2.fit(X2_reshaped, y2)

y2_pred = model2.predict(X2_reshaped)
r_squared2 = model2.score(X2_reshaped, y2)
coefficient2 = model2.coef_[0]
intercept2 = model2.intercept_

# Calculate standard error and confidence interval of regression residuals
residuals2 = y2 - y2_pred
mse2 = np.mean(residuals2**2)
se2 = np.sqrt(mse2 / (len(y2) - 2))

print(f"\nLinear Regression:")
print(f"  R² = {r_squared2:.4f}")
print(f"  Coefficient (β) = {coefficient2:.4f}")
print(f"  Intercept = {intercept2:.4f}")
print(f"  Standard Error = {se2:.4f}")
print(f"  Equation: Excess_vs_SPY = {intercept2:.2f} + {coefficient2:.2f} * Sentiment")

# ========================================================================================#
# MODEL 3: SECTOR-ADJUSTED RETURNS
# Assessing sentiment vs returns adjusted for sector performance
# ========================================================================================#

print("\n" + "=" * 70)
print("MODEL 3: Q2 SENTIMENT vs Q3 EXCESS RETURNS (vs SECTOR)")
print("=" * 70)

# Define X and y for Model 3
X3 = merged_df['q2_2023_avg_sentiment'].values
y3 = merged_df['excess_vs_sector'].values

# Correlation tests
pearson_r3, pearson_p3 = stats.pearsonr(X3, y3)
spearman_r3, spearman_p3 = stats.spearmanr(X3, y3)

print(f"\nCorrelation Analysis:")
print(f"  Pearson r:  {pearson_r3:.4f} (p = {pearson_p3:.4f})")
print(f"  Spearman ρ: {spearman_r3:.4f} (p = {spearman_p3:.4f})")

# Linear regression
X3_reshaped = X3.reshape(-1, 1)
model3 = LinearRegression()
model3.fit(X3_reshaped, y3)

y3_pred = model3.predict(X3_reshaped)
r_squared3 = model3.score(X3_reshaped, y3)
coefficient3 = model3.coef_[0]
intercept3 = model3.intercept_

# Calculate standard error and confidence interval of regression residuals
residuals3 = y3 - y3_pred
mse3 = np.mean(residuals3**2)
se3 = np.sqrt(mse3 / (len(y3) - 2))

print(f"\nLinear Regression:")
print(f"  R² = {r_squared3:.4f}")
print(f"  Coefficient (β) = {coefficient3:.4f}")
print(f"  Intercept = {intercept3:.4f}")
print(f"  Standard Error = {se3:.4f}")
print(f"  Equation: Excess_vs_Sector = {intercept3:.2f} + {coefficient3:.2f} * Sentiment")

# ========================================================================================#
# SUMMARY TABLE of ALL MODELS
# ========================================================================================#

print("\n" + "=" * 70)
print("SUMMARY: ALL THREE MODELS")
print("=" * 70)

# Create a summary DataFrame
summary_data = {
    'Model': ['Model 1: Raw Returns', 'Model 2: vs SPY', 'Model 3: vs Sector'],
    'Pearson r': [pearson_r1, pearson_r2, pearson_r3],
    'Pearson p': [pearson_p1, pearson_p2, pearson_p3],
    'Spearman ρ': [spearman_r1, spearman_r2, spearman_r3],
    'Spearman p': [spearman_p1, spearman_p2, spearman_p3],
    'R²': [r_squared1, r_squared2, r_squared3],
    'Coefficient': [coefficient1, coefficient2, coefficient3],
    'Std Error': [se1, se2, se3]
}

# Display summary table
summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Save summary table to a CSV file
summary_file = "data/regression_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"\n✓ Saved summary to {summary_file}")

# ========================================================================================#
# VISUALIZATIONS
# Consisting of scatter plots, regression lines, residual plots, and comparison charts
# ========================================================================================#

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

# Visualization 1: Three scatter plots with regression lines
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Model 1: Raw Returns
axes[0].scatter(X1, y1, alpha=0.6, s=80, color='steelblue')
axes[0].plot(X1, y1_pred, color='red', linewidth=2, label=f'y = {intercept1:.2f} + {coefficient1:.2f}x')
axes[0].set_xlabel('Q2 2023 Average Sentiment', fontsize=11)
axes[0].set_ylabel('Q3 2023 Return (%)', fontsize=11)
axes[0].set_title(f'Model 1: Raw Returns\nR² = {r_squared1:.4f}, p = {pearson_p1:.4f}', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Model 2: vs SPY
axes[1].scatter(X2, y2, alpha=0.6, s=80, color='green')
axes[1].plot(X2, y2_pred, color='red', linewidth=2, label=f'y = {intercept2:.2f} + {coefficient2:.2f}x')
axes[1].set_xlabel('Q2 2023 Average Sentiment', fontsize=11)
axes[1].set_ylabel('Q3 2023 Excess Return vs SPY (%)', fontsize=11)
axes[1].set_title(f'Model 2: Market-Adjusted\nR² = {r_squared2:.4f}, p = {pearson_p2:.4f}', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Model 3: vs Sector
axes[2].scatter(X3, y3, alpha=0.6, s=80, color='purple')
axes[2].plot(X3, y3_pred, color='red', linewidth=2, label=f'y = {intercept3:.2f} + {coefficient3:.2f}x')
axes[2].set_xlabel('Q2 2023 Average Sentiment', fontsize=11)
axes[2].set_ylabel('Q3 2023 Excess Return vs Sector (%)', fontsize=11)
axes[2].set_title(f'Model 3: Sector-Adjusted\nR² = {r_squared3:.4f}, p = {pearson_p3:.4f}', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Adjust layout and save figures as a PNG file
plt.tight_layout()
plt.savefig('images/figures/hypothesis/three_models_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: images/figures/hypothesis/three_models_comparison.png")
plt.close()

# Visualization 2: Annotated scatter plot (Model 2 - most important)
plt.figure(figsize=(12, 8))
plt.scatter(X2, y2, s=100, alpha=0.6, color='steelblue')

# Add ticker labels to each point
for i, row in merged_df.iterrows():
    plt.annotate(row['ticker'], 
                (row['q2_2023_avg_sentiment'], row['excess_vs_spy']),
                fontsize=9, alpha=0.7, ha='center')

# Regression line for Model 2
plt.plot(X2, y2_pred, color='red', linewidth=2, linestyle='--', 
         label=f'Regression Line: y = {intercept2:.2f} + {coefficient2:.2f}x')

plt.xlabel('Q2 2023 Average Sentiment Score', fontsize=13)
plt.ylabel('Q3 2023 Excess Return vs SPY (%)', fontsize=13)
plt.title(f'Reddit Sentiment vs Stock Performance (Market-Adjusted)\nPearson r = {pearson_r2:.4f}, p = {pearson_p2:.4f}, R² = {r_squared2:.4f}', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

# Adjust layout and save figure as a PNG file
plt.tight_layout()
plt.savefig('images/figures/hypothesis/sentiment_vs_excess_spy_labeled.png', dpi=300, bbox_inches='tight')
print("✓ Saved: images/figures/hypothesis/sentiment_vs_excess_spy_labeled.png")
plt.close()

# Visualization 3: Residual plots for all three models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(y1_pred, residuals1, alpha=0.6, color='steelblue')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Fitted Values', fontsize=11)
axes[0].set_ylabel('Residuals', fontsize=11)
axes[0].set_title('Model 1: Residual Plot', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y2_pred, residuals2, alpha=0.6, color='green')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Fitted Values', fontsize=11)
axes[1].set_ylabel('Residuals', fontsize=11)
axes[1].set_title('Model 2: Residual Plot', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(y3_pred, residuals3, alpha=0.6, color='purple')
axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[2].set_xlabel('Fitted Values', fontsize=11)
axes[2].set_ylabel('Residuals', fontsize=11)
axes[2].set_title('Model 3: Residual Plot', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

# Adjust layout and save figures as a PNG file
plt.tight_layout()
plt.savefig('images/diagnostics/model/residual_plots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: images/diagnostics/model/residual_plots.png")
plt.close()

# Visualization 4: Comparison bar chart of R² values
models = ['Raw Returns', 'vs SPY', 'vs Sector']
r_squared_values = [r_squared1, r_squared2, r_squared3]
colors = ['steelblue', 'green', 'purple']

plt.figure(figsize=(10, 6))
bars = plt.bar(models, r_squared_values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('R² Value', fontsize=13)
plt.title('Model Comparison: Explanatory Power (R²)', fontsize=14, fontweight='bold')
plt.ylim(0, max(r_squared_values) * 1.3)

# Add value labels on bars
for bar, val in zip(bars, r_squared_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')

# Adjust layout and save figure as a PNG file
plt.tight_layout()
plt.savefig('images/diagnostics/model/r_squared_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: images/diagnostics/model/r_squared_comparison.png")
plt.close()

# Visualization 5: Correlation coefficients comparison
fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(models))
width = 0.35

pearson_values = [pearson_r1, pearson_r2, pearson_r3]
spearman_values = [spearman_r1, spearman_r2, spearman_r3]

bars1 = ax.bar(x_pos - width/2, pearson_values, width, label='Pearson r', alpha=0.8, color='coral')
bars2 = ax.bar(x_pos + width/2, spearman_values, width, label='Spearman ρ', alpha=0.8, color='teal')

ax.set_xlabel('Model', fontsize=13)
ax.set_ylabel('Correlation Coefficient', fontsize=13)
ax.set_title('Correlation Coefficients Across Models', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Adjust layout and save figures as a PNG file
plt.tight_layout()
plt.savefig('images/diagnostics/model/correlation_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: images/diagnostics/model/correlation_comparison.png")
plt.close()

# Visualization #6: SECTOR-HIGHLIGHTED CORRELATION PLOT
# Load merged dataset (same file used earlier in this script)
merged_path = "./data/merged_sentiment_returns.csv"
df = pd.read_csv(merged_path)

# Define sectors for highlighting
tech = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "INTC", "CRM", "ORCL", "ADBE", "CSCO", "UBER"]
finance = ["JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP", "SCHW"]

# Map tickers to sectors
def map_sector(ticker):
    if ticker in tech:
        return "Tech"
    elif ticker in finance:
        return "Finance"
    else:
        return "Other"

# Add sector column to DataFrame
df["Sector"] = df["ticker"].apply(map_sector)

# Scatter plot with sector highlights
plt.figure(figsize=(10, 7))

sector_colors = {
    "Tech": "tab:blue",
    "Finance": "tab:red",
    "Other": "tab:green"
}

# Plot each sector with different colors
for sector, color in sector_colors.items():
    subset = df[df["Sector"] == sector]
    plt.scatter(
        subset["q2_2023_avg_sentiment"], 
        subset["q3_return_pct"],
        color=color,
        s=120,
        alpha=0.75,
        label=sector
    )

# Axes & labels
plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)

plt.xlabel("Average Sentiment (Q2 2023)")
plt.ylabel("Q3 Return (%)")
plt.title("Q2 Sentiment vs Q3 Returns Highlighted by Sector")

plt.legend()
plt.tight_layout()

# Save figure as PNG file
plt.savefig("./images/figures/hypothesis/sector_highlight_scatter.png", dpi=300, bbox_inches='tight')
print("✓ Saved: images/figures/hypothesis/sector_highlight_scatter.png")

plt.show()

# Visualization #7: Sector-specific regression lines

plt.figure(figsize=(10, 7))

for sector, color in sector_colors.items():
    subset = df[df["Sector"] == sector]
    x = subset["q2_2023_avg_sentiment"]   # FIXED
    y = subset["q3_return_pct"]           # FIXED

    plt.scatter(x, y, color=color, s=120, alpha=0.6, label=f"{sector} points")

    # Regression line
    if len(subset) >= 2:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m * x_line + b
        plt.plot(x_line, y_line, color=color, linewidth=2, label=f"{sector} trend")

plt.axhline(0, color="black", linewidth=1)
plt.axvline(0, color="black", linewidth=1)

plt.xlabel("Average Sentiment (Q2 2023)")
plt.ylabel("Q3 Return (%)")
plt.title("Sector-Specific Regression Trends")

plt.legend()

# Adjust layout and save figure as PNG file
plt.tight_layout()
plt.savefig("./images/figures/hypothesis/sector_trendlines.png", dpi=300, bbox_inches='tight')
print("✓ Saved: images/figures/hypothesis/sector_trendlines.png")

plt.show()

print("\n" + "=" * 70)
print("✓ ANALYSIS COMPLETE!")
print("=" * 70)

print("\nGenerated files:")
print("  • data/merged_sentiment_returns.csv - Combined dataset")
print("  • data/regression_summary.csv - Statistical summary")
print("  • images/figures/hypothesis/three_models_comparison.png - Side-by-side regression plots")
print("  • images/figures/hypothesis/sentiment_vs_excess_spy_labeled.png - Detailed scatter with tickers")
print("  • images/diagnostics/model/residual_plots.png - Regression diagnostics")
print("  • images/diagnostics/model/r_squared_comparison.png - Model explanatory power")
print("  • images/diagnostics/model/correlation_comparison.png - Pearson vs Spearman")
print("  • images/figures/hypothesis/sector_highlight_scatter.png - Sector-highlighted scatterplot")
print("  • images/figures/hypothesis/sector_trendlines.png - Sector-specific regression lines")

# Interpretation helper
print("\n" + "=" * 70)
print("INTERPRETATION GUIDE")
print("=" * 70)

if pearson_p2 < 0.05:
    print(f"\n✓ Model 2 (vs SPY) shows SIGNIFICANT correlation (p = {pearson_p2:.4f})")
    print(f"  → Reddit sentiment in Q2 predicts market-adjusted returns in Q3")
else:
    print(f"\n✗ Model 2 (vs SPY) shows NO significant correlation (p = {pearson_p2:.4f})")
    print(f"  → Reddit sentiment does NOT predict market-adjusted returns")

if pearson_p3 < 0.05:
    print(f"\n✓ Model 3 (vs Sector) shows SIGNIFICANT correlation (p = {pearson_p3:.4f})")
    print(f"  → Reddit sentiment predicts stock-specific performance beyond sector trends")
else:
    print(f"\n✗ Model 3 (vs Sector) shows NO significant correlation (p = {pearson_p3:.4f})")
    print(f"  → Reddit sentiment does NOT predict sector-adjusted returns")

print("\nR² Interpretation:")
print(f"  Model 1: {r_squared1*100:.2f}% of return variance explained by sentiment")
print(f"  Model 2: {r_squared2*100:.2f}% of market-adjusted return variance explained")
print(f"  Model 3: {r_squared3*100:.2f}% of sector-adjusted return variance explained")

print("\n" + "=" * 70)