# Reddit Sentiment and Stock Performance: A Predictive Analysis

**Authors:** Eshan Prakash, Eduardo Rebollar, and Aaron Good

## Introduction

### Background

Online social media discussion platforms like Reddit have become influential spaces where investors share information and coordinate trading activity. The 2021 GameStop surge demonstrated how subreddits like r/WallStreetBets can drive extreme market volatility, raising questions about whether social media sentiment merely reflects investor mood or actually predicts and influences price movements.

While prior studies have shown correlations between social media sentiment and short-term market activity, the strength, consistency, and predictive power of Reddit sentiment across different stocks and time horizons remains unclear. Understanding this relationship has broad implications for investors, researchers, and policymakers as financial discourse increasingly occurs online.

### Statement of Significance

This research investigates whether Reddit user sentiment about publicly traded stocks can predict subsequent quarterly stock price movements. Finding that Reddit sentiment predicts stock performance would suggest that online stock discussion acts as a legitimate leading indicator of market behavior. Conversely, finding no predictive relationship would suggest that online sentiment is largely reactive to market movements rather than predictive of them.

Either outcome advances our understanding of how information spreads in modern retail investing communities and the role of social media in financial markets.

---

## Research Question

**Does Reddit user sentiment about publicly traded stocks predict subsequent quarterly stock price movements?**

We examine whether sentiment expressed in Reddit discussions during Q2 2023 (April–June) correlates with stock performance in Q3 2023 (July–September). 

**Hypothesis:** We hypothesize that higher average sentiment in Q2 will correspond with increased stock returns in Q3, suggesting that Reddit sentiment functions as a predictor of stock price changes rather than merely reflecting concurrent market trends.

---

## Data Collection

### Data Sources

**Reddit Data:**
- API: Pullpush.io API (archival Reddit data)
- Subreddits: r/WallStreetBets (speculative trading), r/stocks (general analysis), r/StockMarket (macro commentary)
- Time Period: April 1 – June 30, 2023 (Q2 2023)
- Total Posts Collected: **Filtered based on engagement thresholds**

**Stock Data:**
- Source: Yahoo Finance (yfinance Python library)
- Time Period: July 1 – September 30, 2023 (Q3 2023)
- Benchmarks: S&P 500 (SPY) and sector-specific ETFs (XLK, XLF, XLV, XLE, ITA)

### Stock Selection

We analyzed **50 stocks across 5 sectors:**
- **Technology (15 stocks):** AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AMD, NFLX, INTC, CRM, ORCL, ADBE, CSCO, UBER
- **Finance (10 stocks):** JPM, BAC, WFC, GS, MS, C, V, MA, AXP, SCHW
- **Healthcare (10 stocks):** JNJ, UNH, PFE, ABBV, LLY, MRK, TMO, CVS, AMGN, BMY
- **Energy (10 stocks):** XOM, CVX, COP, SLB, EOG, OXY, MPC, PSX, VLO, HAL
- **Aerospace/Defense (5 stocks):** LMT, RTX, BA, NOC, GD

### Filtering Criteria

To ensure data quality, posts were included only if they:
1. Originated from one of the three target subreddits
2. Explicitly mentioned stock tickers or company names
3. Exceeded the subreddit's median upvote or comment count
4. Were written in English
5. Fell within the April 1–June 30, 2023 timeframe

**Median Engagement Thresholds by Subreddit:**
- r/WallStreetBets: 27 upvotes OR 21 comments
- r/stocks: 11 upvotes OR 25 comments
- r/StockMarket: 20 upvotes OR 15 comments

After applying these filters and requiring stocks to have at least the median number of mentions across all stocks, **23 stocks** met the threshold for analysis.

---

## Methodology

### Sentiment Analysis

We used **VADER (Valence Aware Dictionary and sEntiment Reasoner)**, a lexicon-based sentiment analysis tool specifically designed for social media text. VADER produces a compound sentiment score ranging from -1 (most negative) to +1 (most positive).

**Process:**
1. Combined post title and body text for each Reddit post
2. Computed VADER compound sentiment score for each post
3. Aggregated sentiment scores by stock ticker across Q2 2023
4. Calculated mean sentiment score per ticker as the independent variable

**Sentiment Classification:**
- Positive: compound score ≥ 0.05
- Neutral: -0.05 < compound score < 0.05
- Negative: compound score ≤ -0.05

### Stock Return Calculation

For each stock, we calculated three measures of Q3 2023 performance:

1. **Raw Return:** `(Q3_End_Price - Q3_Start_Price) / Q3_Start_Price`
2. **Market-Adjusted Return:** `Raw_Return - SPY_Return` (controls for overall market movement)
3. **Sector-Adjusted Return:** `Raw_Return - Sector_ETF_Return` (controls for sector-specific trends)

### Statistical Analysis

We tested three regression models with increasing levels of control:

**Model 1: Raw Returns**
- Dependent Variable: Q3 raw stock return
- Independent Variable: Q2 average sentiment
- Tests whether sentiment predicts absolute price movement

**Model 2: Market-Adjusted Returns**
- Dependent Variable: Q3 excess return vs. SPY
- Independent Variable: Q2 average sentiment
- Controls for overall market trends
- Tests whether sentiment predicts outperformance relative to the market

**Model 3: Sector-Adjusted Returns**
- Dependent Variable: Q3 excess return vs. sector benchmark
- Independent Variable: Q2 average sentiment
- Controls for both market and sector-specific trends
- Tests whether sentiment predicts stock-specific performance

For each model, we computed:
- **Pearson correlation coefficient** (measures linear relationship)
- **Spearman rank correlation** (non-parametric alternative)
- **Linear regression** (R², coefficient, p-value, standard error)

Statistical significance threshold: α = 0.05

---

## Results

### Descriptive Statistics

**Final Dataset:**
- Number of stocks analyzed: **23**
- Q2 2023 average sentiment range: 0.415 to 0.833
- Q3 2023 raw return range: -11.2% to 9.1%
- SPY (S&P 500) Q3 return: **-3.3%**

**Sample of Merged Data:**

| Ticker | Q2 Avg Sentiment | Q2 Post Count | Q3 Return (%) | Excess vs SPY (%) | Excess vs Sector (%) |
|--------|------------------|---------------|---------------|-------------------|----------------------|
| AAPL   | 0.425           | 197           | -0.11         | -0.08             | -0.06                |
| GOOGL  | 0.625           | 167           | 9.14          | 12.48             | 14.45                |
| CRM    | 0.833           | 13            | -4.19         | -0.86             | 1.12                 |
| CVS    | 0.806           | 8             | 0.89          | 4.23              | 2.69                 |

### Correlation Analysis

#### Model 1: Q2 Sentiment vs Q3 Raw Returns

| Metric | Value | p-value |
|--------|-------|---------|
| **Pearson r** | 0.3112 | 0.1483 |
| **Spearman ρ** | 0.2312 | 0.2884 |
| **R²** | 0.0968 | - |
| **Coefficient (β)** | 0.1355 | - |
| **Standard Error** | 0.0155 | - |

**Regression Equation:** `Q3_Return = -0.10 + 0.14 × Sentiment`

#### Model 2: Q2 Sentiment vs Q3 Excess Returns (vs SPY)

| Metric | Value | p-value |
|--------|-------|---------|
| **Pearson r** | 0.3112 | 0.1483 |
| **Spearman ρ** | 0.2312 | 0.2884 |
| **R²** | 0.0968 | - |
| **Coefficient (β)** | 0.1355 | - |
| **Standard Error** | 0.0155 | - |

**Regression Equation:** `Excess_vs_SPY = -0.07 + 0.14 × Sentiment`

#### Model 3: Q2 Sentiment vs Q3 Excess Returns (vs Sector)

| Metric | Value | p-value |
|--------|-------|---------|
| **Pearson r** | 0.3153 | 0.1428 |
| **Spearman ρ** | 0.2213 | 0.3101 |
| **R²** | 0.0994 | - |
| **Coefficient (β)** | 0.1391 | - |
| **Standard Error** | 0.0157 | - |

**Regression Equation:** `Excess_vs_Sector = -0.06 + 0.14 × Sentiment`

### Summary Across All Models

| Model | Pearson r | p-value | R² | Variance Explained |
|-------|-----------|---------|----|--------------------|
| Model 1: Raw Returns | 0.3112 | 0.1483 | 0.0968 | 9.68% |
| Model 2: vs SPY | 0.3112 | 0.1483 | 0.0968 | 9.68% |
| Model 3: vs Sector | 0.3153 | 0.1428 | 0.0994 | 9.94% |

### Key Findings

1. **No statistically significant relationship** was found between Q2 Reddit sentiment and Q3 stock returns in any of the three models (all p-values > 0.05)

2. **Weak positive correlation** exists across all models (r ≈ 0.31), but this could be due to random chance given the p-values

3. **Low explanatory power:** Reddit sentiment explains only ~10% of the variance in Q3 stock returns

4. **Consistent results across control levels:** The lack of significance holds whether measuring raw returns, market-adjusted returns, or sector-adjusted returns

5. **Regression coefficient interpretation:** For every 1-unit increase in sentiment score, Q3 returns increase by approximately 0.14%, but this relationship is not statistically reliable

---

## Discussion

### Interpretation of Results

Our analysis **does not support the hypothesis** that Reddit sentiment during Q2 2023 predicts stock performance in Q3 2023. While we observed a weak positive correlation (r ≈ 0.31), the relationship was not statistically significant (p > 0.14 for all models), meaning we cannot confidently distinguish this pattern from random noise.

The low R² values (~0.10) indicate that even if a relationship exists, Reddit sentiment explains only about 10% of the variation in quarterly stock returns. The remaining 90% is driven by other factors such as:
- Company fundamentals (earnings, revenue, growth)
- Macroeconomic conditions (interest rates, inflation, GDP growth)
- Industry-specific trends
- Institutional investor activity
- Geopolitical events

### Correlation vs. Causation

Even if our results had shown statistical significance, establishing causation would require additional evidence. Several plausible explanations could account for a correlation between Reddit sentiment and stock returns:

1. **Sentiment causes returns:** Reddit discussions influence investor behavior, driving price movements
2. **Returns cause sentiment:** Stock performance affects what people discuss online (reverse causality)
3. **Third variable:** Both sentiment and returns respond to external news events (confounding)
4. **Selection bias:** Stocks already trending attract more discussion

Our study design cannot definitively distinguish between these mechanisms, highlighting an important limitation in observational social media research.

### Comparison to Prior Research

Our findings differ from some prior studies that found short-term predictive relationships between social media sentiment and stock prices. Potential explanations for this discrepancy include:

1. **Time horizon:** Most studies examine daily or weekly correlations; our quarterly lag may be too long for sentiment signals to persist
2. **Market period:** Q2-Q3 2023 may have unique characteristics that reduce the predictive power of sentiment
3. **Platform differences:** Reddit may function differently than Twitter or StockTwits in terms of information quality
4. **Stock selection:** Our focus on large-cap stocks across multiple sectors may dilute effects that are stronger for specific categories (e.g., meme stocks)

### Implications for Investors

Our results suggest that **retail investors should be cautious about using Reddit sentiment as a standalone predictor** of quarterly stock performance. While online discussions may reflect useful information, they do not appear to reliably forecast returns over a 3-month horizon.

However, this does not mean Reddit is uninformative. Sentiment analysis may still have value for:
- **Shorter time horizons** (daily/weekly trading)
- **Specific stock categories** (small-cap, high-retail-interest stocks)
- **Risk management** (identifying extreme sentiment as a contrarian indicator)
- **Combined with other signals** (fundamental analysis, technical indicators)

### Theoretical Contributions

From an academic perspective, our null findings contribute to the literature by:

1. **Challenging the hype:** Not all social media signals predict market outcomes
2. **Highlighting temporal dynamics:** The predictive power of sentiment may decay rapidly
3. **Emphasizing market efficiency:** For large-cap stocks, publicly available sentiment is likely already priced in by institutional investors
4. **Supporting the reactive interpretation:** Reddit discussion may reflect rather than drive price movements

---

## Limitations

Our study has several important limitations that should be considered when interpreting the results:

### 1. Sample Size
- Only **23 stocks** met our inclusion criteria after filtering
- Small sample size reduces statistical power to detect effects
- Limits generalizability to the broader market

### 2. Time Period Constraints
- Analysis covers only **one quarter** (Q2-Q3 2023)
- Market conditions during this period may not be representative
- Seasonal effects or period-specific events may influence results

### 3. Sentiment Analysis Limitations
- **VADER** is a lexicon-based approach that may miss context, sarcasm, or nuanced language
- Does not account for the credibility or influence of individual users
- Treats all posts equally regardless of engagement quality
- May misclassify complex financial discussions

### 4. Lag Structure
- **3-month lag** between sentiment and returns may be too long
- Sentiment signals may decay or be superseded by new information
- Optimal lag structure unknown and may vary by stock

### 5. Confounding Variables
- Does not control for:
  - Earnings announcements
  - Analyst upgrades/downgrades
  - News events
  - Institutional trading activity
  - Short interest and options activity
  - Changes in fundamentals

### 6. Selection Bias
- Median engagement filter may exclude relevant but less-discussed stocks
- Reddit users are not representative of all investors
- Focus on three subreddits may miss activity elsewhere

### 7. Causality
- Observational design cannot establish causal relationships
- Cannot rule out reverse causality or omitted variable bias

### 8. Data Quality
- Relies on archived Reddit data which may be incomplete
- Deleted posts or comments are excluded
- Bot activity and spam were not explicitly filtered

---

## Conclusion

This study investigated whether Reddit user sentiment about publicly traded stocks during Q2 2023 could predict stock performance in Q3 2023. Using sentiment analysis on posts from r/WallStreetBets, r/stocks, and r/StockMarket, we analyzed 23 stocks across five sectors.

**Key findings:**
1. No statistically significant relationship exists between Q2 Reddit sentiment and Q3 stock returns (p > 0.14 for all models)
2. Reddit sentiment explains only ~10% of variance in quarterly returns
3. Results hold across raw returns, market-adjusted returns, and sector-adjusted returns
4. Evidence suggests Reddit sentiment is more **reactive** than **predictive** over quarterly horizons

**Implications:**
- Investors should exercise caution using Reddit sentiment as a standalone quarterly predictor
- Social media sentiment may be more useful for short-term trading or as part of a multi-signal approach
- For large-cap stocks, sentiment is likely already incorporated into prices by the time it becomes widespread

**Future research directions:**
1. Examine shorter time lags (daily, weekly, monthly)
2. Expand to more stocks and longer time periods
3. Distinguish between different types of posts (DD, memes, news)
4. Weight posts by user credibility or engagement
5. Investigate specific stock categories (meme stocks, small-cap)
6. Use more sophisticated NLP techniques (transformer models, aspect-based sentiment)
7. Control for additional confounding variables

Despite finding null results, this research contributes to our understanding of social media's role in financial markets by providing evidence that quarterly stock returns are not reliably predicted by Reddit sentiment alone. This finding has practical value for retail investors and contributes to academic knowledge about information efficiency in modern markets.

---

## References

### Data Sources
- Pullpush.io API: https://pullpush.io/ (Reddit archival data)
- Yahoo Finance (yfinance): https://pypi.org/project/yfinance/

### Tools & Libraries
- VADER Sentiment Analysis: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14).
- Python Libraries: pandas, numpy, scipy, sklearn, matplotlib, seaborn

### Relevant Literature
- Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market. *Journal of Computational Science*, 2(1), 1-8.
- Sprenger, T. O., Tumasjan, A., Sandner, P. G., & Welpe, I. M. (2014). Tweets and trades: The information content of stock microblogs. *European Financial Management*, 20(5), 926-957.
- Da, Z., Engelberg, J., & Gao, P. (2015). The sum of all FEARS investor sentiment and asset prices. *Review of Financial Studies*, 28(1), 1-32.

---

## Repository Structure

```
reddit-stock-sentiment/
│
├── data/
│   ├── reddit_posts_q2_2023_full.json           # Raw Reddit posts
│   ├── reddit_posts_q2_2023_filtered.json       # Filtered posts
│   ├── reddit_posts_q2_2023_with_sentiment.json # Posts with VADER scores
│   ├── q2_2023_sentiment_by_ticker.json         # Aggregated Q2 sentiment
│   ├── q2_2023_sentiment_by_ticker.csv
│   ├── q3_2023_with_benchmarks.json             # Q3 stock returns
│   ├── q3_2023_with_benchmarks.csv
│   ├── merged_sentiment_returns.csv             # Final merged dataset
│   └── regression_summary.csv                   # Statistical results
│
├── images/
│   ├── sentiment_by_ticker.png                  # Q2 sentiment visualization
│   ├── sentiment_vs_volume.png
│   ├── sentiment_distribution.png
│   ├── top_5_discussed.png
│   ├── sentiment_heatmap.png
│   ├── three_models_comparison.png              # Main regression results
│   ├── sentiment_vs_excess_spy_labeled.png
│   ├── residual_plots.png
│   ├── r_squared_comparison.png
│   └── correlation_comparison.png
│
├── collect_pullpush.py          # Step 1: Collect Reddit posts
├── filter_posts.py              # Step 2: Filter for stock mentions
├── sentiment_analysis.py        # Step 3: VADER sentiment analysis
├── aggregate_sentiment.py       # Step 4: Aggregate by ticker
├── yfinance_fetch_q3.py         # Step 5: Fetch Q3 stock returns
├── correlate_sentiment_returns.py # Step 6: Statistical analysis
│
└── README.md                    # This file
```

## How to Reproduce

1. **Install dependencies:**
   ```bash
   pip install requests vaderSentiment yfinance pandas numpy scipy scikit-learn matplotlib seaborn
   ```

2. **Run scripts in order:**
   ```bash
   python collect_pullpush.py          # Collect Reddit data
   python filter_posts.py              # Filter for stock mentions
   python sentiment_analysis.py        # Compute sentiment scores
   python aggregate_sentiment.py       # Aggregate by ticker
   python yfinance_fetch_q3.py         # Fetch stock returns
   python correlate_sentiment_returns.py # Run statistical analysis
   ```

3. **View results:**
   - Statistical outputs: `data/regression_summary.csv`
   - Visualizations: `images/` folder
   - Merged dataset: `data/merged_sentiment_returns.csv`

---