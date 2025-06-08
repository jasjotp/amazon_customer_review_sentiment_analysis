import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# load preprocessed reviews with VADER sentiment scores
df = pd.read_csv('reviews_with_VADER_sentiment.csv')

# plot a barplot of the average sentiment for each score 
plt.figure(figsize = (10, 8))
ax1 = sns.barplot(data = df, x = 'Score', y = 'compound')
ax1.set_title('Compound Sentiment Score for Each Rating')
plt.xlabel('Review Score')
plt.ylabel('Compound Sentiment Score')
plt.tight_layout()

compound_sentiment_path = os.path.join('graphs', 'vader_sentiment_per_rating.png')
plt.savefig(compound_sentiment_path) # we see a trend that the more positive that the review is (higher stars) results in a more positive sentiment to the review which makes sense

fig, axs = plt.subplots(1, 3, figsize = (15, 8))
sns.barplot(data = df, x = 'Score', y = 'pos', ax = axs[0])
sns.barplot(data = df, x = 'Score', y = 'neu', ax = axs[1])
sns.barplot(data = df, x = 'Score', y = 'neg', ax = axs[2])
axs[0].set_title('Positive Sentiment Score by Review Rating')
axs[1].set_title('Neutral Sentiment Score by Review Rating')
axs[2].set_title('Negative Sentiment Score by Review Rating')

for ax in axs:
    ax.set_xlabel('Review Score')
    ax.set_ylabel('Sentiment Score')

combined_sentiment_path = os.path.join('graphs', 'combined_vader_sentiment_per_rating.png')
plt.tight_layout()
plt.savefig(combined_sentiment_path) # we see a trend that the more positive that the review is (higher stars) results in a more positive sentiment to the review which makes sense

