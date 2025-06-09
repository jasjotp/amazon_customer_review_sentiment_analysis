import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# load preprocessed reviews with VADER sentiment scores
vader_df = pd.read_csv('reviews_with_VADER_sentiment.csv')

# load preprocessed reviews with DistilBERT sentiment scores
distilbert_df = pd.read_csv('reviews_with_DistilBERT_sentiment.csv')

# plot a barplot of the average VADER sentiment score for each score 
plt.figure(figsize = (10, 8))
ax1 = sns.barplot(data = vader_df, x = 'Score', y = 'compound')
ax1.set_title('Compound Sentiment Score for Each Rating (VADER)')
plt.xlabel('Review Score')
plt.ylabel('Compound Sentiment Score')
plt.tight_layout()

compound_sentiment_path = os.path.join('graphs', 'vader_sentiment_per_rating.png')
plt.savefig(compound_sentiment_path) # we see a trend that the more positive that the review is (higher stars) results in a more positive sentiment to the review which makes sense

# plot a barplot of the average DistilBERT sentiment score for each score 

# create a compound DistilBERT score
distilbert_df['distil_compound_score'] = distilbert_df['distil_pos'] - distilbert_df['distil_neg']

plt.figure(figsize = (10, 8))
ax1 = sns.barplot(data = distilbert_df, x = 'Score', y = 'distil_compound_score')
ax1.set_title('Compound Sentiment Score for Each Rating (DistilBERT)')
plt.xlabel('Review Score')
plt.ylabel('Compound Sentiment Score')
plt.tight_layout()

compound_sentiment_distilbert_path = os.path.join('graphs', 'distilbert_sentiment_per_rating.png')
plt.savefig(compound_sentiment_distilbert_path) # we see a trend that the more positive that the review is (higher stars) results in a more positive sentiment to the review which makes sense

# plot 3 subplots of the positive, neutral, and negative VADER sentiment score by each review rating
fig, axs = plt.subplots(1, 3, figsize = (15, 8))
sns.barplot(data = vader_df, x = 'Score', y = 'pos', ax = axs[0])
sns.barplot(data = vader_df, x = 'Score', y = 'neu', ax = axs[1])
sns.barplot(data = vader_df, x = 'Score', y = 'neg', ax = axs[2])
axs[0].set_title('Positive Sentiment Score by Review Rating')
axs[1].set_title('Neutral Sentiment Score by Review Rating')
axs[2].set_title('Negative Sentiment Score by Review Rating')

for ax in axs:
    ax.set_xlabel('Review Score')
    ax.set_ylabel('Sentiment Score')

combined_sentiment_path = os.path.join('graphs', 'combined_vader_sentiment_per_rating.png')
plt.tight_layout()
plt.savefig(combined_sentiment_path) # we see a trend that the more positive that the review is (higher stars) results in a more positive sentiment to the review which makes sense

# plot a boxplot of sentiment score by Star Rating (VADER vs DistilBERT) to show if both models' sentiment scores align with review stars (do higher reviews result in higher sentiment/postiive reviews)
plt.figure(figsize = (14, 6))

# plot a boxplot of the average compound sentiment scores for the VADER model for each score
plt.subplot(1, 2, 1)
sns.boxplot(data = vader_df, x = 'Score', y = 'compound')
plt.title('VADER Compound Score by Review Rating')

# plot a boxplot of the average compound sentiment scores for the DistilBERT model for each score
plt.subplot(1, 2, 2)
sns.boxplot(data = distilbert_df, x = 'Score', y = 'distil_compound_score')
plt.title('DistilBERT Compound Score by Review Rating')

plt.tight_layout()
plt.savefig(os.path.join('graphs', 'boxplot_sentiment_by_score.png'))

# plot a linechart with a trendline to see if sentiment has increased over time (1 linechart for VADER and 1 for DistilBERT)
plt.figure(figsize = (14, 6))

# create a quarter column in each dataframe after ensuring it is in datetime format 
vader_df['Time'] = pd.to_datetime(vader_df['Time'], unit = 's')
distilbert_df['Time'] = pd.to_datetime(distilbert_df['Time'], unit = 's')

vader_df['Quarter'] = vader_df['Time'].dt.to_period('Q')
distilbert_df['Quarter'] = distilbert_df['Time'].dt.to_period('Q')

# format the quarter as Year Qx 
vader_df['Quarter'] = vader_df['Quarter'].apply(lambda x: f'{x.year} Q{x.quarter}')
distilbert_df['Quarter'] = distilbert_df['Quarter'].apply(lambda x: f'{x.year} Q{x.quarter}')

# calculate the average score per quarter/year
vader_quarterly_score = vader_df.groupby('Quarter')['compound'].mean().reset_index()
distilbert_quarterly_score = distilbert_df.groupby('Quarter')['distil_compound_score'].mean().reset_index()

# create a nuemric x axis to plot the trend line 
vader_quarterly_score['QuarterIndex'] = np.arange(len(vader_quarterly_score)) 
distilbert_quarterly_score['QuarterIndex'] = np.arange(len(distilbert_quarterly_score)) 

# fit the trendline using linear regression (y = mx + b - 1st polynomial) to find the best fitting line for each point using the least squares method
vader_trend = np.poly1d(np.polyfit(vader_quarterly_score['QuarterIndex'], vader_quarterly_score['compound'], 1))
distilbert_trend = np.poly1d(np.polyfit(distilbert_quarterly_score['QuarterIndex'], distilbert_quarterly_score['distil_compound_score'], 1))

# linechart for VADER sentiment scores over time 
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# VADER plot
axs[0].plot(vader_quarterly_score['Quarter'], vader_quarterly_score['compound'], marker = 'o', label = 'Vader Avg')
axs[0].plot(vader_quarterly_score['Quarter'], vader_trend(vader_quarterly_score['QuarterIndex']), linestyle='--', color='red', label='Trendline')
axs[0].set_title('VADER Sentiment Over Time (Quarterly)')
axs[0].set_xlabel('Quarter', fontsize = 8)
axs[0].set_ylabel('Avg Compound Sentiment')
axs[0].tick_params(axis = 'x', rotation = 45, labelsize = 8)
axs[0].legend()

# VADER plot
axs[1].plot(distilbert_quarterly_score['Quarter'], distilbert_quarterly_score['distil_compound_score'], marker = 'o', label = 'DistilBERT Avg')
axs[1].plot(distilbert_quarterly_score['Quarter'], distilbert_trend(distilbert_quarterly_score['QuarterIndex']), linestyle = '--', color = 'red', label = 'Trendline')
axs[1].set_title('DistilBERT Sentiment Over Time (Quarterly)')
axs[1].set_xlabel('Quarter', fontsize = 8)
axs[1].set_ylabel('Avg Compound Sentiment')
axs[1].tick_params(axis = 'x', rotation = 45, labelsize = 8)
axs[1].legend()

plt.tight_layout()
plt.savefig(os.path.join('graphs', 'quarterly_sentiment_trends.png'))

# merge the VADER sentiment scores df with the DistilBERT sentiment scores df on Id 
combined_df = pd.merge(vader_df[['Id', 'neg', 'neu', 'pos', 'compound']], distilbert_df, on = 'Id')
print(combined_df.head())
print(combined_df.columns)

# plot a subplot with a histogram and scatter plot to compare the compound score of VADER vs the compound score of DistilBERT
plt.figure(figsize = (12, 6))

# plot a histogram to show the count of occurrences per score 
plt.subplot(1, 2, 1)
sns.histplot(vader_df['compound'], color = 'blue', label = 'VADER', kde = True)
sns.histplot(distilbert_df['distil_compound_score'], color = 'red', label = 'DistilBERT', kde = True)
plt.title('Sentiment Score Distribution (Compound)')
plt.xlabel('Compound Score')
plt.legend()

# plot the second subplot: a scatter plot 
# agreement threshold: that if the compound scores are within +/- x of each other, we can assume they are similar 
threshold = 0.15

# create a sentiment agreement flag, where Agree corresponds to where the compound score of both models are within + / - 0.15 of each other and Diagree otherwise
combined_df['sentiment_agreement'] = np.where(
    np.abs(combined_df['compound'] - combined_df['distil_compound_score']) <= threshold, 
    'Agree', 
    'Disagree'
)
plt.subplot(1, 2, 2)
sns.scatterplot(
    data = combined_df,
    x = 'compound',
    y = 'distil_compound_score', 
    hue = 'sentiment_agreement', 
    palette={'Agree': 'blue', 'Disagree': 'red'},
    alpha = 0.5
)

plt.title('VADER vs DistilBERT Compound Scores')
plt.xlabel('VADER Compound Score')
plt.ylabel('DistilBERT Compound Score')

# add a reference line: y = x, to see perfect matches 
lims = [-1, 1]
plt.plot(lims, lims, 'k--', linewidth=1)

vader_distilbert_path = os.path.join('graphs', 'comparison_vader_distilbert_sentiment.png')

plt.tight_layout()
plt.savefig(vader_distilbert_path) 

# find the top disagreements between each model, to see where VADER (NLTK) and transformer models differ the most
