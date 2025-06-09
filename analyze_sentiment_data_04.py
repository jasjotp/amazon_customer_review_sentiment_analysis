import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from scipy.stats import pearsonr

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

# print out the agreement percentage to see how many reviews both models agree on 
agreement_percentage = (combined_df['sentiment_agreement'] == 'Agree').mean() * 100
print(f'Agreement Percentage Between VADER and DistilBERT Models (+/- 0.15): {agreement_percentage:.3f}')

# plot the scatter plot of the vader and distilbert sentiment scores on opposite axes
plt.subplot(1, 2, 2)
sns.scatterplot(
    data = combined_df,
    x = 'compound',
    y = 'distil_compound_score', 
    hue = 'sentiment_agreement', 
    palette={'Agree': 'blue', 'Disagree': 'red'},
    alpha = 0.5
)


# also calculate the pearson correlation coeficcient to add as an annotation in the scatter plot 
r, _ = pearsonr(combined_df['compound'], combined_df['distil_compound_score'])
print(f'Pearson Correlation Coeficcient Between VADER and DistilBERT sentiment scores: {r:.2f}')

# add the Pearson Correlation Coeficcient annotation
plt.annotate(f"Pearson r = {r:.2f}", 
             xy = (0.05, 0.9),                # relative position in axes coords
             xycoords = 'axes fraction',     # interpret as fraction of axes (0 to 1)
             fontsize = 12,
             bbox = dict(boxstyle = "round,pad=0.3", fc = "lightyellow", ec = "gray", lw = 1))

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
combined_df['diff'] = np.abs(combined_df['compound'] - combined_df['distil_compound_score'])
top_disagreements = combined_df.sort_values('diff', ascending = False).head(10)

# find the summaries of the reviews of the top_diagreements 
labels = top_disagreements['Summary']

vader_scores = top_disagreements['compound'].round(3)
distilbert_scores = top_disagreements['distil_compound_score'].round(3)

# create a table plot with the summary, and vader and distilbert score
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')  # hide plot axes

# create table data
table_data = [
    [label, vader, distil] for label, vader, distil in zip(labels, vader_scores, distilbert_scores)
]

# add a table plot
table = ax.table(
    cellText = table_data,
    colLabels = ["Text", "VADER Compound", "DistilBERT Compound"],
    colColours = ["#f2f2f2", "#d0e1f9", "#f9d0d0"], # got colour codes from: https://www.w3schools.com/colors/colors_picker.asp
    loc = 'center',
    cellLoc = 'center'
)

# set the fontsize and title
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1.2, 1.2)

plt.title("Top 10 Sentiment Disagreements: VADER vs. DistilBERT", fontsize=12)
plt.tight_layout()

top10_sentiment_disagreements_path = os.path.join('graphs', 'top10_sentiment_disagreements.png')
plt.tight_layout()
plt.savefig(top10_sentiment_disagreements_path) 

# create a confusion matrix to see how often both models assign the same label 
# create labels so vader is positive if compound score is >-= 0.05, negative if compound score is <= -0.05, and neutral otherwise, since the compound scores are from -1 to 1, with -1 indicating low sentiment and 1 indicating very positive sentiment
combined_df['vader_label'] = combined_df['compound'].apply(lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral')
combined_df['distilbert_label'] = combined_df['distil_compound_score'].apply(lambda x: 'positive' if x > 0 else 'negative')

# create a confusion matrix to see how many labels from VADER and DistilBERT match 
confusion_matrix = pd.crosstab(combined_df['vader_label'], combined_df['distilbert_label'])

plt.figure(figsize = (10, 6))
sns.heatmap(confusion_matrix, annot = True, fmt = 'd', cmap = 'Blues')
plt.title('VADER vs DistilBERT Sentiment Label Agreement')
plt.savefig(os.path.join('graphs', 'sentiment_label_agreement.png'))
