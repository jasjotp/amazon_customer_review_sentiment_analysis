import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from collections import Counter
from wordcloud import WordCloud
import string

from helpers import load_data, plot_bar_with_annotations, get_clean_text, generate_wordcloud_from_tokens

# define your directory where the data and database is before reading in the full dataabse path with the load_data helper funcction 
database_path = r'data\database.sqlite'
df = load_data(database_path)

# initial analysis: check a preview of the data, columns, and information
print(df.head())
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe(include = 'all'))
print(df['Score'].describe(include = 'all')) # min score of 1 and max score of 5 

# check how many times each score occurs 
print(df['Score'].value_counts().sort_index(ascending = False))

# create graphs directory if it doesn't exist to save the graphs in 
graphs_dir = 'graphs'
os.makedirs(graphs_dir, exist_ok = True)

# plot how many times each score occurs in all the reviews 
plot_bar_with_annotations(data = df['Score'].value_counts().sort_index(ascending = False),
                          title = 'Count of Reviews by Star', 
                          xlabel = 'Review Stars', 
                          ylabel = 'Count',
                          save_path = os.path.join(graphs_dir, 'reviews_star_distribution.png'))

# plot the top 15 most common product ids 
top_15_products = df['ProductId'].value_counts().nlargest(15)

plot_bar_with_annotations(data = top_15_products,
                          title = 'Count of Top 15 Most Reviewed Product IDs', 
                          xlabel = 'Product ID', 
                          ylabel = 'Review Count',
                          rotation = 45,
                          save_path = os.path.join(graphs_dir, 'top15_reviewed_products.png'))

# plot the top 15 most frequent UserId and ProfileNames 
top15_users = df[['UserId', 'ProfileName']].value_counts().nlargest(15)

plot_bar_with_annotations(data = top15_users,
                          title = 'Top 15 Users and Profile Names by Review Count', 
                          xlabel = 'User ID and ProfileName', 
                          ylabel = 'Review Count',
                          rotation = 45,
                          save_path = os.path.join(graphs_dir, 'top15_users_profiles_review_count.png'))

# find the most common years, months, and days of the week the reviews are posted at 

# convert the time from a unix timestamp (default is in nanoseconds) to a datetime format (convert unix timestamp into a date format (seconds))
df['Time'] = pd.to_datetime(df['Time'], unit = 's')

df['Year'] = df['Time'].dt.year
print(df['Year'].value_counts())

# plot how many reviews occur in each year 
plot_bar_with_annotations(data = df['Year'].value_counts().sort_index(ascending = False),
                          title = 'Count of Reviews by Year', 
                          xlabel = 'Year', 
                          ylabel = 'Count',
                          save_path = os.path.join(graphs_dir, 'year_distribution.png'))

df['Month'] = df['Time'].dt.month
print(df['Month'].value_counts())

# plot how many reviews occur in each month 
plot_bar_with_annotations(data = df['Month'].value_counts(),
                          title = 'Count of Reviews by Month', 
                          xlabel = 'Month', 
                          ylabel = 'Count',
                          save_path = os.path.join(graphs_dir, 'month_distribution.png'))

df['DayOfWeek'] = df['Time'].dt.day_name()
print(df['DayOfWeek'].value_counts())

# plot how many reviews occur on each day of the week 
plot_bar_with_annotations(data = df['DayOfWeek'].value_counts(),
                          title = 'Count of Reviews by Day of the Week', 
                          xlabel = 'Day of the Week', 
                          ylabel = 'Count',
                          save_path = os.path.join(graphs_dir, 'dayOfWeek_distribution.png'))

# plot a trendline of the average score of reviews over time (quarterly trend) and forecas

# create a quarter column
df['Quarter'] = df['Time'].dt.to_period('Q')

# format the quarter as Year Qx 
df['Quarter'] = df['Quarter'].apply(lambda x: f'{x.year} Q{x.quarter}')

# calculat the average score per quarter/year
avg_score_per_quarter = df.groupby('Quarter')['Score'].mean().reset_index()

# create a nuemric x axis to plot the trend line 
avg_score_per_quarter['QuarterIndex'] = np.arange(len(avg_score_per_quarter)) 

# fit the trendline using linear regression (y = mx + b - 1st polynomial) to find the best fitting line for each point using the least squares method
z = np.polyfit(avg_score_per_quarter['QuarterIndex'], avg_score_per_quarter['Score'], 1)
p = np.poly1d(z) # turn (x, y, degree) into a a1 polynomial 

# generate the trend line values 
trendline = p(avg_score_per_quarter['QuarterIndex'])

# plot the actual scores 
plt.figure(figsize = (12, 6))
plt.plot(avg_score_per_quarter['Quarter'], avg_score_per_quarter['Score'], marker = 'o', label = 'Actual')

# plot trendline for each quarter 
plt.plot(avg_score_per_quarter['Quarter'], trendline, linestyle = '--', color = 'red', label = 'Trendline')

plt.xticks(rotation = 45, ha = 'right')
plt.xlabel('Quarter/Year')
plt.ylabel('Average Score')
plt.title('Average Review Score by Quarter/Year')
plt.tight_layout()
plt.grid(True)

avg_quarterly_score_path = os.path.join(graphs_dir, 'avg_quarterly_review_score.png')
plt.savefig(avg_quarterly_score_path)

# find the most common words that show up in the body (text) of the review 
# try to already find the punkt and stopwords packages to avoid repeated downloads
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# combine all reviews into one string and tokenize them
all_text = " ".join(df['Text'].astype(str).tolist())
tokens = word_tokenize(all_text.lower(), preserve_line = True)

# remove puntuation and stop words 
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

# count word frequencies 
word_freq = Counter(tokens)

# get top 20 most common words 
top20_most_common_words = word_freq.most_common(20)

for word, freq in top20_most_common_words:
    print(f'{word}: {freq}')

# plot the top 20 words that show up in the reviews 
top_words_df = pd.DataFrame(top20_most_common_words, columns = ['Word', 'Frequency'])

# convert the top words to a series with word as the index (since function expects a series with word as index and freq as value)
top_words_series = top_words_df.set_index('Word')['Frequency']

# set the bar colour to orange to match Amazon colours
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#FF9900'])

# plot the top 20 words using helper function 
plot_bar_with_annotations(
    data=top_words_series,
    title='Top 20 Most Common Words in Review Text',
    xlabel='Word',
    ylabel='Frequency',
    rotation=45,
    save_path=os.path.join(graphs_dir, 'top20_common_words.png'),
    figsize=(10, 6)
)

# find top words in 5-star reviews by making a word cloud 
tokens_5_star_reviews = get_clean_text(df, 5)
generate_wordcloud_from_tokens(tokens = tokens_5_star_reviews, 
                               title = "Top Words in 5-Star Reviews", 
                               wordcount = 100,
                               save_path = os.path.join(graphs_dir, 'wordcloud_5_star.png'))

# find the top words in 1-star reviews by making a word cloud 
tokens_1_star_reviews = get_clean_text(df, 1)
generate_wordcloud_from_tokens(tokens = tokens_1_star_reviews, 
                               title = "Top Words in 1-Star Reviews", 
                               wordcount = 100,
                               save_path = os.path.join(graphs_dir, 'wordcloud_1_star.png'))

# see the average helpfulness ratio across different scores (boxplot)
df['HelpfulnessRatio'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'].replace(0, np.nan)

# drop rows with NaN helpfulness ratio (from 0 denominator)
plot_df = df.dropna(subset = ['HelpfulnessRatio'])

# create boxplot
plt.figure(figsize = (10, 8))
sns.boxplot(data = plot_df, x = 'Score', y = 'HelpfulnessRatio', palette = 'viridis', hue = 'Score', legend = False)

plt.title('Helpfulness Ratio by Review Score')
plt.xlabel('Review Score')
plt.ylabel('Helpfulness Ratio (Numerator / Denominator)')
plt.tight_layout()

helpfulness_boxplot_path = os.path.join(graphs_dir, 'helpfulness_ratio_by_score_boxplot.png')
plt.savefig(helpfulness_boxplot_path)

# plot a heatmap to see if helpfulness score correlates with score and to see if the score distribution and the review (text) length/word count are correlated with higher scores 
df['ReviewLength'] = df['Text'].astype(str).apply(len) # number of characters 
df['WordCount'] = df['Text'].astype(str).apply(lambda x: len(x.split()))

numeric_features = df[['Score', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 
                       'HelpfulnessRatio', 'ReviewLength', 'WordCount', 'Month']]

plt.figure(figsize = (12, 10))
sns.heatmap(numeric_features.corr(), annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.title('Correlation Heatmap for Numerical Features')
plt.tight_layout()

heatmap_path = os.path.join(graphs_dir, 'correlation_heatmap.png')
plt.savefig(heatmap_path)