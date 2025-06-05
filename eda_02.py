import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt

from helpers import load_data

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

# plot how many times each score occurs in all the reviews 
plt.figure(figsize=(10, 8))
ax1 = df['Score'].value_counts().sort_index(ascending = False) \
    .plot(kind = 'bar')
ax1.set_title('Count of Reviews by Star')
ax1.set_xlabel('Review Stars')
ax1.set_ylabel('Count')
plt.xticks(rotation = 0)

# add value labels on top of the bars 
for bar in ax1.patches:
    ax1.annotate(f'{int(bar.get_height())}',
                 (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1),
                 ha='center', va='bottom', fontsize=10)

# create graphs directory if it doesn't exist to save the graphs in 
graphs_dir = 'graphs'
os.makedirs(graphs_dir, exist_ok = True)

# save the plot in the graphs directory 
star_count_path = os.path.join(graphs_dir, 'reviews_star_distribution.png')
plt.tight_layout
plt.savefig(star_count_path)
print(f'Plot saved to {star_count_path}')

# plot the top 15 most common product ids 
plt.figure(figsize=(10, 8))
top_15_products = df['ProductId'].value_counts().nlargest(15)

ax2 = top_15_products.plot(
    kind='bar'
)
ax2.set_title('Count of Top 15 Most Reviewed Product IDs')
ax2.set_xlabel('Product ID')
ax2.set_ylabel('Review Count')
plt.xticks(rotation=45, ha='right')

# annotate the bars 
for bar in ax2.patches:
    ax2.annotate(f'{int(bar.get_height())}',
                 (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1),
                 ha='center', va='bottom', fontsize=10)

# save the plot 
top15_product_path = os.path.join(graphs_dir, 'top15_reviewed_products.png')
plt.tight_layout()
plt.savefig(top15_product_path)
print(f'Plot saved to {top15_product_path}')

# plot the top 10 most frequent UserId and ProfileNames 

# find the most common hours the reviews are posted at 

# plot a trendline of the average score of reviews over time

# find the most common words that show up in the body (text) of the review 

# find top words in 5-star reviews by making a word cloud 

# find the top words in 1-star reviews by making a word cloud 

# see if helpfulness score correlates with score - are helpful reviews usually more positive?

# see the helpfulness ratio across different scores (boxplot)

# see if the score distribution and the review (text) length or word count are correlated with higher scores - check if review lwngth varies by score

# check for duplicate reviews using ProductID, UserId, ProfileName, Score and very similar times 

# try to detect spammy users (many reviews in a short period of time or very similar reviews)