import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt

from helpers import load_data, plot_bar_with_annotations

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

# plot a trendline of the average score of reviews over time (quarterly trend)

# create a quarter column
df['Quarter'] = df['Time'].dt.to_period('Q')

# find the most common words that show up in the body (text) of the review 

# find top words in 5-star reviews by making a word cloud 

# find the top words in 1-star reviews by making a word cloud 

# see if helpfulness score correlates with score - are helpful reviews usually more positive?

# see the helpfulness ratio across different scores (boxplot)

# see if the score distribution and the review (text) length or word count are correlated with higher scores - check if review lwngth varies by score

# check for duplicate reviews using ProductID, UserId, ProfileName, Score and very similar times 

# try to detect spammy users (many reviews in a short period of time or very similar reviews)