import sqlite3 
import pandas as pd 
import matplotlib.pyplot as plt
import os
from nltk.corpus import stopwords
from wordcloud import WordCloud
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
                 
def load_data(database_path):
    """
    Connects to a SQLite database and automatically loads the table
    that contains reviews based on its name.

    Parameters:
    - databsae_path (str): Path to the SQLite .db or .sqlite file.

    Returns:
    - pd.DataFrame: DataFrame containing all rows from the reviews table.
    """
    # connect to the SQLite database 
    conn = sqlite3.connect(database_path)

    # fetch all tables from the database 
    tables = pd.read_sql_query("SELECT name  \
                            FROM sqlite_master \
                            WHERE type = 'table';" , conn)
    
    table_names = tables['name'].tolist()
    
    # search for the table that contains reviews 
    review_table = next((name for name in table_names if 'review' in name.lower()), None)

    # if there is no review table, throw an error
    if review_table is None:
        conn.close()
        raise ValueError("No Table with 'review' in the name was found.")

    # load all rows from the reviews table is found
    df = pd.read_sql_query(f"SELECT * \
                        FROM {review_table};", conn)

    # close the connection 
    conn.close()
    return df 

# function to plot a bar graph with annotations 
def plot_bar_with_annotations(data, title, xlabel, ylabel, rotation = 0, save_path = None, figsize = (10, 8)):
    '''
    plots a bar chart with annotations of the value 

    Parameters: 
     data (series): index is the x-axis like score, and values like count are y-axis heights
     title (str): title of the plot 
     xlabel (str): label for x axis 
     ylabel (str): label for y axis 
     rotation (int): rotation angle for x-axis labels: default reotation of 0
     save_path (str): path to save plot to 
     figsize (tuple): size of the figure: is a default of (10, 8)
    '''
    plt.figure(figsize = figsize)
    ax = data.plot(kind = 'bar')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation = rotation, ha = 'right' if rotation else 'center')

    # label the bars with their values 
    for bar in ax.patches:
        ax.annotate(f'{int(bar.get_height())}',
                    (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1),
                    ha='center', va='bottom', fontsize=10)
    plt.tight_layout()

    # save the plot if a save path was specified 
    if save_path:
        plt.savefig(save_path)
        print(f'Plot saved to {save_path}')
    plt.close()

# function to clean text for a specific score of reviews to get wordclouds of high and low scored reviews to find their top words 
def get_clean_text(df, score):
    '''
    Filters reviews by score and returns cleaned, lowercase review text

    Parameters:
     df (DataFrame): input DataFrame containing a 'Score' and 'Text' column
     score (int): review score to filter on (e.g., 1 or 5)

    Returns:
     str: cleaned string of all words in the selected reviews, with stopwords and non-alphabetic words removed
    '''
    text = " ".join(df[df['Score'] == score]['Text'].astype(str).tolist()).lower()
    tokens = [word for word in text.split() if word.isalpha() and word not in stop_words]
    return tokens

def generate_wordcloud_from_tokens(tokens, title, wordcount, save_path = None, colormap = 'viridis'):
    '''
    Generates and saves a word cloud from a list of tokens

    Parameters:
     tokens (list): list of cleaned, lowercase words to include in the word cloud
     title (str): title displayed above the word cloud plot
     wordcount (int): maxinum number of words to display
     save_path (str): path to save the generated word cloud image (optional)
     colormap (str): color scheme used in the word cloud (default is 'viridis')

    Returns:
     None 
    '''

    text = ' '.join(tokens)
    wordcloud = WordCloud(width = 800, height = 400, background_color = 'white', colormap = colormap, max_words = wordcount, scale = 3, random_state = 42).generate(text)

    plt.figure(figsize = (10, 5))
    plt.imshow(wordcloud, interpolation = 'bilinear')  
    plt.axis('off')
    plt.title(title, fontsize = 16)
    plt.tight_layout()

    # try to save the wordcloud 
    if save_path:
        plt.savefig(save_path)
        print(f'Word Cloud saved to {save_path}')
    plt.close()

def ensure_nltk_resource(resource_name, download_name=None):
    '''
    Ensures that the specified NLTK resource is available. If not found, it downloads the resource.

    Parameters:
     resource_name (str): Path to the NLTK resource (e.g., 'tokenizers/punkt' or 'corpora/stopwords')
     download_name (str): Optional specific name to download using nltk.download(). 
                          If not provided, the function uses the last part of resource_name.

    Returns:
     None
     '''

    try:
        nltk.data.find(resource_name)
    except LookupError:
        print(f"{resource_name} not found. Downloading...")
        nltk.download(download_name or resource_name.split('/')[-1])