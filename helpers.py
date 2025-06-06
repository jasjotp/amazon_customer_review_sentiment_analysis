import sqlite3 
import pandas as pd 
import matplotlib.pyplot as plt
import os

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

