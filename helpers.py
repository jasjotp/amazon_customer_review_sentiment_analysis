import sqlite3 
import pandas as pd 

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