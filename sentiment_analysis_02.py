import pandas as pd 
import numpy as np 
import os 
import sqlite3 

from helpers import load_data

# define your directory where the data and database is before reading in the full dataabse path with the load_data helper funcction 
database_path = r'data\database.sqlite'
df = load_data(database_path)

# initial analysis: check a preview of the data, columns, and information
print(df.head())
print(df.columns)
print(df.info())
print(df.describe(include = 'all'))
print(df['Score'].describe(include = 'all')) # min score of 1 and max score of 5 


