import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk 
import swifter
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm # progress bar trakcer for when we do some loops for benchmarking
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pandarallel import pandarallel
from helpers import load_data, ensure_nltk_resource

pandarallel.initialize()
sia = SentimentIntensityAnalyzer()

# try to already find the punkt and stopwords packages to avoid repeated downloads
ensure_nltk_resource('tokenizers/punkt_tab', 'punkt_tab')
ensure_nltk_resource('corpora/stopwords', 'stopwords')
ensure_nltk_resource('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
ensure_nltk_resource('chunkers/maxent_ne_chunker_tab', 'maxent_ne_chunker_tab')
ensure_nltk_resource('corpora/words', 'words')
ensure_nltk_resource('corpora/words', 'words')
ensure_nltk_resource('sentiment/vader_lexicon', 'vader_lexicon')

database_path = r'data\database.sqlite'

def main():
        
    df = load_data(database_path)

    # print out sample review
    example = df['Text'][50]
    print(example)

    # tokenize the example so that the computer can interpret each word 
    tokens = word_tokenize(example)

    # find the part of speech tag of each token: List of Descriptions: https://medium.com/@faisal-fida/the-complete-list-of-pos-tags-in-nltk-with-examples-eb0485f04321 ex) Oatmeal is a NN: singular noun 
    tagged = pos_tag(tokens)
    print(tagged)

    # take the tags and put them into entities using nltk.chunk 
    entities = nltk.chunk.ne_chunk(tagged)
    print(entities.pprint())

    # try VADER sentiment scoring (Valence Aware Dictionary and SEntiment Reasoner) - Bag of words spparoch 
    # bag of words approach means that stop words are removed and takes all the words in our statement and has a dictionary that it scores each word on -as either negative, neutral or positive 
    # then it combines each words' sentiment to add up how positive, negative or neutral the statement is, based on all those words 
    # ***VADER DOES NOT ACCOUNT FOR RELATIONSHIPS BETWEEN WORDS, WHICH IS VERY IMPORTANT WHEN FINDING SENTIMENT OF A STATEMENT 
    # use the sentiment insensity analyzer object to see what the sentiment is for some block of text 
    print(sia.polarity_scores('I am so happy!')) # for this positive example, negative: 0, neu: 0.32, pos: 0.68, compound: 0.65 so this statement is mostly postiive
    print(sia.polarity_scores('This is the worst thing ever!')) # for this negative example, negative: 0.57, neu: 0.53, pos: 0, compound: -0.65 so this statement is mostly negative

    # print out sentiment score for the example above for the oatmeal review 
    print(f'Sentiment Score for Example: {sia.polarity_scores(example)}') # for this example about oatmeal, negative: 0.22, neu: 0.78, pos: 0, compound: -0.54 so this statement is mostly negative

    # check the polarity score on the entire dataset and convert into a table format using pd.series
    vaders = df['Text'].parallel_apply(sia.polarity_scores).apply(pd.Series)

    print(vaders.head(10))

    # reset the index of the dataframe amd mrege the vaders df with the reivews df on Id so we get each review and its sentiment on the same row 
    vaders = vaders.reset_index().rename(columns = {'index': 'Id'})
    reviews_with_sentiment = vaders.merge(df, on = 'Id', how = 'left')

    print(reviews_with_sentiment.head())

    # check the reviews with sentiment df for any null summaries and remove those 
    print(f'Sum of reviews with an empty Summary: {reviews_with_sentiment['Summary'].isna().sum()}')
    print(f'Missing Summaries: \n\n {reviews_with_sentiment[reviews_with_sentiment['Summary'].isna()]}')

    reviews_with_sentiment = reviews_with_sentiment[~reviews_with_sentiment['Summary'].isna()]

    # export the df to a csv 
    reviews_with_sentiment.to_csv('reviews_with_sentiment.csv', index = False)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
