import pandas as pd 
import numpy as np 
import os 
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import nltk 
import swifter
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm # progress bar trakcer for when we do some loops for benchmarking
from concurrent.futures import ThreadPoolExecutor
from pandarallel import pandarallel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax # to smooth out the scores and make sure they are between 0 and 1

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

    # reset the index of the dataframe amd merge the vaders df with the reivews df on Id so we get each review and its sentiment on the same row 
    vaders = vaders.reset_index().rename(columns = {'index': 'Id'})
    reviews_with_sentiment_vader = vaders.merge(df, on = 'Id', how = 'left')

    print(reviews_with_sentiment_vader.head())

    # check the reviews with sentiment df for any null summaries and remove those 
    print(f'Sum of reviews with an empty Summary: {reviews_with_sentiment_vader['Summary'].isna().sum()}')
    print(f'Missing Summaries: \n\n {reviews_with_sentiment_vader[reviews_with_sentiment_vader['Summary'].isna()]}')

    reviews_with_sentiment_vader = reviews_with_sentiment_vader[~reviews_with_sentiment_vader['Summary'].isna()]

    # export the df that contains the VADER sentiment scores for each review to a csv 
    reviews_with_sentiment_vader.to_csv('reviews_with_VADER_sentiment.csv', index = False)
    print("DistilBERT sentiment scoring complete and saved!")

    # DistilBERT Pretrained Model
    # DistilBERT is a distilled (smaller, faster) version of BERT, a transformer model that captures the context of words based on their surrounding words.
    # This means it can understand sentiment more accurately than simple models like VADER, because it considers word relationships, sarcasm, and nuance.
    # For example, in the sentence "I just love waiting in line for hours", DistilBERT can recognize the sarcasm and classify it as negative, whereas a bag-of-words model may incorrectly classify it as positive due to the word "love".
    # The 'distilbert-base-uncased-finetuned-sst-2-english' model is fine-tuned specifically for binary sentiment classification: positive or negative.
    
    # since I do not have access to a GPU and running 550k+ reviews would take a lot of time on my local machine, use stratified sampling to sample 5,000 reviews (1k from each review star)
    stratified_df = df.groupby('Score', group_keys = False).apply(lambda x: x.sample(n = 1000, random_state = 42)).reset_index(drop = True)

    print(f'Sampled Stratified Subset: {stratified_df['Score'].value_counts().sort_index()}')

    # use the maximum amount of threads available
    torch.set_num_threads(os.cpu_count())
    print(f"Using {os.cpu_count()} CPU threads for PyTorch.")

    # load DistilBERT model
    MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval() # set model to evaluation mode

    # set PyTorch to run on the CPU, if there is no GPU
    device = torch.device('cpu')
    model.to(device)

    # define a batch function to score sentiment
    def get_distilbert_sentiment_batch(text_list):
        try:
            encodings = tokenizer(
                text_list,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            with torch.no_grad():
                outputs = model(**encodings.to(device))
            probs = softmax(outputs.logits.detach().cpu().numpy(), axis=1)
            return probs
        except Exception as e:
            print(f"Batch failed: {e}")
            return [ [np.nan, np.nan] ] * len(text_list)

    # process in batches (e.g., 64 reviews at a time)
    batch_size = 64
    results = []

    print("Running DistilBERT sentiment scoring in batches...")

    for i in tqdm(range(0, len(stratified_df), batch_size)):
        batch_texts = stratified_df['Text'][i:i+batch_size].tolist()
        batch_scores = get_distilbert_sentiment_batch(batch_texts)
        results.extend(batch_scores)
    
    # convert output to a DataFrame
    distilbert_sentiment_df = pd.DataFrame(results, columns=['distil_neg', 'distil_pos'])
    distilbert_sentiment_df['Id'] = stratified_df['Id'].values  # Keep original review IDs

    # merge with original data
    reviews_with_distilbert = pd.merge(distilbert_sentiment_df, df, on='Id')
    reviews_with_distilbert.to_csv('reviews_with_DistilBERT_sentiment.csv', index=False)
    print("DistilBERT sentiment scoring complete and aved!")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
