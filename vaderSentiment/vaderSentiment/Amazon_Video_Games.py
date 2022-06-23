import pandas as pd
import numpy as np
import vaderSentiment
from nltk import tokenize
import time
import swifter

if __name__ == '__main__':

    df = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                     "datasets/preparation_of_datasets/Amazon_Video_Games/new_vader/Preprocessed_Amazon_Video_Games.csv")
    print(df.head(4))

    def get_negations(review):
        analyzer = vaderSentiment.SentimentIntensityAnalyzer()
        sentence_list = tokenize.sent_tokenize(review)
        review_negations = 0
        for sentence in sentence_list:
            vs = analyzer.polarity_scores(sentence)
            review_negations += vs["negations_captured"]
        return review_negations

    def get_sentiment(review):
        analyzer = vaderSentiment.SentimentIntensityAnalyzer()
        sentence_list = tokenize.sent_tokenize(review)
        review_sentiment = 0.0
        for sentence in sentence_list:
            vs = analyzer.polarity_scores(sentence)
            review_sentiment += round(vs["compound"] / len(sentence_list), 4)
        return review_sentiment

    start_time = time.perf_counter()
    print(start_time)

    df['vader_negations'] = list(map(get_negations, df['reviews']))
    print("---------------------negations--done--------------------------")
    df['vader_sentiment_score'] = list(map(get_sentiment, df['reviews']))

    df.to_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
              "datasets/preparation_of_datasets/Amazon_Video_Games/new_vader/Preprocessed_Amazon_Video_Games.csv",
              index=False)

    stop_time = time.perf_counter()
    print("Elapsed time: ", stop_time-start_time)
