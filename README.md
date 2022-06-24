# Hybrid Recommendation system using product reviews

## Overview
Proposed a hybrid recommendation algorithm  called DeepFM based on two types of tasks of recommendation system (rating and ranking prediction tasks) using information extracted from textual product reviews such as sentence embedding vectors, sentiment scores and presence of negations in the reviews.

Datasets used:

DataFiniti Hotel Reviews: 
Description: Hotel Reviews dataset from Datafinti's Business dataset.
URL: https://data.world/datafiniti/hotel-reviews

Amazon Digital Music and Video Games: 
URL: http://jmcauley.ucsd.edu/data/amazon/index_2014.html/


Methods/models used:
BERT: To extract contextual sentence embedding.
VADER: To extract sentiment scores and negations.
DeepFM: Used as a recommendation algorithm.

References:
LibRecommender is an easy-to-use recommender system focused on end-to-end recommendation is utilised for DeepFM model.
GitHub URL: https://github.com/massquantity/LibRecommender/tree/master/examples

Valence Aware Dictionary and sEntiment Reasoner (VADER):
https://github.com/cjhutto/vaderSentiment
