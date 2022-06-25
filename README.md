# Hybrid Recommendation system using product reviews

## Overview
Proposed a hybrid recommendation algorithm  called DeepFM based on two types of tasks of recommendation system (1. predicting desired ratings, and 2. providing recommendations based on rank scores) using information extracted from textual product reviews such as sentence embedding vectors, sentiment scores and presence of negations in the reviews.

#### Datasets used: 

Datafiniti Hotel Reviews: <br />
Description: Hotel Reviews dataset from Datafinti's Business dataset. <br />
URL: https://data.world/datafiniti/hotel-reviews

Amazon Digital Music and Video Games: <br />
Description: From Amazon product reviews dataset. <br />
URL: http://jmcauley.ucsd.edu/data/amazon/index_2014.html/


|**user id** | **item id**|**label**|**reviews**|
|:----------: |:----------:|:-------:|:-------------------:|:-------:|:------: |:-------:|
|80|14|2|The worst experience in my life please never go close to tis hotel ughhhhh nasty rooms smells bad|
|31|9|5|It was a great stay |
|65|38|5|Good: Impeccable service. Beautiful hotel. Don't hesitate to stay here!|
|29|617|5|Part of hotel was under renovation, it was also noisy. Not a good bargain.|






#### Methods/models used: <br />
BERT: To extract contextual sentence embedding of the product reviews. <br />
VADER: To extract sentiment scores and negations of the product reviews. <br />
DeepFM: Used as a recommendation algorithm. <br />





## References:
LibRecommender is an easy-to-use recommender system focused on end-to-end recommendation is utilised for DeepFM model. <br />
GitHub URL: https://github.com/massquantity/LibRecommender/tree/master/examples

Valence Aware Dictionary and sEntiment Reasoner (VADER):
https://github.com/cjhutto/vaderSentiment
