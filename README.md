# Hybrid Recommendation system using product reviews

## Overview
Proposed a hybrid recommendation algorithm  called DeepFM based on two types of tasks of recommendation system (1. predicting desired ratings, and 2. providing recommendations based on rank scores) using information extracted from textual product reviews such as sentence embedding vectors, sentiment scores and presence of negations in the reviews.

#### Datasets used: 

##### Datafiniti Hotel Reviews: <br />
Description: Hotel Reviews dataset from Datafinti's Business dataset. <br />
URL: https://data.world/datafiniti/hotel-reviews <br/>

Sample data: 
|**user id** | **item id**|**label**|**reviews**|
|:------------: |:------------:|:-------:|:-------------------:|
|80|14|2|The worst experience in my life please never go close to tis hotel ughhhhh nasty rooms smells bad|
|31|9|5|It was a great stay |
|65|38|5|Good: Impeccable service. Beautiful hotel. Don't hesitate to stay here!|
|29|617|2|Part of hotel was under renovation, it was also noisy. Not a good bargain.|

##### Amazon Digital Music and Video Games: <br />
Description: From Amazon product reviews dataset. <br />
URL: http://jmcauley.ucsd.edu/data/amazon/index_2014.html/

Sample data of Digital Music: 
|**user id** | **item id**|**label**|**reviews**|
|:------------: |:------------:|:-------:|:----------------------:|
|47|3|5|I loved the Cars, soooo oringinal.the more time passes, the better they are.And this is the set of songs that launched the legendary band.|
|590|284|1|Yet another classic that only became popular because their audience was stoned. Blaring, muddy, unmusical crud. |
|1451|338|1|I cant find one good song on the whole thing, just ignore this record because it's wack.get me against the world though.|
|1397|167|2|This album has plenty of great songs.  It's one of those albums that is great when you first hear it, but after a while, it gets pretty old.|

Sample data of Video Games: 
|**user id** | **item id**|**label**|**reviews**|
|:------------: |:------------:|:-------:|:-------------------:|
|90|8|4|This allows me to preserve whatever precious space I have left on my console for other stuff. I may purchase another one in the near future.|
|15|1|1|Crashed in Vista.  Codemasters told me they don't support it in Windows .  Couldn't get it to work even after looking on the Internet. |
|40|3|5|My oldest grandson loves this game. I can remember playing it when I was young and it was good back then but even better now.|
|72|7|3|Great protection and grip for a while, but like a latex glove, it stretches out after a little time.  For the price, a great deal.|





#### Methods/models used: <br />
BERT: To extract contextual sentence embedding of the product reviews. <br />
VADER: To extract sentiment scores and negations of the product reviews. <br />
DeepFM: Used as a recommendation algorithm. <br />





## References:
LibRecommender is an easy-to-use recommender system focused on end-to-end recommendation is utilised for DeepFM model. <br />
GitHub URL: https://github.com/massquantity/LibRecommender/tree/master/examples

Valence Aware Dictionary and sEntiment Reasoner (VADER):
https://github.com/cjhutto/vaderSentiment
