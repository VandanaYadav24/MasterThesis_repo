import time
import pandas as pd
import matplotlib.pyplot as plt
from libreco.data import split_by_ratio, DatasetFeat
from libreco.algorithms import (
    FM, WideDeep, DeepFM, AutoInt, DIN, YouTuBeRetrieval, YouTubeRanking
)
from libreco.evaluation import evaluate, evaluate_k

# remove unnecessary tensorflow logging
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)


if __name__ == "__main__":
    start_time = time.perf_counter()

    # region No reviews
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/Amazon_Music/"
                       "Amazon_Music_without_negs.csv", sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

    # specify complete columns information
    sparse_col = []
    dense_col = []
    user_col = []
    item_col = []

    train_data, data_info = DatasetFeat.build_trainset(
        train_data, user_col, item_col, sparse_col, dense_col
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    print(data_info)

    reset_state("DeepFM")
    deepfm = DeepFM("ranking", data_info, embed_size=16, n_epochs=5,
                    lr=1e-5, lr_decay=False, reg=None, batch_size=256,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32,16", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["precision", "recall", "map", "ndcg"])

    result_without_reviews = evaluate_k(model=deepfm, data=eval_data,
                                        metrics=["precision", "recall", "map", "ndcg"], ks=[5, 10, 15, 20, 25])
    print("evaluate_result: ", result_without_reviews)
    # endregion

    # region Only reviews' embeddings
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/Amazon_Music/"
                       "Amazon_Music_without_negs.csv", sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

    # specify complete columns information
    sparse_col = []
    dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                 "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                 "review_emb11", "review_emb12", "review_emb13"]
    user_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                "review_emb11", "review_emb12", "review_emb13"]
    item_col = []

    train_data, data_info = DatasetFeat.build_trainset(
        train_data, user_col, item_col, sparse_col, dense_col
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    print(data_info)

    reset_state("DeepFM")
    deepfm = DeepFM("ranking", data_info, embed_size=16, n_epochs=5,
                    lr=1e-5, lr_decay=False, reg=None, batch_size=256,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32,16", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["precision", "recall", "map", "ndcg"])

    result_reviews = evaluate_k(model=deepfm, data=eval_data,
                                        metrics=["precision", "recall", "map", "ndcg"], ks=[5, 10, 15, 20, 25])
    print("evaluate_result: ", result_reviews)
    # endregion

    # region Only Sentiment score from Vader
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/Amazon_Music/"
                       "Amazon_Music_without_negs.csv", sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

    # specify complete columns information
    sparse_col = []
    dense_col = ["vader_sentiment_score"]
    user_col = ["vader_sentiment_score"]
    item_col = []

    train_data, data_info = DatasetFeat.build_trainset(
        train_data, user_col, item_col, sparse_col, dense_col
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    print(data_info)

    reset_state("DeepFM")
    deepfm = DeepFM("ranking", data_info, embed_size=16, n_epochs=5,
                    lr=1e-5, lr_decay=False, reg=None, batch_size=256,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32,16", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["precision", "recall", "map", "ndcg"])

    result_with_only_senti = evaluate_k(model=deepfm, data=eval_data,
                                        metrics=["precision", "recall", "map", "ndcg"], ks=[5, 10, 15, 20, 25])
    print("evaluate_result: ", result_with_only_senti)
    # endregion

    # region reviews' embeddings + Sentiment score from Vader
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/Amazon_Music/"
                       "Amazon_Music_without_negs.csv", sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

    # specify complete columns information
    sparse_col = []
    dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                 "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                 "review_emb11", "review_emb12", "review_emb13", "vader_sentiment_score"]
    user_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                "review_emb11", "review_emb12", "review_emb13", "vader_sentiment_score"]
    item_col = []

    train_data, data_info = DatasetFeat.build_trainset(
        train_data, user_col, item_col, sparse_col, dense_col
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    print(data_info)

    reset_state("DeepFM")
    deepfm = DeepFM("ranking", data_info, embed_size=16, n_epochs=5,
                    lr=1e-5, lr_decay=False, reg=None, batch_size=256,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32,16", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["precision", "recall", "map", "ndcg"])

    result_with_reviews_senti = evaluate_k(model=deepfm, data=eval_data,
                                        metrics=["precision", "recall", "map", "ndcg"], ks=[5, 10, 15, 20, 25])
    print("evaluate_result: ", result_with_reviews_senti)
    # endregion

    # region reviews' embeddings + Negation occurrence
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/Amazon_Music/"
                       "Amazon_Music_without_negs.csv", sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

    # specify complete columns information
    sparse_col = []
    dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                 "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                 "review_emb11", "review_emb12", "review_emb13", "negation_occurrence"]
    user_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                "review_emb11", "review_emb12", "review_emb13", "negation_occurrence"]
    item_col = []

    train_data, data_info = DatasetFeat.build_trainset(
        train_data, user_col, item_col, sparse_col, dense_col
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    print(data_info)

    reset_state("DeepFM")
    deepfm = DeepFM("ranking", data_info, embed_size=16, n_epochs=5,
                    lr=1e-5, lr_decay=False, reg=None, batch_size=256,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32,16", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["precision", "recall", "map", "ndcg"])

    result_with_reviews_negs = evaluate_k(model=deepfm, data=eval_data,
                                        metrics=["precision", "recall", "map", "ndcg"], ks=[5, 10, 15, 20, 25])
    print("evaluate_result: ", result_with_reviews_negs)
    # endregion

    metrics = ["precision", "recall", "map", "ndcg"]
    for i in range(len(metrics)):
        x = []
        y = []
        x_reviews = []
        y_reviews = []
        x_only_senti = []
        y_only_senti = []
        x_reviews_senti_score = []
        y_reviews_senti_score = []
        x_reviews_negs = []
        y_reviews_negs = []
        for f, v in result_without_reviews.items():
            x.append(f)
            y.append(v[metrics[i]])
        for f, v in result_reviews.items():
            x_reviews.append(f)
            y_reviews.append(v[metrics[i]])
        for f, v in result_with_only_senti.items():
            x_only_senti.append(f)
            y_only_senti.append(v[metrics[i]])
        for f, v in result_with_reviews_senti.items():
            x_reviews_senti_score.append(f)
            y_reviews_senti_score.append(v[metrics[i]])
        for f, v in result_with_reviews_negs.items():
            x_reviews_negs.append(f)
            y_reviews_negs.append(v[metrics[i]])
        plt.figure(figsize=(5, 16))
        plt.subplot(4, 1, i + 1)
        plt.plot(x, y, 'g', label='no reviews', marker='s')
        plt.plot(x_reviews, y_reviews, 'r', label='with reviews', marker='s')
        plt.plot(x_only_senti, y_only_senti, '--b', label='vader sentiment score', marker='s')
        plt.plot(x_reviews_senti_score, y_reviews_senti_score, '--y', label='reviews+vader sentiment', marker='s')
        plt.plot(x_reviews_negs, y_reviews_negs, '--c', label='reviews+negation occurrence', marker='s')
        plt.xlabel('recommendation list (N)', fontsize=6)
        plt.ylabel('{}@N'.format(metrics[i]), fontsize=6)
        plt.grid()
        plt.xticks(x)
        plt.yticks(rotation=60)
        # plt.legend(loc=0)
        plt.savefig('{}@N.png'.format(metrics[i]), dpi=800)