import csv
import sys
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


def mergedict(m, n):
    m.update(n)
    return m


if __name__ == "__main__":
    start_time = time.perf_counter()

    #region No reviews
    print("\n")
    print("-------------------------------No reviews-------------------------------------")
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/Amazon_Digital_Music/bert_emb_pca13_senti.csv",
                       sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

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
                    lr=1e-5, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32,16", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["precision", "recall", "map", "ndcg"])

    result_without_reviews = evaluate_k(model=deepfm, data=eval_data,
                                        metrics=["precision", "recall", "map", "ndcg"], ks=[5, 10, 15, 20, 25])
    print("evaluate_result: ", result_without_reviews)

    fields = ["k", "precision", "recall", "map", "ndcg"]
    with open("result_without_reviews.csv", "w", newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for key, val in sorted(result_without_reviews.items()):
            w.writerow(mergedict({'k': key}, val))
    #endregion

    # region Only reviews' embeddings
    print("\n---------------------Only reviews' embeddings-----------------------------------")
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/Amazon_Digital_Music/bert_emb_pca13_senti.csv",
                       sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

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
                    lr=1e-5, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32,16", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["precision", "recall", "map", "ndcg"])

    result_reviews = evaluate_k(model=deepfm, data=eval_data,
                        metrics=["precision", "recall", "map", "ndcg"], ks=[5, 10, 15, 20, 25])
    print("evaluate_result: ", result_reviews)

    fields = ["k", "precision", "recall", "map", "ndcg"]
    with open("result_reviews.csv", "w", newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for key, val in sorted(result_reviews.items()):
            w.writerow(mergedict({'k': key}, val))
    # endregion

    # region Only Sentiment score from Vader
    print("\n----------------------Only Sentiment score from Vader-------------------------")
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/Amazon_Digital_Music/bert_emb_pca13_senti.csv",
                       sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

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
                    lr=1e-5, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32,16", tf_sess_config=None)

    # "loss", "roc_auc", "pr_auc" these metrics are based on y_true & y_prob (1 / (1 + np.exp(-preds)))
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["precision", "recall", "map", "ndcg"])

    # this is based on recommendation list y_true_list compared with y_pred_list
    result_with_only_senti = evaluate_k(model=deepfm, data=eval_data,
                                metrics=["precision", "recall", "map", "ndcg"], ks=[5, 10, 15, 20, 25])
    print("evaluate_result: ", result_with_only_senti)

    fields = ["k", "precision", "recall", "map", "ndcg"]
    with open("result_with_only_senti.csv", "w", newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for key, val in sorted(result_with_only_senti.items()):
            w.writerow(mergedict({'k': key}, val))
    # endregion

    # region reviews' embeddings + Sentiment score from Vader
    print("\n---------------------Reviews' embeddings + Sentiment score from Vader--------------------")
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/Amazon_Digital_Music/bert_emb_pca13_senti.csv",
                       sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

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
                    lr=1e-5, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32,16", tf_sess_config=None)

    # "loss", "roc_auc", "pr_auc" these metrics are based on y_true & y_prob (1 / (1 + np.exp(-preds)))
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["precision", "recall", "map", "ndcg"])

    # this is based on recommendation list y_true_list compared with y_pred_list
    result_with_reviews_senti = evaluate_k(model=deepfm, data=eval_data,
                                        metrics=["precision", "recall", "map", "ndcg"], ks=[5, 10, 15, 20, 25])
    print("evaluate_result: ", result_with_reviews_senti)

    fields = ["k", "precision", "recall", "map", "ndcg"]
    with open("result_with_reviews_senti.csv", "w", newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for key, val in sorted(result_with_reviews_senti.items()):
            w.writerow(mergedict({'k': key}, val))
    # endregion

    # region Reviews  + Sentiment score from Vader + Negation occurrence
    print("\n----------------------reviews' embeddings + Negation occurrence----------------------------")
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/Amazon_Digital_Music/bert_emb_pca13_senti.csv",
                       sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

    # specify complete columns information
    sparse_col = ["negation_occurrence"]
    dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                 "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                 "review_emb11", "review_emb12", "review_emb13", "vader_sentiment_score"]
    user_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                "review_emb11", "review_emb12", "review_emb13", "vader_sentiment_score",
                "negation_occurrence"]
    item_col = []

    train_data, data_info = DatasetFeat.build_trainset(
        train_data, user_col, item_col, sparse_col, dense_col
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    print(data_info)

    reset_state("DeepFM")
    deepfm = DeepFM("ranking", data_info, embed_size=16, n_epochs=5,
                    lr=1e-5, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32,16", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["precision", "recall", "map", "ndcg"])

    result_with_reviews_senti_negs = evaluate_k(model=deepfm, data=eval_data,
                                                metrics=["precision", "recall", "map", "ndcg"],
                                                ks=[5, 10, 15, 20, 25])
    print("evaluate_result: ", result_with_reviews_senti_negs)

    fields = ["k", "precision", "recall", "map", "ndcg"]
    with open("result_with_reviews_senti_negs.csv", "w", newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for key, val in sorted(result_with_reviews_senti_negs.items()):
            w.writerow(mergedict({'k': key}, val))
    # endregion

    # region reviews' embeddings + Negation occurrence
    print("\n---------------------Reviews' embeddings + Negation occurrence------------------------")
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/Amazon_Digital_Music/bert_emb_pca13_senti.csv",
                       sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

    sparse_col = ["negation_occurrence"]
    dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                 "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                 "review_emb11", "review_emb12", "review_emb13"]
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
                    lr=1e-5, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32,16", tf_sess_config=None)

    # "loss", "roc_auc", "pr_auc" these metrics are based on y_true & y_prob (1 / (1 + np.exp(-preds)))
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["precision", "recall", "map", "ndcg"])

    # this is based on recommendation list y_true_list compared with y_pred_list
    result_with_reviews_negs = evaluate_k(model=deepfm, data=eval_data,
                                          metrics=["precision", "recall", "map", "ndcg"], ks=[5, 10, 15, 20, 25])
    print("evaluate_result: ", result_with_reviews_negs)

    fields = ["k", "precision", "recall", "map", "ndcg"]
    with open("result_with_reviews_negs.csv", "w", newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for key, val in sorted(result_with_reviews_negs.items()):
            w.writerow(mergedict({'k': key}, val))
    # endregion

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # region Plot precision metrics only
    # region common variable declare
    x = ["P1", "P2", "P3", "P4", "P5", "P6"]
    y_label = [5, 10, 15, 20, 25]
    x_p1 = []
    x_p2 = []
    x_p3 = []
    x_p4 = []
    x_p5 = []
    x_p6 = []

    y_p1 = []
    y_p2 = []
    y_p3 = []
    y_p4 = []
    y_p5 = []
    y_p6 = []
    # endregion
    for f, v in result_without_reviews.items():
        x_p1.append(f)
        y_p1.append(v["precision"])
    for f, v in result_reviews.items():
        x_p2.append(f)
        y_p2.append(v["precision"])
    for f, v in result_with_only_senti.items():
        x_p3.append(f)
        y_p3.append(v["precision"])
    for f, v in result_with_reviews_senti.items():
        x_p4.append(f)
        y_p4.append(v["precision"])
    for f, v in result_with_reviews_senti_negs.items():
        x_p5.append(f)
        y_p5.append(v["precision"])
    for f, v in result_with_reviews_negs.items():
        x_p6.append(f)
        y_p6.append(v["precision"])

    axes[0, 0].plot(x_p1, y_p1, 'g', label='P1', marker='8')
    axes[0, 0].plot(x_p2, y_p2, 'r', label='P2', marker='8')
    axes[0, 0].plot(x_p3, y_p3, '--b', label='P3', marker='8')
    axes[0, 0].plot(x_p4, y_p4, '--y', label='P4', marker='8')
    axes[0, 0].plot(x_p5, y_p5, '--m', label='P5', marker='8')
    axes[0, 0].plot(x_p6, y_p6, '--c', label='P6', marker='8')
    axes[0, 0].set_xlabel('Recommendation list (N)', fontsize=14)
    axes[0, 0].set_ylabel('Precision@N', fontsize=14)
    axes[0, 0].grid()
    axes[0, 0].set_xticks(x_p1)
    #plt.yticks(rotation=60)
    #axes[0, 0].legend(loc=0)
    # endregion

    # region Plot map metrics only
    # region common variable declare
    x = ["P1", "P2", "P3", "P4", "P5", "P6"]
    y_label = [5, 10, 15, 20, 25]
    x_p1 = []
    x_p2 = []
    x_p3 = []
    x_p4 = []
    x_p5 = []
    x_p6 = []

    y_p1 = []
    y_p2 = []
    y_p3 = []
    y_p4 = []
    y_p5 = []
    y_p6 = []
    # endregion

    for f, v in result_without_reviews.items():
        x_p1.append(f)
        y_p1.append(v["map"])
    for f, v in result_reviews.items():
        x_p2.append(f)
        y_p2.append(v["map"])
    for f, v in result_with_only_senti.items():
        x_p3.append(f)
        y_p3.append(v["map"])
    for f, v in result_with_reviews_senti.items():
        x_p4.append(f)
        y_p4.append(v["map"])
    for f, v in result_with_reviews_senti_negs.items():
        x_p5.append(f)
        y_p5.append(v["map"])
    for f, v in result_with_reviews_negs.items():
        x_p6.append(f)
        y_p6.append(v["map"])

    axes[0, 1].plot(x_p1, y_p1, 'g', label='P1', marker='8')
    axes[0, 1].plot(x_p2, y_p2, 'r', label='P2', marker='8')
    axes[0, 1].plot(x_p3, y_p3, '--b', label='P3', marker='8')
    axes[0, 1].plot(x_p4, y_p4, '--y', label='P4', marker='8')
    axes[0, 1].plot(x_p5, y_p5, '--m', label='P5', marker='8')
    axes[0, 1].plot(x_p6, y_p6, '--c', label='P6', marker='8')
    axes[0, 1].set_xlabel('Recommendation list (N)', fontsize=14)
    axes[0, 1].set_ylabel('Map@N', fontsize=14)
    axes[0, 1].grid()
    axes[0, 1].set_xticks(x_p1)
    #plt.yticks(rotation=60)
    #plt.legend(loc=0)
    # endregion

    # region Plot recall metrics only
    # region common variable declare
    x = ["P1", "P2", "P3", "P4", "P5", "P6"]
    y_label = [5, 10, 15, 20, 25]
    x_p1 = []
    x_p2 = []
    x_p3 = []
    x_p4 = []
    x_p5 = []
    x_p6 = []

    y_p1 = []
    y_p2 = []
    y_p3 = []
    y_p4 = []
    y_p5 = []
    y_p6 = []
    # endregion
    for f, v in result_without_reviews.items():
        x_p1.append(f)
        y_p1.append(v["recall"])
    for f, v in result_reviews.items():
        x_p2.append(f)
        y_p2.append(v["recall"])
    for f, v in result_with_only_senti.items():
        x_p3.append(f)
        y_p3.append(v["recall"])
    for f, v in result_with_reviews_senti.items():
        x_p4.append(f)
        y_p4.append(v["recall"])
    for f, v in result_with_reviews_senti_negs.items():
        x_p5.append(f)
        y_p5.append(v["recall"])
    for f, v in result_with_reviews_negs.items():
        x_p6.append(f)
        y_p6.append(v["recall"])

    axes[1, 0].plot(x_p1, y_p1, 'g', label='P1', marker='8')
    axes[1, 0].plot(x_p2, y_p2, 'r', label='P2', marker='8')
    axes[1, 0].plot(x_p3, y_p3, '--b', label='P3', marker='8')
    axes[1, 0].plot(x_p4, y_p4, '--y', label='P4', marker='8')
    axes[1, 0].plot(x_p5, y_p5, '--m', label='P5', marker='8')
    axes[1, 0].plot(x_p6, y_p6, '--c', label='P6', marker='8')
    axes[1, 0].set_xlabel('Recommendation list (N)', fontsize=14)
    axes[1, 0].set_ylabel('Recall@N', fontsize=14)
    axes[1, 0].grid()
    axes[1, 0].set_xticks(x_p1)
    #plt.yticks(rotation=60)
    axes[1, 0].legend(loc=0)
    # endregion

    # region Plot ndcg metrics only
    # region common variable declare
    x = ["P1", "P2", "P3", "P4", "P5", "P6"]
    y_label = [5, 10, 15, 20, 25]
    x_p1 = []
    x_p2 = []
    x_p3 = []
    x_p4 = []
    x_p5 = []
    x_p6 = []

    y_p1 = []
    y_p2 = []
    y_p3 = []
    y_p4 = []
    y_p5 = []
    y_p6 = []
    # endregion
    for f, v in result_without_reviews.items():
        x_p1.append(f)
        y_p1.append(v["ndcg"])
    for f, v in result_reviews.items():
        x_p2.append(f)
        y_p2.append(v["ndcg"])
    for f, v in result_with_only_senti.items():
        x_p3.append(f)
        y_p3.append(v["ndcg"])
    for f, v in result_with_reviews_senti.items():
        x_p4.append(f)
        y_p4.append(v["ndcg"])
    for f, v in result_with_reviews_senti_negs.items():
        x_p5.append(f)
        y_p5.append(v["ndcg"])
    for f, v in result_with_reviews_negs.items():
        x_p6.append(f)
        y_p6.append(v["ndcg"])

    axes[1, 1].plot(x_p1, y_p1, 'g', label='P1', marker='8')
    axes[1, 1].plot(x_p2, y_p2, 'r', label='P2', marker='8')
    axes[1, 1].plot(x_p3, y_p3, '--b', label='P3', marker='8')
    axes[1, 1].plot(x_p4, y_p4, '--y', label='P4', marker='8')
    axes[1, 1].plot(x_p5, y_p5, '--m', label='P5', marker='8')
    axes[1, 1].plot(x_p6, y_p6, '--c', label='P6', marker='8')
    axes[1, 1].set_xlabel('Recommendation list (N)', fontsize=14)
    axes[1, 1].set_ylabel('NDCG@N', fontsize=14)
    axes[1, 1].grid()
    axes[1, 1].set_xticks(x_p1)
    # plt.yticks(rotation=60)
    # plt.legend(loc=0)
    # endregion

    plt.savefig('Whole_Music_metrics@N.png', dpi=600)

    # region predictors
    # P1: No reviews
    # P2: only with reviews
    # P3: Only sentiment
    # P4: reviews+sentiment
    # P5: reviews+sentiment+negation
    # P6: reviews+negation
    # endregion


