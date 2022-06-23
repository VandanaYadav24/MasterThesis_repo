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
import csv
import numpy as np
import seaborn as sns

def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)

def mergedict(m, n):
    m.update(n)
    return m


if __name__ == "__main__":
    start_time = time.perf_counter()

    #region No reviews
    print("\n-------------------------------No reviews----------------------------------")
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/preparation_of_datasets/hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
                       sep=",", header=0)
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
                    lr=1e-5, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32,16", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["precision", "recall", "map", "ndcg"])

    result_without_reviews = evaluate_k(model=deepfm, data=eval_data,
                                        metrics=["precision", "recall", "map", "ndcg"], ks=[5, 10, 15, 20, 25])
    print("evaluate_result: ", result_without_reviews)
    #endregion

    # region Only reviews' embeddings
    print("\n-------------------------------Only reviews' embeddings----------------------------------")
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/preparation_of_datasets/hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
                       sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

    # specify complete columns information
    sparse_col = []
    dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                 "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                 "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
                 "review_emb16"]
    user_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
                "review_emb16"]
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
    print("\n-------------------------------Only Sentiment score from Vader----------------------------------")
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/preparation_of_datasets/hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
                       sep=",", header=0)
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
                    lr=1e-5, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32,16", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["precision", "recall", "map", "ndcg"])

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
    print("\n-------------------------reviews' embeddings + Sentiment score from Vader----------------------------")
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/preparation_of_datasets/hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
                       sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

    # specify complete columns information
    sparse_col = []
    dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                 "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                 "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
                 "review_emb16", "vader_sentiment_score"]
    user_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
                "review_emb16", "vader_sentiment_score"]
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

    # region reviews' embeddings + Sentiment score from vader + Negation occurrence
    print("\n----------------------reviews' embeddings + Negation occurrence----------------------------")
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/preparation_of_datasets/hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
                       sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

    # specify complete columns information
    sparse_col = ["negation_occurrence"]
    dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                 "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                 "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
                 "review_emb16", "vader_sentiment_score"]
    user_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
                "review_emb16", "vader_sentiment_score", "negation_occurrence"]
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
                                          metrics=["precision", "recall", "map", "ndcg"], ks=[5, 10, 15, 20, 25])
    print("evaluate_result: ", result_with_reviews_senti_negs)
    fields = ["k", "precision", "recall", "map", "ndcg"]
    with open("result_with_reviews_negs.csv", "w", newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for key, val in sorted(result_with_reviews_senti_negs.items()):
            w.writerow(mergedict({'k': key}, val))
    # endregion

    # region reviews' embeddings + Negation occurrence
    print("\n----------------------reviews' embeddings + Negation occurrence----------------------------")
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/preparation_of_datasets/hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
                       sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

    # specify complete columns information
    sparse_col = []
    dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                 "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                 "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
                 "review_emb16", "negation_occurrence"]
    user_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
                "review_emb16", "negation_occurrence"]
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

    # # region Plot all metrics in subplots
    # metrics = ["precision", "recall", "map", "ndcg"]
    # x = []
    # y = []
    # x_reviews = []
    # y_reviews = []
    # x_only_senti = []
    # y_only_senti = []
    # x_reviews_senti = []
    # y_reviews_senti = []
    # x_reviews_senti_negs = []
    # y_reviews_senti_negs = []
    # x_reviews_negs = []
    # y_reviews_negs = []
    # for i in range(len(metrics)):
    #     for f, v in result_without_reviews.items():
    #         x.append(f)
    #         y.append(v[metrics[i]])
    #     for f, v in result_reviews.items():
    #         x_reviews.append(f)
    #         y_reviews.append(v[metrics[i]])
    #     for f, v in result_with_only_senti.items():
    #         x_only_senti.append(f)
    #         y_only_senti.append(v[metrics[i]])
    #     for f, v in result_with_reviews_senti.items():
    #         x_reviews_senti.append(f)
    #         y_reviews_senti.append(v[metrics[i]])
    #     for f, v in result_with_reviews_senti_negs.items():
    #         x_reviews_senti_negs.append(f)
    #         y_reviews_senti_negs.append(v[metrics[i]])
    #     for f, v in result_with_reviews_negs.items():
    #         x_reviews_negs.append(f)
    #         y_reviews_negs.append(v[metrics[i]])
    #     plt.figure(figsize=(12, 9))
    #     if i>1:
    #         plt.subplot(2, 2, i-1)
    #     else:
    #         plt.subplot(2, 1, i + 1)
    #     plt.plot(x, y, 'g', label='P1', marker='s')
    #     plt.plot(x_reviews, y_reviews, 'r', label='P2', marker='s')
    #     plt.plot(x_only_senti, y_only_senti, '--b', label='P3', marker='s')
    #     plt.plot(x_reviews_senti, y_reviews_senti, '--y', label='P4', marker='s')
    #     plt.plot(x_reviews_senti_negs, y_reviews_senti_negs, '--m', label='P5', marker='s')
    #     plt.plot(x_reviews_negs, y_reviews_negs, '--c', label='P6', marker='s')
    #     plt.xlabel('recommendation list (N)', fontsize=6)
    #     plt.ylabel('{}@N'.format(metrics[i]), fontsize=6)
    #     plt.grid()
    #     plt.xticks(x)
    #     plt.yticks(rotation=60)
    #     plt.legend(loc=0)
    #     #plt.savefig('{}@N.png'.format(metrics[i]), dpi=800)
    # plt.savefig('metrics@K.png', dpi=800)
    # # endregion

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
    axes[0, 0].legend(loc=0)
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

    # # region Plot
    # # https://matplotlib.org/3.3.2/gallery/lines_bars_and_markers/barchart.html
    # labels = ["P1", "P2", "P3", "P4", "P5", "P6"]
    #
    # map_10 = [result_without_reviews[10]["map"], result_reviews[10]["map"],
    #           result_with_only_senti[10]["map"], result_with_reviews_senti[10]["map"],
    #           result_with_reviews_senti_negs[10]["map"], result_with_reviews_negs[10]["map"]]
    #
    # xl = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars
    #
    # fig, ax = plt.subplots()
    # rects1 = ax.bar(xl, map_10, width, label='map@10', color="#24d19d")
    #
    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Metrics', fontsize=7)
    # ax.set_xlabel('Predictors used in DeepFM model', fontsize=7)
    # ax.set_xticks(xl)
    # ax.set_xticklabels(labels)
    # ax.legend()
    #
    #
    # def autolabel(rects):
    #     """Attach a text label above each bar in *rects*, displaying its height."""
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(np.round(height, 3)),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 2),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom',
    #                     fontsize=6)
    #
    # autolabel(rects1)
    # fig.tight_layout()
    # plt.savefig('metrics_hotel_ranking_task.png', dpi=800)
    # # endregion

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
    # plt.legend(loc=0)
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
    #plt.yticks(rotation=60)
    # plt.legend(loc=0)
    # endregion

    plt.savefig('Whole_hotel_metrics@N.png', dpi=600)

