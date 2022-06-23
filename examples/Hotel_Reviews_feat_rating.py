import time
import pandas as pd
import matplotlib.pyplot as plt
from libreco.data import split_by_ratio, DatasetFeat, DataInfo
from libreco.algorithms import FM, WideDeep, DeepFM, AutoInt, DIN
from libreco.evaluation import evaluate

# remove unnecessary tensorflow logging
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_WARNINGS"] = "FALSE"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)


if __name__ == "__main__":
    start_time = time.perf_counter()

    # region No reviews
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/preparation_of_datasets/Hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
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
    deepfm = DeepFM("rating", data_info, embed_size=16, n_epochs=15,
                    lr=1e-3, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["rmse", "mae", "r2"])
    result_without_reviews = evaluate(model=deepfm, data=eval_data,
                                        metrics=["rmse", "mae", "r2"])
    print("evaluate_result: ", result_without_reviews)
    # endregion

    # region only with reviews
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/preparation_of_datasets/Hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
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
    deepfm = DeepFM("rating", data_info, embed_size=16, n_epochs=15,
                    lr=1e-3, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["rmse", "mae", "r2"])
    result_reviews = evaluate(model=deepfm, data=eval_data,
                                      metrics=["rmse", "mae", "r2"])
    print("evaluate_result: ", result_reviews)
    # endregion

    # region Sentiment score
    # region Only Sentiment score from Vader
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/preparation_of_datasets/Hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
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
    deepfm = DeepFM("rating", data_info, embed_size=16, n_epochs=15,
                    lr=1e-3, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["rmse", "mae", "r2"])
    result_with_only_senti = evaluate(model=deepfm, data=eval_data,
                              metrics=["rmse", "mae", "r2"])
    print("evaluate_result: ", result_with_only_senti)
    # endregion

    # region reviews' embeddings + Sentiment score from Vader
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/preparation_of_datasets/Hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
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
    deepfm = DeepFM("rating", data_info, embed_size=16, n_epochs=15,
                    lr=1e-3, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["rmse", "mae", "r2"])
    result_with_reviews_senti = evaluate(model=deepfm, data=eval_data,
                              metrics=["rmse", "mae", "r2"])
    print("evaluate_result: ", result_with_reviews_senti)
    # endregion

    # region reviews' embeddings + Sentiment score from Vader + Negation occurrence
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/preparation_of_datasets/Hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
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
    deepfm = DeepFM("rating", data_info, embed_size=16, n_epochs=15,
                    lr=1e-3, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["rmse", "mae", "r2"])
    result_with_reviews_senti_negs = evaluate(model=deepfm, data=eval_data,
                              metrics=["rmse", "mae", "r2"])
    print("evaluate_result: ", result_with_reviews_senti_negs)
    # endregion

    # region reviews' embeddings + Negation occurrence
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
                       "datasets/preparation_of_datasets/Hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
                       sep=",", header=0)
    train_data, eval_data = split_by_ratio(data, test_size=0.2)

    # specify complete columns information
    sparse_col = ["negation_occurrence"]
    dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                 "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                 "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
                 "review_emb16"]
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
    deepfm = DeepFM("rating", data_info, embed_size=16, n_epochs=15,
                    lr=1e-3, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["rmse", "mae", "r2"])
    result_with_reviews_negs = evaluate(model=deepfm, data=eval_data,
                                              metrics=["rmse", "mae", "r2"])
    print("evaluate_result: ", result_with_reviews_negs)
    # endregion
    # endregion

    # # region sentiment
    # # region Only Sentiment from Vader
    # data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
    #                    "datasets/preparation_of_datasets/Hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
    #                    sep=",", header=0)
    # train_data, eval_data = split_by_ratio(data, test_size=0.2)
    #
    # # specify complete columns information
    # sparse_col = ["sentiment"]
    # dense_col = []
    # user_col = ["sentiment"]
    # item_col = []
    #
    # train_data, data_info = DatasetFeat.build_trainset(
    #     train_data, user_col, item_col, sparse_col, dense_col
    # )
    # eval_data = DatasetFeat.build_testset(eval_data)
    # print(data_info)
    #
    # reset_state("DeepFM")
    # deepfm = DeepFM("rating", data_info, embed_size=16, n_epochs=15,
    #                 lr=1e-3, lr_decay=False, reg=None, batch_size=64,
    #                 num_neg=1, use_bn=False, dropout_rate=None,
    #                 hidden_units="128,64,32", tf_sess_config=None)
    # deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
    #            metrics=["rmse", "mae", "r2"])
    # result_with_only_senti = evaluate(model=deepfm, data=eval_data,
    #                           metrics=["rmse", "mae", "r2"])
    # print("evaluate_result: ", result_with_only_senti)
    # # endregion
    #
    # # region reviews' embeddings + Sentiment score from Vader
    # data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
    #                    "datasets/preparation_of_datasets/Hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
    #                    sep=",", header=0)
    # train_data, eval_data = split_by_ratio(data, test_size=0.2)
    #
    # # specify complete columns information
    # sparse_col = ["sentiment"]
    # dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
    #              "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
    #              "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
    #              "review_emb16"]
    # user_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
    #             "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
    #             "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
    #             "review_emb16", "sentiment"]
    # item_col = []
    #
    # train_data, data_info = DatasetFeat.build_trainset(
    #     train_data, user_col, item_col, sparse_col, dense_col
    # )
    # eval_data = DatasetFeat.build_testset(eval_data)
    # print(data_info)
    #
    # reset_state("DeepFM")
    # deepfm = DeepFM("rating", data_info, embed_size=16, n_epochs=15,
    #                 lr=1e-3, lr_decay=False, reg=None, batch_size=64,
    #                 num_neg=1, use_bn=False, dropout_rate=None,
    #                 hidden_units="128,64,32", tf_sess_config=None)
    # deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
    #            metrics=["rmse", "mae", "r2"])
    # result_with_reviews_senti = evaluate(model=deepfm, data=eval_data,
    #                           metrics=["rmse", "mae", "r2"])
    # print("evaluate_result: ", result_with_reviews_senti)
    # # endregion
    #
    # # region reviews' embeddings + Sentiment score from Vader + Negation occurrence
    # data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
    #                    "datasets/preparation_of_datasets/Hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
    #                    sep=",", header=0)
    # train_data, eval_data = split_by_ratio(data, test_size=0.2)
    #
    # # specify complete columns information
    # sparse_col = ["negation_occurrence", "sentiment"]
    # dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
    #              "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
    #              "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
    #              "review_emb16"]
    # user_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
    #             "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
    #             "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
    #             "review_emb16", "sentiment", "negation_occurrence"]
    # item_col = []
    #
    # train_data, data_info = DatasetFeat.build_trainset(
    #     train_data, user_col, item_col, sparse_col, dense_col
    # )
    # eval_data = DatasetFeat.build_testset(eval_data)
    # print(data_info)
    #
    # reset_state("DeepFM")
    # deepfm = DeepFM("rating", data_info, embed_size=16, n_epochs=15,
    #                 lr=1e-3, lr_decay=False, reg=None, batch_size=64,
    #                 num_neg=1, use_bn=False, dropout_rate=None,
    #                 hidden_units="128,64,32", tf_sess_config=None)
    # deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
    #            metrics=["rmse", "mae", "r2"])
    # result_with_reviews_senti_negs = evaluate(model=deepfm, data=eval_data,
    #                           metrics=["rmse", "mae", "r2"])
    # print("evaluate_result: ", result_with_reviews_senti_negs)
    # # endregion
    #
    # # region reviews' embeddings + Negation occurrence
    # data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
    #                    "datasets/preparation_of_datasets/Hotel/new_vader/hotel_bert_emb_pca16_senti.csv",
    #                    sep=",", header=0)
    # train_data, eval_data = split_by_ratio(data, test_size=0.2)
    #
    # # specify complete columns information
    # sparse_col = ["negation_occurrence"]
    # dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
    #              "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
    #              "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
    #              "review_emb16"]
    # user_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
    #             "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
    #             "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
    #             "review_emb16", "negation_occurrence"]
    # item_col = []
    #
    # train_data, data_info = DatasetFeat.build_trainset(
    #     train_data, user_col, item_col, sparse_col, dense_col
    # )
    # eval_data = DatasetFeat.build_testset(eval_data)
    # print(data_info)
    #
    # reset_state("DeepFM")
    # deepfm = DeepFM("rating", data_info, embed_size=16, n_epochs=15,
    #                 lr=1e-3, lr_decay=False, reg=None, batch_size=64,
    #                 num_neg=1, use_bn=False, dropout_rate=None,
    #                 hidden_units="128,64,32", tf_sess_config=None)
    # deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
    #            metrics=["rmse", "mae", "r2"])
    # result_with_reviews_negs = evaluate(model=deepfm, data=eval_data,
    #                                           metrics=["rmse", "mae", "r2"])
    # print("evaluate_result: ", result_with_reviews_negs)
    # # endregion
    # # endregion

    # region Plot
    # https://matplotlib.org/3.3.2/gallery/lines_bars_and_markers/barchart.html
    labels = ["P1", "P2", "P3", "P4", "P5", "P6"]
    rmse = [result_without_reviews["rmse"], result_reviews["rmse"],
            result_with_only_senti["rmse"], result_with_reviews_senti["rmse"],
            result_with_reviews_senti_negs["rmse"], result_with_reviews_negs["rmse"]]

    mae = [result_without_reviews["mae"], result_reviews["mae"],
           result_with_only_senti["mae"], result_with_reviews_senti["mae"],
           result_with_reviews_senti_negs["mae"], result_with_reviews_negs["mae"]]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, rmse, width, label='rmse', color="#ffadad")
    rects2 = ax.bar(x + width / 2, mae, width, label='mae', color="#ffd6a5")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Metrics', fontsize=7)
    ax.set_xlabel('Predictors used in DeepFM model', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(np.round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=7)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.savefig('rmse_mae_hotel_rating_task.png', dpi=800)
    # endregion
