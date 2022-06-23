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


def reset_state(name):
    tf.compat.v1.reset_default_graph()
    print("\n", "=" * 30, name, "=" * 30)


if __name__ == "__main__":
    start_time = time.perf_counter()

    # data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/"
    #                    "datasets/preparation_of_datasets/Hotel/new_vader/Hotel_Reviews_with_negs.csv",
    #                    sep=",", header=0)
    data = pd.read_csv("G:/My Drive/Univ Of Oulu/masters_thesis/preprocess_dataset/datasets/"
                       "preparation_of_datasets/hotel/new_vader/Hotel_Reviews_with_negs.csv", sep=",", header=0)

    train_data, eval_data = split_by_ratio(data, test_size=0.2)

    # specify complete columns information
    sparse_col = []
    # dense_col = ["vader_sentiment_score"]
    # user_col = ["vader_sentiment_score"]
    dense_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                 "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                 "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
                 "review_emb16", "vader_negations"]
    user_col = ["review_emb1", "review_emb2", "review_emb3", "review_emb4", "review_emb5",
                "review_emb6", "review_emb7", "review_emb8", "review_emb9", "review_emb10",
                "review_emb11", "review_emb12", "review_emb13", "review_emb14", "review_emb15",
                "review_emb16", "vader_negations"]
    item_col = []

    train_data, data_info = DatasetFeat.build_trainset(
        train_data, user_col, item_col, sparse_col, dense_col
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    print(data_info)


    reset_state("DeepFM")
    deepfm = DeepFM("rating", data_info, embed_size=16, n_epochs=20,
                    lr=1e-3, lr_decay=False, reg=None, batch_size=64,
                    num_neg=1, use_bn=False, dropout_rate=None,
                    hidden_units="128,64,32", tf_sess_config=None)
    deepfm.fit(train_data, verbose=2, shuffle=True, eval_data=eval_data,
               metrics=["rmse", "mae", "r2"])

    print("evaluate_result: ", evaluate(model=deepfm, data=eval_data,
                                        metrics=["rmse", "mae", "r2"]))

