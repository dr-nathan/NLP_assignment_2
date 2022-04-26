# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 19:00:32 2022

@author: nvs690
"""

import numpy as np
import pandas as pd
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

olid_train = pd.read_csv("data/olid-train.csv")

pos = np.sum(olid_train["labels"] == 1)
neg = np.sum(olid_train["labels"] == 0)


# NV: input as series
def baseline_scores(
    testinput: pd.core.series.Series,
    testlabels: pd.core.series.Series,
    baseline_type: str,
):

    if baseline_type == "majority":
        # NV: count 1's and 0's in testlabels
        counts = testlabels.value_counts()
        # NV: take index of max
        majority_class = counts.idxmax()
        # NV: make series of predictions with majority class
        predictions = pd.Series([majority_class for t in testinput])

    elif baseline_type == "random":
        # NV: randomly give each sentence a 1 or 0
        predictions = pd.Series([random.choice([1, 0]) for t in testinput])

    goldlabels = testlabels

    # NV: use sklearn metrics
    accuracy = accuracy_score(goldlabels, predictions)

    (
        (precision_pos, precision_neg),
        (recall_pos, recall_neg),
        (F1_pos, F1_neg),
        (_, _)
    ) = precision_recall_fscore_support(
        goldlabels, predictions, average=None, labels=[1, 0]
    )

    weighted_F1 = f1_score(goldlabels, predictions, average="weighted")
    macro_F1 = f1_score(goldlabels, predictions, average="macro")

    return (
        predictions,
        accuracy,
        precision_pos,
        precision_neg,
        recall_pos,
        recall_neg,
        F1_pos,
        F1_neg,
        weighted_F1,
        macro_F1
    )

if __name__ == '__main__':
    (
        predictions_majority,
        accuracy_majority,
        precision_pos_majority,
        precision_neg_majority,
        recall_pos_majority,
        recall_neg_majority,
        F1_pos_majority,
        F1_neg_majority,
        weighted_F1_majority,
        macro_F1_majority
    ) = baseline_scores(olid_train["text"], olid_train["labels"], "majority")

    (
        predictions_random,
        accuracy_random,
        precision_pos_random,
        precision_neg_random,
        recall_pos_random,
        recall_neg_random,
        F1_pos_random,
        F1_neg_random,
        weighted_F1_random,
        macro_F1_random
    ) = baseline_scores(olid_train["text"], olid_train["labels"], "random")
