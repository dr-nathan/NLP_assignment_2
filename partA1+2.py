# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 19:00:32 2022

@author: nvs690
"""

import numpy as np
import pandas as pd
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,\
    f1_score, precision_score, recall_score

# NV: input as series, type of baseline is string


def baseline_scores(
    traininput: pd.core.series.Series,
    trainlabels: pd.core.series.Series,
    testinput: pd.core.series.Series,
    testlabels: pd.core.series.Series,
    baseline_type: str
):

    if baseline_type == "majority":
        # NV: count 1's and 0's on train data
        counts = trainlabels.value_counts()
        # NV: take index of max
        majority_class = counts.idxmax()
        # NV: make series of predictions with majority class
        predictions = pd.Series([majority_class for t in testinput])

    elif baseline_type == "random":
        # NV: randomly give each sentence a 1 or 0, convert to Series
        predictions = pd.Series([random.choice([1, 0]) for t in testinput])

    else:
        raise NotImplementedError("this baseline is not implemented!")

    # NV: just a rename for clarity
    goldlabels = testlabels

    # NV: use sklearn metrics
    # accuracy
    accuracy = accuracy_score(goldlabels, predictions)

    #precision, recall, F1
    (
        (precision_pos, precision_neg),
        (recall_pos, recall_neg),
        (F1_pos, F1_neg),
        (_, _)  # ignore support
    ) = precision_recall_fscore_support(
        goldlabels, predictions, average=None, labels=[1, 0]
    )

    # averages for precision
    weighted_precision = precision_score(goldlabels, predictions, average="weighted")
    macro_precision = precision_score(goldlabels, predictions, average="macro")

    # averages for recall
    weighted_recall = recall_score(goldlabels, predictions, average="weighted")
    macro_recall = recall_score(goldlabels, predictions, average="macro")

    # averages for F1
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
        macro_F1,
        weighted_precision,
        macro_precision,
        weighted_recall,
        macro_recall
    )


if __name__ == '__main__':

    # to the reader: the results as stored as variables. Access value by printing them.

    # NV: load train and test (train only for majority class determination)
    olid_train = pd.read_csv("data/olid-train.csv")
    olid_test = pd.read_csv("data/olid-test.csv")

    # NV: total amount
    pos = np.sum(olid_train["labels"] == 1)
    neg = np.sum(olid_train["labels"] == 0)

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
        macro_F1_majority,
        weighted_precision_majority,
        macro_precision_majority,
        weighted_recall_majority,
        macro_recall_majority
    ) = baseline_scores(olid_train["text"], olid_train["labels"], olid_test["text"], olid_test["labels"], "majority")

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
        macro_F1_random,
        weighted_precision_random,
        macro_precision_random,
        weighted_recall_random,
        macro_recall_random
    ) = baseline_scores(olid_train["text"], olid_train["labels"], olid_test["text"], olid_test["labels"], "random")
