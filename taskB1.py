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
from simpletransformers.classification import ClassificationModel

olid_subset = pd.read_csv("data/olid-subset-diagnostic-tests.csv")

model = ClassificationModel(
    "bert", "outputs/checkpoint-8000", use_cuda=False, from_flax=True
)
