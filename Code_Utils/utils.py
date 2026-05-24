
"""Utility functions for the experiments in the thesis.
Master's Thesis: "Automatic Personality Trait Prediction for Clinical Decision Support"
Author: André Morgado
ID: ist199888
Instituto Superior Técnico, Universidade de Lisboa
24/05/2026

Version 1.0 - Lacks commenting (To implement).
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
import os
import torch
from transformers import set_seed

def set_model_seed(seed_value=29):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


def yn_to_binary(val):
    return 1.0 if val.lower() == 'y' else 0.0


def calculate_metrics(preds, labels, threshold=0.5):
    preds = preds > threshold
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return acc, f1


def split_into_chunks(text):
    return text.split("|||")