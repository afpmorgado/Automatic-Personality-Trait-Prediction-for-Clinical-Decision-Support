"""
Psycholinguistic Functions and Data adapted from the source code of the following paper:
Bottom-Up and Top-Down: Predicting Personality with Psycholinguistic and Language Model Features, Yash Mehta et al., 2020
DOI: 10.1109/ICDM50108.2020.00146

https://github.com/yashsmehta/personality-prediction
"""

"""
Assembles the psycholinguistic features for the users of both datasets
Master's Thesis: "Automatic Personality Trait Prediction for Clinical Decision Support"
Author: André Morgado
ID: ist199888
Instituto Superior Técnico, Universidade de Lisboa
24/05/2026

Version 1.0 - Lacks commenting (To implement).
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import re
from scipy.io import arff

paraphrasing = False

def read_and_process(path):
 
    arff = open(path, "r")
    attributes = []
    values = []
    is_attr = True

    arff.readline()
    arff.readline()

    while is_attr:
        line = arff.readline()
        if len(line.split()) == 0:
            is_attr = False
            continue
        type = line.split()[0]
        attr = " ".join(line.split()[1:])
        if type == "@attribute":
            attributes.append(attr)
        else:
            is_attr = False

    for line in arff.readlines():
        if len(line.split(",")) < 10:
            continue
        else:
            components = line.split(",")
            values.append(components)

            name = components[0].replace("'", "").split("\\\\")[-1]
            values[-1][0] = name

    df = pd.DataFrame(columns=attributes, data=values)
    df["idx"] = [int(re.sub("id_", "", i)) for i in df[df.columns[0]]]
    df = df.drop(df.columns[0], axis=1)
    df = df.set_index(["idx"])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_index()

    return df


def load_features(dir, dataset):
    idx = "id"

    if dataset == "Kaggle":
        drop_cols = [
            "BROWN-FREQ numeric",
            "K-F-FREQ numeric",
            "K-F-NCATS numeric",
            "K-F-NSAMP numeric",
            "T-L-FREQ numeric",
            "Extraversion numeric",
            "'Emotional stability' numeric",
            "Agreeableness numeric",
            "Conscientiousness numeric",
            "'Openness to experience' numeric",
        ]

        mairesse = read_and_process(dir + dataset + "_mairesse_labeled.arff")
        mairesse = mairesse.drop(drop_cols, axis=1)

    elif dataset == "Essays":
        idx = "#AUTHID"
        mairesse = pd.read_csv(dir + dataset + "_mairesse_labeled.csv")
        mairesse = mairesse.drop(columns=["#AUTHID", "filename string"], errors="ignore")

    if dataset == "Kaggle":
        if paraphrasing:
            nrc = pd.read_csv(dir + dataset + "_nrc_paraphrased.csv").set_index([idx])
            nrc_vad = pd.read_csv(dir + dataset + "_nrc_vad_paraphrased.csv").set_index([idx])
            readability = pd.read_csv(dir + dataset + "_readability_paraphrased.csv").set_index([idx])
        else:
            nrc = pd.read_csv(dir + dataset + "_nrc.csv").set_index([idx])
            nrc_vad = pd.read_csv(dir + dataset + "_nrc_vad.csv").set_index([idx])
            readability = pd.read_csv(dir + dataset + "_readability.csv").set_index([idx])
    elif dataset == "Essays":
        if paraphrasing:
            nrc = pd.read_csv(dir + dataset + "_nrc_paraphrased.csv")
            nrc_vad = pd.read_csv(dir + dataset + "_nrc_vad_paraphrased.csv")
            readability = pd.read_csv(dir + dataset + "_readability_paraphrased.csv")
        else:
            nrc = pd.read_csv(dir + dataset + "_nrc.csv")
            nrc_vad = pd.read_csv(dir + dataset + "_nrc_vad.csv")
            readability = pd.read_csv(dir + dataset + "_readability.csv")
        for df in [nrc, nrc_vad, readability]:
            df.drop(columns=["#AUTHID", "filename string"], errors="ignore", inplace=True)

    return [nrc, nrc_vad, readability, mairesse]



def get_inputs(dataset):

    print("Getting psycholinguistic features...")
    features_list = load_features("./Psycholinguistic Utils/Data/" + dataset + "/", dataset)

    features_df = pd.concat(features_list, axis=1)
    features_array = features_df.values

    return features_array, features_df


if __name__ == "__main__":
    MODEL_INPUT = "psycholinguist_features"
    dataset = "Kaggle"

    features, features_df = get_inputs(dataset)
    print("Number of samples:", features.shape[0])
    print("Total number of features:", features.shape[1])

    if paraphrasing:
        save_path = f"Features/{dataset}/{MODEL_INPUT}_paraphrased.npy"
    else:
        save_path = f"Features/{dataset}/{MODEL_INPUT}.npy"
    np.save(save_path, features)
    print(f"Features saved to {save_path}")

    if paraphrasing:
        csv_path = f"Features/{dataset}/{MODEL_INPUT}_paraphrased.csv"
    else:
        csv_path = f"Features/{dataset}/{MODEL_INPUT}.csv"
    features_df.to_csv(csv_path)
    print(f"Feature CSV saved to {csv_path}")
