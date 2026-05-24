"""
Psycholinguistic Functions and Data adapted from the source code of the following paper:
Bottom-Up and Top-Down: Predicting Personality with Psycholinguistic and Language Model Features, Yash Mehta et al., 2020
DOI: 10.1109/ICDM50108.2020.00146

https://github.com/yashsmehta/personality-prediction
"""

"""Extracts psycholinguistic features for the users of both datasets
Master's Thesis: "Automatic Personality Trait Prediction for Clinical Decision Support"
Author: André Morgado
ID: ist199888
Instituto Superior Técnico, Universidade de Lisboa
24/05/2026

Version 1.0 - Lacks commenting (To implement).
"""

import pandas as pd
from collections import Counter
import readability #Used version 0.3.1, version 0.3.2 yields different results
import re
import numpy as np
from scipy import stats
dataset_type = "Kaggle"
paraphrasing = False
split = True

if split and paraphrasing:
    print("Only one of the modes can be active: Paraphrasing, Split")
    

if dataset_type == "Kaggle":
    op_dir = "../Data/Kaggle/"
    if paraphrasing:
        datafile = "../../Data/Kaggle/kaggle_paraphrased.csv"
    elif split:
        datafile = "../../Data/Kaggle/kaggle_split.csv"
    else:
        datafile = "../../Data/Kaggle/kaggle_none.csv"
elif dataset_type == "Essays":
    op_dir = "../Data/Essays/"
    if paraphrasing:
        datafile = "../../Data/Essays/essays_paraphrased.csv"
    elif split:
        datafile = "../../Data/Essays/essays_split.csv"
    else:
        datafile = "../../Data/Essays/essays_none.csv"


def normalization(df):
    for idx, col in enumerate(df.columns):
        df[col] = np.nan_to_num(stats.zscore(df[col]))
    return df

def extract_readability_features(text):
    text = re.sub(r"\.", ".\n", text)
    text = re.sub(r"\?", "?\n", text)
    text = re.sub(r"!", "!\n", text)
    features = dict(readability.getmeasures(text, lang="en"))
    result = {}
    for d in features:
        result.update(features[d])
    del result["paragraphs"]
    result = pd.Series(result)
    return result

def extract_NRC_VAD_features(x, vad_df):
    tokens = x.split()
    tokens = Counter(tokens)
    df = pd.DataFrame.from_dict(tokens, orient="index", columns=["count"])
    merged_df = pd.merge(df, vad_df, left_index=True, right_index=True)
    if merged_df.empty:
        return pd.Series(0.0, index=vad_df.columns)
    for col in merged_df.columns[1:]:
        merged_df[col] *= merged_df["count"]

    result = merged_df.sum()
    result /= result["count"]
    result = result.iloc[1:]
    return result



def extract_NRC_features(x, nrc_df):
    tokens = x.split()
    tokens = Counter(tokens)
    df = pd.DataFrame.from_dict(tokens, orient="index", columns=["count"])
    merged_df = pd.merge(df, nrc_df, left_index=True, right_index=True)
    if merged_df.empty:
        return pd.Series(0.0, index=nrc_df.columns)
    for col in merged_df.columns[1:]:
        merged_df[col] *= merged_df["count"]
    result = merged_df.sum()
    result /= result["count"]
    result = result.iloc[1:]
    return result


if __name__ == "__main__":
    count_df = pd.read_csv(datafile)

    NRC_path = "Lexicons/NRC-Emotion-Lexicon.xlsx"
    NRC_VAD_path = "Lexicons/NRC-VAD-Lexicon.txt"
    NRC_df = pd.read_excel(NRC_path, index_col=0)
    NRC_VAD_df = pd.read_csv(NRC_VAD_path, index_col=["Word"], sep="\t")

    NRC_df.index = NRC_df.index.str.lower()

    if dataset_type == "Kaggle":
        tmp = count_df["posts"].apply(lambda x: extract_NRC_features(x, NRC_df))
        tmp_VAD = count_df["posts"].apply(lambda x: extract_NRC_VAD_features(x, NRC_VAD_df))
        tmp_readability = count_df["posts"].apply(lambda x: extract_readability_features(x))
    elif dataset_type == "Essays":
        tmp = count_df["text"].apply(lambda x: extract_NRC_features(x, NRC_df))
        tmp_VAD = count_df["text"].apply(lambda x: extract_NRC_VAD_features(x, NRC_VAD_df))
        tmp_readability = count_df["text"].apply(lambda x: extract_readability_features(x))
    tmp_readability = normalization(tmp_readability)
    if dataset_type == "Kaggle":
        ids = pd.Series(range(len(count_df)), name="id")
        result = pd.concat([ids, tmp], axis=1)
        result_VAD = pd.concat([ids, tmp_VAD], axis=1)
        result_readability = pd.concat([ids, tmp_readability], axis=1)
        result = result[
            ["id",
            "Positive", "Negative",
            "Anger", "Anticipation", "Disgust", "Fear",
            "Joy", "Sadness", "Surprise", "Trust"]
        ]
        result_VAD = result_VAD[
            ["id",
            "Valence", "Arousal", "Dominance"]
        ]

        result_readability = result_readability[["id","Kincaid","ARI","Coleman-Liau","FleschReadingEase",
                                                "GunningFogIndex","LIX","SMOGIndex","RIX","DaleChallIndex",
                                                "characters_per_word","syll_per_word","words_per_sentence",
                                                "sentences_per_paragraph","type_token_ratio","characters","syllables",
                                                "words","wordtypes","sentences","long_words","complex_words","complex_words_dc",
                                                "tobeverb","auxverb","conjunction","pronoun","preposition","nominalization",
                                                "interrogative","article","subordination"]]
    elif dataset_type == "Essays":
        result = pd.concat([count_df["#AUTHID"], tmp], axis=1)
        result_VAD = pd.concat([count_df["#AUTHID"], tmp_VAD], axis=1)
        result_readability = pd.concat([count_df["#AUTHID"], tmp_readability], axis=1)
        result = result[
            ["#AUTHID",
            "Positive", "Negative",
            "Anger", "Anticipation", "Disgust", "Fear",
            "Joy", "Sadness", "Surprise", "Trust"]
        ]
        result_VAD = result_VAD[
            ["#AUTHID",
            "Valence", "Arousal", "Dominance"]
        ]

        result_readability = result_readability[["#AUTHID","Kincaid","ARI","Coleman-Liau","FleschReadingEase",
                                                "GunningFogIndex","LIX","SMOGIndex","RIX","DaleChallIndex",
                                                "characters_per_word","syll_per_word","words_per_sentence",
                                                "sentences_per_paragraph","type_token_ratio","characters","syllables",
                                                "words","wordtypes","sentences","long_words","complex_words","complex_words_dc",
                                                "tobeverb","auxverb","conjunction","pronoun","preposition","nominalization",
                                                "interrogative","article","subordination"]]

    if paraphrasing:
        print("Saved NRC features to", op_dir + dataset_type + "_nrc_paraphrased.csv")
        output_file = op_dir + dataset_type + "_nrc_paraphrased.csv"
        print("Saved NRC VAD features to", op_dir + dataset_type + "_nrc_vad_paraphrased.csv")
        output_file_VAD = op_dir + dataset_type + "_nrc_vad_paraphrased.csv"
        print("Saved Readability features to", op_dir + dataset_type + "_readability_paraphrased.csv")
        output_file_readability = op_dir + dataset_type + "_readability_paraphrased.csv"
    elif split:
        print("Saved NRC features to", op_dir + dataset_type + "_nrc_split.csv")
        output_file = op_dir + dataset_type + "_nrc_split.csv"
        print("Saved NRC VAD features to", op_dir + dataset_type + "_nrc_vad_split.csv")
        output_file_VAD = op_dir + dataset_type + "_nrc_vad_split.csv"
        print("Saved Readability features to", op_dir + dataset_type + "_readability_split.csv")
        output_file_readability = op_dir + dataset_type + "_readability_split.csv"
    else:
        print("Saved NRC features to", op_dir + dataset_type + "_nrc.csv")
        output_file = op_dir + dataset_type + "_nrc.csv"
        print("Saved NRC VAD features to", op_dir + dataset_type + "_nrc_vad.csv")
        output_file_VAD = op_dir + dataset_type + "_nrc_vad.csv"
        print("Saved Readability features to", op_dir + dataset_type + "_readability.csv")
        output_file_readability = op_dir + dataset_type + "_readability.csv"
    result.to_csv(output_file, index=False)
    result_VAD.to_csv(output_file_VAD, index=False)
    result_readability.to_csv(output_file_readability, index=False)