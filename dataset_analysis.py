"""
Produces the analysis of both datasets
Master's Thesis: "Automatic Personality Trait Prediction for Clinical Decision Support"
Author: André Morgado
ID: ist199888
Instituto Superior Técnico, Universidade de Lisboa
24/05/2026

Version 1.0 - Lacks commenting (To implement).
"""

#Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
from transformers import DistilBertTokenizerFast
import numpy as np
import re
import spacy


nlp = spacy.load("en_core_web_sm")
dataset = 'Essays'  # 'Kaggle' or 'Essays'
preprocessing = "Heavy" #"Light", "Heavy", "None"

Kaggle_traits = {
    "I/E": ("I", "E"),
    "N/S": ("N", "S"),
    "T/F": ("T", "F"),
    "J/P": ("J", "P"),
}

Big_five_traits = ["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]



#Utility functions
def split_phrases(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def word_count(texts):
    return [len(text.split()) if isinstance(text, str) else 0 for text in texts]




df_kaggle = pd.read_csv(f"./Data/Kaggle/kaggle_{preprocessing.lower()}.csv")
df_kaggle_aux = df_kaggle.copy() #Used for stats
df_essays = pd.read_csv(f"./Data/Essays/essays_{preprocessing.lower()}.csv")
df_essays_aux = df_essays.copy() #Used for stats

for trait in Big_five_traits:
    df_essays[trait] = df_essays[trait].map({"y": 1, "n": 0})

if dataset == 'Kaggle':
    output_dir = "./Results/Dataset_Analysis/Kaggle"
elif dataset == 'Essays':
    output_dir = "./Results/Dataset_Analysis/Essays"


os.makedirs(output_dir, exist_ok=True)


if dataset == 'Kaggle': 


    df_kaggle["post_list"] = df_kaggle["posts"].apply(lambda x: x.split("|||"))
    df_kaggle["post_count"] = df_kaggle["post_list"].apply(len)
    avg_post_count = df_kaggle["post_count"].mean()

    df_kaggle["post_word_counts"] = df_kaggle["post_list"].apply(word_count)
    df_kaggle["avg_words_per_post"] = df_kaggle["post_word_counts"].apply(lambda x: sum(x)/len(x) if len(x) > 0 else 0)
    df_kaggle["total_words_per_user"] = df_kaggle["post_word_counts"].apply(sum)
    avg_words_per_user = df_kaggle["total_words_per_user"].mean()


    trait_counts = {
        pair: [
            (df_kaggle["type"].str[i] == t1).sum(),
            (df_kaggle["type"].str[i] == t2).sum()
        ]
        for i, (pair, (t1, t2)) in enumerate(Kaggle_traits.items())
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(Kaggle_traits))
    width = 0.15


    for idx, (pair, counts) in enumerate(trait_counts.items()):
        c1 = cm.Blues(0.4)
        c2 = cm.Blues(0.7)
        ax.bar(x[idx] - width/2, counts[0], width, label=Kaggle_traits[pair][0], color=c1)
        ax.bar(x[idx] + width/2, counts[1], width, label=Kaggle_traits[pair][1], color=c2)

    ax.set_xticks(x)
    ax.set_xticklabels(Kaggle_traits.keys())
    ax.set_ylabel("Count")
    ax.set_title("MBTI Trait Distribution")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Trait",
            loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{output_dir}/mbti_trait_distribution_{preprocessing.lower()}.png", dpi=300)
    plt.close()


    plt.figure(figsize=(10, 6))
    min_val = int(df_kaggle["avg_words_per_post"].min())
    max_val = int(df_kaggle["avg_words_per_post"].max())
    bins = np.arange(min_val - 0.5, max_val + 1.5, max(1, (max_val - min_val) // 50))
    sns.histplot(df_kaggle["avg_words_per_post"], bins=bins, kde=True)
    plt.title("Distribution of Post's Average Word Count, per User")
    plt.xlabel("Average Words per Post")
    plt.ylabel("Frequency")
    plt.text(0.3, 0.85, f"Avg posts per user: {avg_post_count:.2f}",
            transform=plt.gca().transAxes, fontsize=11,
            bbox=dict(facecolor="white", alpha=0.85))

    plt.text(0.3, 0.78, 
            f"Avg words per post: {df_kaggle['avg_words_per_post'].mean():.2f}",
            transform=plt.gca().transAxes, fontsize=11,
            bbox=dict(facecolor="white", alpha=0.85))
    plt.text(
        0.3, 0.71,
        f"Avg words per user: {avg_words_per_user:.2f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.85)
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mbti_avg_word_distribution_per_user_{preprocessing.lower()}.png", dpi=300)
    plt.close()


    rows = []
    for idx, (trait_pair, (t1, t2)) in enumerate(Kaggle_traits.items()):
        for trait in (t1, t2):
            subset = df_kaggle[df_kaggle["type"].str[idx] == trait]
            all_post_words = [t for sublist in subset["post_word_counts"] for t in sublist]
            rows.append({
                "Trait Pair": trait_pair,
                "Trait": trait,
                "Avg Words per Post": sum(all_post_words)/len(all_post_words)
            })
    avg_trait_df = pd.DataFrame(rows)


    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(Kaggle_traits))
    width = 0.15

    for idx, (trait_pair, (t1, t2)) in enumerate(Kaggle_traits.items()):
        v1 = avg_trait_df.loc[(avg_trait_df["Trait Pair"] == trait_pair) & 
                            (avg_trait_df["Trait"] == t1), "Avg Words per Post"].values[0]
        v2 = avg_trait_df.loc[(avg_trait_df["Trait Pair"] == trait_pair) & 
                            (avg_trait_df["Trait"] == t2), "Avg Words per Post"].values[0]

        c1 = cm.Blues(0.4)
        c2 = cm.Blues(0.7) 
        ax.bar(x[idx] - width/2, v1, width, label=t1, color=c1)
        ax.bar(x[idx] + width/2, v2, width, label=t2, color=c2)

    ax.set_xticks(x)
    ax.set_xticklabels(Kaggle_traits.keys())
    ax.set_ylabel("Average Words per Post")
    ax.set_xlabel("Trait Pair")
    ax.set_title("Average Word Count per Post by MBTI Trait")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Trait",
            loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{output_dir}/mbti_avg_words_per_trait_{preprocessing.lower()}.png", dpi=300)
    plt.close()

    type_counts = df_kaggle["type"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))


    colors = cm.Blues(np.linspace(0.3, 0.8, len(type_counts)))
    ax.bar(type_counts.index, type_counts.values, color=colors)

    ax.set_title("Distribution of MBTI Personality Types")
    ax.set_xlabel("MBTI Type")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/mbti_type_distribution_{preprocessing.lower()}.png", dpi=300)
    plt.close()

elif dataset == 'Essays':
    if preprocessing == "None":
        unit_name = "Phrase"
        unit_name_lower = "phrase"
        unit_name_plural = "Phrases"
        
        essay_cols = [f"essay{i}" for i in range(10)]
        df_essays["full_text"] = df_essays["text"].fillna("")
        df_essays["phrase_list"] = df_essays["full_text"].apply(split_phrases)

        df_essays["unit_list"] = df_essays["phrase_list"]
        df_essays["unit_count"] = df_essays["phrase_list"].apply(len)
        avg_unit_count = df_essays["unit_count"].mean()
        
        df_essays["unit_word_counts"] = df_essays["phrase_list"].apply(word_count)

        df_essays["avg_words_per_unit"] = df_essays["unit_word_counts"].apply(
            lambda x: sum(x)/len(x) if len(x) > 0 else 0
        )

    else:
        unit_name = "Chunk"
        unit_name_lower = "chunk"
        unit_name_plural = "Chunks"
        
        df_essays["chunk_list"] = df_essays["text"].apply(lambda x: x.split("|||") if pd.notna(x) else [])
        df_essays["unit_list"] = df_essays["chunk_list"]
        df_essays["unit_count"] = df_essays["chunk_list"].apply(len)
        avg_unit_count = df_essays["unit_count"].mean()

        df_essays["unit_word_counts"] = df_essays["chunk_list"].apply(word_count)

        df_essays["avg_words_per_unit"] = df_essays["unit_word_counts"].apply(
            lambda x: sum(x)/len(x) if len(x) > 0 else 0
        )

    df_essays["total_words_per_user"] = df_essays["unit_word_counts"].apply(sum)
    avg_words_per_user = df_essays["total_words_per_user"].mean()
        

    plt.figure(figsize=(10, 6))

    min_val = int(df_essays["avg_words_per_unit"].min())
    max_val = int(df_essays["avg_words_per_unit"].max())
    bins = np.arange(min_val - 0.5, max_val + 1.5, max(1, (max_val - min_val) // 50))
    sns.histplot(df_essays["avg_words_per_unit"], bins=bins, kde=True)

    plt.title(f"Distribution of {unit_name}'s Average Word Count per User")
    plt.xlabel(f"Average Words per {unit_name}")
    plt.ylabel("Frequency")

    plt.text(
        0.4, 0.92,
        f"Avg {unit_name_lower}s per user: {avg_unit_count:.2f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.85)
    )

    plt.text(
        0.4, 0.85,
        f"Avg words per {unit_name_lower}: {df_essays['avg_words_per_unit'].mean():.2f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.85)
    )
    
    plt.text(
        0.4, 0.78,
        f"Avg words per user: {avg_words_per_user:.2f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.85)
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ocean_avg_word_distribution_per_user_{preprocessing.lower()}.png", dpi=300)
    plt.close()

    rows = []

    for trait in Big_five_traits:
        for value in [1, 0]:  # y vs n
            subset = df_essays[df_essays[trait] == value]

            avg_words_per_user = subset["avg_words_per_unit"].mean() if len(subset) > 0 else 0

            rows.append({
                "Trait": trait,
                "Value": "y" if value == 1 else "n",
                f"Avg Words per {unit_name}": avg_words_per_user
            })

    avg_trait_df = pd.DataFrame(rows)


    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(Big_five_traits))
    width = 0.2

    for idx, trait in enumerate(Big_five_traits):
        v_y = avg_trait_df.loc[
            (avg_trait_df["Trait"] == trait) &
            (avg_trait_df["Value"] == "y"),
            f"Avg Words per {unit_name}"
        ].values[0]

        v_n = avg_trait_df.loc[
            (avg_trait_df["Trait"] == trait) &
            (avg_trait_df["Value"] == "n"),
            f"Avg Words per {unit_name}"
        ].values[0]

        ax.bar(x[idx] - width/2, v_y, width, label=f"{trait}=y", color=cm.Blues(0.4))
        ax.bar(x[idx] + width/2, v_n, width, label=f"{trait}=n", color=cm.Blues(0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(Big_five_traits)
    ax.set_ylabel(f"Average Words per {unit_name}")
    ax.set_title(f"Average Word Count per {unit_name} by OCEAN Trait")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(),
            loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{output_dir}/ocean_avg_words_per_trait_{preprocessing.lower()}.png", dpi=300)
    plt.close()


    plt.figure(figsize=(10, 6))

    sns.histplot(df_essays["unit_count"], bins=50, kde=True, discrete=True)

    plt.title(f"Distribution of Number of {unit_name_plural} per User")
    plt.xlabel(f"Number of {unit_name_plural} per User")
    plt.ylabel("Frequency")


    plt.text(
        0.62, 0.85,
        f"Avg {unit_name_lower}s per user: {avg_unit_count:.2f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.85)
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ocean_avg_{unit_name_lower}_count_per_user_{preprocessing.lower()}.png", dpi=300)
    plt.close()

    trait_counts = {
        trait: [
            (df_essays[trait] == 1).sum(),
            (df_essays[trait] == 0).sum()
        ]
        for trait in Big_five_traits
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(Big_five_traits))
    width = 0.2

    for idx, trait in enumerate(Big_five_traits):

        counts = trait_counts[trait]

        ax.bar(x[idx] - width/2, counts[0], width,
                label=f"{trait}=y", color=cm.Blues(0.4))

        ax.bar(x[idx] + width/2, counts[1], width,
                label=f"{trait}=n", color=cm.Blues(0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(Big_five_traits)
    ax.set_ylabel("Count")
    ax.set_title("OCEAN Trait Distribution")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(),
              loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"{output_dir}/ocean_trait_distribution_{preprocessing.lower()}.png", dpi=300)
    plt.close()


if dataset == 'Kaggle':
    df_kaggle_aux["post_list"] = df_kaggle_aux["posts"].apply(lambda x: x.split("|||") if pd.notna(x) else [])
    df_kaggle_aux["num_posts"] = df_kaggle_aux["post_list"].apply(len)
    df_kaggle_aux["post_word_counts"] = df_kaggle_aux["post_list"].apply(lambda lst: [len(p.split()) for p in lst])
    df_kaggle_aux["num_words_user"] = df_kaggle_aux["post_word_counts"].apply(sum)
    df_kaggle_aux["avg_words_per_post"] = df_kaggle_aux.apply(
        lambda row: np.mean(row["post_word_counts"]) if len(row["post_word_counts"]) > 0 else 0,
        axis=1
    )
    df_kaggle_aux.drop(columns=["post_list", "post_word_counts"], inplace=True)
elif dataset == 'Essays':
    df_essays_aux["chunk_list"] = df_essays_aux["text"].apply(lambda x: x.split("|||") if pd.notna(x) else [])
    df_essays_aux["num_chunks"] = df_essays_aux["chunk_list"].apply(len)
    df_essays_aux["chunk_word_counts"] = df_essays_aux["chunk_list"].apply(lambda lst: [len(c.split()) for c in lst])
    df_essays_aux["num_words_user"] = df_essays_aux["chunk_word_counts"].apply(sum)
    df_essays_aux["avg_words_per_chunk"] = df_essays_aux.apply(
        lambda row: np.mean(row["chunk_word_counts"]) if len(row["chunk_word_counts"]) > 0 else 0,
        axis=1
    )
    df_essays_aux.drop(columns=["chunk_list", "chunk_word_counts"], inplace=True)

if dataset == 'Kaggle':
    if preprocessing == "Light":
        df_kaggle_aux.to_csv("./Data/Kaggle/kaggle_light.csv", index=False)
    elif preprocessing == "Heavy":
        df_kaggle_aux.to_csv("./Data/Kaggle/kaggle_heavy.csv", index=False)
elif dataset == 'Essays':
    if preprocessing == "Light":
        df_essays_aux.to_csv("./Data/Essays/essays_light.csv", index=False)
    elif preprocessing == "Heavy":
        df_essays_aux.to_csv("./Data/Essays/essays_heavy.csv", index=False)