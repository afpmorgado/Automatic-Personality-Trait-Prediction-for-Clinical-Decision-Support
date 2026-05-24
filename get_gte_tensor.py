"""Creates the GTE embeddings for the users of both datasets
Master's Thesis: "Automatic Personality Trait Prediction for Clinical Decision Support"
Author: André Morgado
ID: ist199888
Instituto Superior Técnico, Universidade de Lisboa
24/05/2026

Version 1.0 - Lacks commenting (To implement).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse

#%cd /kaggle/input/models/andrmorgado/main21/pytorch/main21/1 #Change to current directory

BATCH_SIZE = 4

preprocessing = "Light"
dataset = "Kaggle"
paraphrasing = False

gte_model_path = "/kaggle/input/etmopt2/pytorch/etmopt2/1/ETMOPT/gte-multilingual-base" #change path to model Path


default_output_dir = f"/kaggle/working/{dataset}/" #change path to output directory
if paraphrasing:
    default_csv_path = f"./Data/{dataset}/{dataset.lower()}_paraphrased.csv"
    default_output_filename = f"GTE_{dataset.lower()}_{preprocessing.lower()}_paraphrased.pt"
else:
    default_csv_path = f"./Data/{dataset}/{dataset.lower()}_{preprocessing.lower()}.csv"
    default_output_filename = f"GTE_{dataset.lower()}_{preprocessing.lower()}.pt"



print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
SEED = 29
DM = 3

MBTIs_tuple = (
    'INTJ', 'INTP', 'INFP', 'ENTP', 'ISTP', 'ISFP',
    'ESTJ', 'ISTJ', 'ESTP', 'ISFJ', 'ENFP', 'ESFP',
    'ESFJ', 'ENFJ', 'INFJ', 'ENTJ'
)

TOKEN_MASK = ""

def find_all_MBTIs_lower(post_lower: str, mbti_list: tuple) -> list:
    found_indices = []
    for mbti_type in mbti_list:
        for match in re.finditer(mbti_type.lower(), post_lower):
            found_indices.append((match.start(), match.end()))
    return sorted(found_indices)



def generate_gte_embeddings(csv_path, output_dir, output_filename):
    print("Starting GTE embedding generation")
    print(f"SEED={SEED}, DM={DM}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(gte_model_path, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        gte_model_path,
        trust_remote_code=True,
        output_hidden_states=True,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16
    ).to(device)

    model.eval()
    MAX_LEN = tokenizer.model_max_length
    print(f"GTE loaded on {device}")

    try:
        data_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Loaded {len(data_df)} users")

    all_gte_embeddings = []
    texts_buffer = []

    pbar = tqdm(total=len(data_df), desc="Encoding users")
    MAX_POSTS_PER_USER = 50



    rows = []
    for idx, row in data_df.iterrows():
        rows.append(row)

        if dataset == 'Kaggle':
            raw_posts = str(row["posts"]).split("|||")
        elif dataset == 'Essays':
            raw_posts = str(row["text"]).split("|||")
        
        limited_posts = raw_posts[:MAX_POSTS_PER_USER]
        
        joined_posts = " ".join(limited_posts)
        
        texts_buffer.append(joined_posts)
        pbar.update(1)

        if len(texts_buffer) == BATCH_SIZE or idx == len(data_df) - 1:
            inputs = tokenizer(
                texts_buffer,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            hidden_states = outputs.hidden_states
            attention_mask = inputs["attention_mask"]  # [B, T]
            selected_layers = hidden_states[-DM:]       # list of [B, T, H]

            mask = attention_mask.unsqueeze(-1)         # [B, T, 1]
            layer_vectors = []

            for layer_hs in selected_layers:
                masked_hs = layer_hs * mask
                summed = masked_hs.sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-6)
                layer_vectors.append(summed / denom)

            user_vectors = torch.stack(layer_vectors).mean(dim=0)  # [B, H]
            batch_count = user_vectors.size(0)
            all_gte_embeddings.extend(user_vectors.cpu())
            texts_buffer = []
    pbar.close()

    embeddings_tensor = torch.stack(all_gte_embeddings)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_filename)
    torch.save(embeddings_tensor, output_path)
    print (len(embeddings_tensor))
    print(f"Saved embeddings: {embeddings_tensor.shape}")
    print(f"Path: {output_path}")


if __name__ == "__main__":
    generate_gte_embeddings(default_csv_path, default_output_dir, default_output_filename )