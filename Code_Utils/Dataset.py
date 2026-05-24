import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from .utils import yn_to_binary, split_into_chunks

MAX_CHUNKS = 35

class MBTIDataset_baseline(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        label = self.labels[item]

        encodings = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(label, dtype=torch.float)
        }


class MBTIDataset_FM_Abl(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        label = self.labels[item]

        posts = split_into_chunks(sentence)[:MAX_CHUNKS]

        encodings = self.tokenizer(
            posts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        num_posts = len(posts)
        if num_posts < MAX_CHUNKS:
            pad_size = MAX_CHUNKS - num_posts
            zero_pad = torch.zeros(pad_size, self.max_len, dtype=torch.long)
            encodings['input_ids'] = torch.cat([encodings['input_ids'], zero_pad], dim=0)
            encodings['attention_mask'] = torch.cat([encodings['attention_mask'], zero_pad], dim=0)

        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(label, dtype=torch.float)
        }

class featureDataset_FM_Abl(Dataset):
    def __init__(self, original_dataset, gte_embeds, psycholinguistic_feats):
        self.original_dataset = original_dataset
        self.gte_embeds = gte_embeds
        self.psycholinguistic_feats = psycholinguistic_feats
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        item = self.original_dataset[idx]
        item['gte_embedding'] = self.gte_embeds[idx]
        item['psycholinguistic_feats'] = self.psycholinguistic_feats[idx]
        return item


class featureDataset_baseline(Dataset):
    def __init__(self, original_dataset, psycho_feats):
        self.original_dataset = original_dataset
        self.psycho_feats = psycho_feats
        
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        item = self.original_dataset[idx]
        item['psycholinguistic_feats'] = self.psycho_feats[idx]
        return item


def dataset_config(dataset_name):
    if dataset_name == "Kaggle":
        N_AXIS = 4
        axes = ["I-E", "N-S", "T-F", "J-P"]
        classes = {"I": 0, "E": 1, "N": 0, "S": 1, "T": 0, "F": 1, "J": 0, "P": 1}
        label_cols = "type"
        text_col = "posts"
        alphas = [0.768, 0.863, 0.46, 0.39]
        
    elif dataset_name == "Essays":
        N_AXIS = 5
        axes = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
        classes = None
        label_cols = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
        text_col = "text_overlap"
        alphas = [0.5, 0.5, 0.5, 0.5, 0.5]
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Must be 'Kaggle' or 'Essays'")
    
    return N_AXIS, axes, classes, label_cols, text_col, alphas


def load_data_FM_Abl(dataset_name, preprocessing, classes, label_cols, text_col, paraphrased=False, split=False):
    if dataset_name == "Kaggle":
        if preprocessing == "Light":
            data = pd.read_csv(f"./Data/Kaggle/kaggle_light.csv")
            gte_embeddings = torch.load(f"./Features/Kaggle/GTE_kaggle_light.pt")
        elif preprocessing == "Heavy":
            data = pd.read_csv(f"./Data/Kaggle/kaggle_heavy.csv")
            gte_embeddings = torch.load(f"./Features/Kaggle/GTE_kaggle_heavy.pt")
        
        labels = np.array([[classes[c] for c in pers] for pers in data["type"]], dtype="float32")
        sentences = data[text_col].tolist()
        psycholinguistic_feats = np.load(f'./Features/Kaggle/psycholinguist_features.npy')
        if paraphrased and preprocessing == "Light":
            data_paraphrased = pd.read_csv("./Data/Kaggle/kaggle_paraphrased.csv")
            gte_embeddings_paraphrased = torch.load(f"./Features/Kaggle/GTE_kaggle_light_paraphrased.pt")
            sentences_paraphrased = data_paraphrased[text_col].tolist()
            psycholinguistic_feats_paraphrased = np.load(f'./Features/Kaggle/psycholinguist_features_paraphrased.npy')
        if split and preprocessing == "Light":
            data_split = pd.read_csv("./Data/Kaggle/kaggle_split.csv")
            gte_embeddings_split = torch.load(f"./Features/Kaggle/GTE_kaggle_light_split.pt")
            sentences_split = data_split[text_col].tolist()
            psycholinguistic_feats_split = np.load(f'./Features/Kaggle/psycholinguist_features_split.npy')
    elif dataset_name == "Essays":
        if preprocessing == "Light":
            data = pd.read_csv(f"./Data/Essays/essays_light.csv")
            gte_embeddings = torch.load(f"./Features/Essays/GTE_essays_light.pt")
        elif preprocessing == "Heavy":
            data = pd.read_csv(f"./Data/Essays/essays_heavy.csv")
            gte_embeddings = torch.load(f"./Features/Essays/GTE_essays_heavy.pt")
        
        labels = np.array([
            [yn_to_binary(row[col]) for col in label_cols]
            for _, row in data.iterrows()
        ], dtype="float32")
        
        sentences = data[text_col].tolist()
        psycholinguistic_feats = np.load(f'./Features/Essays/psycholinguist_features.npy')
        if paraphrased and preprocessing == "Light":
            data_paraphrased = pd.read_csv("./Data/Essays/essays_paraphrased.csv")
            gte_embeddings_paraphrased = torch.load(f"./Features/Essays/GTE_essays_light_paraphrased.pt")
            sentences_paraphrased = data_paraphrased[text_col].tolist()
            psycholinguistic_feats_paraphrased = np.load(f'./Features/Essays/psycholinguist_features_paraphrased.npy')
        if split and preprocessing == "Light":
            data_split = pd.read_csv("./Data/Essays/essays_split.csv")
            gte_embeddings_split = torch.load(f"./Features/Essays/GTE_essays_light_split.pt")
            sentences_split = data_split[text_col].tolist()
            psycholinguistic_feats_split = np.load(f'./Features/Essays/psycholinguist_features_split.npy')
        
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Must be 'Kaggle' or 'Essays'")

    psycholinguistic_feats = np.nan_to_num(psycholinguistic_feats, nan=0.0, posinf=0.0, neginf=0.0)
    psycholinguistic_feats = np.clip(psycholinguistic_feats, -1e10, 1e10)
    psycholinguistic_feats = torch.from_numpy(psycholinguistic_feats).float().contiguous()
    if paraphrased and preprocessing == "Light":
        psycholinguistic_feats_paraphrased = np.nan_to_num(psycholinguistic_feats_paraphrased, nan=0.0, posinf=0.0, neginf=0.0)
        psycholinguistic_feats_paraphrased = np.clip(psycholinguistic_feats_paraphrased, -1e10, 1e10)
        psycholinguistic_feats_paraphrased = torch.from_numpy(psycholinguistic_feats_paraphrased).float().contiguous()
        psycholinguistic_feats_paraphrased = torch.where(torch.isfinite(psycholinguistic_feats_paraphrased), psycholinguistic_feats_paraphrased, torch.zeros_like(psycholinguistic_feats_paraphrased))
    if split and preprocessing == "Light":
        psycholinguistic_feats_split = np.nan_to_num(psycholinguistic_feats_split, nan=0.0, posinf=0.0, neginf=0.0)
        psycholinguistic_feats_split = np.clip(psycholinguistic_feats_split, -1e10, 1e10)
        psycholinguistic_feats_split = torch.from_numpy(psycholinguistic_feats_split).float().contiguous()
        psycholinguistic_feats_split = torch.where(torch.isfinite(psycholinguistic_feats_split), psycholinguistic_feats_split, torch.zeros_like(psycholinguistic_feats_split))
    # if paraphrased return both original and paraphrased data, otherwise return just original
    if paraphrased and split and preprocessing == "Light":
        return data, sentences, labels, gte_embeddings, psycholinguistic_feats, data_paraphrased, sentences_paraphrased, gte_embeddings_paraphrased, psycholinguistic_feats_paraphrased, data_split, sentences_split, gte_embeddings_split, psycholinguistic_feats_split
    else:
        return data, sentences, labels, gte_embeddings, psycholinguistic_feats 


def load_data_baseline(dataset_name, classes, label_cols, text_col):
    if dataset_name == "Kaggle":
        data = pd.read_csv(f"./Data/Kaggle/kaggle_light.csv")
        labels = np.array([[classes[c] for c in pers] for pers in data["type"]], dtype="float32")
        sentences = data[text_col].tolist()
        psycholinguistic_feats = np.load(f'./Features/Kaggle/psycholinguist_features.npy')
    elif dataset_name == "Essays":
        data = pd.read_csv(f"./Data/Essays/essays_light.csv")  # Adjust path as needed
        
        labels = np.array([
            [yn_to_binary(row[col]) for col in label_cols]
            for _, row in data.iterrows()
        ], dtype="float32")
        sentences = data[text_col].tolist()
        psycholinguistic_feats = np.load(f'./Features/Essays/psycholinguist_features.npy')  # Adjust path as needed
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Must be 'Kaggle' or 'Essays'")
    psycholinguistic_feats = np.nan_to_num(psycholinguistic_feats, nan=0.0, posinf=0.0, neginf=0.0)
    psycholinguistic_feats = np.clip(psycholinguistic_feats, -1e10, 1e10)
    psycholinguistic_feats = torch.from_numpy(psycholinguistic_feats).float().contiguous()
    return data, sentences, labels, psycholinguistic_feats