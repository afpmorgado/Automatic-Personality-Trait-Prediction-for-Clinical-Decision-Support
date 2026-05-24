
"""Preprocesses and creates the necessary datasets for the experiments
Master's Thesis: "Automatic Personality Trait Prediction for Clinical Decision Support"
Author: André Morgado
ID: ist199888
Instituto Superior Técnico, Universidade de Lisboa
24/05/2026

Version 1.0 - Lacks commenting (To implement).
"""

#%cd /kaggle/input/models/andrmorgado/main21/pytorch/main21/1 #Change to current directory
import re
import pandas as pd
import string
import contractions
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import random


seed = 29
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def add_user_id(df, id_col="user_id"):
    df = df.copy()
    df[id_col] = range(1, len(df) + 1)
    return df

token = "<mask>"
Keep_MBTI_tokens = False
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
POST_SEP = "POSTSEP"
nlp = spacy.load("en_core_web_sm")

def create_overlapping_from_chunks(chunked_text, chunk_size=50, overlap=10):
    if pd.isna(chunked_text) or not chunked_text.strip():
        return ""

    continuous_text = chunked_text.replace("|||", " ")
    
    words = continuous_text.split()
    if len(words) <= chunk_size:
        return " ".join(words)
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        
        if end >= len(words):
            break
        start = end - overlap
    
    return "|||".join(chunks)

def essays_sliding_window_nonoverlap(text, chunk_size=40):
    if pd.isna(text) or not text.strip():
        return ""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start:start+chunk_size]
        chunks.append(" ".join(chunk))
        start += chunk_size
    return "|||".join(chunks)

def count_chunks(text):
    if pd.isna(text) or text.strip() == "":
        return 0
    return len(text.split("|||"))

MBTIs = ('INTJ','INTP','INFP','ENTP','ISTP','ISFP','ESTJ','ISTJ','ESTP','ISFJ','ENFP','ESFP','ESFJ','ENFJ','INFJ','ENTJ')
mbti_pattern = re.compile(r'\b(' + '|'.join(m.lower() for m in MBTIs) + r')\b')
def text_preprocessing_light_Kaggle(text):
    if pd.isna(text):
        return ""

    text = text.replace("|||", f" {POST_SEP} ")
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    if not Keep_MBTI_tokens:
        text = mbti_pattern.sub(token, text)
    text = text.replace(POST_SEP.lower(), "|||")
    return text

def text_preprocessing_light_Essays(text):
    if pd.isna(text):
        return ""
    text = text.replace("|||", f" {POST_SEP} ")
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    text = text.replace(POST_SEP.lower(), "|||")
    return text

def text_preprocessing_heavy_Kaggle(text):
    if pd.isna(text):
        return ""

    text = text.replace("|||", f" {POST_SEP} ")
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    if not Keep_MBTI_tokens:
        text = mbti_pattern.sub(token, text)
    text = contractions.fix(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    text = ' '.join(tokens)
    text = text.replace(POST_SEP.lower(), "|||")
    return text

def text_preprocessing_heavy_Essays(text):
    if pd.isna(text):
        return ""

    text = text.replace("|||", f" {POST_SEP} ")
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    text = contractions.fix(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    text = ' '.join(tokens)
    text = text.replace(POST_SEP.lower(), "|||")
    return text


kaggle_raw = pd.read_csv("./Data/Kaggle/kaggle_none.csv")
essays_raw = pd.read_csv("./Data/Essays/essays_none.csv")
def clean_raw_text(text):
    if pd.isna(text):
        return ""
    text = text.replace("\t", " ").replace("\n", " ").replace(";", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

kaggle_raw["posts"] = kaggle_raw["posts"].apply(clean_raw_text)
essays_raw["text"] = essays_raw["text"].apply(clean_raw_text)
essays_raw["text"] = essays_raw["text"].apply(lambda x: essays_sliding_window_nonoverlap(x, chunk_size=40))

print ("Original number of users (Essays):", len(essays_raw))

MIN_CHUNKS = 5
MAX_CHUNKS = 35

essays_raw = essays_raw[
    (essays_raw["text"].apply(count_chunks) >= MIN_CHUNKS) & 
    (essays_raw["text"].apply(count_chunks) <= MAX_CHUNKS)
]

print ("Number of users after filtering (Essays):", len(essays_raw))
print ("Number of Kaggle users:", len(kaggle_raw))
kaggle_dataset_light = kaggle_raw.copy()
kaggle_dataset_light["posts"] = kaggle_dataset_light["posts"].apply(text_preprocessing_light_Kaggle)

kaggle_dataset_heavy = kaggle_raw.copy()
kaggle_dataset_heavy["posts"] = kaggle_dataset_heavy["posts"].apply(text_preprocessing_heavy_Kaggle)

essays_dataset_light = essays_raw.copy()
essays_dataset_light["text"] = essays_dataset_light["text"].apply(text_preprocessing_light_Essays)
essays_dataset_heavy = essays_raw.copy()
essays_dataset_heavy["text"] = essays_dataset_heavy["text"].apply(text_preprocessing_heavy_Essays)


essays_dataset_light["text_overlap"] = essays_dataset_light["text"].apply(
    lambda x: create_overlapping_from_chunks(x, chunk_size=50, overlap=10)
)

essays_dataset_heavy["text_overlap"] = essays_dataset_heavy["text"].apply(
    lambda x: create_overlapping_from_chunks(x, chunk_size=50, overlap=5)
)

def clean_df_for_csv(df):
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]

kaggle_dataset_light = add_user_id(kaggle_dataset_light)
kaggle_dataset_heavy = add_user_id(kaggle_dataset_heavy)

essays_dataset_light = add_user_id(essays_dataset_light)
essays_dataset_heavy = add_user_id(essays_dataset_heavy)

clean_df_for_csv(kaggle_dataset_light).to_csv("./Data/Kaggle/kaggle_light.csv", index=False)
clean_df_for_csv(kaggle_dataset_heavy).to_csv("./Data/Kaggle/kaggle_heavy.csv", index=False)

clean_df_for_csv(essays_dataset_light).to_csv("./Data/Essays/essays_light.csv", index=False)
clean_df_for_csv(essays_dataset_heavy).to_csv("./Data/Essays/essays_heavy.csv", index=False)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import time

class OpusMTBackTranslator:

    def __init__(self, pivot_lang="fr", device=None):

        self.pivot = pivot_lang
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing Opus-MT for EN <-> {pivot_lang.upper()}")
        print(f"Using device: {self.device}")

        model_en_to = f"Helsinki-NLP/opus-mt-en-{pivot_lang}"
        model_to_en = f"Helsinki-NLP/opus-mt-{pivot_lang}-en"
        
        print(f"⬇Loading {model_en_to}...")
        self.model_en_pivot = AutoModelForSeq2SeqLM.from_pretrained(
            model_en_to,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        self.tokenizer_en_pivot = AutoTokenizer.from_pretrained(model_en_to)
        
        print(f"⬇Loading {model_to_en}...")
        self.model_pivot_en = AutoModelForSeq2SeqLM.from_pretrained(
            model_to_en,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        ).to(self.device)
        self.tokenizer_pivot_en = AutoTokenizer.from_pretrained(model_to_en)
        
        if self.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def translate_batch(self, texts, model, tokenizer, max_length=85):
        """Translate a batch of texts"""
        if not texts:
            return []
        
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        with torch.no_grad():
            if self.device == 'cuda':
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    num_beams=1,
                    early_stopping=False,
                    do_sample=False,
                    use_cache=True
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    num_beams=1,
                    early_stopping=False,
                    do_sample=False,
                    use_cache=True
                )
        
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def back_translate_chunks(self, chunks, batch_size=64):

        if not chunks:
            return []
        
        results = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"Processing {len(chunks)} chunks in {total_batches} batches...")
        start_time = time.time()
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Back-translating"):
            batch = chunks[i:i+batch_size]
            
            pivot_texts = self.translate_batch(
                batch, 
                self.model_en_pivot, 
                self.tokenizer_en_pivot
            )
            
            back_texts = self.translate_batch(
                pivot_texts,
                self.model_pivot_en,
                self.tokenizer_pivot_en
            )
            
            results.extend(back_texts)
            
            if self.device == 'cuda' and i % (batch_size * 10) == 0 and i > 0:
                torch.cuda.empty_cache()
        
        elapsed = time.time() - start_time
        chunks_per_sec = len(chunks) / elapsed
        print(f"Completed in {elapsed:.1f}s ({chunks_per_sec:.1f} chunks/sec)")
        
        return results

def back_translate_dataset(df, text_col, batch_size=64, pivot_lang="fr"):
    translator = OpusMTBackTranslator(pivot_lang=pivot_lang)
    
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Back-translating users ({pivot_lang})"):
        text = row[text_col]
        
        if pd.isna(text) or not text.strip():
            results.append(text)
            continue
        if "|||" in text:
            chunks = text.split("|||")
        else:
            chunks = [text]
        bt_chunks = translator.back_translate_chunks(chunks, batch_size=batch_size)

        results.append("|||".join(bt_chunks))
    
    return pd.Series(results, index=df.index)


print("\n" + "="*80)
print("BACK-TRANSLATION DATA AUGMENTATION (Opus-MT)")
print("="*80)

print("\n=== Back-translating Kaggle Dataset ===")
kaggle_paraphrased = kaggle_dataset_light.copy()
kaggle_paraphrased["posts"] = back_translate_dataset(
    kaggle_paraphrased, 
    text_col="posts",
    batch_size=256,
    pivot_lang="de"
)

print("\n=== Back-translating Essays Dataset ===")
essays_paraphrased = essays_dataset_light.copy()
essays_paraphrased["text"] = back_translate_dataset(
    essays_paraphrased,
    text_col="text", 
    batch_size=256,
    pivot_lang="de"
)

essays_paraphrased["text_overlap"] = essays_paraphrased["text"].apply(
    lambda x: create_overlapping_from_chunks(x, chunk_size=50, overlap=10)
)



print("\n=== Saving paraphrased datasets ===")
clean_df_for_csv(kaggle_paraphrased).to_csv("/kaggle/working/kaggle_paraphrased.csv", index=False) #change directories
clean_df_for_csv(essays_paraphrased).to_csv("/kaggle/working/essays_paraphrased.csv", index=False) #change directories
