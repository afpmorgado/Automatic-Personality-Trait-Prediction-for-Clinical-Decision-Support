"""Robustness and Reliability tests
Master's Thesis: "Automatic Personality Trait Prediction for Clinical Decision Support"
Author: André Morgado
ID: ist199888
Instituto Superior Técnico, Universidade de Lisboa
24/05/2026

Version 1.0 - Lacks commenting (To implement).
"""

#%cd /kaggle/input/models/andrmorgado/main58/pytorch/main58/1 #Change to the correct path before running
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import DistilBertModel, DistilBertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import random
import os
from scipy import stats
from scipy.interpolate import make_interp_spline
from Code_Utils.utils import yn_to_binary, set_model_seed, calculate_metrics
from Code_Utils.Dataset import featureDataset_FM_Abl, MBTIDataset_FM_Abl, dataset_config, load_data_FM_Abl
from Code_Utils.Model import BERTClassifier_FM_Essays, BERTClassifier_FM_Kaggle
import pingouin as pg
from sklearn.model_selection import train_test_split


torch.cuda.empty_cache()


dataset_name = "Essays"
preprocessing = "Light"
chunk_pooling = "simple_mean"
threshold_tuning = True

scenario_name = f"Optimized_{preprocessing}_{chunk_pooling}{'_TN' if threshold_tuning else ''}"

if not os.path.exists(f"/kaggle/input/models/andrmorgado/main58/pytorch/main58/1/Results/{dataset_name}/{scenario_name}"):
    raise ValueError(f"Scenario not found: /kaggle/input/models/andrmorgado/main58/pytorch/main58/1/Results/{dataset_name}/{scenario_name}. Please run the training script for this scenario before running inference tests.")

base_results_dir = os.path.join(f"/kaggle/input/models/andrmorgado/main58/pytorch/main58/1/Results/{dataset_name}", scenario_name)

test_a_dir = os.path.join(f"/kaggle/working/Results/{dataset_name}", "word_count_analysis")
calibration_dir = os.path.join(f"/kaggle/working//Results/{dataset_name}", "calibration_analysis")
paraphrased_dir = os.path.join(f"/kaggle/working//Results/{dataset_name}", "paraphrased_analysis")
os.makedirs(test_a_dir, exist_ok=True)
os.makedirs(calibration_dir, exist_ok=True)
os.makedirs(paraphrased_dir, exist_ok=True)


seed_value = 29
set_model_seed(seed_value)


MAX_SEQ_LEN = 64
MAX_CHUNKS = 35
BERT_NAME = "/kaggle/input/models/andrmorgado/distil/pytorch/distil/1/distilbert-base-uncased"
batch_size = 1

N_AXIS, axes, classes, label_cols, text_col, alphas = dataset_config(dataset_name)
print(f"\n{'='*80}")
print(f"UNIFIED TEST A: Word Count Sweep + Calibration Analysis")
print(f"Dataset: {dataset_name}")
print(f"{'='*80}\n")


def calculate_rank_biserial(group1, group2, u_statistic):

    n1 = len(group1)
    n2 = len(group2)
    
    r = 1 - (2 * u_statistic) / (n1 * n2)
    
    return r

def interpret_rank_biserial(r):
    abs_r = abs(r)
    if abs_r < 0.1:
        return 'Negligible'
    elif abs_r < 0.3:
        return 'Small'
    elif abs_r < 0.5:
        return 'Medium'
    else:
        return 'Large'

data, sentences, labels, bge_embeddings, psycholinguistic_feats, data_paraphrased, sentences_paraphrased, bge_embeddings_paraphrased, psycholinguistic_feats_paraphrased, data_split, sentences_split, bge_embeddings_split, psycholinguistic_feats_split = load_data_FM_Abl(dataset_name, preprocessing, classes, label_cols, text_col, paraphrased=True, split=True)

if dataset_name == "Kaggle":
    stratify_col = data[label_cols]
elif dataset_name == "Essays":
    stratify_col = data[label_cols].apply(lambda row: ''.join(row.values), axis=1)

train_indices, val_test_indices = train_test_split(
    np.arange(len(data)), 
    test_size=0.4, 
    random_state=seed_value,
    stratify=stratify_col
)


test_indices, val_indices = train_test_split(
    val_test_indices, 
    test_size=0.5, 
    random_state=seed_value,
    stratify=stratify_col[val_test_indices]
)
tokenizer = DistilBertTokenizer.from_pretrained(BERT_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if dataset_name == "Kaggle":
    stratify_col = data[label_cols]
elif dataset_name == "Essays":
    stratify_col = data[label_cols].apply(lambda row: ''.join(row.values), axis=1)

min_words = data['num_words_user'].min()
max_words = data['num_words_user'].max()
print(f"Word count range: {min_words} to {max_words}")

bin_size = 50
word_bins = list(range(int(min_words), int(max_words) + bin_size, bin_size))
print(f"Word bins: {len(word_bins)} bins from {word_bins[0]} to {word_bins[-1]}\n")


user_results = []
user_results_paraphrased = []
calibration_results = []
user_embedding_correlations = []

for (train_idx, test_idx, val_idx) in [(train_indices, test_indices, val_indices)]:
    user_test_ids = data.iloc[test_indices]['user_id'].tolist()
    fold_test_df = data.iloc[test_indices].reset_index(drop=True)
    fold_test_df_paraphrased = data_paraphrased.iloc[test_indices].reset_index(drop=True)

    if dataset_name == "Kaggle":
        test_sentences = fold_test_df["posts"].tolist()
        test_sentences_paraphrased = fold_test_df_paraphrased["posts"].tolist()
        y_test = np.array([[classes[c] for c in t] for t in fold_test_df["type"]], dtype="float32")
    elif dataset_name == "Essays":
        test_sentences = fold_test_df[text_col].tolist()
        test_sentences_paraphrased = fold_test_df_paraphrased[text_col].tolist()
        y_test = np.array([[yn_to_binary(row[col]) for col in label_cols] for _, row in fold_test_df.iterrows()], dtype="float32")
    
    test_bge = bge_embeddings[test_indices]
    test_bge_paraphrased = bge_embeddings_paraphrased[test_indices]
    test_psycholinguistic_feats_fold = psycholinguistic_feats[test_indices]
    test_psycholinguistic_feats_fold_paraphrased = psycholinguistic_feats_paraphrased[test_indices]
    test_dataset = featureDataset_FM_Abl(MBTIDataset_FM_Abl(test_sentences, y_test, tokenizer, MAX_SEQ_LEN), test_bge, test_psycholinguistic_feats_fold)
    test_dataset_paraphrased = featureDataset_FM_Abl(MBTIDataset_FM_Abl(test_sentences_paraphrased, y_test, tokenizer, MAX_SEQ_LEN), test_bge_paraphrased, test_psycholinguistic_feats_fold_paraphrased)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader_paraphrased = DataLoader(test_dataset_paraphrased, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    fold_dir = os.path.join(base_results_dir, f"Run Results")
    model_path = os.path.join(fold_dir, "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        continue
    
    bert_model = DistilBertModel.from_pretrained(BERT_NAME)
    if dataset_name == "Kaggle":
        model = BERTClassifier_FM_Kaggle(bert_model, N_AXIS, bge_embed_dim=768, psych_dim=psycholinguistic_feats.shape[1], chunk_pooling=chunk_pooling).to(device)
    elif dataset_name == "Essays":
        model = BERTClassifier_FM_Essays(bert_model, N_AXIS, bge_embed_dim=768, psych_dim=psycholinguistic_feats.shape[1], chunk_pooling=chunk_pooling).to(device)
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    thresholds = [0.5] * N_AXIS
    if threshold_tuning:
        possible_paths = [
            os.path.join(fold_dir, 'final_test_avg_metrics.xlsx'), 
            os.path.join(fold_dir, 'final_test_per_trait_metrics.xlsx'), 
            os.path.join(fold_dir, 'metrics.xlsx')
        ]
        for file_path in possible_paths:
            if not os.path.exists(file_path):
                continue
            try:
                df = pd.read_excel(file_path)
                threshold_cols = [f'Threshold_{axis}' for axis in axes]
                if all(col in df.columns for col in threshold_cols):
                    row_idx = -1 if 'final' in file_path.lower() else 0
                    thresholds = [df[col].values[row_idx] for col in threshold_cols]
                    break
                elif 'Threshold' in df.columns and 'Trait' in df.columns:
                    thresholds = []
                    for axis in axes:
                        trait_rows = df[df['Trait'] == axis]
                        if len(trait_rows) > 0:
                            thresholds.append(trait_rows['Threshold'].values[-1])
                        else:
                            thresholds.append(0.5)
                    break
            except:
                continue
        print(f"Loaded thresholds: {[f'{t:.3f}' for t in thresholds]}")
    
    user_embedding_normal = []
    with torch.no_grad():
        for user_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            bge_emb = batch['bge_embedding'].to(device)
            psych_batch = batch['psycholinguistic_feats'].to(device)
            
            cls_out, user_embedding = model(input_ids, attention_mask, bge_emb, psych_batch)
            probs = torch.sigmoid(cls_out).cpu().numpy()[0]   # shape (N_AXIS,)
            labels_np = labels_batch.cpu().numpy()[0]          # shape (N_AXIS,)
            user_embedding_normal.append(user_embedding.cpu().numpy())
            user_word_count = fold_test_df.iloc[user_idx]['num_words_user']

            original_user_id = fold_test_df.iloc[user_idx]['user_id']
            
            preds = (probs > np.array(thresholds)).astype(int)
            user_acc = accuracy_score(labels_np, preds)
            
            user_results.append({
                'user_id': original_user_id,
                'num_words': user_word_count,
                'accuracy': user_acc,
                'probs': probs.tolist(),
                'thresholds': thresholds,
                'predictions': preds.tolist(),
                'labels': labels_np.tolist()
            })
            for trait_idx, trait in enumerate(axes):
                calibration_results.append({
                    'user_id': original_user_id,
                    'trait': trait,
                    'prob': float(probs[trait_idx]),
                    'label': float(labels_np[trait_idx])
                })
    
    print(f"Collected data for {len(fold_test_df)} users")
    del test_dataset, test_loader
    torch.cuda.empty_cache()
    user_embeddings_normal = np.vstack(user_embedding_normal)



    user_embedding_paraphrased = []
    with torch.no_grad():
        for user_idx, batch in enumerate(test_loader_paraphrased):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            bge_emb = batch['bge_embedding'].to(device)
            psych_batch = batch['psycholinguistic_feats'].to(device)
            
            cls_out, user_embedding = model(input_ids, attention_mask, bge_emb, psych_batch)
            probs = torch.sigmoid(cls_out).cpu().numpy()[0]   # shape (N_AXIS,)
            labels_np = labels_batch.cpu().numpy()[0]          # shape (N_AXIS,)
            user_embedding_paraphrased.append(user_embedding.cpu().numpy())
            
            user_word_count = fold_test_df.iloc[user_idx]['num_words_user']

            original_user_id = fold_test_df.iloc[user_idx]['user_id']
            
            preds = (probs > np.array(thresholds)).astype(int)
            user_acc = accuracy_score(labels_np, preds)
            
            user_results_paraphrased.append({
                'user_id': original_user_id,
                'num_words': user_word_count,
                'accuracy': user_acc,
                'predictions': preds.tolist(),
                'labels': labels_np.tolist()
            })
    
    print(f"Collected data for {len(fold_test_df)} users")
    del test_dataset_paraphrased, test_loader_paraphrased
    torch.cuda.empty_cache()
    user_embeddings_paraphrased = np.vstack(user_embedding_paraphrased)

    del model
    torch.cuda.empty_cache()
    

    for user_idx in range(len(user_embeddings_normal)):
        emb_normal = user_embeddings_normal[user_idx]
        emb_paraphrased = user_embeddings_paraphrased[user_idx]
        
        original_user_id = fold_test_df.iloc[user_idx]['user_id']
        
        correlation = np.corrcoef(emb_normal, emb_paraphrased)[0, 1]
        
        cosine_sim = np.dot(emb_normal, emb_paraphrased) / (np.linalg.norm(emb_normal) * np.linalg.norm(emb_paraphrased))
        
        user_embedding_correlations.append({
            'user_id': original_user_id,
            'pearson_correlation': correlation,
            'cosine_similarity': cosine_sim
        })
    
    print(f"Computed correlations for {len(user_embeddings_normal)} users")


user_results_df = pd.DataFrame(user_results)
user_results_paraphrased_df = pd.DataFrame(user_results_paraphrased)
calib_df = pd.DataFrame(calibration_results)
embedding_corr_df = pd.DataFrame(user_embedding_correlations)

embedding_corr_df.to_excel(os.path.join(paraphrased_dir, 'user_embedding_correlations.xlsx'), index=False)


corr_summary = pd.DataFrame({
    'Metric': ['Pearson Correlation', 'Cosine Similarity'],
    'Mean': [embedding_corr_df['pearson_correlation'].mean(), embedding_corr_df['cosine_similarity'].mean()],
    'Median': [embedding_corr_df['pearson_correlation'].median(), embedding_corr_df['cosine_similarity'].median()]
})
corr_summary.to_excel(os.path.join(paraphrased_dir, 'embedding_correlation_summary.xlsx'), index=False)

print(f"\nEmbedding correlation results saved")

print(f"\n{'='*60}")
print(f"DATA COLLECTION COMPLETE")
print(f"{'='*60}")
print(f"Total users analyzed: {len(user_results_df)}")
print(f"Total calibration records: {len(calib_df)}")
print(f"{'='*60}\n")


print(f"\n{'='*80}")
print("PER-USER ACCURACY ANALYSIS")
print(f"{'='*80}\n")

mw_metric = 'accuracy'
mw_metric_label = 'Accuracy'

user_results_df['word_bin'] = pd.cut(
    user_results_df['num_words'],
    bins=word_bins,
    include_lowest=True
)

user_results_df['word_bin_center'] = user_results_df['word_bin'].apply(
    lambda interval: interval.mid if pd.notna(interval) else np.nan
)

bin_stats = user_results_df.groupby('word_bin_center').agg({
    mw_metric: ['mean', 'std', 'count']
}).reset_index()
bin_stats.columns = ['word_bin_center', f'{mw_metric}_mean', f'{mw_metric}_std', f'{mw_metric}_count']

print(f"Average {mw_metric_label} by Word Count Bin:")
print(f"{'Bin Center':<12} {'N Users':<10} {f'Avg {mw_metric_label}':<20}")
print(f"{'-'*50}")
for _, row in bin_stats.iterrows():
    print(f"{row['word_bin_center']:<12.0f} {row[f'{mw_metric}_count']:<10.0f} {row[f'{mw_metric}_mean']:.4f} ± {row[f'{mw_metric}_std']:.4f}")


user_results_df.to_excel(os.path.join(test_a_dir, 'word_count_sweep_per_user_results.xlsx'), index=False)
bin_stats.to_excel(os.path.join(test_a_dir, 'word_count_sweep_bin_stats.xlsx'), index=False)


q25 = user_results_df['num_words'].quantile(0.25)
q75 = user_results_df['num_words'].quantile(0.75)

group_low = user_results_df[user_results_df['num_words'] <= q25]
group_high = user_results_df[user_results_df['num_words'] >= q75]

print(f"\n{'='*60}")
print(f"MANN-WHITNEY U TEST: Low vs High Word Count Users")
print(f"Metric: {mw_metric_label}")
print(f"{'='*60}\n")
print(f"Low group:  ≤{q25:.0f} words (bottom 25%, N={len(group_low)})")
print(f"High group: ≥{q75:.0f} words (top 25%, N={len(group_high)})")

mw_acc_stat, mw_acc_p = stats.mannwhitneyu(group_low[mw_metric], group_high[mw_metric], alternative='two-sided')
rank_biserial_acc = calculate_rank_biserial(group_low[mw_metric].values, group_high[mw_metric].values, mw_acc_stat)

sem_low_acc = group_low[mw_metric].sem()
sem_high_acc = group_high[mw_metric].sem()

print(f"\n{mw_metric_label}:")
print(f"  Low group:  {group_low[mw_metric].mean():.4f} ± {group_low[mw_metric].std():.4f} (SD)")
print(f"              {group_low[mw_metric].mean():.4f} ± {sem_low_acc:.4f} (SEM)")
print(f"  High group: {group_high[mw_metric].mean():.4f} ± {group_high[mw_metric].std():.4f} (SD)")
print(f"              {group_high[mw_metric].mean():.4f} ± {sem_high_acc:.4f} (SEM)")
print(f"  Difference: {abs(group_low[mw_metric].mean() - group_high[mw_metric].mean()):.4f}")
print(f"  Mann-Whitney U: {mw_acc_stat:.2f}, p-value: {mw_acc_p:.4f}")
print(f"  Rank-Biserial Correlation: {rank_biserial_acc:.4f}")

if mw_acc_p < 0.001:
    print(f"  HIGHLY SIGNIFICANT difference (p < 0.001)")
elif mw_acc_p < 0.01:
    print(f"  SIGNIFICANT difference (p < 0.01)")
elif mw_acc_p < 0.05:
    print(f"  Significant difference (p < 0.05)")
else:
    print(f"  NO significant difference (p ≥ 0.05)")

stats_results = pd.DataFrame({
    'Metric': [mw_metric_label],
    'Low_Group_N': [len(group_low)],
    'Low_Group_Mean': [group_low[mw_metric].mean()],
    'Low_Group_SD': [group_low[mw_metric].std()],
    'Low_Group_SEM': [group_low[mw_metric].sem()],
    'High_Group_N': [len(group_high)],
    'High_Group_Mean': [group_high[mw_metric].mean()],
    'High_Group_SD': [group_high[mw_metric].std()],
    'High_Group_SEM': [group_high[mw_metric].sem()],
    'Difference': [abs(group_low[mw_metric].mean() - group_high[mw_metric].mean())],
    'Mann_Whitney_U': [mw_acc_stat],
    'P_Value': [mw_acc_p],
    'Significant': [mw_acc_p < 0.05],
    'Rank_Biserial_r': [rank_biserial_acc],
    'Effect_Size': [interpret_rank_biserial(rank_biserial_acc)]
})
stats_results.to_excel(os.path.join(test_a_dir, 'statistical_tests.xlsx'), index=False)


print(f"\n{'='*60}")
print("Generating accuracy visualizations...")
print(f"{'='*60}\n")


print(" Creating bar plot with trend line...")
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

bin_centers = bin_stats['word_bin_center'].values
acc_means = bin_stats[f'{mw_metric}_mean'].values
acc_stds = bin_stats[f'{mw_metric}_std'].values

bars = ax.bar(bin_centers, acc_means, width=bin_size * 0.95, alpha=0.6, color='steelblue',
              edgecolor='black', linewidth=0.5, label=f'Mean {mw_metric_label} per bin', align='center')

ax.errorbar(bin_centers, acc_means, yerr=acc_stds, fmt='none',
           ecolor='black', capsize=3, alpha=0.5)

overall_mean = user_results_df[mw_metric].mean()
ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2,
          label=f'Overall mean {mw_metric_label}: {overall_mean:.4f}')
ax.set_xticks(bin_stats['word_bin_center'])
ax.set_xticklabels([f'{int(x)}' for x in bin_stats['word_bin_center']], rotation=45, ha='right')
ax.set_xlabel('Number of Words (bin centers)', fontsize=12)
ax.set_ylabel(f'Mean {mw_metric_label}', fontsize=12)
ax.set_title(f'{mw_metric_label} by Word Count', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(test_a_dir, 'word_count_bar_with_trend.png'), dpi=300, bbox_inches='tight')
plt.close()
print(" Saved: word_count_bar_with_trend.png")


print(f"\n{'='*80}")
print("CALIBRATION ANALYSIS - Overall Reliability Diagrams + Brier Scores")
print(f"{'='*80}\n")

calib_df.to_excel(os.path.join(calibration_dir, 'calibration_raw.xlsx'), index=False)


print(f"{'='*60}")
print(f"BRIER SCORES PER TRAIT (Overall - All Users)")
print(f"{'='*60}")

brier_records = []
for trait in axes:
    trait_df = calib_df[calib_df['trait'] == trait]
    brier = np.mean((trait_df['prob'].values - trait_df['label'].values) ** 2)
    brier_records.append({'Trait': trait, 'Brier_Score': brier})
    print(f"  {trait}: {brier:.4f}")

brier_df = pd.DataFrame(brier_records)
brier_df.to_excel(os.path.join(calibration_dir, 'brier_scores.xlsx'), index=False)


n_calibration_bins = 10
prob_bin_edges = np.linspace(0.0, 1.0, n_calibration_bins + 1)

print(f"\nGenerating overall reliability diagrams for {N_AXIS} traits...")

for trait in axes:
    trait_df = calib_df[calib_df['trait'] == trait].copy()
    brier_score = brier_df[brier_df['Trait'] == trait]['Brier_Score'].values[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    bin_centers_fixed = (prob_bin_edges[:-1] + prob_bin_edges[1:]) / 2
    
    bin_accs = []
    bin_ns = []
    
    for idx, (b_lo, b_hi) in enumerate(zip(prob_bin_edges[:-1], prob_bin_edges[1:])):
        
        if b_hi == 1.0:
            mask = (trait_df['prob'] >= b_lo) & (trait_df['prob'] <= b_hi)
        else:
            mask = (trait_df['prob'] >= b_lo) & (trait_df['prob'] < b_hi)
        
        n_samples = mask.sum()
        bin_ns.append(n_samples)
        
        if n_samples == 0:
            bin_accs.append(0)
        else:
            bin_accs.append(trait_df.loc[mask, 'label'].mean())

    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration', zorder=1)
    
    bin_width = 1.0 / n_calibration_bins
    
    for center, acc, n in zip(bin_centers_fixed, bin_accs, bin_ns):
        if n > 0:
            ax.bar(center, acc,
                   width=bin_width,
                   alpha=0.7,
                   color='steelblue',
                   edgecolor='black',
                   linewidth=1,
                   zorder=2)
            
            ax.text(center, acc + 0.03,
                    f'n={n}',
                    ha='center',
                    va='bottom',
                    fontsize=9)

    for center, n in zip(bin_centers_fixed, bin_ns):
        if n == 0:
            ax.text(center, 0.02,
                    'n=0',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='gray',
                    style='italic')

    from matplotlib.patches import Patch
    legend_elements = [
        ax.get_lines()[0],
        Patch(facecolor='steelblue', edgecolor='black', alpha=0.7, label='Observed frequency')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(prob_bin_edges)
    ax.set_xlabel('Predicted Probability Bin', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(f'Reliability Diagram: {trait}\nBrier Score: {brier_score:.4f}',
                 fontsize=14)
    ax.grid(True, alpha=0.3)

    textstr = f'Total samples: {len(trait_df)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=props)
    
    plt.tight_layout()
    save_path = os.path.join(calibration_dir, f'reliability_{trait}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: reliability_{trait}.png")

print("\n  Creating Brier scores summary...")
fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.Blues(np.linspace(0.3, 0.6, len(axes)))

bars = ax.bar(brier_df['Trait'], brier_df['Brier_Score'], 
             color=colors, edgecolor='steelblue', linewidth=1.2)

for bar, val in zip(bars, brier_df['Brier_Score']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11)

ax.set_xlabel('Trait', fontsize=13)
ax.set_ylabel('Brier Score', fontsize=13)
ax.set_title('Brier Score per Trait', fontsize=14)
ax.set_ylim(0, brier_df['Brier_Score'].max() * 1.15)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(calibration_dir, 'brier_scores_summary.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: brier_scores_summary.png")


print(f"\n{'='*80}")
print("PER-TRAIT STABILITY - Normal vs Paraphrased")
print(f"{'='*80}\n")

trait_results = []
for _, row in user_results_df.iterrows():
    for trait_idx, trait in enumerate(axes):
        trait_results.append({
            'user_id': row['user_id'],
            'trait': trait,
            'prediction_normal': row['predictions'][trait_idx],
            'label': row['labels'][trait_idx]
        })

trait_results_para = []
for _, row in user_results_paraphrased_df.iterrows():
    for trait_idx, trait in enumerate(axes):
        trait_results_para.append({
            'user_id': row['user_id'],
            'trait': trait,
            'prediction_paraphrased': row['predictions'][trait_idx]
        })

trait_df = pd.DataFrame(trait_results)
trait_df_para = pd.DataFrame(trait_results_para)

trait_merged = trait_df.merge(
    trait_df_para,
    on=['user_id', 'trait'],
    how='inner'
)

print(f"{'='*60}")
print("TRAIT-LEVEL AGREEMENT RATES")
print(f"{'='*60}\n")

agreement_results = []
for trait in axes:
    trait_data = trait_merged[trait_merged['trait'] == trait]
    agreement = (trait_data['prediction_normal'] == trait_data['prediction_paraphrased']).sum()
    total = len(trait_data)
    agreement_rate = agreement / total * 100
    agreement_results.append({'Trait': trait, 'Agreement_Rate': agreement_rate, 'N': total})
    print(f"  {trait}: {agreement}/{total} ({agreement_rate:.1f}% agreement)")

total_predictions = len(trait_merged)
total_agreement = (trait_merged['prediction_normal'] == trait_merged['prediction_paraphrased']).sum()
overall_agreement_rate = total_agreement / total_predictions * 100
print(f"\n  Overall (all traits): {total_agreement}/{total_predictions} ({overall_agreement_rate:.1f}% agreement)")

print(f"\n{'='*60}")
print("PER-USER TRAIT FLIP DISTRIBUTION")
print(f"{'='*60}\n")

user_flips = trait_merged.groupby('user_id').apply(
    lambda x: (x['prediction_normal'] != x['prediction_paraphrased']).sum()
).reset_index()
user_flips.columns = ['user_id', 'num_flips']

print("Distribution of trait flips per user:")
for num_flips in range(N_AXIS + 1):
    count = (user_flips['num_flips'] == num_flips).sum()
    pct = count / len(user_flips) * 100
    print(f"  {num_flips} traits flipped: {count} users ({pct:.1f}%)")

print(f"\nAverage traits flipped per user: {user_flips['num_flips'].mean():.2f}")
print(f"Median traits flipped per user: {user_flips['num_flips'].median():.1f}")

print(f"\n{'='*60}")
print("COHEN'S KAPPA PER TRAIT (Intra-rater reliability)")
print(f"{'='*60}\n")

from sklearn.metrics import cohen_kappa_score

kappa_results = []
for trait in axes:
    trait_data = trait_merged[trait_merged['trait'] == trait]
    kappa = cohen_kappa_score(trait_data['prediction_normal'], trait_data['prediction_paraphrased'])
    kappa_results.append({'Trait': trait, 'Cohen_Kappa': kappa})
    
    if kappa < 0:
        interpretation = "Poor (worse than chance)"
    elif kappa < 0.20:
        interpretation = "Slight agreement"
    elif kappa < 0.40:
        interpretation = "Fair agreement"
    elif kappa < 0.60:
        interpretation = "Moderate agreement"
    elif kappa < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"
    
    print(f"  {trait}: κ = {kappa:.3f} ({interpretation})")

kappa_df = pd.DataFrame(kappa_results)
avg_kappa = kappa_df['Cohen_Kappa'].mean()
print(f"\n  Average Cohen's Kappa: {avg_kappa:.3f}")

agreement_df = pd.DataFrame(agreement_results)
agreement_df.to_excel(os.path.join(paraphrased_dir, 'trait_agreement_rates.xlsx'), index=False)
kappa_df.to_excel(os.path.join(paraphrased_dir, 'cohen_kappa_per_trait.xlsx'), index=False)

user_flips_df = pd.DataFrame({
    'Num_Flips': range(N_AXIS + 1),
    'Count': [(user_flips['num_flips'] == i).sum() for i in range(N_AXIS + 1)],
    'Percentage': [(user_flips['num_flips'] == i).sum() / len(user_flips) * 100 for i in range(N_AXIS + 1)]
})
user_flips_df.to_excel(os.path.join(paraphrased_dir, 'trait_flips_distribution.xlsx'), index=False)

trait_merged.to_excel(os.path.join(paraphrased_dir, 'trait_level_predictions.xlsx'), index=False)

print(f"\nPer-trait stability results saved")


print(f"\n  Creating trait agreement rates bar plot...")
fig, ax = plt.subplots(figsize=(10, 6))
colors_agreement = plt.cm.Blues(np.linspace(0.3, 0.6, len(axes)))
bars = ax.bar(agreement_df['Trait'], agreement_df['Agreement_Rate'],
              color=colors_agreement, edgecolor='steelblue', linewidth=1.2)
for bar, val in zip(bars, agreement_df['Agreement_Rate']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
ax.axhline(y=overall_agreement_rate, color='red', linestyle='--', linewidth=1.8,
           label=f'Overall mean: {overall_agreement_rate:.1f}%')
ax.set_xlabel('Trait', fontsize=13)
ax.set_ylabel('Agreement Rate (%)', fontsize=13)
ax.set_title('Trait Prediction Agreement Rate\n(Normal vs Paraphrased)', fontsize=14)
ax.set_ylim(0, 115)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(paraphrased_dir, 'trait_agreement_rates.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: trait_agreement_rates.png")

print(f"Creating Cohen's Kappa bar plot...")
fig, ax = plt.subplots(figsize=(10, 6))
colors_kappa = plt.cm.Blues(np.linspace(0.3, 0.6, len(axes)))
bars = ax.bar(kappa_df['Trait'], kappa_df['Cohen_Kappa'],
              color=colors_kappa, edgecolor='steelblue', linewidth=1.2)
for bar, val in zip(bars, kappa_df['Cohen_Kappa']):
    y_pos = val + 0.01 if val >= 0 else val - 0.03
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11)
ax.axhline(y=avg_kappa, color='red', linestyle='--', linewidth=1.8,
           label=f'Average κ: {avg_kappa:.3f}')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Trait', fontsize=13)
ax.set_ylabel("Cohen's Kappa (κ)", fontsize=13)
ax.set_title("Cohen's Kappa per Trait\n(Normal vs Paraphrased)", fontsize=14)

kappa_max = kappa_df['Cohen_Kappa'].max()
ax.set_ylim(0, kappa_max * 1.2 + 0.05)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(paraphrased_dir, 'cohen_kappa_per_trait.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: cohen_kappa_per_trait.png")

print(f"Creating trait flips distribution bar plot...")
fig, ax = plt.subplots(figsize=(10, 6))
flip_counts = [(user_flips['num_flips'] == i).sum() for i in range(N_AXIS + 1)]
flip_pcts = [c / len(user_flips) * 100 for c in flip_counts]
x_positions = list(range(N_AXIS + 1))
colors_flips = plt.cm.Blues(np.linspace(0.25, 0.55, N_AXIS + 1))
bars = ax.bar(x_positions, flip_counts, color=colors_flips, edgecolor='steelblue', linewidth=1.2)
for bar, count, pct in zip(bars, flip_counts, flip_pcts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)
ax.set_xticks(x_positions)
ax.set_xticklabels([f'{i} trait{"s" if i != 1 else ""}' for i in x_positions], fontsize=11)
ax.set_xlabel('Number of Traits Flipped', fontsize=13)
ax.set_ylabel('Number of Users', fontsize=13)
ax.set_title('Distribution of Trait Flips per User\n(Normal vs Paraphrased)', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(paraphrased_dir, 'trait_flips_distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: trait_flips_distribution.png")