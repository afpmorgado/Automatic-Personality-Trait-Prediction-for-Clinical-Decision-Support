"""Phase 1 Optimization Study
Master's Thesis: "Automatic Personality Trait Prediction for Clinical Decision Support"
Author: André Morgado
ID: ist199888
Instituto Superior Técnico, Universidade de Lisboa
24/05/2026

This script implements the phase 1 optimization study on the proposed model.
Version 1.0 - Lacks commenting (To implement).
"""

#%cd /kaggle/input/models/andrmorgado/main34/pytorch/main34/1
#Change to current directory

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import DistilBertModel, DistilBertTokenizer
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import random
import os
import datetime
import json
import re
import torch.nn.functional as F
import os
from sklearn.model_selection import StratifiedKFold, KFold
import seaborn as sns
torch.cuda.empty_cache()
from Code_Utils.utils import yn_to_binary, set_model_seed
from Code_Utils.Dataset import featureDataset_FM_Abl, MBTIDataset_FM_Abl, dataset_config, load_data_FM_Abl
from Code_Utils.Model import BERTClassifier_ablation, FocalLoss,evaluate 
from Code_Utils.Plots_Metrics import fold_plots, cross_validation_summary, plot_confusion_matrix


dataset_name = "Kaggle"  # Options: "Kaggle" or "Essays"

#Ablation Study Flags:
ablation_flags = {
    #Default Reference Model: all scenarios set to False
    "Scenario_1": False, #If true: Final Model w/o cosine scheduler, uses linear learning rate 
    "Scenario_2": False, #If true: Final Model w/o Focal Loss, uses BCE Loss 
    "Scenario_3": False, #If true: Normal Kfold instead of Stratified 
    "Scenario_4": False, #If true: Final Model w/o GTE fusion gate and non linear transformation 
    "Scenario_5": False, #If true: Final Model w/o GTE User embeddings 
    "Scenario_6": False, #If true: Final Model w/o Psycholinguistic Features 
    "Scenario_7": False, #If true: Final Model w/ scenarios 5,6 ablations combined 
    "Scenario_8": False, #If true: Final Model w/o Token average of distilbert, uses CLS instead. 
    "Scenario_9": False, #If true: Final Model w/ Dropout and L2 penalty in the Final Classifier 
    }

for scenario, active in ablation_flags.items():
    if active:
        print(f"Active Ablation Scenario: {scenario}")

active_scenarios = [scenario for scenario, active in ablation_flags.items() if active]
if len(active_scenarios) > 1:
    print("Warning: More than one ablation scenario is active: (may not be intended)")
    for scenario in active_scenarios:
        print(f"- {scenario}")

scenario_name = [k for k, v in ablation_flags.items() if v and k.startswith("Scenario_")]
if len(scenario_name) > 1:
    scenario_name = "Combined_Configuration"
elif len(scenario_name) == 1:
    scenario_name = scenario_name[0]
else:
    scenario_name = "Initial Configuration"


base_results_dir = os.path.join(f"/kaggle/working/Results/{dataset_name}", scenario_name) #change to intended output directory
os.makedirs(base_results_dir, exist_ok=True)

seed_value = 29
set_model_seed(seed_value)


MAX_SEQ_LEN = 64
MAX_CHUNKS = 35
BERT_NAME = "/kaggle/input/etmopt2/pytorch/etmopt2/1/ETMOPT/distilbert-base-uncased" #change to intended llm directory
batch_size = 4
max_epochs = 8
learning_rate = 3e-5
accum_steps = 2 
curr_epoch = 0


N_AXIS, axes, classes, label_cols, text_col, alphas = dataset_config(dataset_name)


print(f"\n{'='*60}")
print(f"Running phase 1 optimization study on dataset: {dataset_name}")
print(f"Number of axes: {N_AXIS}")
print(f"Axes: {axes}")
print(f"{'='*60}\n")


data, sentences, labels, gte_embeddings, psycholinguistic_feats = load_data_FM_Abl(dataset_name, "Light", classes, label_cols, text_col)

psycholinguistic_feats = torch.where(
    torch.isfinite(psycholinguistic_feats),
    psycholinguistic_feats,
    torch.zeros_like(psycholinguistic_feats)
)


if dataset_name == "Kaggle":
    stratify_col = data[label_cols]
elif dataset_name == "Essays":
    stratify_col = data[label_cols].apply(lambda row: ''.join(row.values), axis=1)

if ablation_flags["Scenario_3"]:
    train_indices, val_test_indices = train_test_split(
        np.arange(len(data)), 
        test_size=0.4, 
        random_state=seed_value
    )


    test_indices, val_indices = train_test_split(
        val_test_indices, 
        test_size=0.5, 
        random_state=seed_value
    )
else:
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

all_folds_test_metrics = []
all_folds_val_metrics = []
tokenizer = DistilBertTokenizer.from_pretrained(BERT_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = DistilBertModel.from_pretrained(BERT_NAME).to(device)
bert_model.eval()


scaler = torch.amp.GradScaler('cuda')
print("Starting training", flush=True)

for (train_idx, test_idx, val_idx) in [(train_indices, test_indices, val_indices)]:

    fold_train_df = data.iloc[train_indices].reset_index(drop=True)
    fold_test_df  = data.iloc[test_indices].reset_index(drop=True)
    fold_val_df   = data.iloc[val_indices].reset_index(drop=True)

    if dataset_name == "Kaggle":
        train_sentences = fold_train_df["posts"].tolist()
        y_train = np.array([[classes[c] for c in t] for t in fold_train_df["type"]], dtype="float32")
        
        val_sentences = fold_val_df["posts"].tolist()
        y_val = np.array([[classes[c] for c in t] for t in fold_val_df["type"]], dtype="float32")

        test_sentences = fold_test_df["posts"].tolist()
        y_test = np.array([[classes[c] for c in t] for t in fold_test_df["type"]], dtype="float32")
    elif dataset_name == "Essays":
        train_sentences = fold_train_df[text_col].tolist()
        y_train = np.array([[yn_to_binary(row[col]) for col in label_cols] for _, row in fold_train_df.iterrows()], dtype="float32")
        
        val_sentences = fold_val_df[text_col].tolist()
        y_val = np.array([[yn_to_binary(row[col]) for col in label_cols] for _, row in fold_val_df.iterrows()], dtype="float32")

        test_sentences = fold_test_df[text_col].tolist()
        y_test = np.array([[yn_to_binary(row[col]) for col in label_cols] for _, row in fold_test_df.iterrows()], dtype="float32")
    

    train_gte = gte_embeddings[train_indices]
    train_psycholinguistic_feats = psycholinguistic_feats[train_indices]
    
    val_gte = gte_embeddings[val_indices]
    val_psycholinguistic_feats = psycholinguistic_feats[val_indices]

    test_gte = gte_embeddings[test_indices]
    test_psycholinguistic_feats = psycholinguistic_feats[test_indices]
        
    train_dataset = featureDataset_FM_Abl(MBTIDataset_FM_Abl(train_sentences, y_train, tokenizer, MAX_SEQ_LEN), train_gte, train_psycholinguistic_feats)
    val_dataset = featureDataset_FM_Abl(MBTIDataset_FM_Abl(val_sentences, y_val, tokenizer, MAX_SEQ_LEN), val_gte, val_psycholinguistic_feats)
    test_dataset = featureDataset_FM_Abl(MBTIDataset_FM_Abl(test_sentences, y_test, tokenizer, MAX_SEQ_LEN), test_gte, test_psycholinguistic_feats)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    print(f"  Train: {len(train_dataset)} samples\n")
    print(f"  Val: {len(val_dataset)} samples\n")
    print(f"  Test: {len(test_dataset)} samples\n")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_model = DistilBertModel.from_pretrained(BERT_NAME)

    model = BERTClassifier_ablation(bert_model, N_AXIS, gte_embed_dim=768, psych_dim=psycholinguistic_feats.shape[1], ablation_flags=ablation_flags).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")

    model = model.to(device)
    if ablation_flags["Scenario_9"]:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4 )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


    if ablation_flags["Scenario_1"]:
        scheduler = optim.lr_scheduler.ConstantLR(optimizer)
    else:
        num_training_steps = len(train_loader) * max_epochs // accum_steps
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    if ablation_flags["Scenario_2"]:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = FocalLoss(alphas=alphas)

    fold_dir = os.path.join(base_results_dir, "Run Results")
    os.makedirs(fold_dir, exist_ok=True)

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    
    patience = 2
    min_epochs = 5
    patience_counter = 0
    
    metrics_xlsx_path = os.path.join(fold_dir, f"metrics.xlsx")
    writer = pd.ExcelWriter(metrics_xlsx_path, engine="openpyxl")
    best_model_path = os.path.join(fold_dir, f"best_model.pth")
    val_losses_per_epoch = []
    avg_val_f1s_per_epoch = []
    avg_val_accs_per_epoch = []
    train_losses_per_epoch = []
    for epoch in range(max_epochs):
        model.train()
        total_class_loss = 0
        optimizer.zero_grad()
        
        for idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            gte_emb = batch['gte_embedding'].to(device)
            psycholinguistic_feats_batch = batch['psycholinguistic_feats'].to(device)

            with torch.amp.autocast('cuda'):
                cls_out, bert_emb = model(input_ids, attention_mask, gte_emb, psycholinguistic_feats_batch)
                cls_loss = criterion(cls_out, labels)

                loss = cls_loss

                scaler.scale(loss / accum_steps).backward()

            total_class_loss += cls_loss.item()

            if (idx + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        avg_class_loss = total_class_loss / len(train_loader)

        val_loss, val_accs, val_f1s, _, _, current_thresholds = evaluate("ablation", model, val_loader, N_AXIS, device, criterion)
        avg_f1_val = np.mean(val_f1s)
        avg_acc_val = np.mean(val_accs)
        
        val_losses_per_epoch.append(val_loss)
        avg_val_f1s_per_epoch.append(avg_f1_val)
        avg_val_accs_per_epoch.append(avg_acc_val)
        train_losses_per_epoch.append(avg_class_loss)
        
        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_class_loss:.4f}", flush=True)
        print(f"Val Loss: {val_loss:.4f} | Avg Val Acc: {avg_acc_val:.4f} | Avg Val Macro-F1: {avg_f1_val:.4f}", flush=True)
        for i, axis in enumerate(axes):
            print(f"{axis} | Acc: {val_accs[i]:.4f} | F1: {val_f1s[i]:.4f}", flush=True)

        per_trait_df = pd.DataFrame({
            "Epoch": epoch + 1,
            "Trait": axes,
            "Val_Acc": val_accs,
            "Val_F1": val_f1s
        })
        
        sheet_name = "per_trait_metrics"
        if sheet_name not in writer.sheets:
            per_trait_df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            startrow = writer.sheets[sheet_name].max_row
            per_trait_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                header=False,
                startrow=startrow
            )
        
        avg_metrics_df = pd.DataFrame({
            "Epoch": [epoch + 1],
            "Avg_Val_Acc": [avg_acc_val],
            "Avg_Val_F1": [avg_f1_val]
        })
        
        sheet_name = "avg_metrics"
        if epoch == 0:
            avg_metrics_df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            startrow = writer.sheets[sheet_name].max_row
            avg_metrics_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                header=False,
                startrow=startrow
            )
        
        if dataset_name == "Kaggle":
            is_best = avg_f1_val > best_val_f1
            if is_best:
                best_val_f1 = avg_f1_val
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
            
                print(
                    f"Saved best model at epoch {best_epoch} "
                    f"with Avg Macro-F1 {best_val_f1:.4f}"
                )
        elif dataset_name == "Essays":
            is_best = avg_acc_val > best_val_acc
            
            if is_best:
                best_val_acc = avg_acc_val
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_model_path)
            
                print(
                    f"Saved best model at epoch {best_epoch} "
                    f"with Avg Acc {best_val_acc:.4f}"
                )
        
        if epoch + 1 >= min_epochs:
            if is_best:
                patience_counter = 0
            else:
                patience_counter += 1
        
            if patience_counter >= patience:
                if dataset_name == "Kaggle":
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(best epoch: {best_epoch}, best macro-F1: {best_val_f1:.4f})"
                    )
                elif dataset_name == "Essays":
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(best epoch: {best_epoch}, best Acc: {best_val_acc:.4f})"
                    )
                break

    fold_idx = 0
    writer.close()
    actual_epochs = len(train_losses_per_epoch)
    if dataset_name == "Kaggle":
        fold_plots(dataset_name,actual_epochs, train_losses_per_epoch, val_losses_per_epoch, avg_val_f1s_per_epoch, fold_idx, fold_dir,False)
    else:
        fold_plots(dataset_name,actual_epochs, train_losses_per_epoch, val_losses_per_epoch, avg_val_accs_per_epoch, fold_idx, fold_dir,False)
    
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(best_model_path))

    val_loss, val_accs, val_f1s, val_preds, val_labels, current_thresholds = evaluate("ablation", model, val_loader, N_AXIS, device, criterion)
    avg_f1_val = np.mean(val_f1s)
    avg_acc_val = np.mean(val_accs)
    
    print(f"\nFinal Val Results (Best Model from Epoch {best_epoch}):")
    print(f"Val Loss: {val_loss:.4f} | Avg Val Acc: {avg_acc_val:.4f} | Avg Val Macro-F1: {avg_f1_val:.4f}")
    for i, axis in enumerate(axes):
        print(f"{axis} | Val Acc: {val_accs[i]:.4f} | Val F1: {val_f1s[i]:.4f}")
    
    Val_per_trait_path = os.path.join(fold_dir, f'final_Val_per_trait_metrics.xlsx')
    Val_per_trait_df = pd.DataFrame({
        "Trait": axes,
        "Val_Acc": val_accs,
        "Val_F1": val_f1s
    })
    Val_per_trait_df.to_excel(Val_per_trait_path, index=False)
    print(f"Saved final val per-trait metrics to {Val_per_trait_path}")
    
    val_avg_path = os.path.join(fold_dir, f'final_val_avg_metrics.xlsx')
    val_avg_df = pd.DataFrame({
        "Best_Epoch": [best_epoch],
        "Avg_Val_Acc": [avg_acc_val],
        "Avg_Val_F1": [avg_f1_val]
    })
    val_avg_df.to_excel(val_avg_path, index=False)
    print(f"Saved final val average metrics to {val_avg_path}")
    
    final_cm_dir = os.path.join(fold_dir, 'final_val_confusion_matrices')
    os.makedirs(final_cm_dir, exist_ok=True)
    
    for i, axis in enumerate(axes):
        cm = confusion_matrix(
            np.array(val_labels[i]),
            (np.array(val_preds[i]) > 0.5).astype(int)
        )
        
        png_path = os.path.join(final_cm_dir, f'Val_CM_{axis}_counts.png')
        plot_confusion_matrix(cm, axis, png_path, 
                            title_suffix=f'Best Epoch {best_epoch}',
                            normalize=False)
        
        png_path_norm = os.path.join(final_cm_dir, f'Val_CM_{axis}_normalized.png')
        plot_confusion_matrix(cm, axis, png_path_norm, 
                            title_suffix=f'Best Epoch {best_epoch}',
                            normalize=True)
    
    all_folds_val_metrics.append({
        'best_epoch': best_epoch,
        'val_loss': val_loss,
        'val_avg_acc': avg_acc_val,
        'val_avg_f1': avg_f1_val,
        'val_accs': val_accs,
        'val_f1s': val_f1s,
        'val_preds': val_preds,
        'val_labels': val_labels
    })

    test_loss, test_accs, test_f1s, test_preds, test_labels, current_thresholds = evaluate("ablation", model, test_loader, N_AXIS, device, criterion)
    avg_f1_test = np.mean(test_f1s)
    avg_acc_test = np.mean(test_accs)
    
    print(f"\nFinal Test Results (Best Model from Epoch {best_epoch}):")
    print(f"Test Loss: {test_loss:.4f} | Avg Acc: {avg_acc_test:.4f} | Avg Test Macro-F1: {avg_f1_test:.4f}")
    for i, axis in enumerate(axes):
        print(f"{axis} | Test Acc: {test_accs[i]:.4f} | Test F1: {test_f1s[i]:.4f}")
    
    test_per_trait_path = os.path.join(fold_dir, f'final_test_per_trait_metrics.xlsx')
    test_per_trait_df = pd.DataFrame({
        "Trait": axes,
        "Test_Acc": test_accs,
        "Test_F1": test_f1s
    })
    test_per_trait_df.to_excel(test_per_trait_path, index=False)
    print(f"Saved final test per-trait metrics to {test_per_trait_path}")
    
    test_avg_path = os.path.join(fold_dir, f'final_test_avg_metrics.xlsx')
    test_avg_df = pd.DataFrame({
        "Best_Epoch": [best_epoch],
        "Avg_Test_Acc": [avg_acc_test],
        "Avg_Test_F1": [avg_f1_test]
    })
    test_avg_df.to_excel(test_avg_path, index=False)
    print(f"Saved final test average metrics to {test_avg_path}")
    
    final_cm_dir = os.path.join(fold_dir, 'final_confusion_matrices')
    os.makedirs(final_cm_dir, exist_ok=True)
    
    for i, axis in enumerate(axes):
        cm = confusion_matrix(
            np.array(test_labels[i]),
            (np.array(test_preds[i]) > 0.5).astype(int)
        )
        
        png_path = os.path.join(final_cm_dir, f'CM_{axis}_counts.png')
        plot_confusion_matrix(cm, axis, png_path, 
                            title_suffix=f'Best Epoch {best_epoch}',
                            normalize=False)
        
        png_path_norm = os.path.join(final_cm_dir, f'CM_{axis}_normalized.png')
        plot_confusion_matrix(cm, axis, png_path_norm, 
                            title_suffix=f'Best Epoch {best_epoch}',
                            normalize=True)
    
    all_folds_test_metrics.append({
        'best_epoch': best_epoch,
        'test_loss': test_loss,
        'test_avg_acc': avg_acc_test,
        'test_avg_f1': avg_f1_test,
        'test_accs': test_accs,
        'test_f1s': test_f1s,
        'test_preds': test_preds,
        'test_labels': test_labels
    })
    del model
    del train_dataset
    del test_dataset
    del val_dataset
    del train_loader
    del test_loader
    del val_loader
    torch.cuda.empty_cache()
    torch.cuda.synchronize()