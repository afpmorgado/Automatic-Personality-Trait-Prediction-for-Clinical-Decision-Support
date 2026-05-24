"""Baseline model
Master's Thesis: "Automatic Personality Trait Prediction for Clinical Decision Support"
Author: André Morgado
ID: ist199888
Instituto Superior Técnico, Universidade de Lisboa
24/05/2026

This script implements Baseline model.
Version 1.0 - Lacks commenting (To implement).
"""

#%cd /kaggle/input/models/andrmorgado/main34/pytorch/main34/1
#Change to current directory

#Note: Psycholinguistic is imported but not used to classify personality in the baseline model.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import torch.nn.functional as F
from sklearn.model_selection import KFold
from Code_Utils.utils import yn_to_binary, set_model_seed
from Code_Utils.Dataset import featureDataset_baseline, MBTIDataset_baseline, dataset_config, load_data_baseline
from Code_Utils.Model import BERTClassifier_baseline, evaluate
from Code_Utils.Plots_Metrics import fold_plots, cross_validation_summary, plot_confusion_matrix
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()

dataset_name = "Kaggle"  # Options: "Kaggle" or "Essays"

base_results_dir = os.path.join(f"/kaggle/working/Results/{dataset_name}", f"baseline_Bert_Concat_{dataset_name}") #change to intended results directory

os.makedirs(base_results_dir, exist_ok=True)


seed_value = 29
set_model_seed(seed_value)


MAX_SEQ_LEN = 512
BERT_NAME = "/kaggle/input/etmopt2/pytorch/etmopt2/1/ETMOPT/distilbert-base-uncased"
batch_size = 4
max_epochs = 8
learning_rate = 3e-5
accum_steps = 2 
curr_epoch = 0


N_AXIS, axes, classes, label_cols, text_col, alphas = dataset_config(dataset_name)

print(f"\n{'='*60}")
print(f"Running baseline on dataset: {dataset_name}")
print(f"Number of axes: {N_AXIS}")
print(f"Axes: {axes}")


PSYCHO_FEAT_DIM = 123
data, sentences, labels, psycholinguistic_feats = load_data_baseline(dataset_name, classes, label_cols, text_col)


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

all_folds_test_metrics = []
all_folds_val_metrics = []
tokenizer = DistilBertTokenizer.from_pretrained(BERT_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_model = DistilBertModel.from_pretrained(BERT_NAME).to(device)
bert_model.eval()

scaler = torch.amp.GradScaler('cuda')
print("Starting training", flush=True)

run_dir = base_results_dir
os.makedirs(run_dir, exist_ok=True)


num_workers = 0
for (train_idx, test_idx, val_idx) in [(train_indices, test_indices, val_indices)]:

    fold_train_df = data.iloc[train_indices].reset_index(drop=True)
    fold_val_df = data.iloc[val_indices].reset_index(drop=True)
    fold_test_df  = data.iloc[test_indices].reset_index(drop=True)
    
    if dataset_name == "Kaggle":
        train_sentences = fold_train_df["posts"].tolist()
        y_train = np.array([[classes[c] for c in t] for t in fold_train_df["type"]], dtype="float32")
        
        test_sentences = fold_test_df["posts"].tolist()
        y_test = np.array([[classes[c] for c in t] for t in fold_test_df["type"]], dtype="float32")

        val_sentences = fold_val_df["posts"].tolist()
        y_val = np.array([[classes[c] for c in t] for t in fold_val_df["type"]], dtype="float32")
    elif dataset_name == "Essays":
        train_sentences = fold_train_df[text_col].tolist()
        y_train = np.array([[yn_to_binary(row[col]) for col in label_cols] for _, row in fold_train_df.iterrows()], dtype="float32")
        
        test_sentences = fold_test_df[text_col].tolist()
        y_test = np.array([[yn_to_binary(row[col]) for col in label_cols] for _, row in fold_test_df.iterrows()], dtype="float32")
    
        val_sentences = fold_val_df[text_col].tolist()
        y_val = np.array([[yn_to_binary(row[col]) for col in label_cols] for _, row in fold_val_df.iterrows()], dtype="float32")

    train_psycholinguistic_feats = psycholinguistic_feats[train_indices]
    
    test_psycholinguistic_feats = psycholinguistic_feats[test_indices]

    val_psycholinguistic_feats = psycholinguistic_feats[val_indices]

    train_dataset = featureDataset_baseline(MBTIDataset_baseline(train_sentences, y_train, tokenizer, MAX_SEQ_LEN), train_psycholinguistic_feats)
    test_dataset = featureDataset_baseline(MBTIDataset_baseline(test_sentences, y_test, tokenizer, MAX_SEQ_LEN), test_psycholinguistic_feats)
    val_dataset = featureDataset_baseline(MBTIDataset_baseline(val_sentences, y_val, tokenizer, MAX_SEQ_LEN), val_psycholinguistic_feats)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples\n")
    print(f"  Validation: {len(val_dataset)} samples\n")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_model = DistilBertModel.from_pretrained(BERT_NAME)


    bert_model_copy = DistilBertModel.from_pretrained(BERT_NAME)
    model = BERTClassifier_baseline(
        bert_model_copy, 
        N_AXIS, 
        psycho_feat_dim=PSYCHO_FEAT_DIM,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)



    criterion = nn.BCEWithLogitsLoss()

    fold_dir = os.path.join(base_results_dir, "Run Results")
    os.makedirs(fold_dir, exist_ok=True)

    best_val_f1 = 0.0
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
            psycholinguistic_feats_batch = batch['psycholinguistic_feats'].to(device)

            with torch.amp.autocast('cuda'):
                cls_out = model(input_ids, attention_mask)
                cls_loss = criterion(cls_out, labels)

                loss = cls_loss

                scaler.scale(loss / accum_steps).backward()

            total_class_loss += cls_loss.item()

            if (idx + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_class_loss = total_class_loss / len(train_loader)

        val_loss, val_accs, val_f1s, _, _, current_thresholds = evaluate("baseline", model, val_loader, N_AXIS, device, criterion)
        avg_f1_val = np.mean(val_f1s)
        avg_acc_val = np.mean(val_accs)
        
        val_losses_per_epoch.append(val_loss)
        avg_val_f1s_per_epoch.append(avg_f1_val)
        avg_val_accs_per_epoch.append(avg_acc_val)
        train_losses_per_epoch.append(avg_class_loss)
        
        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {avg_class_loss:.4f}", flush=True)
        print(f"Val Loss: {val_loss:.4f} | Avg Val Acc: {avg_acc_val:.4f} | Avg Val F1: {avg_f1_val:.4f}", flush=True)
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
        
        is_best = avg_f1_val > best_val_f1
        
        if is_best:
            best_val_f1 = avg_f1_val
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
        
            print(
                f"Saved best model at epoch {best_epoch} "
                f"with Avg Macro-F1 {best_val_f1:.4f}"
            )
        
        if epoch + 1 >= min_epochs:
            if is_best:
                patience_counter = 0
            else:
                patience_counter += 1
        
            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best epoch: {best_epoch}, best macro-F1: {best_val_f1:.4f})"
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

    val_loss, val_accs, val_f1s, val_preds, val_labels, current_thresholds = evaluate("baseline", model, val_loader, N_AXIS, device, criterion)
    avg_f1_val = np.mean(val_f1s)
    avg_acc_val = np.mean(val_accs)
    
    print(f"\nFinal Val Results (Best Model from Epoch {best_epoch}):")
    print(f"Val Loss: {val_loss:.4f} | Avg Val Acc: {avg_acc_val:.4f} | Avg Val F1: {avg_f1_val:.4f}")
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

    test_loss, test_accs, test_f1s, test_preds, test_labels, current_thresholds = evaluate("baseline", model, test_loader, N_AXIS, device, criterion)
    avg_f1_test = np.mean(test_f1s)
    avg_acc_test = np.mean(test_accs)
    
    print(f"\nFinal Test Results (Best Model from Epoch {best_epoch}):")
    print(f"Test Loss: {test_loss:.4f} | Avg Acc: {avg_acc_test:.4f} | Avg F1: {avg_f1_test:.4f}")
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
