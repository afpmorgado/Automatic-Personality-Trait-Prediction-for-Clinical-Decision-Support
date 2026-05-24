
"""Model classes and evaluation functions for the experiments in the thesis.
Master's Thesis: "Automatic Personality Trait Prediction for Clinical Decision Support"
Author: André Morgado
ID: ist199888
Instituto Superior Técnico, Universidade de Lisboa
24/05/2026

Version 1.0 - Lacks commenting (To implement).
"""


import torch
import torch.nn as nn
import numpy as np
from .utils import calculate_metrics


class FocalLoss(nn.Module):
    def __init__(self, alphas, gamma=2.0):
        super().__init__()
        self.alphas = alphas
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        F_loss = torch.zeros_like(BCE)
        for i in range(len(self.alphas)):
            alpha_factor = self.alphas[i]*targets[:,i] + (1-self.alphas[i])*(1-targets[:,i])
            F_loss[:,i] = alpha_factor * (1-pt[:,i])**self.gamma * BCE[:,i]
        return torch.mean(F_loss)


class BERTClassifier_baseline(nn.Module):
    def __init__(self, bert_model, n_classes, psycho_feat_dim):
        super().__init__()
        
        self.bert = bert_model
        self.out = nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]

        cls_output = self.out(pooled_output)

        return cls_output

class BERTClassifier_FM_Kaggle(nn.Module):
    def __init__(self, bert_model, n_classes, gte_embed_dim, psych_dim, chunk_pooling):
        super().__init__()
        self.bert = bert_model
        self.chunk_pooling = chunk_pooling
        self.gte_fc = nn.Linear(gte_embed_dim, 768)
        self.relu = nn.ReLU()
        self.out_1 = nn.Linear(768, n_classes)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.chunk_scorer = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.gate_fc = nn.Linear(768 * 2, 768)
        self.user_query = nn.Parameter(torch.randn(1, 1, 768))

        self.chunk_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
    def forward(self, input_ids, attention_mask, gte_embeddings, psycholinguistic_feats):
        if input_ids.dim() == 3:
            batch_size, num_chunks, seq_len = input_ids.size()
            input_ids = input_ids.view(-1, seq_len)
            attention_mask = attention_mask.view(-1, seq_len)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # [B*C, S, H]
        cls_id = self.bert.config.bos_token_id
        sep_id = self.bert.config.eos_token_id
        token_mask = (attention_mask == 1) & (input_ids != cls_id) & (input_ids != sep_id)  # [B*C, S]
        token_mask = token_mask.unsqueeze(-1).float()  # [B*C, S, 1]
        masked_hidden = hidden_states * token_mask
        token_counts = token_mask.sum(dim=1).clamp(min=1.0)
        chunk_embeddings = masked_hidden.sum(dim=1) / token_counts  # [B*C, H]
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, -1)  # [B, C, H]

        if self.chunk_pooling == "simple_mean":
            user_embedding = chunk_embeddings.mean(dim=1)
        elif self.chunk_pooling == "attention_pooling":
            chunk_mask = (attention_mask.view(batch_size, num_chunks, -1).sum(dim=2) > 0)
            query = self.user_query.expand(batch_size, -1, -1)

            attn_output, attn_weights = self.chunk_attention(
                query,
                chunk_embeddings,
                chunk_embeddings,
                key_padding_mask=~chunk_mask
            )
            user_embedding = attn_output.squeeze(1)
        elif self.chunk_pooling == "chunk_scorer":
            scores = self.chunk_scorer(chunk_embeddings)  # [B, C, 1]
            chunk_mask = (attention_mask.view(batch_size, num_chunks, -1).sum(dim=2) > 0)
            scores = scores.masked_fill(
                ~chunk_mask.unsqueeze(-1),
                torch.finfo(scores.dtype).min
            )
            weights = torch.softmax(scores.float(), dim=1).type_as(chunk_embeddings)  # [B, C, 1]
            user_embedding = torch.sum(weights * chunk_embeddings, dim=1)  # [B, H]

        cls_output = self.out_1(user_embedding)
        
        return cls_output, user_embedding
    

class BERTClassifier_FM_Essays(nn.Module):
    def __init__(self, bert_model, n_classes, gte_embed_dim, psych_dim, chunk_pooling):
        super().__init__()
        self.bert = bert_model
        self.chunk_pooling = chunk_pooling
        self.gte_fc = nn.Linear(gte_embed_dim, 768)
        self.relu = nn.ReLU()
        self.out_1 = nn.Linear(768, n_classes)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.chunk_scorer = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.gate_fc = nn.Linear(768 * 2, 768)
        self.user_query = nn.Parameter(torch.randn(1, 1, 768))

        self.chunk_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
    def forward(self, input_ids, attention_mask, gte_embeddings, psycholinguistic_feats):
        if input_ids.dim() == 3:
            batch_size, num_chunks, seq_len = input_ids.size()
            input_ids = input_ids.view(-1, seq_len)
            attention_mask = attention_mask.view(-1, seq_len)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        chunk_embeddings = outputs.last_hidden_state[:, 0].view(batch_size, num_chunks, -1)  # [B, C, H]

        if self.chunk_pooling == "simple_mean":
            user_embedding = chunk_embeddings.mean(dim=1)
        elif self.chunk_pooling == "attention_pooling":
            chunk_mask = (attention_mask.view(batch_size, num_chunks, -1).sum(dim=2) > 0)
            query = self.user_query.expand(batch_size, -1, -1)

            attn_output, attn_weights = self.chunk_attention(
                query,
                chunk_embeddings,
                chunk_embeddings,
                key_padding_mask=~chunk_mask
            )
            user_embedding = attn_output.squeeze(1)
        elif self.chunk_pooling == "chunk_scorer":
            scores = self.chunk_scorer(chunk_embeddings)  # [B, C, 1]
            chunk_mask = (attention_mask.view(batch_size, num_chunks, -1).sum(dim=2) > 0)
            scores = scores.masked_fill(
                ~chunk_mask.unsqueeze(-1),
                torch.finfo(scores.dtype).min
            )
            weights = torch.softmax(scores.float(), dim=1).type_as(chunk_embeddings)  # [B, C, 1]
            user_embedding = torch.sum(weights * chunk_embeddings, dim=1)  # [B, H]


        gte_transformed = self.relu(self.gte_fc(gte_embeddings))
        gate = torch.sigmoid(
            self.gate_fc(torch.cat([user_embedding, gte_transformed], dim=1))
        )

        final_user_emb = gate * user_embedding + (1 - gate) * gte_transformed

        cls_output = self.out_1(final_user_emb)
        
        return cls_output, user_embedding
    




class BERTClassifier_ablation(nn.Module):
    def __init__(self, bert_model, n_classes, gte_embed_dim, psych_dim, ablation_flags):
        super().__init__()
        self.ablation_flags = ablation_flags
        self.bert = bert_model
        self.gte_fc = nn.Linear(gte_embed_dim, 768)
        self.relu = nn.ReLU()
        self.out_1 = nn.Linear(891, n_classes) if not (self.ablation_flags["Scenario_6"] or self.ablation_flags["Scenario_7"]) else nn.Linear(768, n_classes)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.chunk_scorer = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.gate_fc = nn.Linear(768 * 2, 768)
        self.user_query = nn.Parameter(torch.randn(1, 1, 768))

        self.chunk_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
    def forward(self, input_ids, attention_mask, gte_embeddings, psycholinguistic_feats):
        if input_ids.dim() == 3:
            batch_size, num_chunks, seq_len = input_ids.size()
            input_ids = input_ids.view(-1, seq_len)
            attention_mask = attention_mask.view(-1, seq_len)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # [B*C, S, H]
        if self.ablation_flags["Scenario_8"]: #CLS
            chunk_embeddings = outputs.last_hidden_state[:, 0].view(batch_size, num_chunks, -1)  # [B, C, H]
        else:
            cls_id = self.bert.config.bos_token_id   # treat <s> as CLS
            sep_id = self.bert.config.eos_token_id   # treat </s> as SEP
            token_mask = (attention_mask == 1) & (input_ids != cls_id) & (input_ids != sep_id)  # [B*C, S]
            token_mask = token_mask.unsqueeze(-1).float()  # [B*C, S, 1]
            masked_hidden = hidden_states * token_mask
            token_counts = token_mask.sum(dim=1).clamp(min=1.0)
            chunk_embeddings = masked_hidden.sum(dim=1) / token_counts  # [B*C, H]
            chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, -1)  # [B, C, H]
        
        user_embedding = chunk_embeddings.mean(dim=1)

        if self.ablation_flags["Scenario_5"] or self.ablation_flags["Scenario_7"]:
            final_user_emb = user_embedding
        else:
            if self.ablation_flags["Scenario_4"]:
                gte_transformed = self.relu(self.gte_fc(gte_embeddings))
                final_user_emb = user_embedding + gte_transformed  # [B, 768]
            else:
                gte_transformed = self.relu(self.gte_fc(gte_embeddings))
                gate = torch.sigmoid(
                    self.gate_fc(torch.cat([user_embedding, gte_transformed], dim=1))
                )

                final_user_emb = gate * user_embedding + (1 - gate) * gte_transformed
        
        if self.ablation_flags["Scenario_6"] or self.ablation_flags["Scenario_7"]:
            final_representation = final_user_emb
        else:
            final_representation = torch.cat([final_user_emb, psycholinguistic_feats], dim=1)  # [B, 768 + P]
        if self.ablation_flags["Scenario_9"]:
            cls_output = self.dropout(self.out_1(final_representation))
        else:
            cls_output = self.out_1(final_representation)
        
        return cls_output, user_embedding


def evaluate_with_threshold_sweep(dataset_name,model,dataloader,n_axis, device, criterion,threshold_range=np.arange(0.1, 1.01, 0.05)):
    model.eval()
    total_loss = 0

    all_probs = [[] for _ in range(n_axis)]
    all_labels = [[] for _ in range(n_axis)]

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            gte_emb = batch['gte_embedding'].to(device)
            psych_feats = batch['psycholinguistic_feats'].to(device)

            logits, _ = model(
                input_ids,
                attention_mask,
                gte_emb,
                psych_feats
            )

            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(n_axis):
                all_probs[i].extend(probs[:, i])
                all_labels[i].extend(labels_np[:, i])

    best_thresholds = []
    best_f1s = []
    best_accs = []

    for i in range(n_axis):
        probs_i = np.array(all_probs[i])
        labels_i = np.array(all_labels[i])

        best_f1 = -1
        best_acc = -1
        best_thr = 0.5

        for thr in threshold_range:
            acc, f1 = calculate_metrics(
                probs_i,
                labels_i,
                threshold=thr
            )
            if dataset_name == "Essays":
                if acc > best_acc:
                    best_f1 = f1
                    best_acc = acc
                    best_thr = thr
            if dataset_name == "Kaggle":
                if f1 > best_f1:
                    best_f1 = f1
                    best_acc = acc
                    best_thr = thr

        best_thresholds.append(best_thr)
        best_f1s.append(best_f1)
        best_accs.append(best_acc)

    return (
        total_loss / len(dataloader),
        best_accs,
        best_f1s,
        best_thresholds
    )



def evaluate(file, model, dataloader, n_axis, device, criterion, thresholds=None):
    if thresholds is None:
        thresholds = [0.5] * n_axis
    elif isinstance(thresholds, (int, float)):
        thresholds = [thresholds] * n_axis
    
    model.eval()
    total_loss = 0
    all_preds = [[] for _ in range(n_axis)]
    all_labels = [[] for _ in range(n_axis)]
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            if file != "baseline":
                gte_emb = batch['gte_embedding'].to(device)
            psycholinguistic_feats_batch = batch['psycholinguistic_feats'].to(device)
            
            if file == "baseline":
                cls_out = model(input_ids, attention_mask)
            elif file == "FM" or file == "ablation":
                cls_out, _ = model(input_ids, attention_mask, gte_emb, psycholinguistic_feats_batch)
            else:
                raise ValueError(f"Unknown file type: {file}")
            loss = criterion(cls_out, labels)
            total_loss += loss.item()
            
            preds = torch.sigmoid(cls_out).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for i in range(n_axis):
                all_preds[i].extend(preds[:, i])
                all_labels[i].extend(labels_np[:, i])
    
    accs, f1s = [], []
    for i in range(n_axis):
        acc, f1 = calculate_metrics(
            np.array(all_preds[i]), 
            np.array(all_labels[i]),
            threshold=thresholds[i]
        )
        accs.append(acc)
        f1s.append(f1)
    
    return total_loss / len(dataloader), accs, f1s, all_preds, all_labels, thresholds