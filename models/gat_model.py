#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph Attention Network (GAT) model for process mining
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class NextTaskGAT(nn.Module):
    """
    Graph Attention Network for next task prediction
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
        for _ in range(num_layers-1):
            self.convs.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=True))
        self.fc = nn.Linear(hidden_dim*heads, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.nn.functional.elu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.fc(x)

def train_gat_model(model, train_loader, val_loader, criterion, optimizer, 
                   device, num_epochs=20, model_path="best_gnn_model.pth"):
    """
    Train the GAT model
    """
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        for batch_data in train_loader:
            out = model(batch_data.x.to(device),
                       batch_data.edge_index.to(device),
                       batch_data.batch.to(device))
            graph_labels = compute_graph_label(batch_data.y, batch_data.batch).to(device, dtype=torch.long)
            loss = criterion(out, graph_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                out = model(batch_data.x.to(device),
                          batch_data.edge_index.to(device),
                          batch_data.batch.to(device))
                glabels = compute_graph_label(batch_data.y, batch_data.batch).to(device, dtype=torch.long)
                val_loss += criterion(out, glabels).item()
        avg_val_loss = val_loss/len(val_loader)
        
        print(f"[Epoch {epoch}/{num_epochs}] train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  Saved best model (val_loss={best_val_loss:.4f})")
    
    return model

def compute_graph_label(y, batch):
    """Modal next_task across all events of each case → graph-level label.

    NOTE: this collapses a node-level signal (next_task differs per node)
    into a single graph label, which is a coarser objective than typical
    next-event prediction. A node-level head with a "predict at every
    prefix" loss would be more aligned with the literature; this helper
    is kept for backwards compatibility with the published configuration.
    """
    unique_batches = batch.unique()
    labels_out = []
    for bidx in unique_batches:
        mask = (batch == bidx)
        yvals_cpu = y[mask].detach().cpu()
        vals, counts = torch.unique(yvals_cpu, return_counts=True)
        labels_out.append(vals[torch.argmax(counts)])
    return torch.stack(labels_out)

def evaluate_gat_model(model, val_loader, device):
    """
    Evaluate GAT model and return predictions and probabilities
    """
    model.eval()
    y_true_all, y_pred_all, y_prob_all = [], [], []
    
    with torch.no_grad():
        for batch_data in val_loader:
            logits = model(batch_data.x.to(device),
                         batch_data.edge_index.to(device),
                         batch_data.batch.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            glabels = compute_graph_label(batch_data.y, batch_data.batch)
            
            for i in range(logits.size(0)):
                y_pred_all.append(int(torch.argmax(logits[i]).cpu()))
                y_prob_all.append(probs[i])
                y_true_all.append(int(glabels[i]))
    
    return (
        torch.tensor(y_true_all),
        torch.tensor(y_pred_all),
        torch.tensor(y_prob_all)
    ) 