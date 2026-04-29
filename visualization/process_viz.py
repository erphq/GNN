#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization module for process mining analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from sklearn.manifold import TSNE

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_embeddings(embeddings, method="tsne", save_path=None):
    """Plot task embeddings using t-SNE or UMAP"""
    if method == "tsne":
        tsne_perp = min(30, embeddings.shape[0]-1)
        coords = TSNE(n_components=2, perplexity=tsne_perp, random_state=42).fit_transform(embeddings)
        title = "Task Embeddings - t-SNE"
    else:  # umap
        import umap  # optional dep — only imported when this branch runs
        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
        title = "Task Embeddings - UMAP"
    
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=coords[:,0], y=coords[:,1])
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_cycle_time_distribution(durations, save_path="cycle_time_distribution.png"):
    """Plot cycle time distribution"""
    plt.figure(figsize=(6,4))
    plt.hist(durations, bins=30, color="skyblue", edgecolor="black")
    plt.title("Cycle Time Distribution (hours)")
    plt.xlabel("Hours")
    plt.ylabel("Number of Cases")
    mean_c = np.mean(durations)
    plt.axvline(mean_c, color="red", linestyle="--", label=f"Mean={mean_c:.1f}h")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_process_flow(bottleneck_stats, le_task, top_bottlenecks, 
                     save_path="process_flow_bottlenecks.png"):
    """Plot process flow with bottlenecks highlighted"""
    G_flow = nx.DiGraph()
    for i, row in bottleneck_stats.iterrows():
        src = int(row["task_id"])
        dst = int(row["next_task_id"])
        G_flow.add_edge(src, dst, freq=int(row["count"]), mean_hours=row["mean_hours"])
    
    btop_edges = set((int(src), int(dst)) for src, dst in zip(
        top_bottlenecks["task_id"], top_bottlenecks["next_task_id"]
    ))
    
    edge_cols, edge_wids = [], []
    for (u,v) in G_flow.edges():
        if (u,v) in btop_edges:
            edge_cols.append("red")
            edge_wids.append(2.0)
        else:
            edge_cols.append("gray")
            edge_wids.append(1.0)

    plt.figure(figsize=(9,7))
    pos = nx.spring_layout(G_flow, seed=42)
    nx.draw_networkx_nodes(G_flow, pos, node_color="lightblue", node_size=600)
    
    labels_dict = {n: le_task.inverse_transform([int(n)])[0] for n in G_flow.nodes()}
    nx.draw_networkx_labels(G_flow, pos, labels_dict, font_size=8)
    nx.draw_networkx_edges(G_flow, pos, edge_color=edge_cols, width=edge_wids, arrows=True)

    edge_lbl = {}
    for (u,v) in btop_edges:
        edge_lbl[(u,v)] = f"{G_flow[u][v]['mean_hours']:.1f}h"
    nx.draw_networkx_edge_labels(G_flow, pos, edge_labels=edge_lbl, 
                                font_color="red", font_size=7)
    
    plt.title("Process Flow with Bottlenecks (Red edges)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_transition_heatmap(transitions, le_task, save_path="transition_probability_heatmap.png"):
    """Plot transition probability heatmap"""
    trans_count = transitions.groupby(["task_id","next_task_id"]).size().unstack(fill_value=0)
    prob_matrix = trans_count.div(trans_count.sum(axis=1), axis=0)
    
    plt.figure(figsize=(10,8))
    xticklabels = [le_task.inverse_transform([int(c)])[0] for c in prob_matrix.columns]
    yticklabels = [le_task.inverse_transform([int(r)])[0] for r in prob_matrix.index]
    
    sns.heatmap(prob_matrix, cmap="YlGnBu", annot=False,
                xticklabels=xticklabels,
                yticklabels=yticklabels)
    plt.title("Transition Probability Heatmap")
    plt.xlabel("Next Activity")
    plt.ylabel("Current Activity")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_sankey_diagram(df, le_task, save_path="process_flow_sankey.html"):
    """Create Sankey diagram of process flow"""
    start_counts = df.groupby("case_id").first()["task_id"].value_counts().to_dict()
    end_counts = df.groupby("case_id").last()["task_id"].value_counts().to_dict()
    
    trans_count = df.groupby(["task_id","next_task_id"]).size().unstack(fill_value=0)
    arr = trans_count.stack().reset_index().values
    
    unique_nodes = ["Start"] + list(le_task.classes_) + ["End"]
    node_idx = {n:i for i,n in enumerate(unique_nodes)}

    sources, targets, values = [], [], []
    
    # Start transitions
    for act_id, ct in start_counts.items():
        sources.append(node_idx["Start"])
        act_name = le_task.inverse_transform([int(act_id)])[0]
        targets.append(node_idx[act_name])
        values.append(int(ct))
    
    # End transitions
    for act_id, ct in end_counts.items():
        act_name = le_task.inverse_transform([int(act_id)])[0]
        sources.append(node_idx[act_name])
        targets.append(node_idx["End"])
        values.append(int(ct))
    
    # Internal transitions
    for row in arr:
        sid, tid, ccount = row
        sid_name = le_task.inverse_transform([int(sid)])[0]
        tid_name = le_task.inverse_transform([int(tid)])[0]
        sources.append(node_idx[sid_name])
        targets.append(node_idx[tid_name])
        values.append(int(ccount))

    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(label=unique_nodes),
        link=dict(source=sources, target=targets, value=values)
    )])
    
    sankey_fig.update_layout(title_text="Process Flow Sankey Diagram", font_size=10)
    sankey_fig.write_html(save_path) 