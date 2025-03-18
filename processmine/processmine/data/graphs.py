"""
High-performance graph building utilities with DGL, optimized for efficient memory usage
and vectorized operations with minimal memory overhead.
"""

import torch
import numpy as np
import dgl
import pandas as pd
import time
import logging
import gc
import psutil
from typing import List, Dict, Optional, Union, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

def build_graph_data(
    df, 
    enhanced: bool = False, 
    batch_size: Optional[int] = None, 
    num_workers: int = 0, 
    verbose: bool = True,
    bidirectional: bool = True,
    limit_nodes: Optional[int] = None,
    mode: str = 'auto',
    use_edge_features: bool = True
) -> List[dgl.DGLGraph]:
    """
    Build graph data with optimized memory usage and vectorized operations
    
    Args:
        df: Process data dataframe
        enhanced: Whether to include edge features
        batch_size: Batch size for memory-efficient processing 
                   (auto-determined if None)
        num_workers: Number of workers for parallel processing
        verbose: Whether to print progress information
        bidirectional: Whether to create bidirectional edges
        limit_nodes: Maximum number of nodes per graph (None for no limit)
        mode: Graph building strategy ('auto', 'standard', 'sparse')
        
    Returns:
        List of DGL graph objects
    """
    if verbose:
        logger.info(f"Building {'enhanced ' if enhanced else ''}graph data with DGL")
    start_time = time.time()
    
    # Identify feature columns
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    
    if not feature_cols:
        logger.warning("No feature columns found. Ensure feature extraction was performed.")
        feature_cols = ["task_id", "resource_id"]  # Use basic features as fallback
    
    # Determine optimal batch size based on available memory if not specified
    if batch_size is None:
        batch_size = _determine_optimal_batch_size(df, feature_cols, enhanced)
        if verbose:
            logger.info(f"Auto-determined batch size: {batch_size} cases")
    
    # Get unique case IDs
    case_ids = df["case_id"].unique()
    num_cases = len(case_ids)
    
    # Function for tracking progress
    progress_tracker = _progress_tracker(num_cases, verbose, "Building graphs")
    
    # Statistics for logging
    stats = {"node_counts": [], "edge_counts": []}
    
    # Auto-determine mode if not specified
    if mode == 'auto':
        # Use sparse for larger datasets
        use_sparse = len(df) > 50000 or len(case_ids) > 1000
        mode = 'sparse' if use_sparse else 'standard'
        
        if verbose:
            logger.info(f"Selected graph building mode: {mode}")
    
    # Process in batches with optimized memory usage
    graphs = []
    for i in range(0, num_cases, batch_size):
        batch_end = min(i + batch_size, num_cases)
        batch_case_ids = case_ids[i:batch_end]
        
        # Create local dataframe view - filter with query for efficiency
        batch_df = df.loc[df["case_id"].isin(batch_case_ids)].copy()
        
        # Pre-sort all data to avoid sorting in loop
        batch_df.sort_values(["case_id", "timestamp"], inplace=True)
        
        # Process batch using the specified mode
        if mode == 'sparse':
            batch_graphs = _build_graphs_sparse(batch_df, feature_cols, enhanced, 
                                          bidirectional, limit_nodes, stats)
        else:  # 'standard' mode
            batch_graphs = _build_graphs_standard(batch_df, feature_cols, enhanced, 
                                              bidirectional, limit_nodes, stats)
            
        # Add batch graphs to overall list
        graphs.extend(batch_graphs)
        
        # Update progress
        progress_tracker(batch_end - i)
        
        # Force garbage collection after each batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Log statistics
    _log_graph_statistics(graphs, stats, start_time, verbose)
    
    return graphs

def _determine_optimal_batch_size(df, feature_cols, enhanced):
    """Determine optimal batch size based on memory availability"""
    # Estimate memory requirements per case
    avg_events_per_case = len(df) / df["case_id"].nunique()
    bytes_per_event = len(feature_cols) * 4  # 4 bytes per float32
    
    if enhanced:
        # Enhanced graphs need more memory for edge features
        bytes_per_event += 8  # Additional memory for edge features
    
    estimated_memory_per_case = avg_events_per_case * bytes_per_event
    
    # Get available memory with safety margin (use 30% of available)
    available_memory = psutil.virtual_memory().available * 0.3
    
    # Calculate batch size, with sensible min/max
    batch_size = max(50, min(5000, int(available_memory / (estimated_memory_per_case * 1.5))))
    
    return batch_size

def _progress_tracker(total, verbose, desc):
    """Create simple progress tracker function"""
    if not verbose:
        return lambda x: None
        
    try:
        from tqdm import tqdm
        progress_bar = tqdm(total=total, desc=desc)
        return lambda x: progress_bar.update(x)
    except ImportError:
        # Simple fallback
        start_time = time.time()
        processed = 0
        
        def update(x):
            nonlocal processed
            processed += x
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / rate if rate > 0 else 0
            print(f"\r{desc}: {processed}/{total} ({100*processed/total:.1f}%) "
                  f"[{elapsed:.1f}s elapsed, {remaining:.1f}s remaining]", end="")
            if processed >= total:
                print()  # New line at end
        
        return update

def _build_graphs_standard(batch_df, feature_cols, enhanced, bidirectional, limit_nodes, stats):
    """Build graphs with standard approach for smaller batches using DGL"""
    batch_graphs = []
    
    # Process each case
    for case_id, case_df in batch_df.groupby("case_id"):
        # Sort by timestamp for proper sequence
        case_df = case_df.sort_values("timestamp")
        
        # Apply node limit if specified
        if limit_nodes and len(case_df) > limit_nodes:
            case_df = case_df.iloc[:limit_nodes]
        
        n_nodes = len(case_df)
        stats["node_counts"].append(n_nodes)
        
        # Create node features
        node_features = torch.tensor(case_df[feature_cols].values, dtype=torch.float32)
        
        # Store targets as a tensor
        labels = torch.tensor(case_df["next_task"].values, dtype=torch.long)
        
        if n_nodes > 1:
            # Create sequential edges efficiently
            src = torch.arange(n_nodes-1, dtype=torch.long)
            dst = torch.arange(1, n_nodes, dtype=torch.long)
            
            # For DGL we use lists of source and destination nodes
            edges_src = src.tolist()
            edges_dst = dst.tolist()
            
            if bidirectional:
                # Add reverse edges
                edges_src.extend(dst.tolist())
                edges_dst.extend(src.tolist())
                edge_count = 2 * (n_nodes - 1)
            else:
                edge_count = n_nodes - 1
                
            stats["edge_counts"].append(edge_count)
            
            # Create DGL graph
            g = dgl.graph((edges_src, edges_dst), num_nodes=n_nodes)
            
            # Add node features
            g.ndata['feat'] = node_features
            g.ndata['label'] = labels
            
            # Add edge features if enhanced
            if enhanced:
                # Calculate time differences
                timestamps = case_df["timestamp"].values
                time_diffs = np.array([
                    (timestamps[i+1] - timestamps[i]) / np.timedelta64(1, 'h')  # Hours
                    for i in range(n_nodes-1)
                ], dtype=np.float32)
                
                # Normalize time differences
                max_time = max(time_diffs.max(), 1e-6)  # Avoid division by zero
                norm_time_diffs = time_diffs / max_time
                
                # Create edge features tensor
                if bidirectional:
                    # Edge features: forward edges have time diff, backward edges have -time diff
                    edge_feats = torch.tensor(
                        np.concatenate([norm_time_diffs, -norm_time_diffs]), 
                        dtype=torch.float32
                    ).view(-1, 1)
                else:
                    edge_feats = torch.tensor(norm_time_diffs, dtype=torch.float32).view(-1, 1)
                
                # Add edge features to the graph
                g.edata['feat'] = edge_feats
        else:
            # Handle single-node case (no edges)
            g = dgl.graph(([], []), num_nodes=1)
            g.ndata['feat'] = node_features
            g.ndata['label'] = labels
            
            if enhanced:
                # Add empty edge features for consistency
                g.edata['feat'] = torch.empty((0, 1), dtype=torch.float32)
            
            stats["edge_counts"].append(0)
        
        batch_graphs.append(g)
    
    return batch_graphs

def _build_graphs_sparse(batch_df, feature_cols, enhanced, bidirectional, limit_nodes, stats):
    """
    Build graphs using sparse matrix operations for larger batches
    This is more memory efficient for large batches
    """
    # Group case IDs
    case_groups = batch_df.groupby("case_id")
    all_graphs = []
    
    # Extract all node features at once
    all_features = batch_df[feature_cols].values
    
    # Track node offsets
    node_offset = 0
    edge_indices_src = []
    edge_indices_dst = []
    edge_attrs = []
    graph_slices = []
    y_values = []
    
    # Process each case
    for case_id, indices in case_groups.indices.items():
        case_df = batch_df.iloc[indices].sort_values("timestamp")
        
        # Apply node limit if specified
        if limit_nodes and len(case_df) > limit_nodes:
            case_df = case_df.iloc[:limit_nodes]
            indices = indices[:limit_nodes]
        
        n_nodes = len(case_df)
        stats["node_counts"].append(n_nodes)
        
        if n_nodes > 1:
            # Create sequential edges efficiently
            src = np.arange(n_nodes-1) + node_offset
            tgt = np.arange(1, n_nodes) + node_offset
            
            if bidirectional:
                edge_indices_src.extend(src)
                edge_indices_src.extend(tgt)
                edge_indices_dst.extend(tgt)
                edge_indices_dst.extend(src)
                edge_count = 2 * (n_nodes - 1)
            else:
                edge_indices_src.extend(src)
                edge_indices_dst.extend(tgt)
                edge_count = n_nodes - 1
                
            stats["edge_counts"].append(edge_count)
            
            # Add edge features if enhanced
            if enhanced:
                # Calculate time differences
                timestamps = case_df["timestamp"].values
                time_diffs = np.array([
                    (timestamps[i+1] - timestamps[i]) / np.timedelta64(1, 'h')  # Hours
                    for i in range(n_nodes-1)
                ], dtype=np.float32)
                
                # Normalize time differences
                max_time = max(time_diffs.max(), 1e-6)  # Avoid division by zero
                norm_time_diffs = time_diffs / max_time
                
                if bidirectional:
                    # Edge features: forward edges have time diff, backward edges have -time diff
                    edge_attrs.extend(norm_time_diffs)
                    edge_attrs.extend(-norm_time_diffs)
                else:
                    edge_attrs.extend(norm_time_diffs)
        else:
            stats["edge_counts"].append(0)
        
        # Get target values for this case
        y_values.extend(case_df["next_task"].values)
        
        # Store slice boundaries
        graph_slices.append((node_offset, node_offset + n_nodes))
        
        # Update offset
        node_offset += n_nodes
    
    # Create graphs from slices
    for i, (start, end) in enumerate(graph_slices):
        # Extract node features for this graph
        node_features = torch.tensor(all_features[start:end], dtype=torch.float32)
        
        # Extract labels for this graph
        labels = torch.tensor(y_values[start:end], dtype=torch.long)
        
        # Find edges for this graph
        if edge_indices_src and edge_indices_dst: # Check if there are any edges
            mask = ((start <= np.array(edge_indices_src)) & (np.array(edge_indices_src) < end) & 
                    (start <= np.array(edge_indices_dst)) & (np.array(edge_indices_dst) < end))
            
            # Extract edges for this graph
            if np.any(mask):
                src = np.array(edge_indices_src)[mask] - start
                dst = np.array(edge_indices_dst)[mask] - start
                
                # Create DGL graph
                g = dgl.graph((src, dst), num_nodes=end-start)
                
                # Add node features
                g.ndata['feat'] = node_features
                g.ndata['label'] = labels
                
                if enhanced and edge_attrs:
                    # Extract edge attributes for this graph
                    edge_feats = torch.tensor(np.array(edge_attrs)[mask], dtype=torch.float32).view(-1, 1)
                    g.edata['feat'] = edge_feats
            else:
                # Create a graph with no edges
                g = dgl.graph(([], []), num_nodes=end-start)
                g.ndata['feat'] = node_features
                g.ndata['label'] = labels
                
                if enhanced:
                    # Add empty edge features for consistency
                    g.edata['feat'] = torch.empty((0, 1), dtype=torch.float32)
        else:
            # Create a graph with no edges
            g = dgl.graph(([], []), num_nodes=end-start)
            g.ndata['feat'] = node_features
            g.ndata['label'] = labels
            
            if enhanced:
                # Add empty edge features for consistency
                g.edata['feat'] = torch.empty((0, 1), dtype=torch.float32)
        
        all_graphs.append(g)
    
    return all_graphs

def _log_graph_statistics(graphs, stats, start_time, verbose):
    """Log detailed statistics about the built graphs"""
    if not verbose:
        return
    
    # Calculate statistics
    if stats["node_counts"] and stats["edge_counts"]:
        avg_nodes = np.mean(stats["node_counts"])
        avg_edges = np.mean(stats["edge_counts"])
        max_nodes = np.max(stats["node_counts"])
        max_edges = np.max(stats["edge_counts"]) if stats["edge_counts"] else 0
        min_nodes = np.min(stats["node_counts"])
        min_edges = np.min(stats["edge_counts"]) if stats["edge_counts"] else 0
        median_nodes = np.median(stats["node_counts"])
        median_edges = np.median(stats["edge_counts"]) if stats["edge_counts"] else 0
        total_nodes = sum(stats["node_counts"])
        total_edges = sum(stats["edge_counts"]) if stats["edge_counts"] else 0
    else:
        avg_nodes = avg_edges = max_nodes = max_edges = 0
        min_nodes = min_edges = median_nodes = median_edges = 0
        total_nodes = total_edges = 0
    
    # Log detailed statistics
    logger.info(f"Graph statistics:")
    logger.info(f"  Total graphs: {len(graphs):,}")
    logger.info(f"  Total nodes: {total_nodes:,}, Total edges: {total_edges:,}")
    logger.info(f"  Avg nodes per graph: {avg_nodes:.2f} (min={min_nodes}, median={median_nodes}, max={max_nodes})")
    logger.info(f"  Avg edges per graph: {avg_edges:.2f} (min={min_edges}, median={median_edges}, max={max_edges})")
    logger.info(f"  Sparsity: {total_edges/(total_nodes**2):.6f}")
    logger.info(f"Graphs built in {time.time() - start_time:.2f}s")
    
    # Check for potential memory issues
    if max_nodes > 1000 or max_edges > 5000:
        logger.warning(f"Very large graphs detected. Consider limiting graph size with limit_nodes parameter.")

def build_heterogeneous_graph(
    df: pd.DataFrame, 
    node_types: Optional[Dict[str, List[str]]] = None, 
    edge_types: Optional[List[Tuple[str, str, str]]] = None,  # Updated to use canonical DGL edge type tuples
    batch_size: int = 1000, 
    verbose: bool = True,
    use_edge_attr: bool = True
) -> List[dgl.DGLGraph]:
    """
    Build heterogeneous graph data with optimized memory management
    
    Args:
        df: Process data dataframe
        node_types: Dictionary mapping node types to feature columns
                    (e.g., {'task': ['task_id', 'feat_task_id'], 'resource': ['resource_id']})
        edge_types: List of edge types in (src_type, relation, dst_type) format
                    (e.g., [('task', 'to', 'task'), ('task', 'uses', 'resource')])
        batch_size: Batch size for memory-efficient processing
        verbose: Whether to print progress information
        use_edge_attr: Whether to include edge attributes
        
    Returns:
        List of heterogeneous DGL graphs
    """
    if verbose:
        logger.info("Building heterogeneous graph data with DGL")
    start_time = time.time()
    
    # Define default node types if not provided
    if node_types is None:
        task_cols = [col for col in df.columns if col.startswith('feat_task') or col == 'task_id']
        resource_cols = [col for col in df.columns if col.startswith('feat_resource') or col == 'resource_id']
        
        node_types = {
            "task": task_cols,
            "resource": resource_cols
        }
    
    # Define default edge types if not provided
    if edge_types is None:
        edge_types = [
            ('task', 'to', 'task'),
            ('task', 'uses', 'resource'),
            ('resource', 'performs', 'task')
        ]
    
    # Get unique case IDs
    case_ids = df["case_id"].unique()
    num_cases = len(case_ids)
    
    # Function for tracking progress
    progress_tracker = _progress_tracker(num_cases, verbose, "Building heterogeneous graphs")
    
    # Process in batches
    het_graphs = []
    
    for i in range(0, num_cases, batch_size):
        batch_end = min(i + batch_size, num_cases)
        batch_case_ids = case_ids[i:batch_end]
        
        # Filter dataframe for current batch
        batch_df = df.loc[df["case_id"].isin(batch_case_ids)].copy()
        
        # Pre-sort by case_id and timestamp
        batch_df.sort_values(["case_id", "timestamp"], inplace=True)
        
        # Process each case
        for case_id, case_group in batch_df.groupby("case_id"):
            # Dictionaries to store node features and mappings by type
            node_features = {}
            node_mappings = {}
            node_counts = {}
            
            # Data dictionaries for heterograph construction - using canonical DGL format
            graph_data = {}
            
            # Extract nodes and features for each node type
            for node_type, feature_cols in node_types.items():
                # Get valid feature columns
                valid_cols = [col for col in feature_cols if col in case_group.columns]
                
                if not valid_cols:
                    if verbose:
                        logger.warning(f"No valid columns found for node type {node_type}")
                    continue
                
                # Get unique node IDs based on node type
                if node_type == "task":
                    id_col = "task_id"
                elif node_type == "resource":
                    id_col = "resource_id"
                else:
                    # For custom node types, use the first column as ID
                    id_col = valid_cols[0]
                
                # Get unique nodes with features
                unique_nodes = case_group[id_col].unique()
                node_counts[node_type] = len(unique_nodes)
                
                # Create mapping from original ID to graph node index
                node_mappings[node_type] = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
                
                # Extract features
                feat_cols = [col for col in valid_cols if col != id_col]
                if feat_cols:
                    # Create feature matrix, handle each unique node once
                    features = np.zeros((len(unique_nodes), len(feat_cols)), dtype=np.float32)
                    
                    for idx, node_id in enumerate(unique_nodes):
                        # Use the first occurrence of this node for features
                        node_data = case_group[case_group[id_col] == node_id].iloc[0]
                        features[idx] = node_data[feat_cols].values
                else:
                    # Use node ID as single feature
                    features = np.array(unique_nodes).reshape(-1, 1).astype(np.float32)
                
                # Convert to tensor
                node_features[node_type] = torch.tensor(features, dtype=torch.float32)
            
            # Process edges for each edge type
            for src_type, rel_type, dst_type in edge_types:
                # Skip if source or destination type is missing
                if src_type not in node_mappings or dst_type not in node_mappings:
                    continue
                
                # Create edge lists and attributes
                src_nodes = []
                dst_nodes = []
                edge_attrs = []
                
                if src_type == dst_type == "task" and rel_type == "to":
                    # Task to task transitions (sequential in process)
                    task_ids = case_group["task_id"].values
                    task_map = node_mappings["task"]
                    
                    for i in range(len(task_ids) - 1):
                        src_id = task_ids[i]
                        dst_id = task_ids[i+1]
                        
                        if src_id in task_map and dst_id in task_map:
                            src_nodes.append(task_map[src_id])
                            dst_nodes.append(task_map[dst_id])
                            
                            # Add edge attributes if enabled
                            if use_edge_attr and "timestamp" in case_group.columns:
                                time_diff = (case_group["timestamp"].iloc[i+1] - case_group["timestamp"].iloc[i]).total_seconds() / 3600
                                edge_attrs.append([float(time_diff)])
                
                elif src_type == "task" and dst_type == "resource" and rel_type == "uses":
                    # Task uses resource relationship
                    for _, row in case_group.iterrows():
                        task_id = row["task_id"]
                        resource_id = row["resource_id"]
                        
                        task_map = node_mappings["task"]
                        resource_map = node_mappings["resource"]
                        
                        if task_id in task_map and resource_id in resource_map:
                            src_nodes.append(task_map[task_id])
                            dst_nodes.append(resource_map[resource_id])
                            
                            # Add edge attributes if needed
                            if use_edge_attr:
                                edge_attrs.append([1.0])  # Default weight
                
                elif src_type == "resource" and dst_type == "task" and rel_type == "performs":
                    # Resource performs task relationship
                    for _, row in case_group.iterrows():
                        resource_id = row["resource_id"]
                        task_id = row["task_id"]
                        
                        resource_map = node_mappings["resource"]
                        task_map = node_mappings["task"]
                        
                        if resource_id in resource_map and task_id in task_map:
                            src_nodes.append(resource_map[resource_id])
                            dst_nodes.append(task_map[task_id])
                            
                            # Add edge attributes if needed
                            if use_edge_attr:
                                edge_attrs.append([1.0])  # Default weight
                
                # Add other custom edge types here...
                
                # Store the edge data if we have edges
                if src_nodes and dst_nodes:
                    # Create canonical edge type tuple for heterograph
                    etype = (src_type, rel_type, dst_type)
                    graph_data[etype] = (src_nodes, dst_nodes)
                    
                    # Store edge attributes if available
                    if use_edge_attr and edge_attrs:
                        if 'edge_features' not in locals():
                            edge_features = {}
                        edge_features[etype] = torch.tensor(edge_attrs, dtype=torch.float32)
            
            # Create heterogeneous graph if we have nodes and edges
            if node_features and graph_data:
                # Specify number of nodes for each type to avoid issues with isolated nodes
                num_nodes_dict = {ntype: count for ntype, count in node_counts.items()}
                
                # Create the heterograph with appropriate number of nodes
                g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
                
                # Add node features
                for ntype, features in node_features.items():
                    if g.num_nodes(ntype) > 0:
                        g.nodes[ntype].data['feat'] = features
                
                # Add edge features if available
                if use_edge_attr and 'edge_features' in locals():
                    for etype, features in edge_features.items():
                        if g.num_edges(etype) > 0:
                            g.edges[etype].data['feat'] = features
                
                # Add node labels (for task nodes)
                if 'task' in node_features and 'next_task' in case_group.columns:
                    # Create label mapping from task IDs to indices
                    task_map = node_mappings['task']
                    task_ids = list(task_map.keys())
                    
                    # Initialize labels tensor with -1 (padding value)
                    labels = torch.full((len(task_ids),), -1, dtype=torch.long)
                    
                    # Fill in known labels
                    for i, task_id in enumerate(task_ids):
                        # Find rows where this is the current task
                        task_rows = case_group[case_group['task_id'] == task_id]
                        if not task_rows.empty and 'next_task' in task_rows.columns:
                            # Use the first valid next_task as label
                            next_task = task_rows['next_task'].iloc[0]
                            if not pd.isna(next_task):
                                labels[i] = int(next_task)
                    
                    # Only assign valid labels (not -1)
                    valid_mask = labels >= 0
                    if valid_mask.any():
                        g.nodes['task'].data['label'] = labels.masked_fill(~valid_mask, 0)
                
                het_graphs.append(g)
            else:
                # No valid edges, create minimal graph with just nodes
                if verbose:
                    logger.warning(f"No valid edges found for case {case_id}, creating node-only heterograph")
                
                # Create a minimal heterograph with at least one node type
                for ntype, count in node_counts.items():
                    if count > 0:
                        # Create heterograph with single node type and no edges
                        g = dgl.heterograph({(ntype, 'self', ntype): ([], [])}, 
                                            num_nodes_dict={ntype: count})
                        
                        # Add features for this node type
                        if ntype in node_features:
                            g.nodes[ntype].data['feat'] = node_features[ntype]
                        
                        het_graphs.append(g)
                        break
        
        # Update progress
        progress_tracker(batch_end - i)
    
    # Log completion
    if verbose:
        logger.info(f"Built {len(het_graphs)} heterogeneous graphs in {time.time() - start_time:.2f}s")
        
        # Log graph statistics
        if het_graphs:
            total_nodes = 0
            total_edges = 0
            node_types_used = set()
            edge_types_used = set()
            
            for g in het_graphs[:min(100, len(het_graphs))]:  # Sample at most 100 graphs
                for ntype in g.ntypes:
                    total_nodes += g.num_nodes(ntype)
                    node_types_used.add(ntype)
                
                for etype in g.canonical_etypes:
                    total_edges += g.num_edges(etype)
                    edge_types_used.add(etype)
            
            logger.info(f"Graph statistics (sample of {min(100, len(het_graphs))} graphs):")
            logger.info(f"  Average nodes per graph: {total_nodes / min(100, len(het_graphs)):.1f}")
            logger.info(f"  Average edges per graph: {total_edges / min(100, len(het_graphs)):.1f}")
            logger.info(f"  Node types used: {sorted(node_types_used)}")
            logger.info(f"  Edge types used: {sorted(edge_types_used)}")
    
    return het_graphs

def build_dgl_graph_batch(node_features, edge_indices, edge_features=None):
    """
    Build a batched DGL graph from node features and edge indices
    
    Args:
        node_features: Tensor of node features [num_nodes, feature_dim]
        edge_indices: Tuple of (src, dst) tensors for edges
        edge_features: Optional tensor of edge features [num_edges, feature_dim]
        
    Returns:
        DGL graph
    """
    # Create graph
    g = dgl.graph((edge_indices[0], edge_indices[1]))
    
    # Add node features
    g.ndata['feat'] = node_features
    
    # Add edge features if provided
    if edge_features is not None:
        g.edata['feat'] = edge_features
    
    return g

def add_self_loops_to_dgl(g):
    """
    Add self-loops to a DGL graph
    
    Args:
        g: DGL graph
        
    Returns:
        DGL graph with self-loops
    """
    return dgl.add_self_loop(g)

def create_dgl_heterograph(data_dict, num_nodes_dict=None):
    """
    Create a heterogeneous DGL graph
    
    Args:
        data_dict: Dictionary mapping edge types to edge data
        num_nodes_dict: Optional dictionary mapping node types to node counts
        
    Returns:
        Heterogeneous DGL graph
    """
    return dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

def extract_subgraphs_dgl(g, nodes_list):
    """
    Extract multiple node-induced subgraphs from a DGL graph
    
    Args:
        g: DGL graph
        nodes_list: List of lists of nodes for each subgraph
        
    Returns:
        List of DGL subgraphs
    """
    return [dgl.node_subgraph(g, nodes) for nodes in nodes_list]

def convert_to_bidirectional_dgl(g):
    """
    Convert a directed DGL graph to bidirectional by adding reverse edges
    
    Args:
        g: DGL graph
        
    Returns:
        Bidirectional DGL graph
    """
    # Get edges
    src, dst = g.edges()
    
    # Add reverse edges
    src_reverse, dst_reverse = dst, src
    
    # Combine original and reverse edges
    new_src = torch.cat([src, src_reverse])
    new_dst = torch.cat([dst, dst_reverse])
    
    # Create new graph with bidirectional edges
    new_g = dgl.graph((new_src, new_dst), num_nodes=g.num_nodes())
    
    # Copy node features
    for key, value in g.ndata.items():
        new_g.ndata[key] = value
    
    # Copy and extend edge features
    for key, value in g.edata.items():
        # Duplicate edge features for reverse edges
        new_g.edata[key] = torch.cat([value, value])
    
    return new_g