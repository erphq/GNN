"""
Data loading and processing utilities for DGL-based graph models.
"""

import psutil
import gc
from random import random
import torch
import dgl
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

def collate_dgl_graphs(graphs):
    """
    Collate function for batching DGL graphs
    
    Args:
        graphs: List of DGL graphs to batch
        
    Returns:
        Batched DGL graph
    """
    return dgl.batch(graphs)

def get_graph_dataloader(graphs, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader for DGL graphs
    
    Args:
        graphs: List of DGL graphs
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader for DGL graphs
    """
    from dgl.dataloading import GraphDataLoader
    
    return GraphDataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_dgl_graphs
    )

def get_graph_targets(g):
    """
    Extract graph-level targets from a DGL graph or batched graph
    
    Args:
        g: DGL graph or batched graph
        
    Returns:
        Graph-level target tensor
    """
    if 'label' in g.ndata:
        # Get node labels
        node_labels = g.ndata['label']
        
        if hasattr(g, 'batch_size') and g.batch_size > 1:
            # For batched graph, extract one label per graph
            batch_num_nodes = g.batch_num_nodes()
            graph_labels = []
            
            node_offset = 0
            for num_nodes in batch_num_nodes:
                # Get labels for this graph
                graph_node_labels = node_labels[node_offset:node_offset + num_nodes]
                
                # Use mode (most common label) as graph label
                if len(graph_node_labels) > 0:
                    values, counts = torch.unique(graph_node_labels, return_counts=True)
                    mode_idx = torch.argmax(counts)
                    graph_labels.append(values[mode_idx])
                else:
                    # Fallback if no labels
                    graph_labels.append(torch.tensor(0, device=node_labels.device))
                
                # Update offset
                node_offset += num_nodes
            
            return torch.stack(graph_labels)
        else:
            # For single graph, use mode of node labels
            values, counts = torch.unique(node_labels, return_counts=True)
            mode_idx = torch.argmax(counts)
            return values[mode_idx].unsqueeze(0)
    else:
        # No labels found
        return None

def get_batch_graphs_from_indices(graphs, indices):
    """
    Get a list of graphs from indices with proper error handling
    
    Args:
        graphs: List of DGL graphs
        indices: List or NumPy array of indices to extract
        
    Returns:
        List of DGL graphs at the specified indices
    """
    import dgl
    
    if not isinstance(graphs, list):
        raise TypeError(f"Expected list of graphs, got {type(graphs)}")
    
    if len(graphs) == 0:
        raise ValueError("Empty graph list provided")
    
    # Convert NumPy array to list if needed
    if isinstance(indices, np.ndarray):
        indices = indices.tolist()
    
    # Check if indices are valid
    max_idx = max(indices) if indices else -1
    if max_idx >= len(graphs):
        raise IndexError(f"Index {max_idx} out of range for graph list of length {len(graphs)}")
    
    # Extract graphs
    result = [graphs[i] for i in indices]
    
    # Verify all are DGL graphs
    for i, g in enumerate(result):
        if not isinstance(g, dgl.DGLGraph):
            raise TypeError(f"Graph at index {indices[i]} is not a DGL graph: {type(g)}")
    
    return result

def apply_to_nodes(g, func):
    """
    Apply a function to all nodes in a graph
    
    Args:
        g: DGL graph
        func: Function to apply to node features
        
    Returns:
        Updated graph
    """
    # Create a new graph to avoid modifying the original
    new_g = g.clone()
    
    # Apply function to node features
    if 'feat' in g.ndata:
        new_g.ndata['feat'] = func(g.ndata['feat'])
    
    return new_g

def apply_to_edges(g, func):
    """
    Apply a function to all edges in a graph
    
    Args:
        g: DGL graph
        func: Function to apply to edge features
        
    Returns:
        Updated graph
    """
    # Create a new graph to avoid modifying the original
    new_g = g.clone()
    
    # Apply function to edge features
    if 'feat' in g.edata:
        new_g.edata['feat'] = func(g.edata['feat'])
    
    return new_g

def create_node_masks(g, mask_ratio=0.1):
    """
    Create node feature masks for self-supervised learning
    
    Args:
        g: DGL graph
        mask_ratio: Ratio of nodes to mask
        
    Returns:
        Graph with added mask
    """
    num_nodes = g.num_nodes()
    mask_indices = torch.randperm(num_nodes)[:int(num_nodes * mask_ratio)]
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[mask_indices] = True
    
    # Add mask to graph
    g.ndata['mask'] = mask
    
    # Store original features for masked nodes
    if 'feat' in g.ndata:
        g.ndata['orig_feat'] = g.ndata['feat'].clone()
        
        # Apply masking (replace with zeros)
        masked_feat = g.ndata['feat'].clone()
        masked_feat[mask] = 0.0
        g.ndata['feat'] = masked_feat
    
    return g

def prepare_graph_data(batch_data, device=None):
    """
    Prepare graph data for model processing
    
    Args:
        batch_data: Input graph data (must be DGL graph or list of DGL graphs)
        device: Device to move data to
        
    Returns:
        DGL graph or batched graph on specified device
    """
    # Handle device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle DGL graph directly
    if isinstance(batch_data, dgl.DGLGraph):
        return batch_data.to(device)
    
    # Handle list of DGL graphs
    if isinstance(batch_data, list):
        if not all(isinstance(g, dgl.DGLGraph) for g in batch_data):
            raise TypeError("All items in the list must be DGL graphs")
        return dgl.batch(batch_data).to(device)
        
    # Unsupported type
    raise TypeError(f"Unsupported batch data type: {type(batch_data)}. Must be DGL graph or list of DGL graphs.")

def apply_to_nodes(g, func):
    """
    Apply a function to all nodes in a graph with DGL optimized approach
    
    Args:
        g: DGL graph
        func: Function to apply to node features
        
    Returns:
        Updated graph
    """
    # Using DGL's in-place feature modification when possible
    if 'feat' in g.ndata:
        # Create a new graph only if necessary
        if func.__code__.co_argcount > 1 or g.is_readonly():
            # Create a new graph to avoid modifying the original
            new_g = g.clone()
            new_g.ndata['feat'] = func(g.ndata['feat'])
            return new_g
        else:
            # Apply function in-place to save memory
            g.ndata['feat'] = func(g.ndata['feat'])
            return g
    return g

def apply_to_edges(g, func):
    """
    Apply a function to all edges in a graph using DGL optimized approach
    
    Args:
        g: DGL graph
        func: Function to apply to edge features
        
    Returns:
        Updated graph
    """
    if 'feat' in g.edata:
        # Use in-place operations when possible
        if func.__code__.co_argcount > 1 or g.is_readonly():
            # Create a new graph only when necessary
            new_g = g.clone()
            new_g.edata['feat'] = func(g.edata['feat'])
            return new_g
        else:
            # Apply function in-place
            g.edata['feat'] = func(g.edata['feat'])
            return g
    return g

def create_node_masks(g, mask_ratio=0.1):
    """
    Create node feature masks for self-supervised learning with DGL-optimized approach
    
    Args:
        g: DGL graph
        mask_ratio: Ratio of nodes to mask
        
    Returns:
        Graph with added mask
    """
    num_nodes = g.num_nodes()
    mask_indices = torch.randperm(num_nodes)[:int(num_nodes * mask_ratio)]
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=g.device)
    mask[mask_indices] = True
    
    # Add mask to graph
    g.ndata['mask'] = mask
    
    # Store original features for masked nodes if they exist
    if 'feat' in g.ndata:
        # Save original features only for masked nodes to save memory
        orig_feat = g.ndata['feat'].clone()
        g.ndata['orig_feat'] = orig_feat
        
        # Apply masking (replace with zeros) using DGL's efficient indexing
        if mask.any():
            masked_feat = g.ndata['feat'].clone()
            masked_feat[mask] = 0.0
            g.ndata['feat'] = masked_feat
    
    return g

def prepare_graph_batch(batch_data, device=None):
    """
    Prepare a batch of graphs for model processing with DGL
    
    Args:
        batch_data: Graph data (DGL graph or list of DGL graphs)
        device: Target device for the graphs
        
    Returns:
        DGL graph or batched graph on specified device
    """
    import torch
    import dgl
    
    # Handle device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle DGL graph or batched graph directly
    if isinstance(batch_data, dgl.DGLGraph):
        return batch_data.to(device)
    
    # Handle list of DGL graphs
    if isinstance(batch_data, list):
        if not all(isinstance(g, dgl.DGLGraph) for g in batch_data):
            raise TypeError("All items in the list must be DGL graphs")
        return dgl.batch(batch_data).to(device)
    
    # Unsupported input type
    raise TypeError(f"Unsupported input type: {type(batch_data)}. Must be DGL graph or list of DGL graphs.")

class MemoryEfficientDataLoader:
    """
    Memory-efficient data loader for DGL graphs with dynamic batch sizing
    """
    def __init__(
        self, 
        dataset, 
        batch_size=32, 
        shuffle=True, 
        pin_memory=True,
        prefetch_factor=2, 
        memory_threshold=0.85,
        drop_last=False,
        collate_fn=None,
        num_workers=0
    ):
        """
        Initialize memory-efficient data loader
        
        Args:
            dataset: Dataset or list of DGL graphs
            batch_size: Initial batch size
            shuffle: Whether to shuffle the data
            pin_memory: Whether to use pinned memory
            prefetch_factor: Prefetch factor for asynchronous loading
            memory_threshold: Memory usage threshold to trigger adjustments
            drop_last: Whether to drop the last batch if incomplete
            collate_fn: Custom collate function (defaults to dgl.batch)
            num_workers: Number of worker processes
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.memory_threshold = memory_threshold
        self.drop_last = drop_last
        self.num_workers = num_workers
        
        # Default to DGL batch collate if not provided
        self.collate_fn = collate_fn or (lambda graphs: dgl.batch(graphs))
        
        # Calculate initial indices
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        
        # Track memory usage for adaptive batch sizing
        self.batch_size_history = []
        self.memory_usage_history = []
        
        # Set up DGL DataLoader for multi-process loading if workers > 0
        if num_workers > 0:
            try:
                from dgl.dataloading import GraphDataLoader
                self.dataloader = GraphDataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=drop_last,
                    collate_fn=self.collate_fn
                )
                self.use_dgl_loader = True
            except ImportError:
                self.use_dgl_loader = False
        else:
            self.use_dgl_loader = False
        
        # Track current position
        self.position = 0
        self._iterator = None
    
    def __iter__(self):
        """Create iterator for batches"""
        # If using DGL DataLoader, return its iterator
        if self.use_dgl_loader:
            self._iterator = iter(self.dataloader)
            return self
        
        # Reset position
        self.position = 0
        
        # Reshuffle if needed
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        
        return self
    
    def __next__(self):
        """Get next batch with memory-aware adaptive sizing"""
        if self.use_dgl_loader:
            try:
                return next(self._iterator)
            except StopIteration:
                raise StopIteration
        
        if self.position >= len(self.indices):
            raise StopIteration
        
        # Check current memory usage
        current_memory = self._get_memory_usage()
        
        # Adjust batch size if needed
        current_batch_size = self._adjust_batch_size(current_memory)
        
        # Calculate end position for this batch
        end_position = min(self.position + current_batch_size, len(self.indices))
        
        # Handle drop_last
        if self.drop_last and end_position - self.position < current_batch_size:
            self.position = len(self.indices)  # Move to end
            raise StopIteration
        
        # Get batch indices
        batch_indices = self.indices[self.position:end_position]
        
        # Update position for next batch
        self.position = end_position
        
        # Track memory and batch size for analysis
        self.batch_size_history.append(current_batch_size)
        self.memory_usage_history.append(current_memory)
        
        # Extract and collate batch
        batch = [self.dataset[i] for i in batch_indices]
        
        # Run garbage collection if memory usage is high
        if current_memory > self.memory_threshold * 0.95:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return self.collate_fn(batch)
    
    def _get_memory_usage(self):
        """Get current memory usage as a fraction of total available memory"""
        try:
            # Check GPU memory if available
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                return max(allocated, reserved) / total
            else:
                # Fall back to CPU memory
                vm = psutil.virtual_memory()
                return vm.percent / 100.0
        except:
            # Default to a safe value if measurement fails
            return 0.7
    
    def _adjust_batch_size(self, current_memory):
        """Adjust batch size based on current memory usage"""
        if current_memory > self.memory_threshold:
            # Reduce batch size when memory usage is high
            new_batch_size = max(1, int(self.batch_size * 0.8))
            if new_batch_size != self.batch_size:
                self.batch_size = new_batch_size
        elif current_memory < self.memory_threshold * 0.7:
            # Increase batch size when memory usage is low
            new_batch_size = min(1024, int(self.batch_size * 1.2))
            if new_batch_size != self.batch_size:
                self.batch_size = new_batch_size
        
        return self.batch_size

def adaptive_normalization(features, feature_statistics=None):
    """
    Apply appropriate normalization based on data characteristics
    Fixed version that ensures proper type handling
    
    Args:
        features: Feature tensor or array to normalize
        feature_statistics: Optional pre-computed statistics
        
    Returns:
        Normalized features
    """
    import numpy as np
    import torch
    
    # Convert to numpy if tensor
    is_tensor = torch.is_tensor(features)
    if is_tensor:
        device = features.device
        features_np = features.cpu().numpy()
    else:
        features_np = features
    
    # Ensure features are float type for normalization
    if features_np.dtype.kind in 'iu':  # If integer type
        features_np = features_np.astype(np.float32)
    
    # Calculate statistics if not provided
    if feature_statistics is None:
        feature_statistics = {
            'mean': np.mean(features_np, axis=0),
            'std': np.std(features_np, axis=0),
            'min': np.min(features_np, axis=0),
            'max': np.max(features_np, axis=0),
            'skewness': _calculate_skewness(features_np)
        }
    
    # Get statistics and ensure they're float arrays
    skewness = np.asarray(feature_statistics['skewness'], dtype=np.float32)
    min_vals = np.asarray(feature_statistics['min'], dtype=np.float32)
    max_vals = np.asarray(feature_statistics['max'], dtype=np.float32)
    
    # Calculate range ratio (avoiding division by zero)
    epsilon = 1e-8
    range_ratio = np.divide(
        max_vals, 
        np.maximum(min_vals, epsilon),
        out=np.ones_like(max_vals, dtype=np.float32),  # Explicitly make this a float array
        where=min_vals>epsilon
    )
    
    # Choose normalization strategy based on data properties
    if np.any(np.abs(skewness) > 1.5) or np.any(range_ratio > 10):
        # Highly skewed with large range differences - use robust scaling
        median = np.median(features_np, axis=0)
        q1 = np.percentile(features_np, 25, axis=0)
        q3 = np.percentile(features_np, 75, axis=0)
        iqr = q3 - q1
        # Avoid division by zero
        iqr = np.maximum(iqr, epsilon)
        normalized = (features_np - median) / iqr
    elif np.any(np.abs(features_np) > 5.0):
        # Large magnitudes - use L2 normalization
        norms = np.sqrt(np.sum(features_np**2, axis=1, keepdims=True))
        norms = np.maximum(norms, epsilon)  # Avoid division by zero
        normalized = features_np / norms
    else:
        # Well-behaved features - use MinMax
        range_vals = np.maximum(max_vals - min_vals, epsilon)
        normalized = (features_np - min_vals) / range_vals
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        normalized = torch.tensor(normalized, dtype=torch.float32, device=device)
    
    return normalized

def _calculate_skewness(arr):
    """Calculate skewness of array elements along first axis"""
    import numpy as np
    
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    # Avoid division by zero
    std = np.maximum(std, 1e-8)
    
    # Calculate skewness (third moment)
    n = arr.shape[0]
    m3 = np.sum((arr - mean)**3, axis=0) / n
    return m3 / (std**3)

def apply_dgl_sampling(g, method='neighbor', fanout=10, k=16):
    """
    Apply DGL's graph sampling techniques for memory-efficient processing
    
    Args:
        g: DGL graph to sample
        method: Sampling method ('neighbor', 'topk', 'random', 'khop')
        fanout: Number of neighbors to sample in neighbor sampling
        k: Number of nodes to select in topk sampling
        
    Returns:
        Sampled DGL graph
    """
    import dgl
    
    try:
        if method == 'neighbor':
            # Sample neighbors with importance weights
            if 'weight' in g.edata:
                # Use edge weights for importance sampling
                frontier = dgl.sampling.sample_neighbors(
                    g, 
                    torch.arange(g.num_nodes()), 
                    fanout, 
                    edge_dir='out',
                    prob='weight'
                )
                return frontier
            else:
                # Uniform sampling if no weights
                frontier = dgl.sampling.sample_neighbors(
                    g, 
                    torch.arange(g.num_nodes()), 
                    fanout, 
                    edge_dir='out'
                )
                return frontier
                
        elif method == 'topk':
            # Select important nodes based on connectivity or features
            # Either use in-degree for importance or node features
            if g.in_degrees().sum() > 0:
                scores = g.in_degrees().float()
            else:
                # Use feature magnitude as importance if available
                if 'feat' in g.ndata:
                    scores = g.ndata['feat'].sum(dim=1)
                else:
                    scores = torch.ones(g.num_nodes())
                    
            # Select top-k nodes
            _, indices = torch.topk(scores, min(k, g.num_nodes()))
            return g.subgraph(indices)
            
        elif method == 'khop':
            # Get k-hop subgraph from a set of seed nodes
            # Choose important nodes as seeds (e.g., with highest degree)
            if g.in_degrees().sum() > 0:
                scores = g.in_degrees().float()
                _, seeds = torch.topk(scores, min(10, g.num_nodes()))
            else:
                seeds = torch.arange(min(10, g.num_nodes()))
                
            # Extract k-hop subgraph
            nodes, edges = dgl.khop_in_subgraph(g, seeds, k=2)
            subg = g.subgraph(nodes)
            return subg
            
        elif method == 'random':
            # Simple random node sampling
            sample_size = min(g.num_nodes() // 2 + 1, g.num_nodes())
            nodes = torch.randperm(g.num_nodes())[:sample_size]
            return g.subgraph(nodes)
            
        else:
            return g  # Return original graph if method not supported
            
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"DGL sampling failed: {e}")
        return g  # Return original graph on error
    
def create_dgl_batch_from_graphs(graphs):
    """
    Create a batched DGL graph from a list of graphs
    
    Args:
        graphs: List of DGL graphs
        
    Returns:
        Batched DGL graph
    """
    return dgl.batch(graphs)

def split_dgl_batched_graph(batched_graph):
    """
    Split a batched DGL graph back into individual graphs
    
    Args:
        batched_graph: Batched DGL graph
        
    Returns:
        List of individual DGL graphs
    """
    return dgl.unbatch(batched_graph)

def convert_networkx_to_dgl(nx_graph, node_attrs=None, edge_attrs=None):
    """
    Convert a NetworkX graph to a DGL graph
    
    Args:
        nx_graph: NetworkX graph
        node_attrs: List of node attributes to copy
        edge_attrs: List of edge attributes to copy
        
    Returns:
        DGL graph
    """
    return dgl.from_networkx(nx_graph, node_attrs=node_attrs, edge_attrs=edge_attrs)

def sample_dgl_neighbor_graph(g, seeds, fanout, edge_dir='in'):
    """
    Sample a subgraph by randomly choosing neighbors
    
    Args:
        g: Input DGL graph
        seeds: Seed nodes
        fanout: Number of neighbors to sample per node
        edge_dir: Edge direction ('in', 'out', or 'both')
        
    Returns:
        Sampled subgraph
    """
    return dgl.sampling.sample_neighbors(g, seeds, fanout, edge_dir=edge_dir)

def sample_dgl_khop_subgraph(g, seeds, k, edge_dir='in'):
    """
    Extract k-hop subgraph for given seeds
    
    Args:
        g: Input DGL graph
        seeds: Seed nodes
        k: Number of hops
        edge_dir: Edge direction ('in', 'out', or 'both')
        
    Returns:
        k-hop subgraph and nodes IDs
    """
    nodes, edges, inverse_index = dgl.sampling.sample_neighbors(
        g, seeds, -1, edge_dir=edge_dir, return_edges=True
    )
    return dgl.node_subgraph(g, nodes), nodes