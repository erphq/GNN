"""
Memory-optimized data loading and preprocessing with advanced chunking, 
vectorization, and minimal memory footprint.
"""
import pandas as pd
import numpy as np
import torch
import time
import logging
import gc
import os
import psutil
import dask
from typing import Tuple, Dict, Any, Optional, List, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler
from functools import partial

logger = logging.getLogger(__name__)

def load_and_preprocess_data(
    data_path: str, 
    norm_method: str = 'l2',
    chunk_size: Optional[int] = None,
    cache_dir: Optional[str] = None,
    use_dtypes: bool = True,
    memory_limit_gb: float = 4.0,
    use_dask: bool = False,
    selected_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Load and preprocess event log data with maximal memory efficiency
    
    Args:
        data_path: Path to the data file (CSV)
        norm_method: Normalization method ('l2', 'standard', 'minmax', or None)
        chunk_size: Size of chunks to process (auto-detected if None)
        cache_dir: Directory to cache intermediate results (None for no caching)
        use_dtypes: Whether to optimize dtypes to reduce memory usage
        memory_limit_gb: Memory limit in GB for chunking calculation
        use_dask: Whether to use Dask for out-of-memory processing
        selected_columns: Only load specific columns (None for all columns)
        
    Returns:
        Tuple of (preprocessed_df, task_encoder, resource_encoder)
    """
    start_time = time.time()
    logger.info(f"Loading data from {data_path}")

    # Check cache first
    cached_file = _check_cache(data_path, cache_dir)
    if cached_file:
        try:
            df, task_encoder, resource_encoder = pd.read_pickle(cached_file)
            logger.info(f"Loaded cached data: {len(df):,} rows, {df['case_id'].nunique():,} cases")
            return df, task_encoder, resource_encoder
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
    
    # Use Dask for very large files
    if use_dask:
        return _load_with_dask(data_path, norm_method, cache_dir, selected_columns)
    
    # Determine optimal chunk size based on file size and available memory
    if chunk_size is None:
        chunk_size = _calculate_chunk_size(data_path, memory_limit_gb)
    
    # Optimize dtypes if requested
    dtype_optimizers = _optimize_dtypes(data_path, selected_columns) if use_dtypes else None
    
    # Load data with chunking for large files
    if chunk_size:
        df = _load_chunked(data_path, chunk_size, dtype_optimizers, selected_columns)
    else:
        df = _load_single_pass(data_path, dtype_optimizers, selected_columns)
    
    # Standardize columns
    df = _standardize_columns(df)
    
    # Process df in-place to reduce memory usage
    task_encoder, resource_encoder = _preprocess_data_inplace(df, norm_method)
    
    # Cache result if requested
    if cached_file:
        try:
            os.makedirs(os.path.dirname(cached_file), exist_ok=True)
            df.to_pickle(cached_file)
            logger.info(f"Cached preprocessed data to {cached_file}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    # Log data statistics
    preprocessing_time = time.time() - start_time
    logger.info(f"Preprocessing completed in {preprocessing_time:.2f}s")
    _log_data_statistics(df)
    
    return df, task_encoder, resource_encoder

def _check_cache(data_path: str, cache_dir: Optional[str]) -> Optional[str]:
    """Check if cached preprocessing exists and return cache path"""
    if not cache_dir:
        return None
        
    os.makedirs(cache_dir, exist_ok=True)
    file_hash = hash(os.path.abspath(data_path) + str(os.path.getmtime(data_path)))
    return os.path.join(cache_dir, f"processed_{file_hash}.pkl")

def _calculate_chunk_size(data_path: str, memory_limit_gb: float) -> Optional[int]:
    """Calculate optimal chunk size based on file size and memory constraints"""
    file_size_gb = os.path.getsize(data_path) / (1024 ** 3)
    # Estimate memory needed (CSV parsing typically expands by 2-4x)
    estimated_memory_gb = file_size_gb * 3
    
    if estimated_memory_gb > memory_limit_gb:
        # Need to use chunking - estimate rows per GB based on file size
        estimated_rows = os.path.getsize(data_path) / 200  # Rough estimate: 200 bytes per row
        rows_per_chunk = int(estimated_rows * (memory_limit_gb / estimated_memory_gb))
        # Round to nearest 10000
        chunk_size = max(10000, round(rows_per_chunk, -4))
        logger.info(f"File size: {file_size_gb:.2f} GB, using chunk size: {chunk_size:,} rows")
        return chunk_size
    else:
        logger.info(f"File size: {file_size_gb:.2f} GB, loading in single pass")
        return None

def _optimize_dtypes(data_path: str, selected_columns: Optional[List[str]]) -> Dict[str, Any]:
    """Create optimized dtypes for loading CSV files efficiently"""
    # Sample first few rows to infer dtypes
    sample_size = 10000
    usecols = selected_columns
    
    sample = pd.read_csv(data_path, nrows=sample_size, usecols=usecols)
    dtype_map = {}
    
    for col in sample.columns:
        # Skip timestamp columns
        if col.lower().endswith('timestamp') or col.lower().endswith('time'):
            continue
            
        # Check if column is categorical or ID
        if col.lower().endswith('id') or sample[col].nunique() < len(sample) * 0.1:
            sample_val = sample[col].iloc[0]
            
            if pd.api.types.is_integer_dtype(sample[col]):
                # Use smallest possible int dtype
                max_val = sample[col].max()
                min_val = sample[col].min()
                
                if min_val >= 0:
                    # Unsigned integer
                    if max_val < 2**8:
                        dtype_map[col] = 'uint8'
                    elif max_val < 2**16:
                        dtype_map[col] = 'uint16'
                    elif max_val < 2**32:
                        dtype_map[col] = 'uint32'
                    else:
                        dtype_map[col] = 'uint64'
                else:
                    # Signed integer
                    if min_val > -2**7 and max_val < 2**7:
                        dtype_map[col] = 'int8'
                    elif min_val > -2**15 and max_val < 2**15:
                        dtype_map[col] = 'int16'
                    elif min_val > -2**31 and max_val < 2**31:
                        dtype_map[col] = 'int32'
                    else:
                        dtype_map[col] = 'int64'
            
            elif pd.api.types.is_string_dtype(sample[col]):
                # Use category for string columns with few unique values
                if sample[col].nunique() < min(1000, len(sample) * 0.1):
                    dtype_map[col] = 'category'
        
        # For float columns
        elif pd.api.types.is_float_dtype(sample[col]):
            # Check if float32 is sufficient
            if sample[col].dropna().between(-3.4e38, 3.4e38).all():
                dtype_map[col] = 'float32'
    
    # Release memory
    del sample
    gc.collect()
    
    return dtype_map

def _load_chunked(data_path: str, chunk_size: int, dtypes: Optional[Dict[str, Any]], 
                  selected_columns: Optional[List[str]]) -> pd.DataFrame:
    """Load data in chunks to minimize memory usage"""
    chunks = []
    total_rows = 0
    
    # Use iterator for large files
    for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunk_size, 
                                         dtype=dtypes, usecols=selected_columns)):
        # Standardize columns
        chunk = _standardize_columns(chunk)
        
        # Track progress
        total_rows += len(chunk)
        logger.info(f"Loaded chunk {i+1}: {len(chunk):,} rows, total: {total_rows:,} rows")
        
        chunks.append(chunk)
        
        # Force garbage collection after each chunk
        if i % 5 == 4:  # Every 5 chunks
            gc.collect()
    
    # Combine chunks efficiently
    logger.info(f"Combining {len(chunks)} chunks...")
    df = pd.concat(chunks, ignore_index=True, copy=False)
    
    # Clear chunk references to free memory
    chunks.clear()
    gc.collect()
    
    return df

def _load_single_pass(data_path: str, dtypes: Optional[Dict[str, Any]], 
                     selected_columns: Optional[List[str]]) -> pd.DataFrame:
    """Load entire file at once when memory permits"""
    df = pd.read_csv(data_path, dtype=dtypes, usecols=selected_columns)
    df = _standardize_columns(df)
    logger.info(f"Loaded {len(df):,} rows in single pass")
    return df

def _load_with_dask(data_path: str, norm_method: str, cache_dir: Optional[str],
                   selected_columns: Optional[List[str]]) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Use Dask for out-of-memory processing of very large files"""
    try:
        import dask.dataframe as dd
        logger.info("Using Dask for out-of-memory processing")
        
        # Read data with Dask
        ddf = dd.read_csv(data_path, usecols=selected_columns)
        
        # Standardize column names
        ddf = ddf.rename(columns={
            "case:concept:name": "case_id",
            "concept:name": "task_name",
            "org:resource": "resource",
            "time:timestamp": "timestamp",
            "case:id": "case_id"
        })
        
        # Convert to pandas in chunks for processing
        meta_df = ddf.head(100)
        task_encoder = LabelEncoder().fit(meta_df["task_name"])
        resource_encoder = LabelEncoder().fit(meta_df["resource"])
        
        # Process in chunks
        processed_chunks = []
        for chunk in ddf.partitions:
            # Convert chunk to pandas
            pdf = chunk.compute()
            
            # Process chunk (without encoders to avoid refitting)
            _, _, _ = _preprocess_data_inplace(pdf, norm_method, task_encoder, resource_encoder)
            
            processed_chunks.append(pdf)
            del pdf
            gc.collect()
        
        # Combine processed chunks
        df = pd.concat(processed_chunks, ignore_index=True, copy=False)
        processed_chunks.clear()
        gc.collect()
        
        return df, task_encoder, resource_encoder
        
    except ImportError:
        logger.warning("Dask not available. Falling back to chunked loading.")
        chunk_size = 100000  # Default large chunk size for Dask fallback
        return load_and_preprocess_data(data_path, norm_method, chunk_size, 
                                       cache_dir, True, 4.0, False, selected_columns)

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to handle different naming conventions (e.g., XES format)"""
    # Handle different column naming conventions (support XES format)
    column_mappings = {
        "case:concept:name": "case_id",
        "concept:name": "task_name",
        "org:resource": "resource",
        "time:timestamp": "timestamp",
    }
    
    # If case:id exists but case:concept:name doesn't, map it to case_id
    if "case:id" in df.columns and "case:concept:name" not in df.columns:
        column_mappings["case:id"] = "case_id"
    elif "case:id" in df.columns:
        # If both exist, map case:id to a different name
        column_mappings["case:id"] = "case_id_alt"
    
    # Only rename columns that exist in the dataframe
    existing_mappings = {k: v for k, v in column_mappings.items() if k in df.columns}
    if existing_mappings:
        df = df.rename(columns=existing_mappings)
    
    # Check for required columns
    required_columns = ["case_id", "task_name", "timestamp"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df

def _preprocess_data_inplace(df: pd.DataFrame, norm_method: str = 'l2',
                           task_encoder: Optional[LabelEncoder] = None,
                           resource_encoder: Optional[LabelEncoder] = None) -> Tuple:
    """
    Preprocess data in-place to minimize memory usage
    
    Args:
        df: Input dataframe
        norm_method: Normalization method
        task_encoder: Existing task encoder (None to create new)
        resource_encoder: Existing resource encoder (None to create new)
        
    Returns:
        Tuple of (task_encoder, resource_encoder)
    """
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Remove rows with invalid timestamps
    n_before = len(df)
    df.dropna(subset=["timestamp"], inplace=True)
    n_after = len(df)
    
    if n_before > n_after:
        logger.warning(f"Removed {n_before - n_after:,} rows with invalid timestamps")
    
    # Sort by case_id and timestamp - critical for correct sequence
    df.sort_values(["case_id", "timestamp"], inplace=True)
    
    # Create encoders if not provided
    if task_encoder is None:
        task_encoder = LabelEncoder()
        df["task_id"] = task_encoder.fit_transform(df["task_name"])
    else:
        # Handle new categories in incremental processing
        df["task_id"] = df["task_name"].map(
            lambda x: task_encoder.transform([x])[0] if x in task_encoder.classes_ else -1
        )
        # Add new categories
        mask = df["task_id"] == -1
        if mask.any():
            new_items = df.loc[mask, "task_name"].unique()
            old_classes = task_encoder.classes_
            task_encoder.classes_ = np.append(old_classes, new_items)
            for i, item in enumerate(new_items):
                df.loc[df["task_name"] == item, "task_id"] = len(old_classes) + i
    
    # Same for resource encoder
    if resource_encoder is None:
        resource_encoder = LabelEncoder()
        df["resource_id"] = resource_encoder.fit_transform(df["resource"])
    else:
        df["resource_id"] = df["resource"].map(
            lambda x: resource_encoder.transform([x])[0] if x in resource_encoder.classes_ else -1
        )
        mask = df["resource_id"] == -1
        if mask.any():
            new_items = df.loc[mask, "resource"].unique()
            old_classes = resource_encoder.classes_
            resource_encoder.classes_ = np.append(old_classes, new_items)
            for i, item in enumerate(new_items):
                df.loc[df["resource"] == item, "resource_id"] = len(old_classes) + i
    
    # Add derived temporal features efficiently using vectorized operations
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_weekend"] = (df["day_of_week"] >= 5).astype('uint8')  # Use uint8 to save memory
    
    # Add next_task by case efficiently
    df["next_task"] = df.groupby("case_id")["task_id"].shift(-1)
    
    # Remove rows with no next task (last event in each case)
    n_before = len(df)
    df.dropna(subset=["next_task"], inplace=True)
    n_after = len(df)
    
    logger.info(f"Removed {n_before - n_after:,} end events (no next task)")
    
    # Convert next_task to int - do this safely to handle potential NaNs
    df["next_task"] = df["next_task"].astype(int)
    
    # Create feature columns
    feature_cols = ["task_id", "resource_id", "day_of_week", "hour_of_day", "is_weekend"]
    
    # Add amount feature if available
    if "amount" in df.columns:
        feature_cols.append("amount")
    
    # Normalize features efficiently
    if norm_method and norm_method.lower() != 'none':
        # Extract features as numpy array for faster processing
        features = df[feature_cols].values
        
        # Normalize features without creating unnecessary copies
        norm_features = _normalize_features(features, method=norm_method)
        
        # Add normalized features back to dataframe
        for i, col in enumerate(feature_cols):
            df[f"feat_{col}"] = norm_features[:, i]
            
        # Free memory
        del norm_features
    else:
        # Just add the feature columns with original values
        for col in feature_cols:
            df[f"feat_{col}"] = df[col]
    
    # Add case-level features
    _add_case_features_inplace(df)
    
    # Force garbage collection
    gc.collect()
    
    return task_encoder, resource_encoder

def _normalize_features(features: np.ndarray, method: str = 'l2') -> np.ndarray:
    """Normalize features with different methods, optimized for memory efficiency"""
    
    if features.dtype.kind in 'iu':  # If integer type
        features = features.astype(np.float32)
    
    if method.lower() == 'l2':
        # L2 normalization - compute norms first
        norms = np.sqrt(np.sum(features**2, axis=1, keepdims=True))
        np.maximum(norms, 1e-10, out=norms)  # Avoid division by zero in-place
        
        # Normalize inplace if the array is writeable, otherwise make a copy
        if features.flags.writeable:
            features = np.divide(features, norms, out=features)
            return features
        else:
            return features / norms
    
    elif method.lower() == 'standard':
        # Standard scaling - use float32 to save memory
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        np.maximum(std, 1e-10, out=std)  # Avoid division by zero
        
        # In-place operations if possible
        if features.flags.writeable:
            features = np.subtract(features, mean, out=features)
            features = np.divide(features, std, out=features)
            return features
        else:
            return (features - mean) / std
    
    elif method.lower() == 'minmax':
        # Min-max scaling to [0, 1]
        min_vals = np.min(features, axis=0, keepdims=True)
        max_vals = np.max(features, axis=0, keepdims=True)
        denom = np.maximum(max_vals - min_vals, 1e-10)
        
        # In-place operations if possible
        if features.flags.writeable:
            features = np.subtract(features, min_vals, out=features)
            features = np.divide(features, denom, out=features)
            return features
        else:
            return (features - min_vals) / denom
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def _add_case_features_inplace(df: pd.DataFrame) -> None:
    """Add case-level features for better prediction, optimized for memory efficiency"""
    # Calculate case-level statistics
    case_stats = df.groupby('case_id').agg({
        'timestamp': ['min', 'max', 'count'],
        'task_id': 'nunique',
        'resource_id': 'nunique'
    })
    
    # Flatten multi-level columns
    case_stats.columns = ['_'.join(col).strip() for col in case_stats.columns.values]
    
    # Calculate duration in seconds
    case_stats['case_duration_seconds'] = (
        case_stats['timestamp_max'] - case_stats['timestamp_min']
    ).dt.total_seconds()
    
    # Rename columns for clarity
    case_stats = case_stats.rename(columns={
        'timestamp_count': 'case_events',
        'task_id_nunique': 'case_unique_tasks',
        'resource_id_nunique': 'case_unique_resources'
    })
    
    # Select columns to add back
    case_features = case_stats[[
        'case_events', 
        'case_unique_tasks', 
        'case_unique_resources', 
        'case_duration_seconds'
    ]]
    
    # Add back to original dataframe - use merge for memory efficiency
    df_case_features = case_features.reset_index()
    
    # Add case features using efficient merge instead of join
    for col in case_features.columns:
        # Create mapping dictionary for faster lookup
        case_feature_dict = dict(zip(df_case_features['case_id'], df_case_features[col]))
        
        # Vectorized mapping
        df[col] = df['case_id'].map(case_feature_dict)
    
    # Free memory
    del case_stats, case_features, df_case_features
    gc.collect()

def _log_data_statistics(df: pd.DataFrame) -> None:
    """Log detailed statistics about the processed dataframe"""
    logger.info(f"Data statistics:")
    logger.info(f"  Cases: {df['case_id'].nunique():,}")
    logger.info(f"  Activities: {df['task_id'].nunique():,}")
    logger.info(f"  Resources: {df['resource_id'].nunique():,}")
    logger.info(f"  Events: {len(df):,}")
    
    try:
        logger.info(f"  Time range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    except:
        pass
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(f"  Memory usage: {memory_usage:.2f} MB")
    
    # Activity distribution
    top_activities = df['task_id'].value_counts().nlargest(5)
    logger.info(f"  Top activities:")
    for act, count in top_activities.items():
        logger.info(f"    Task {act}: {count:,} events ({count/len(df)*100:.1f}%)")

def create_sequence_dataset(df, max_seq_len=50, min_seq_len=2, feature_cols=None, pad_sequences=True):
    """
    Create sequence dataset for LSTM models with optimized memory usage
    
    Args:
        df: Preprocessed dataframe
        max_seq_len: Maximum sequence length
        min_seq_len: Minimum sequence length
        feature_cols: Feature columns to use (default: all feat_ columns)
        pad_sequences: Whether to pad sequences to the same length
        
    Returns:
        Tuple of (sequences, targets, seq_lengths)
    """
    logger.info("Creating sequence dataset")
    start_time = time.time()
    
    # If feature columns not specified, use all feat_ columns
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col.startswith('feat_')]
    
    # Create efficient groupby object
    groups = df.groupby('case_id')
    
    # Pre-allocate memory for better efficiency
    num_cases = df['case_id'].nunique()
    sequences = []
    targets = []
    seq_lengths = []
    
    # Estimate memory to handle batching if needed
    approx_mem_per_seq = len(feature_cols) * 4 * max_seq_len / (1024 * 1024)  # in MB
    batch_size = num_cases
    
    # If memory footprint is too large, process in batches
    if approx_mem_per_seq * num_cases > 1000:  # More than 1GB
        # Reduce batch size to keep memory usage reasonable
        batch_size = max(100, int(1000 / approx_mem_per_seq))
        logger.info(f"Processing sequences in batches of {batch_size} for memory efficiency")
    
    # Process in batches if needed
    case_ids = list(groups.groups.keys())
    
    for i in range(0, len(case_ids), batch_size):
        batch_ids = case_ids[i:i+batch_size]
        
        # Process each case in batch
        batch_sequences = []
        batch_targets = []
        batch_lengths = []
        
        for case_id in batch_ids:
            group = groups.get_group(case_id)
            
            # Skip if sequence is too short
            if len(group) < min_seq_len:
                continue
            
            # Sort by timestamp
            group = group.sort_values('timestamp')
            
            # Create sequence
            seq = group[feature_cols].values
            
            # Truncate if too long
            if len(seq) > max_seq_len:
                seq = seq[:max_seq_len]
            
            # Get target (next task after sequence)
            target = group['next_task'].values
            if len(target) > max_seq_len:
                target = target[:max_seq_len]
            
            # Store sequence, target, and length
            batch_sequences.append(seq)
            batch_targets.append(target)
            batch_lengths.append(len(seq))
        
        # Process batch efficiently
        if pad_sequences:
            # Find max length in batch
            batch_max_len = max(batch_lengths)
            
            # Pad sequences and targets in batch
            for i in range(len(batch_sequences)):
                seq_len = batch_lengths[i]
                if seq_len < batch_max_len:
                    # Pad sequence with zeros
                    pad_seq = np.zeros((batch_max_len - seq_len, len(feature_cols)), dtype=np.float32)
                    batch_sequences[i] = np.vstack([batch_sequences[i], pad_seq])
                    
                    # Pad targets with zeros
                    pad_target = np.zeros(batch_max_len - seq_len, dtype=np.int64)
                    batch_targets[i] = np.concatenate([batch_targets[i], pad_target])
        
        # Convert to torch tensors
        batch_sequences = [torch.FloatTensor(seq) for seq in batch_sequences]
        batch_targets = [torch.LongTensor(target) for target in batch_targets]
        
        # Add to main lists
        sequences.extend(batch_sequences)
        targets.extend(batch_targets)
        seq_lengths.extend(batch_lengths)
        
        # Force garbage collection after each batch
        if i % (batch_size * 5) == 0:
            gc.collect()
    
    logger.info(f"Created {len(sequences)} sequences in {time.time() - start_time:.2f}s")
    logger.info(f"Sequence length statistics: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={sum(seq_lengths)/len(seq_lengths):.1f}")
    
    return sequences, targets, seq_lengths