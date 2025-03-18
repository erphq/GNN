"""
Core runner functions for analysis, training, and optimization workflows.
These functions provide a programmatic API for ProcessMine functionality independent of the CLI.
Updated to use DGL for graph operations.
"""

import logging
import time
import torch
import os
import json
import numpy as np
import dgl
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

def ensure_continuous_labels(y):
    """
    Ensure labels are continuous integers starting from 0

    Args:
        y: Input labels array

    Returns:
        Tuple of (remapped_labels, label_mapping, reverse_mapping)
    """
    unique_classes = np.sort(np.unique(y))
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
    reverse_mapping = {new_class: old_class for old_class, new_class in class_mapping.items()}

    # Apply mapping to create continuous labels
    y_continuous = np.array([class_mapping[label] for label in y])

    return y_continuous, class_mapping, reverse_mapping

def run_analysis(
    data_path: str,
    output_dir: Optional[Union[str, Path]] = None,
    bottleneck_threshold: float = 90.0,
    freq_threshold: int = 5,
    max_variants: int = 10,
    viz_format: str = "both",
    device: Optional[Union[str, torch.device]] = None,
    mem_efficient: bool = False,
    cache_dir: Optional[str] = None,
    skip_conformance: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Run process analysis workflow

    Args:
        data_path: Path to process data CSV file
        output_dir: Directory to save results (None for no saving)
        bottleneck_threshold: Percentile threshold for bottleneck detection
        freq_threshold: Minimum frequency for significant transitions
        max_variants: Maximum number of process variants to analyze
        viz_format: Visualization format ("static", "interactive", "both")
        device: Computing device to use
        mem_efficient: Whether to use memory-efficient mode
        cache_dir: Directory to cache processed data
        skip_conformance: Whether to skip conformance checking
        **kwargs: Additional arguments

    Returns:
        Dictionary with analysis results
    """
    # Set up device if provided as string
    if device is not None and isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Import necessary modules
        from processmine.data.loader import load_and_preprocess_data
        from processmine.process_mining.analysis import (
            analyze_bottlenecks,
            analyze_cycle_times,
            analyze_transition_patterns,
            identify_process_variants,
            analyze_resource_workload
        )
        from processmine.utils.memory import log_memory_usage

        # Log initial memory usage
        log_memory_usage()

        # Load and preprocess data
        logger.info(f"Loading data from {data_path}")
        df, task_encoder, resource_encoder = load_and_preprocess_data(
            data_path,
            norm_method='l2',
            cache_dir=cache_dir,
            use_dtypes=True,
            memory_limit_gb=8.0 if not mem_efficient else 2.0
        )

        # Log basic data statistics
        logger.info(f"Data loaded: {len(df):,} events, {df['case_id'].nunique():,} cases, " +
                   f"{df['task_id'].nunique():,} activities, {df['resource_id'].nunique():,} resources")

        # Run analysis pipeline
        start_time = time.time()

        # Step 1: Bottleneck analysis
        logger.info("Analyzing bottlenecks...")
        bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(
            df,
            freq_threshold=freq_threshold,
            percentile_threshold=bottleneck_threshold
        )

        # Step 2: Cycle time analysis
        logger.info("Analyzing cycle times...")
        case_stats, long_cases, p95 = analyze_cycle_times(df)

        # Step 3: Transition pattern analysis
        logger.info("Analyzing transition patterns...")
        transitions, trans_count, prob_matrix = analyze_transition_patterns(df)

        # Step 4: Process variant analysis
        logger.info("Identifying process variants...")
        variant_stats, variant_sequences = identify_process_variants(
            df,
            max_variants=max_variants
        )

        # Step 5: Resource workload analysis
        logger.info("Analyzing resource workload...")
        resource_stats = analyze_resource_workload(df)

        # Step 6: Conformance checking if not skipped
        conformance_results = None
        if not skip_conformance:
            logger.info("Performing conformance checking...")
            try:
                from processmine.process_mining.conformance import ConformanceChecker
                checker = ConformanceChecker(df)
                conformance_results = checker.check_conformance()
                logger.info(f"Conformance ratio: {conformance_results.get('conformance_ratio', 0):.2f}")
            except Exception as e:
                logger.warning(f"Error in conformance checking: {e}")

        # Create visualizations if output directory provided
        if output_dir is not None:
            logger.info("Creating visualizations...")
            from processmine.visualization.viz import ProcessVisualizer

            # Determine visualization format
            use_interactive = viz_format in ["interactive", "both"]
            use_static = viz_format in ["static", "both"]

            viz = ProcessVisualizer(
                output_dir=output_dir / "visualizations",
                force_static=not use_interactive
            )

            # Create visualizations
            if use_static or not use_interactive:
                # Static visualizations
                viz.cycle_time_distribution(
                    case_stats["duration_h"].values,
                    filename="cycle_time_distribution.png"
                )

                viz.bottleneck_analysis(
                    bottleneck_stats,
                    significant_bottlenecks,
                    task_encoder,
                    filename="bottleneck_analysis.png"
                )

                viz.process_flow(
                    bottleneck_stats,
                    task_encoder,
                    significant_bottlenecks,
                    filename="process_flow.png"
                )

                viz.transition_heatmap(
                    transitions,
                    task_encoder,
                    filename="transition_heatmap.png"
                )

            if use_interactive:
                # Interactive visualizations
                viz.cycle_time_distribution(
                    case_stats["duration_h"].values,
                    filename="cycle_time_distribution.html"
                )

                viz.bottleneck_analysis(
                    bottleneck_stats,
                    significant_bottlenecks,
                    task_encoder,
                    filename="bottleneck_analysis.html"
                )

                viz.process_flow(
                    bottleneck_stats,
                    task_encoder,
                    significant_bottlenecks,
                    filename="process_flow.html"
                )

                viz.transition_heatmap(
                    transitions,
                    task_encoder,
                    filename="transition_heatmap.html"
                )

                # Create dashboard
                viz.create_dashboard(
                    df=df,
                    cycle_times=case_stats["duration_h"].values,
                    bottleneck_stats=bottleneck_stats,
                    significant_bottlenecks=significant_bottlenecks,
                    task_encoder=task_encoder,
                    case_stats=case_stats,
                    resource_stats=resource_stats,
                    variant_stats=variant_stats,
                    filename="dashboard.html"
                )

        # Compile metrics
        metrics = {
            "cases": df["case_id"].nunique(),
            "events": len(df),
            "activities": df["task_id"].nunique(),
            "resources": df["resource_id"].nunique(),
            "variants": len(variant_stats),
            "bottlenecks": len(significant_bottlenecks),
            "perf": {
                "top_bottleneck_wait": significant_bottlenecks["mean_hours"].iloc[0] if len(significant_bottlenecks) > 0 else 0,
                "median_cycle_time": case_stats["duration_h"].median(),
                "p95_cycle_time": p95,
                "resource_gini": resource_stats.attrs.get("gini_coefficient", 0),
                "top_variant_pct": variant_stats["percentage"].iloc[0] if len(variant_stats) > 0 else 0
            }
        }

        if conformance_results:
            metrics["conformance"] = conformance_results

        # Save metrics if output directory provided
        if output_dir is not None:
            metrics_dir = output_dir / "metrics"
            os.makedirs(metrics_dir, exist_ok=True)

            with open(metrics_dir / "analysis_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else (int(o) if isinstance(o, np.integer) else str(o)))

        # Log completion
        analysis_time = time.time() - start_time
        logger.info(f"Analysis completed in {analysis_time:.2f}s")

        return {
            "df": df,
            "task_encoder": task_encoder,
            "resource_encoder": resource_encoder,
            "bottleneck_stats": bottleneck_stats,
            "significant_bottlenecks": significant_bottlenecks,
            "case_stats": case_stats,
            "transitions": transitions,
            "variant_stats": variant_stats,
            "resource_stats": resource_stats,
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def run_training(
    data_path: str,
    model: str = "enhanced_gnn",
    output_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
    mem_efficient: bool = True,
    cache_dir: Optional[str] = None,
    analysis_results: Optional[Dict[str, Any]] = None,
    graphs_path: Optional[str] = None,
    save_graphs: bool = False,
    dgl_sampling: str = "neighbor",
    use_edge_features: bool = True,
    input_dim: Optional[int] = None,  # Added explicit input_dim parameter
    **kwargs
) -> Dict[str, Any]:
    """
    Run model training workflow with enhanced DGL integration

    Args:
        data_path: Path to process data CSV file
        model: Model type ("gnn", "lstm", "enhanced_gnn", etc.)
        output_dir: Directory to save results (None for no saving)
        device: Computing device to use
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        seed: Random seed
        mem_efficient: Whether to use memory-efficient mode
        cache_dir: Directory to cache processed data
        analysis_results: Optional results from analysis step
        graphs_path: Path to load/save processed DGL graphs
        save_graphs: Whether to save processed DGL graphs
        dgl_sampling: Graph sampling method for large graphs
        use_edge_features: Whether to use edge features
        input_dim: Optional explicit input dimension (to avoid duplication)
        **kwargs: Additional model-specific arguments

    Returns:
        Dictionary with training results
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set DGL random seed
    dgl.random.seed(seed)

    # Set up device if provided as string
    if device is not None and isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        os.makedirs(output_dir / "models", exist_ok=True)
        os.makedirs(output_dir / "metrics", exist_ok=True)

        # Create graphs directory if saving graphs
        if save_graphs:
            os.makedirs(output_dir / "graphs", exist_ok=True)

    try:
        # Import necessary modules
        from processmine.data.loader import load_and_preprocess_data
        from processmine.data.graphs import build_graph_data
        from processmine.utils.memory import log_memory_usage
        from processmine.core.training import (
            train_model, evaluate_model, create_optimizer,
            create_lr_scheduler, compute_class_weights
        )
        from processmine.models.factory import create_model, get_model_config, ensure_continuous_labels
        from processmine.utils.dataloader import get_graph_dataloader

        # Load data if not provided from analysis
        if analysis_results is None or "df" not in analysis_results:
            logger.info(f"Loading data from {data_path}")
            df, task_encoder, resource_encoder = load_and_preprocess_data(
                data_path,
                norm_method='l2',
                cache_dir=cache_dir,
                use_dtypes=True
            )
        else:
            df = analysis_results["df"]
            task_encoder = analysis_results["task_encoder"]
            resource_encoder = analysis_results["resource_encoder"]

        # Log memory usage
        log_memory_usage()

        # Build appropriate model and datasets
        logger.info(f"Creating {model} model...")

        # Get model default config and update with kwargs
        model_config = get_model_config(model)
        model_config.update(kwargs)

        # If input_dim is explicitly provided, use it (avoids duplication issue)
        if input_dim is not None:
            model_config['input_dim'] = input_dim
            
        # Make sure model_type is not in kwargs to avoid conflicts
        model_config.pop('model_type', None)

        if model in ["gnn", "enhanced_gnn", "positional_gnn", "diverse_gnn"]:
            # Check for pre-processed graphs
            graphs = None
            if graphs_path and os.path.exists(graphs_path):
                try:
                    # Try to load preprocessed DGL graphs
                    logger.info(f"Loading preprocessed DGL graphs from {graphs_path}")
                    result = dgl.load_graphs(graphs_path)
                    graphs = result[0]  # First element contains the list of graphs
                    logger.info(f"Loaded {len(graphs)} preprocessed DGL graphs")
                except Exception as e:
                    logger.warning(f"Failed to load preprocessed graphs: {e}")
                    graphs = None

            # Build graph data if not loaded
            if graphs is None:
                logger.info("Building graph data with DGL...")
                graphs = build_graph_data(
                    df,
                    enhanced=(model == "enhanced_gnn"),
                    batch_size=500 if not mem_efficient else 100,
                    num_workers=kwargs.get('num_workers', 0),
                    bidirectional=True,
                    use_edge_features=use_edge_features,
                    verbose=True
                )

                # Save graphs if requested
                if save_graphs and output_dir:
                    graph_save_path = str(output_dir / "graphs" / "processed_graphs.bin")
                    logger.info(f"Saving processed DGL graphs to {graph_save_path}")
                    try:
                        dgl.save_graphs(graph_save_path, graphs)
                        logger.info(f"Saved {len(graphs)} DGL graphs")
                    except Exception as e:
                        logger.warning(f"Failed to save graphs: {e}")

            # Create model with optimal parameters
            if 'input_dim' not in model_config:
                input_dim = len([col for col in df.columns if col.startswith("feat_")])
                model_config['input_dim'] = input_dim
                
            if 'output_dim' not in model_config:
                output_dim = len(task_encoder.classes_)
                model_config['output_dim'] = output_dim

            # Now create model with a single set of parameters
            model_instance = create_model(model, **model_config)

            # Split indices with stratification for graph classification tasks
            try:
                # Extract GRAPH-LEVEL labels for stratification (common in DGL graph classification)
                graph_labels = []
                for g in graphs:
                    if 'graph_label' in g.ndata:
                        graph_labels.append(g.ndata['graph_label'][0].item())
                    elif 'label' in g.ndata:
                        graph_labels.append(g.ndata['label'][0].item())
                    else:
                        # Handle case with no explicit graph-level label
                        raise KeyError("No suitable graph label found")
                
                from sklearn.model_selection import train_test_split

                # First split: 70% train, 30% temp (val+test)
                train_idx, temp_idx = train_test_split(
                    range(len(graphs)),
                    test_size=0.3,
                    random_state=seed,
                    stratify=graph_labels
                )

                # Second split: 15% val, 15% test (half of 30%)
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size=0.5,
                    random_state=seed,
                    stratify=[graph_labels[i] for i in temp_idx]
                )
            except Exception as e:
                # Fallback to random split with warning
                logger.warning(f"Stratified split failed: {e}. Using random split.")
                indices = np.random.permutation(len(graphs))
                train_idx = indices[:int(0.7 * len(indices))]
                val_idx = indices[int(0.7 * len(indices)):int(0.85 * len(indices))]
                test_idx = indices[int(0.85 * len(indices)):]

            # Apply DGL sampling if requested (for very large graphs)
            if dgl_sampling != "none" and any(g.num_nodes() > 1000 for g in graphs):
                logger.info(f"Applying DGL {dgl_sampling} sampling for large graphs")

                try:
                    from processmine.utils.dataloader import apply_dgl_sampling

                    # Apply sampling to all graphs
                    sampled_graphs = []
                    for g in graphs:
                        if g.num_nodes() > 1000:  # Only sample large graphs
                            if dgl_sampling == 'neighbor':
                                # Use neighbor sampling for maintaining local structure
                                sampled_g = apply_dgl_sampling(g, method='neighbor', fanout=50)
                            elif dgl_sampling == 'topk':
                                # Use top-k sampling for selecting important nodes
                                sampled_g = apply_dgl_sampling(g, method='topk', k=500)
                            elif dgl_sampling == 'khop':
                                # Use k-hop sampling for preserving neighborhood
                                sampled_g = apply_dgl_sampling(g, method='khop')
                            else:
                                # Default to random sampling
                                sampled_g = apply_dgl_sampling(g, method='random')
                            sampled_graphs.append(sampled_g)
                        else:
                            sampled_graphs.append(g)

                        graphs = sampled_graphs
                        logger.info("Applied DGL sampling to reduce graph sizes")
                except Exception as e:
                    logger.warning(f"DGL sampling failed: {e}")

            # Create data loaders with memory efficiency in mind
            num_workers = kwargs.get('num_workers', 0) if not mem_efficient else 0

            # Get train/val/test graphs
            from processmine.utils.dataloader import get_batch_graphs_from_indices
            train_graphs = get_batch_graphs_from_indices(graphs, train_idx)
            val_graphs = get_batch_graphs_from_indices(graphs, val_idx)
            test_graphs = get_batch_graphs_from_indices(graphs, test_idx)

            # Create DGL DataLoaders
            train_loader = get_graph_dataloader(
                train_graphs,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )

            val_loader = get_graph_dataloader(
                val_graphs,
                batch_size=batch_size,
                num_workers=num_workers
            )

            test_loader = get_graph_dataloader(
                test_graphs,
                batch_size=batch_size,
                num_workers=num_workers
            )

            logger.info(f"Created data loaders with {len(train_idx)} train, " +
                    f"{len(val_idx)} validation, {len(test_idx)} test samples")

            # Create class weights for handling imbalance
            class_weights = compute_class_weights(
                df,
                len(task_encoder.classes_),
                method=kwargs.get('class_weight_method', 'balanced')
            )

            # Move weights to device
            class_weights = class_weights.to(device)

            # Create loss function with class weights
            if 'use_multi_objective_loss' in kwargs and kwargs['use_multi_objective_loss']:
                # Use ProcessLoss for combined task, time, and structure prediction
                from processmine.models.gnn.architectures import ProcessLoss
                criterion = ProcessLoss(
                    task_weight=kwargs.get('task_weight', 0.5),
                    time_weight=kwargs.get('time_weight', 0.3),
                    structure_weight=kwargs.get('structure_weight', 0.2)
                )
            else:
                # Standard cross entropy loss
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

            # Create optimizer
            optimizer = create_optimizer(
                model_instance,
                optimizer_type="adamw",
                lr=lr,
                weight_decay=kwargs.get('weight_decay', 5e-4)
            )

            # Create scheduler
            lr_scheduler = create_lr_scheduler(
                optimizer,
                scheduler_type=kwargs.get('scheduler', 'cosine'),
                epochs=epochs,
                warmup_epochs=kwargs.get('warmup_epochs', 3),
                patience=kwargs.get('patience', 5)
            )

            # Train model
            model_path = output_dir / "models" / f"{model}_best.pt" if output_dir else None

            model_instance, metrics = train_model(
                model=model_instance,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epochs=epochs,
                patience=kwargs.get('patience', 5),
                model_path=model_path,
                use_amp=kwargs.get('use_amp', False),
                clip_grad_norm=kwargs.get('clip_grad'),
                lr_scheduler=lr_scheduler,
                memory_efficient=mem_efficient,
                track_memory=True
            )

            # Evaluate model
            logger.info("Evaluating model on test set...")
            eval_metrics, predictions, true_labels = evaluate_model(
                model_instance,
                test_loader,
                device=device,
                criterion=criterion,
                detailed=True
            )

            # Save training history and metrics if output directory provided
            if output_dir is not None:
                # Save training history
                with open(output_dir / "metrics" / "training_history.json", "w") as f:
                    history = {k: [float(v) for v in vals] for k, vals in metrics.items() if isinstance(vals, list)}
                    json.dump(history, f, indent=2)

                # Save evaluation metrics
                with open(output_dir / "metrics" / "model_metrics.json", "w") as f:
                    json.dump(eval_metrics, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, float)) else (int(o) if isinstance(o, (np.integer, int)) else str(o)))

            return {
                "model": model_instance,
                "metrics": eval_metrics,
                "history": metrics,
                "predictions": predictions,
                "true_labels": true_labels
            }

        elif model in ["lstm", "enhanced_lstm"]:
            # Create model for LSTM - use output dim from task encoder
            output_dim = len(task_encoder.classes_)

            # Update model_config with output dimension
            model_config['output_dim'] = output_dim
            model_config['num_cls'] = output_dim

            # Create the model with appropriate parameters
            model_instance = create_model(model, **model_config)

            # Prepare feature columns for sequence dataset
            feature_cols = [col for col in df.columns if col.startswith("feat_")]

            # Create sequence dataset
            from processmine.data.loader import create_sequence_dataset
            sequences, targets, seq_lengths = create_sequence_dataset(
                df,
                max_seq_len=kwargs.get('max_seq_len', 50),
                feature_cols=feature_cols
            )

            # Split data
            from sklearn.model_selection import train_test_split

            train_idx, temp_idx = train_test_split(
                range(len(sequences)),
                test_size=0.3,
                random_state=seed
            )

            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.5,
                random_state=seed
            )

            # Create dataset tensors
            from torch.utils.data import TensorDataset, DataLoader

            def create_tensor_dataloader(indices, batch_size, shuffle=False):
                batch_sequences = [sequences[i] for i in indices]
                batch_targets = [targets[i] for i in indices]
                batch_lengths = [seq_lengths[i] for i in indices]

                # Find max sequence length in this batch
                max_len = max(batch_lengths)

                # Pad sequences to same length if needed
                padded_seqs = []
                padded_targets = []

                for seq, target, length in zip(batch_sequences, batch_targets, batch_lengths):
                    if length < max_len:
                        padded_seq = torch.cat([
                            seq,
                            torch.zeros(max_len - length, seq.size(1), dtype=seq.dtype, device=seq.device)
                        ], dim=0)
                        padded_target = torch.cat([
                            target,
                            torch.zeros(max_len - length, dtype=target.dtype, device=target.device)
                        ], dim=0)
                    else:
                        padded_seq = seq
                        padded_target = target

                    padded_seqs.append(padded_seq)
                    padded_targets.append(padded_target)

                # Stack into tensors
                seq_tensor = torch.stack(padded_seqs)
                target_tensor = torch.stack(padded_targets)
                length_tensor = torch.tensor(batch_lengths)

                # Create dataset and dataloader
                dataset = TensorDataset(seq_tensor, target_tensor, length_tensor)
                return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

            # Create dataloaders
            train_loader = create_tensor_dataloader(train_idx, batch_size, shuffle=True)
            val_loader = create_tensor_dataloader(val_idx, batch_size)
            test_loader = create_tensor_dataloader(test_idx, batch_size)

            logger.info(f"Created sequence data loaders with {len(train_idx)} train, " +
                    f"{len(val_idx)} validation, {len(test_idx)} test samples")

            # Create optimizer
            optimizer = create_optimizer(
                model_instance,
                optimizer_type="adamw",
                lr=lr,
                weight_decay=kwargs.get('weight_decay', 5e-4)
            )

            # Create scheduler
            lr_scheduler = create_lr_scheduler(
                optimizer,
                scheduler_type=kwargs.get('scheduler', 'cosine'),
                epochs=epochs,
                warmup_epochs=kwargs.get('warmup_epochs', 3),
                patience=kwargs.get('patience', 5)
            )

            # Create loss function
            criterion = torch.nn.CrossEntropyLoss()

            # Define custom training function for sequence models
            def train_lstm_model(model, train_loader, val_loader, optimizer, criterion, device, epochs):
                # Training history
                history = {
                    'train_loss': [],
                    'val_loss': [],
                    'train_accuracy': [],
                    'val_accuracy': []
                }

                # Move model to device
                model.to(device)

                # Training loop
                for epoch in range(epochs):
                    # Training phase
                    model.train()
                    train_loss = 0.0
                    train_correct = 0
                    train_total = 0

                    for batch_idx, (sequences, targets, lengths) in enumerate(train_loader):
                        # Move to device
                        sequences = sequences.to(device)
                        targets = targets.to(device)
                        lengths = lengths.to(device)

                        # Zero gradients
                        optimizer.zero_grad()

                        # Forward pass
                        outputs = model(sequences, lengths)

                        # Get predictions
                        if isinstance(outputs, dict):
                            logits = outputs.get('task_pred', next(iter(outputs.values())))
                        else:
                            logits = outputs

                        # Calculate loss
                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                        # Backward pass
                        loss.backward()
                        optimizer.step()

                        # Track metrics
                        train_loss += loss.item()

                        # Calculate accuracy
                        _, predicted = torch.max(logits.view(-1, logits.size(-1)), 1)
                        mask = targets.view(-1) > 0  # Don't count padding
                        train_total += mask.sum().item()
                        train_correct += (predicted[mask] == targets.view(-1)[mask]).sum().item()

                    # Calculate average metrics
                    avg_train_loss = train_loss / len(train_loader)
                    train_accuracy = train_correct / max(train_total, 1) * 100

                    # Validation phase
                    model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0

                    with torch.no_grad():
                        for batch_idx, (sequences, targets, lengths) in enumerate(val_loader):
                            # Move to device
                            sequences = sequences.to(device)
                            targets = targets.to(device)
                            lengths = lengths.to(device)

                            # Forward pass
                            outputs = model(sequences, lengths)

                            # Get predictions
                            if isinstance(outputs, dict):
                                logits = outputs.get('task_pred', next(iter(outputs.values())))
                            else:
                                logits = outputs

                            # Calculate loss
                            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                            # Track metrics
                            val_loss += loss.item()

                            # Calculate accuracy
                            _, predicted = torch.max(logits.view(-1, logits.size(-1)), 1)
                            mask = targets.view(-1) > 0  # Don't count padding
                            val_total += mask.sum().item()
                            val_correct += (predicted[mask] == targets.view(-1)[mask]).sum().item()

                    # Calculate average metrics
                    avg_val_loss = val_loss / len(val_loader)
                    val_accuracy = val_correct / max(val_total, 1) * 100

                    # Update history
                    history['train_loss'].append(avg_train_loss)
                    history['val_loss'].append(avg_val_loss)
                    history['train_accuracy'].append(train_accuracy)
                    history['val_accuracy'].append(val_accuracy)

                    # Update scheduler
                    if lr_scheduler is not None:
                        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            lr_scheduler.step(avg_val_loss)
                        else:
                            lr_scheduler.step()

                    # Log progress
                    logger.info(f"Epoch {epoch+1}/{epochs}: "
                            f"train_loss={avg_train_loss:.4f}, "
                            f"val_loss={avg_val_loss:.4f}, "
                            f"train_acc={train_accuracy:.2f}%, "
                            f"val_acc={val_accuracy:.2f}%")

                return model, history

            # Train model
            model_instance, history = train_lstm_model(
                model_instance,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                device,
                epochs
            )

            # Evaluate model on test set
            model_instance.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch_idx, (sequences, targets, lengths) in enumerate(test_loader):
                    # Move to device
                    sequences = sequences.to(device)
                    targets = targets.to(device)
                    lengths = lengths.to(device)

                    # Forward pass
                    outputs = model_instance(sequences, lengths)

                    # Get predictions
                    if isinstance(outputs, dict):
                        logits = outputs.get('task_pred', next(iter(outputs.values())))
                    else:
                        logits = outputs

                    # Calculate loss
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                    # Track metrics
                    test_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(logits.view(-1, logits.size(-1)), 1)
                    mask = targets.view(-1) > 0  # Don't count padding
                    test_total += mask.sum().item()
                    test_correct += (predicted[mask] == targets.view(-1)[mask]).sum().item()

                    # Collect predictions and targets
                    all_preds.extend(predicted[mask].cpu().numpy())
                    all_targets.extend(targets.view(-1)[mask].cpu().numpy())

            # Calculate test metrics
            test_accuracy = test_correct / max(test_total, 1) * 100
            test_loss = test_loss / len(test_loader)

            # Calculate detailed metrics
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score, recall_score
            )

            eval_metrics = {
                'accuracy': accuracy_score(all_targets, all_preds),
                'f1_weighted': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
                'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
                'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }

            # Log results
            logger.info(f"Test results: "
                    f"loss={test_loss:.4f}, "
                    f"accuracy={test_accuracy:.2f}%, "
                    f"f1_weighted={eval_metrics['f1_weighted']:.4f}")

            # Save model and metrics if output directory provided
            if output_dir is not None:
                # Save model
                model_path = output_dir / "models" / f"{model}_best.pt"
                torch.save(model_instance.state_dict(), model_path)

                # Save history
                with open(output_dir / "metrics" / f"{model}_history.json", "w") as f:
                    json.dump(history, f, indent=2)

                # Save metrics
                with open(output_dir / "metrics" / f"{model}_metrics.json", "w") as f:
                    json.dump(eval_metrics, f, indent=2)

            return {
                "model": model_instance,
                "metrics": eval_metrics,
                "history": history,
                "predictions": np.array(all_preds),
                "true_labels": np.array(all_targets)
            }

        elif model in ["xgboost", "random_forest", "decision_tree"]:
            # Traditional ML models
            logger.info(f"Training {model} model...")

            # Prepare feature data
            feature_cols = [col for col in df.columns if col.startswith("feat_")]
            X = df[feature_cols].values
            y = df["next_task"].values

            # Ensure continuous class labels with LabelEncoder
            from sklearn.preprocessing import LabelEncoder
            y_mapped, class_mapping, reverse_mapping = ensure_continuous_labels(y)

            # Update class count for XGBoost
            if model == "xgboost":
                model_config["num_class"] = len(np.unique(y_mapped))

            # Split data
            from sklearn.model_selection import train_test_split

            X_train, X_temp, y_train, y_temp = train_test_split(X, y_mapped, test_size=0.3, random_state=seed)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

            # Create model
            model_instance = create_model(model, **model_config)
            
            # Store mapping in model for XGBoost
            if model == "xgboost":
                model_instance.class_mapping = class_mapping
                model_instance.reverse_mapping = reverse_mapping

            # Train model
            model_instance.fit(X_train, y_train)

            # Evaluate
            y_pred = model_instance.predict(X_test)

            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
                "precision_weighted": float(precision_score(y_test, y_pred, average="weighted")),
                "recall_weighted": float(recall_score(y_test, y_pred, average="weighted"))
            }

            logger.info(f"{model} model performance: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_weighted']:.4f}")

            # Save model and metrics if output directory provided
            if output_dir is not None:
                if model == "xgboost":
                    model_instance.save_model(str(output_dir / "models" / "xgboost_model.json"))
                else:
                    import pickle
                    with open(output_dir / "models" / f"{model}_model.pkl", "wb") as f:
                        pickle.dump(model_instance, f)

                # Save metrics
                with open(output_dir / "metrics" / "model_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

            return {
                "model": model_instance,
                "metrics": metrics,
                "predictions": y_pred,
                "true_labels": y_test,
                "class_mapping": class_mapping,
                "reverse_mapping": reverse_mapping
            }

        else:
            logger.error(f"Unknown model type: {model}")
            return {"error": f"Unknown model type: {model}"}

    except Exception as e:
        logger.error(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def run_optimization(
    data_path: str,
    output_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    rl_episodes: int = 30,
    rl_alpha: float = 0.1,
    rl_gamma: float = 0.9,
    rl_epsilon: float = 0.1,
    mem_efficient: bool = False,
    cache_dir: Optional[str] = None,
    analysis_results: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run process optimization with reinforcement learning

    Args:
        data_path: Path to process data CSV file
        output_dir: Directory to save results (None for no saving)
        device: Computing device to use
        rl_episodes: Number of RL training episodes
        rl_alpha: RL learning rate
        rl_gamma: RL discount factor
        rl_epsilon: RL exploration rate
        mem_efficient: Whether to use memory-efficient mode
        cache_dir: Directory to cache processed data
        analysis_results: Optional results from analysis step
        **kwargs: Additional arguments

    Returns:
        Dictionary with optimization results
    """
    # Set up device if provided as string
    if device is not None and isinstance(device, str):
        device = torch.device(device)

    # Set up output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        os.makedirs(output_dir / "policies", exist_ok=True)
        os.makedirs(output_dir / "visualizations", exist_ok=True)

    try:
        # Import necessary modules
        from processmine.data.loader import load_and_preprocess_data
        from processmine.process_mining.optimization import ProcessEnv, run_q_learning

        # Load data if not provided from analysis
        if analysis_results is None or "df" not in analysis_results:
            logger.info(f"Loading data from {data_path}")
            df, task_encoder, resource_encoder = load_and_preprocess_data(
                data_path,
                norm_method='l2',
                cache_dir=cache_dir,
                use_dtypes=True
            )
        else:
            df = analysis_results["df"]
            task_encoder = analysis_results["task_encoder"]
            resource_encoder = analysis_results["resource_encoder"]

        # Create environment with resource constraints
        logger.info("Creating process optimization environment...")
        env = ProcessEnv(
            df,
            task_encoder,
            resources=list(range(min(5, df["resource_id"].nunique())))
        )

        # Run Q-learning
        logger.info(f"Running Q-learning for {rl_episodes} episodes...")
        viz_dir = output_dir / "visualizations" if output_dir else None
        policy_dir = output_dir / "policies" if output_dir else None

        q_table = run_q_learning(
            env,
            episodes=rl_episodes,
            alpha=rl_alpha,
            gamma=rl_gamma,
            epsilon=rl_epsilon,
            viz_dir=viz_dir,
            policy_dir=policy_dir
        )

        return {
            "q_table": q_table,
            "env": env
        }

    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def run_full_pipeline(
    data_path: str,
    output_dir: Optional[Union[str, Path]] = None,
    model: str = "enhanced_gnn",
    device: Optional[Union[str, torch.device]] = None,
    seed: int = 42,
    mem_efficient: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Run complete process mining pipeline: analysis, training, and optimization

    Args:
        data_path: Path to process data CSV file
        output_dir: Directory to save results
        model: Model type for training
        device: Computing device to use
        seed: Random seed
        mem_efficient: Whether to use memory-efficient mode
        **kwargs: Additional arguments for individual steps

    Returns:
        Dictionary with pipeline results
    """
    # Set up output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        dgl.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    results = {}

    # Step 1: Analysis
    logger.info("Starting analysis phase...")
    analysis_results = run_analysis(
        data_path=data_path,
        output_dir=output_dir,
        device=device,
        mem_efficient=mem_efficient,
        **kwargs
    )

    results["analysis"] = analysis_results

    if "error" in analysis_results:
        logger.error(f"Analysis failed: {analysis_results['error']}")
        return results

    # Step 2: Training
    logger.info("Starting training phase...")
    training_results = run_training(
        data_path=data_path,
        model=model,
        output_dir=output_dir,
        device=device,
        mem_efficient=mem_efficient,
        seed=seed,
        analysis_results=analysis_results,
        **kwargs
    )

    results["training"] = training_results

    if "error" in training_results:
        logger.error(f"Training failed: {training_results['error']}")
        # Continue with optimization

    # Step 3: Optimization
    logger.info("Starting optimization phase...")
    optimization_results = run_optimization(
        data_path=data_path,
        output_dir=output_dir,
        device=device,
        mem_efficient=mem_efficient,
        analysis_results=analysis_results,
        **kwargs
    )

    results["optimization"] = optimization_results

    if "error" in optimization_results:
        logger.error(f"Optimization failed: {optimization_results['error']}")

    # Generate final report
    if output_dir is not None:
        logger.info("Creating final report...")
        try:
            from processmine.utils.reporting import generate_report
            report_path = generate_report(
                data_path=data_path,
                df=analysis_results.get("df"),
                model_type=model,
                metrics=training_results.get("metrics"),
                output_dir=output_dir
            )
            logger.info(f"Generated report at {report_path}")
            results["report_path"] = str(report_path)
        except Exception as e:
            logger.warning(f"Error generating final report: {e}")

    return results