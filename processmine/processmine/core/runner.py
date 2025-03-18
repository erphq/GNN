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
        from processmine.models.factory import create_model, get_model_config
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
        
        if model in ["gnn", "enhanced_gnn"]:
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
                        save_graphs(graph_save_path, graphs)
                        logger.info(f"Saved {len(graphs)} DGL graphs")
                    except Exception as e:
                        logger.warning(f"Failed to save graphs: {e}")
            
            # Create model with optimal parameters
            input_dim = len([col for col in df.columns if col.startswith("feat_")])
            model_kwargs = kwargs.copy()
            if 'input_dim' in model_kwargs:
                model_kwargs.pop('input_dim')
            if 'output_dim' in model_kwargs:
                model_kwargs.pop('output_dim')
            
            # Now call create_model with explicit parameters
            model_instance = create_model(
                                    model,
                                    input_dim=input_dim,  # Keep this if it exists in scope
                                    **kwargs  # Just pass through kwargs without adding output_dim
                                )
            
            # Prepare for training
            
            # Split indices with stratification for graph classification tasks
            try:
                # Extract GRAPH-LEVEL labels for stratification (common in DGL graph classification)
                # Adjust attribute name if your dataset uses different field (e.g., 'label', 'y', etc.)
                graph_labels = [g.ndata['graph_label'][0].item() for g in graphs]  # Common pattern if using node data for graph labels
                # Alternative if using proper graph attributes:
                # graph_labels = [g.graph['label'].item() for g in graphs]
                
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
            # LSTM models - implementation pending
            logger.warning("LSTM models not fully implemented in runner")
            return {"error": "LSTM models not fully implemented"}
            
        elif model in ["xgboost", "random_forest"]:
            # Traditional ML models
            logger.info(f"Training {model} model...")
            
            # Prepare feature data
            feature_cols = [col for col in df.columns if col.startswith("feat_")]
            X = df[feature_cols].values
            y = df["next_task"].values
            
            # Split data
            from sklearn.model_selection import train_test_split
            
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=seed)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
            
            # Create model with appropriate parameters
            model_instance = create_model(
                model,
                n_estimators=epochs if model == "xgboost" else 100,
                max_depth=kwargs.get('num_layers', 3) * 3 if model == "xgboost" else 10,
                learning_rate=lr if model == "xgboost" else 0.1,
                n_jobs=-1 if not mem_efficient else 1
            )
            
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
                "true_labels": y_test
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