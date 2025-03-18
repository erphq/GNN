"""
Advanced workflow for ProcessMine integrating all improvements from the enhancement plan.
This module provides a comprehensive workflow combining positional encoding, diverse attention,
adaptive normalization, and multi-objective loss functions with DGL-optimized implementation.
"""

import os
import torch
import numpy as np
import logging
import time
import dgl
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)

def run_advanced_workflow(
    data_path: str,
    output_dir: Optional[Union[str, Path]] = None,
    model: str = "enhanced_gnn",
    device: Optional[Union[str, torch.device]] = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
    mem_efficient: bool = True,
    use_positional_encoding: bool = True,
    use_diverse_attention: bool = True,
    use_multi_objective_loss: bool = True,
    use_adaptive_normalization: bool = True,
    cache_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run advanced ProcessMine workflow with all improvements from the enhancement plan

    Args:
        data_path: Path to process data CSV file
        output_dir: Directory to save results (None for no saving)
        model: Model type (default: "enhanced_gnn")
        device: Computing device to use
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        seed: Random seed
        mem_efficient: Whether to use memory-efficient mode
        use_positional_encoding: Whether to use positional encoding
        use_diverse_attention: Whether to use diverse attention mechanism
        use_multi_objective_loss: Whether to use multi-objective loss
        use_adaptive_normalization: Whether to use adaptive normalization
        cache_dir: Directory to cache processed data
        **kwargs: Additional model-specific arguments

    Returns:
        Dictionary with workflow results
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
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "models", exist_ok=True)
        os.makedirs(output_dir / "metrics", exist_ok=True)

    try:
        # Import necessary modules
        from processmine.data.loader import load_and_preprocess_data
        from processmine.data.graphs import build_graph_data
        from processmine.utils.memory import log_memory_usage
        from processmine.utils.dataloader import adaptive_normalization
        from processmine.models.factory import create_model, get_model_config

        # Log start of workflow
        logger.info("Starting advanced ProcessMine workflow")
        start_time = time.time()

        # Load and preprocess data
        logger.info(f"Loading data from {data_path}")
        df, task_encoder, resource_encoder = load_and_preprocess_data(
            data_path,
            norm_method='l2' if not use_adaptive_normalization else None,
            cache_dir=cache_dir,
            use_dtypes=True,
            memory_limit_gb=8.0 if not mem_efficient else 2.0
        )

        # Apply adaptive normalization if enabled
        if use_adaptive_normalization:
            logger.info("Applying adaptive normalization")
            feature_cols = [col for col in df.columns if col.startswith("feat_")]
            features = df[feature_cols].values

            # Calculate feature statistics
            feature_statistics = {
                'mean': np.mean(features, axis=0),
                'std': np.std(features, axis=0),
                'min': np.min(features, axis=0),
                'max': np.max(features, axis=0),
                'skewness': _calculate_skewness(features)
            }

            # Apply normalization
            normalized_features = adaptive_normalization(features, feature_statistics)

            # Update dataframe with normalized features
            for i, col in enumerate(feature_cols):
                df[col] = normalized_features[:, i]

        # Log memory usage after data loading
        log_memory_usage()

        # Build graph data with enhanced features
        logger.info("Building graph data with DGL...")
        graphs = build_graph_data(
            df,
            enhanced=True,  # Always use enhanced graph features
            batch_size=500 if not mem_efficient else 100,
            num_workers=kwargs.get('num_workers', 0),
            bidirectional=True,
            use_edge_features=True,
            verbose=True
        )

        # Split indices for training, validation, and testing
        from sklearn.model_selection import train_test_split

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

        # Create model configuration with advanced features
        model_config = get_model_config(model)

        # Update with advanced features from improvement plan
        model_config.update({
            'use_positional_encoding': use_positional_encoding,
            'use_diverse_attention': use_diverse_attention,
            'attention_type': 'combined' if use_diverse_attention else 'basic',
            'diversity_weight': kwargs.get('diversity_weight', 0.1),
            'pos_enc_dim': kwargs.get('pos_enc_dim', 16),
            'use_residual': kwargs.get('use_residual', True),
            'use_batch_norm': kwargs.get('use_batch_norm', True),
            'use_layer_norm': kwargs.get('use_layer_norm', False),
            'predict_time': use_multi_objective_loss,  # Enable time prediction for multi-objective loss
        })

        # Update with user-provided kwargs
        model_config.update(kwargs)

        # Create model
        input_dim = len([col for col in df.columns if col.startswith("feat_")])
        model_instance = create_model(
            model,
            input_dim=input_dim,
            output_dim=len(task_encoder.classes_),
            **model_config
        )

        # Create data loaders with memory efficiency
        from processmine.utils.dataloader import get_batch_graphs_from_indices

        # Get train/val/test graphs
        train_graphs = get_batch_graphs_from_indices(graphs, train_idx)
        val_graphs = get_batch_graphs_from_indices(graphs, val_idx)
        test_graphs = get_batch_graphs_from_indices(graphs, test_idx)

        # Use memory-efficient dataloader if enabled
        if mem_efficient:
            from processmine.utils.dataloader import MemoryEfficientDataLoader

            train_loader = MemoryEfficientDataLoader(
                train_graphs,
                batch_size=batch_size,
                shuffle=True,
                memory_threshold=0.85
            )

            val_loader = MemoryEfficientDataLoader(
                val_graphs,
                batch_size=batch_size,
                memory_threshold=0.85
            )

            test_loader = MemoryEfficientDataLoader(
                test_graphs,
                batch_size=batch_size,
                memory_threshold=0.85
            )
        else:
            # Use standard graph dataloader
            from processmine.utils.dataloader import get_graph_dataloader

            train_loader = get_graph_dataloader(
                train_graphs,
                batch_size=batch_size,
                shuffle=True,
                num_workers=kwargs.get('num_workers', 0)
            )

            val_loader = get_graph_dataloader(
                val_graphs,
                batch_size=batch_size,
                num_workers=kwargs.get('num_workers', 0)
            )

            test_loader = get_graph_dataloader(
                test_graphs,
                batch_size=batch_size,
                num_workers=kwargs.get('num_workers', 0)
            )

        # Create class weights
        from processmine.core.training import compute_class_weights
        class_weights = compute_class_weights(
            df,
            len(task_encoder.classes_),
            method=kwargs.get('class_weight_method', 'balanced')
        )

        # Move weights to device
        class_weights = class_weights.to(device)

        # Create loss function based on configuration
        if use_multi_objective_loss:
            # Use ProcessLoss for multi-objective loss
            from processmine.models.gnn.architectures import ProcessLoss
            criterion = ProcessLoss(
                task_weight=kwargs.get('task_weight', 0.5),
                time_weight=kwargs.get('time_weight', 0.3),
                structure_weight=kwargs.get('structure_weight', 0.2)
            )
        else:
            # Use standard cross entropy loss
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Create optimizer
        from processmine.core.training import create_optimizer, create_lr_scheduler
        optimizer = create_optimizer(
            model_instance,
            optimizer_type=kwargs.get('optimizer', 'adamw'),
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

        # Define model path
        model_path = output_dir / "models" / f"{model}_best.pt" if output_dir else None

        # Train model
        from processmine.core.training import train_model, evaluate_model
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
                import json
                history = {k: [float(v) for v in vals] for k, vals in metrics.items() if isinstance(vals, list)}
                json.dump(history, f, indent=2)

            # Save evaluation metrics
            with open(output_dir / "metrics" / "model_metrics.json", "w") as f:
                import json
                json.dump(eval_metrics, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, float)) else (int(o) if isinstance(o, (np.integer, int)) else str(o)))

        # Log completion
        workflow_time = time.time() - start_time
        logger.info(f"Advanced workflow completed in {workflow_time:.2f}s")

        return {
            "model": model_instance,
            "metrics": eval_metrics,
            "history": metrics,
            "predictions": predictions,
            "true_labels": true_labels,
            "training_time": workflow_time,
            "config": {
                "model": model,
                "input_dim": input_dim,
                "output_dim": len(task_encoder.classes_),
                "use_positional_encoding": use_positional_encoding,
                "use_diverse_attention": use_diverse_attention,
                "use_multi_objective_loss": use_multi_objective_loss,
                "use_adaptive_normalization": use_adaptive_normalization,
            }
        }

    except Exception as e:
        logger.error(f"Error in advanced workflow: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}        

def _calculate_skewness(arr):
    """Calculate skewness of array elements along first axis"""
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    # Avoid division by zero
    std = np.maximum(std, 1e-8)
    
    # Calculate skewness (third moment)
    n = arr.shape[0]
    m3 = np.sum((arr - mean)**3, axis=0) / n
    return m3 / (std**3)


'''"""
Enhanced integration of ablation runner with the main workflow.
This ensures all improvement plan components can be tested properly.
"""

import os
import torch
import numpy as np
import logging
import time
import dgl
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)

def run_comprehensive_ablation(
    data_path: str,
    base_model: str = "enhanced_gnn",
    output_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    epochs: int = 10,  # Reduced epochs for faster ablation study
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
    mem_efficient: bool = True,
    components_to_test: Optional[List[str]] = None,
    include_combinations: bool = False,
    test_mode: str = "disable",  # 'disable' or 'enable'
    max_workers: int = 0,  # 0 for sequential, >0 for parallel
    cache_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run comprehensive ablation study to evaluate all components from the improvement plan
    
    Args:
        data_path: Path to process data CSV file
        base_model: Base model type ("gnn", "lstm", "enhanced_gnn", etc.)
        output_dir: Directory to save results (None for no saving)
        device: Computing device to use
        epochs: Number of training epochs (reduced for ablation)
        batch_size: Batch size for training
        lr: Learning rate
        seed: Random seed
        mem_efficient: Whether to use memory-efficient mode
        components_to_test: List of components to test (None for all)
        include_combinations: Whether to test combinations of components
        test_mode: How to test components ('disable' or 'enable')
        max_workers: Number of workers for parallel execution
        cache_dir: Directory to cache processed data
        **kwargs: Additional model-specific arguments
        
    Returns:
        Dictionary with ablation study results
    """
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set DGL random seed
    dgl.random.seed(seed)
    
    # Set up device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Set up output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        os.makedirs(output_dir / "ablation_study", exist_ok=True)
    
    # Import necessary modules
    from processmine.data.loader import load_and_preprocess_data
    from processmine.utils.memory import log_memory_usage
    from processmine.core.ablation import AblationStudy
    from processmine.core.ablation_runner import run_ablation_study
    
    logger.info(f"Starting comprehensive ablation study for {base_model} model")
    
    # Define default components if not provided
    if components_to_test is None:
        components_to_test = [
            "use_positional_encoding",  # Graph position encoding
            "use_diverse_attention",     # Attention diversity mechanism
            "use_batch_norm",           # Batch normalization
            "use_residual",             # Residual connections
            "use_layer_norm",           # Layer normalization
            "use_adaptive_normalization", # Adaptive feature normalization
            "use_multi_objective_loss"    # Multi-objective loss function
        ]

    # Create ablation configuration
    ablation_config = {
        "components": components_to_test,
        "disable": test_mode == "disable",
        "include_combinations": include_combinations
    }
    
    # Add grid search for diversity_weight if testing diverse attention
    if "use_diverse_attention" in components_to_test and kwargs.get("test_diversity_weights", False):
        ablation_config["grid_search"] = {
            "diversity_weight": [0.05, 0.1, 0.2]
        }
    
    # Run the ablation study
    results = run_ablation_study(
        data_path=data_path,
        base_model=base_model,
        output_dir=output_dir / "ablation_study" if output_dir else None,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        mem_efficient=mem_efficient,
        ablation_config=ablation_config,
        cache_dir=cache_dir,
        parallel=max_workers > 0,
        max_workers=max_workers,
        **kwargs
    )
    
    # Create ablation summary
    if output_dir is not None and "results" in results:
        _create_ablation_summary(results, output_dir / "ablation_study")
    
    return results

def _create_ablation_summary(results: Dict[str, Any], output_dir: Path) -> None:
    """Create human-readable summary of ablation results"""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results into a DataFrame for easier analysis
    data = []
    component_impacts = {}
    
    for exp_name, exp_results in results.get("results", {}).items():
        row = {"experiment": exp_name}
        
        # Extract modifications
        mods = exp_results.get("modifications", {})
        for comp, value in mods.items():
            row[comp] = value
        
        # Extract metrics
        for metric in ["test_acc", "test_f1", "val_loss", "training_time"]:
            if f"{metric}_mean" in exp_results:
                row[metric] = exp_results[f"{metric}_mean"]
        
        data.append(row)
        
        # Estimate component impact
        if "baseline" in results.get("results", {}):
            baseline = results["results"]["baseline"]
            if len(mods) == 1 and "test_acc_mean" in baseline and "test_acc_mean" in exp_results:
                comp = list(mods.keys())[0]
                impact = baseline["test_acc_mean"] - exp_results["test_acc_mean"]
                component_impacts[comp] = impact
    
    # Create DataFrame
    if data:
        df = pd.DataFrame(data)
        csv_path = output_dir / "ablation_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved ablation results to {csv_path}")
    
    # Create component impact visualization
    if component_impacts:
        plt.figure(figsize=(10, 6))
        components = list(component_impacts.keys())
        impacts = [component_impacts[c] for c in components]
        
        # Sort by absolute impact
        sorted_indices = np.argsort(np.abs(impacts))[::-1]
        components = [components[i] for i in sorted_indices]
        impacts = [impacts[i] for i in sorted_indices]
        
        # Create bar chart
        bars = plt.bar(components, impacts, alpha=0.7)
        
        # Color bars by sign (positive impact is good when disabling hurts performance)
        for i, v in enumerate(impacts):
            if v < 0:
                bars[i].set_color('green')  # Negative impact (disabling hurts, component is useful)
            else:
                bars[i].set_color('red')    # Positive impact (disabling helps, component may not be useful)
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel("Component")
        plt.ylabel("Impact on Performance (disabling)")
        plt.title("Component Importance Analysis")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_dir / "component_impact.png", dpi=100)
        logger.info(f"Saved component impact visualization to {output_dir / 'component_impact.png'}")'''
        