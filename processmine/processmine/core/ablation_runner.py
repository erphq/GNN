"""
Integration of ablation study functionality with ProcessMine workflows.
Enables systematic testing of model components with DGL compatibility.
"""

import os
import torch
import numpy as np
import logging
import time
import dgl
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from processmine.core.ablation import AblationStudy
from processmine.utils.memory import log_memory_usage

logger = logging.getLogger(__name__)

def run_ablation_study(
    data_path: str,
    base_model: str = "enhanced_gnn",
    output_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
    mem_efficient: bool = True,
    ablation_config: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced ablation study runner that properly handles all improvement components
    
    Args:
        data_path: Path to process data CSV file
        base_model: Base model type ("gnn", "lstm", "enhanced_gnn", etc.)
        output_dir: Directory to save results (None for no saving)
        device: Computing device to use
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        seed: Random seed
        mem_efficient: Whether to use memory-efficient mode
        ablation_config: Configuration for ablation study (components to test, etc.)
        cache_dir: Directory to cache processed data
        **kwargs: Additional model-specific arguments
        
    Returns:
        Dictionary with ablation study results
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set DGL random seed
    import dgl
    dgl.random.seed(seed)
    
    # Set up device if provided as string
    if device is not None and isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        os.makedirs(output_dir / "ablation_study", exist_ok=True)
    
    try:
        # Import necessary modules
        from processmine.data.loader import load_and_preprocess_data
        from processmine.data.graphs import build_graph_data
        from processmine.utils.memory import log_memory_usage
        from processmine.core.ablation import AblationStudy
        from processmine.models.factory import get_model_config
        from processmine.utils.dataloader import adaptive_normalization
        
        # Set up base configuration with default values for all components
        base_config = {
            # Add default values for all components from the improvement plan
            'use_positional_encoding': True,
            'use_diverse_attention': True,
            'attention_type': 'combined',
            'diversity_weight': kwargs.get('diversity_weight', 0.1),
            'pos_enc_dim': kwargs.get('pos_enc_dim', 16),
            'use_residual': True,
            'use_batch_norm': True,
            'use_layer_norm': False,
            'use_adaptive_normalization': True,
            'use_multi_objective_loss': True,
            'predict_time': True  # Enable time prediction for multi-objective loss
        }
        
        # Merge with model-specific config
        model_config = get_model_config(base_model)
        base_config.update(model_config)
        
        # Override with any user-provided kwargs
        base_config.update(kwargs)
        
        # Load data - determine normalization method based on config
        logger.info(f"Loading data from {data_path}")
        if base_config.get('use_adaptive_normalization', True):
            # Adaptive normalization is handled in advanced_workflow
            norm_method = None  # Using None triggers adaptive normalization later
        else:
            # Use standard L2 normalization
            norm_method = 'l2'
            
        df, task_encoder, resource_encoder = load_and_preprocess_data(
            data_path,
            norm_method=norm_method,
            cache_dir=cache_dir,
            use_dtypes=True
        )
        
        # If using adaptive normalization, apply it now
        if base_config.get('use_adaptive_normalization', True) and norm_method is None:
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
        
        # Store the normalization decision for reference
        base_config['used_adaptive_normalization'] = (norm_method is None)
        
        # Log memory usage
        log_memory_usage()
        
        # Build graph data with DGL
        logger.info("Building graph data with DGL...")
        graphs = build_graph_data(
            df,
            enhanced=(base_model == "enhanced_gnn"),
            batch_size=500 if not mem_efficient else 100,
            num_workers=kwargs.get('num_workers', 0),
            bidirectional=True,
            use_edge_features=kwargs.get('use_edge_features', True),
            verbose=True
        )
        
        # Split data for training, validation, and testing
        from sklearn.model_selection import train_test_split
        
        # Extract labels for stratification if possible
        try:
            graph_labels = [g.ndata['label'][0].item() if 'label' in g.ndata else 0 for g in graphs]
            
            train_idx, temp_idx = train_test_split(
                range(len(graphs)), 
                test_size=0.3, 
                random_state=seed,
                stratify=graph_labels
            )
            val_idx, test_idx = train_test_split(
                temp_idx, 
                test_size=0.5, 
                random_state=seed,
                stratify=[graph_labels[i] for i in temp_idx]
            )
        except Exception as e:
            # Fallback to random split
            logger.warning(f"Stratified split failed, using random split: {e}")
            indices = np.arange(len(graphs))
            np.random.shuffle(indices)
            train_idx = indices[:int(0.7 * len(indices))]
            val_idx = indices[int(0.7 * len(indices)):int(0.85 * len(indices))]
            test_idx = indices[int(0.85 * len(indices)):]
        
        # Update base configuration with common training parameters
        base_config.update({
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'seed': seed,
            'train_idx': train_idx.tolist(),  # Convert to list for serialization
            'val_idx': val_idx.tolist(),
            'test_idx': test_idx.tolist(),
            'input_dim': len([col for col in df.columns if col.startswith("feat_")]),
            'output_dim': len(task_encoder.classes_),
            'mem_efficient': mem_efficient,
        })
        
        # Initialize ablation study
        study = AblationStudy(
            base_config=base_config,
            output_dir=output_dir,
            experiment_name=f"ablation_{base_model}",
            device=device,
            save_models=True,
            parallel=kwargs.get('parallel', False),
            max_workers=kwargs.get('max_workers', 4),
            random_seeds=[seed, seed+1, seed+2] if kwargs.get('multiple_seeds', False) else [seed]
        )
        
        # Add ablation experiments based on configuration
        if ablation_config is None:
            # Default ablation: test key components from improvement plan
            study.add_ablation_experiments(
                components=[
                    'use_positional_encoding',
                    'use_diverse_attention',
                    'use_residual',
                    'use_batch_norm',
                    'use_layer_norm',
                    'use_adaptive_normalization',  # Include adaptive normalization
                    'use_multi_objective_loss'     # Include multi-objective loss
                ],
                disable=True  # Test by disabling each component
            )
        else:
            # Custom ablation configuration
            if 'components' in ablation_config:
                study.add_ablation_experiments(
                    components=ablation_config['components'],
                    disable=ablation_config.get('disable', True),
                    include_combinations=ablation_config.get('include_combinations', False)
                )
            
            if 'grid_search' in ablation_config:
                study.add_grid_search(ablation_config['grid_search'])
        
        # Always add a baseline experiment
        study.add_experiment("baseline", {})
        
        # Define run function for each experiment
        def run_experiment(config, device):
            # Create model
            from processmine.models.factory import create_model
            model_type = config.get('model_type', base_model)
            model = create_model(
                model_type,
                input_dim=config['input_dim'],
                output_dim=config['output_dim'],
                **{k: v for k, v in config.items() if k not in 
                   ['input_dim', 'output_dim', 'model_type', 'epochs', 'batch_size', 
                    'lr', 'seed', 'train_idx', 'val_idx', 'test_idx', 'mem_efficient',
                    'used_adaptive_normalization']}
            )
            
            # Get data indices
            train_idx = np.array(config['train_idx'])
            val_idx = np.array(config['val_idx'])
            test_idx = np.array(config['test_idx'])
            
            # Create data loaders
            from processmine.utils.dataloader import get_graph_dataloader, get_batch_graphs_from_indices
            
            train_graphs = get_batch_graphs_from_indices(graphs, train_idx)
            val_graphs = get_batch_graphs_from_indices(graphs, val_idx)
            test_graphs = get_batch_graphs_from_indices(graphs, test_idx)
            
            # Use MemoryEfficientDataLoader if memory efficiency is enabled
            if config.get('mem_efficient', True):
                from processmine.utils.dataloader import MemoryEfficientDataLoader
                
                train_loader = MemoryEfficientDataLoader(
                    train_graphs,
                    batch_size=config['batch_size'],
                    shuffle=True,
                    memory_threshold=0.85
                )
                
                val_loader = MemoryEfficientDataLoader(
                    val_graphs,
                    batch_size=config['batch_size'],
                    memory_threshold=0.85
                )
                
                test_loader = MemoryEfficientDataLoader(
                    test_graphs,
                    batch_size=config['batch_size'],
                    memory_threshold=0.85
                )
            else:
                # Use standard graph dataloader
                train_loader = get_graph_dataloader(
                    train_graphs,
                    batch_size=config['batch_size'],
                    shuffle=True,
                    num_workers=config.get('num_workers', 0)
                )
                
                val_loader = get_graph_dataloader(
                    val_graphs,
                    batch_size=config['batch_size'],
                    num_workers=config.get('num_workers', 0)
                )
                
                test_loader = get_graph_dataloader(
                    test_graphs,
                    batch_size=config['batch_size'],
                    num_workers=config.get('num_workers', 0)
                )
            
            # Create class weights
            from processmine.core.training import compute_class_weights
            class_weights = compute_class_weights(
                df, 
                config['output_dim'],
                method=config.get('class_weight_method', 'balanced')
            ).to(device)
            
            # Use appropriate loss function based on configuration
            if config.get('use_multi_objective_loss', False):
                # Use ProcessLoss for multi-objective loss
                from processmine.models.gnn.architectures import ProcessLoss
                criterion = ProcessLoss(
                    task_weight=config.get('task_weight', 0.5),
                    time_weight=config.get('time_weight', 0.3),
                    structure_weight=config.get('structure_weight', 0.2)
                )
            else:
                # Use standard cross entropy loss
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            
            # Create optimizer
            from processmine.core.training import create_optimizer, create_lr_scheduler
            optimizer = create_optimizer(
                model,
                optimizer_type=config.get('optimizer', 'adamw'),
                lr=config['lr'],
                weight_decay=config.get('weight_decay', 5e-4)
            )
            
            # Create scheduler
            lr_scheduler = create_lr_scheduler(
                optimizer,
                scheduler_type=config.get('scheduler', 'cosine'),
                epochs=config['epochs'],
                warmup_epochs=config.get('warmup_epochs', 3),
                patience=config.get('patience', 5)
            )
            
            # Train model
            from processmine.core.training import train_model, evaluate_model
            model_path = None
            if 'output_dir' in config:
                model_path = Path(config['output_dir']) / "model_best.pt"
            
            model, metrics = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epochs=config['epochs'],
                patience=config.get('patience', 5),
                model_path=model_path,
                use_amp=config.get('use_amp', False),
                clip_grad_norm=config.get('clip_grad'),
                lr_scheduler=lr_scheduler,
                memory_efficient=config.get('mem_efficient', True),
                track_memory=True
            )
            
            # Evaluate model
            eval_metrics, _, _ = evaluate_model(
                model,
                test_loader,
                device=device,
                criterion=criterion,
                detailed=True
            )
            
            # Combine training and evaluation metrics
            result = {
                'val_loss': min(metrics.get('val_loss', [float('inf')])),
                'val_acc': max(metrics.get('val_accuracy', [0.0])),
                'test_acc': eval_metrics.get('accuracy', 0.0),
                'test_f1': eval_metrics.get('f1_weighted', 0.0),
                'training_time': sum(metrics.get('epoch_times', [0.0]))
            }
            
            return result
        
        # Run ablation study
        results = study.run_experiments(
            run_function=run_experiment,
            aggregate_metrics=['test_acc', 'test_f1', 'val_loss', 'training_time']
        )
        
        return {
            'study': study,
            'results': results,
            'base_config': base_config,
            'ablation_config': ablation_config,
            'model_counts': {
                'train': len(train_idx),
                'val': len(val_idx),
                'test': len(test_idx)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in ablation study: {e}")
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