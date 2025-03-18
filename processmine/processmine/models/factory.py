"""
Factory module for creating model instances with a consistent interface.
"""

import logging
from typing import Any, Dict, Optional, Union, List

logger = logging.getLogger(__name__)

def create_model(model_type: str, **kwargs) -> Any:
    """
    Factory function to create models with consistent interface

    Args:
        model_type: Type of model ('gnn', 'lstm', 'enhanced_gnn', 'xgboost', etc.)
        **kwargs: Model-specific parameters
            - input_dim: Input feature dimension
            - output_dim: Output dimension (required for neural models)
            - hidden_dim: Hidden layer dimension
            - num_layers: Number of layers
            - dropout: Dropout rate
            - attention_type: Attention mechanism ('basic', 'positional', 'diverse', 'combined')
            - use_positional_encoding: Whether to use positional encoding
            - use_diverse_attention: Whether to use diverse attention
            - diversity_weight: Weight for diversity loss
            - pos_enc_dim: Positional encoding dimension
            - pooling: Graph pooling method
            - predict_time: Whether to predict time in addition to task
            - use_batch_norm: Whether to use batch normalization
            - use_layer_norm: Whether to use layer normalization
            - use_residual: Whether to use residual connections
            - mem_efficient: Whether to use memory-efficient implementation

    Returns:
        Model instance
    """
    # Create a clean copy of kwargs to avoid modifying the original
    model_kwargs = kwargs.copy()
    
    # Check required parameters for neural models
    if model_type in ['gnn', 'enhanced_gnn', 'positional_gnn', 'diverse_gnn']:
        if 'input_dim' not in model_kwargs:
            raise ValueError(f"input_dim is required for {model_type} model")
        if 'output_dim' not in model_kwargs:
            raise ValueError(f"output_dim is required for {model_type} model")

    # Standardize parameter names
    # Map num_cls to output_dim for LSTM models if output_dim not already present
    if model_type in ['lstm', 'enhanced_lstm'] and 'num_cls' in model_kwargs and 'output_dim' not in model_kwargs:
        model_kwargs['output_dim'] = model_kwargs['num_cls']

    # Handle special parameters and enhancements for GNN models
    if model_type in ['gnn', 'enhanced_gnn', 'positional_gnn', 'diverse_gnn']:
        # Process attention_type parameter
        attention_type = model_kwargs.get('attention_type')

        # Handle use_positional_encoding override
        if 'use_positional_encoding' in model_kwargs:
            use_pos = model_kwargs.pop('use_positional_encoding')
            if use_pos and attention_type == 'basic':
                attention_type = 'positional'
                model_kwargs['attention_type'] = 'positional'
            elif not use_pos and attention_type in ['positional', 'combined']:
                # Remove positional component if explicitly disabled
                if attention_type == 'positional':
                    attention_type = 'basic'
                elif attention_type == 'combined':
                    attention_type = 'diverse'
                model_kwargs['attention_type'] = attention_type

        # Handle use_diverse_attention override
        if 'use_diverse_attention' in model_kwargs:
            use_diverse = model_kwargs.pop('use_diverse_attention')
            if use_diverse and attention_type == 'basic':
                attention_type = 'diverse'
                model_kwargs['attention_type'] = 'diverse'
            elif use_diverse and attention_type == 'positional':
                attention_type = 'combined'
                model_kwargs['attention_type'] = 'combined'
            elif not use_diverse and attention_type in ['diverse', 'combined']:
                # Remove diversity component if explicitly disabled
                if attention_type == 'diverse':
                    attention_type = 'basic'
                elif attention_type == 'combined':
                    attention_type = 'positional'
                model_kwargs['attention_type'] = attention_type

    # Create specific model based on type
    if model_type == 'gnn':
        from processmine.models.gnn.architectures import MemoryEfficientGNN
        return MemoryEfficientGNN(
            attention_type=model_kwargs.pop('attention_type', 'basic'),
            **model_kwargs
        )

    elif model_type == 'enhanced_gnn':
        from processmine.models.gnn.architectures import MemoryEfficientGNN
        return MemoryEfficientGNN(
            attention_type=model_kwargs.pop('attention_type', 'combined'),
            **model_kwargs
        )

    elif model_type == 'positional_gnn':
        from processmine.models.gnn.architectures import MemoryEfficientGNN
        return MemoryEfficientGNN(
            attention_type='positional',
            **model_kwargs
        )

    elif model_type == 'diverse_gnn':
        from processmine.models.gnn.architectures import MemoryEfficientGNN
        return MemoryEfficientGNN(
            attention_type='diverse',
            **model_kwargs
        )

    elif model_type == 'gat_layer':
        # Direct use of GAT layer
        layer_type = model_kwargs.pop('layer_type', 'basic')

        if layer_type == 'basic':
            from processmine.models.gnn.architectures import MemoryEfficientGATLayer
            return MemoryEfficientGATLayer(**model_kwargs)
        elif layer_type == 'positional':
            from processmine.models.gnn.architectures import PositionalGATLayer
            return PositionalGATLayer(**model_kwargs)
        elif layer_type == 'diverse':
            from processmine.models.gnn.architectures import DiverseGATLayer
            return DiverseGATLayer(**model_kwargs)
        elif layer_type == 'combined':
            from processmine.models.gnn.architectures import CombinedGATLayer
            return CombinedGATLayer(**model_kwargs)
        elif layer_type == 'expressive':
            from processmine.models.gnn.architectures import ExpressiveGATConv
            return ExpressiveGATConv(**model_kwargs)
        else:
            raise ValueError(f"Unknown GAT layer type: {layer_type}")

    elif model_type == 'lstm':
        from processmine.models.sequence.lstm import NextActivityLSTM
        
        # Create a clean copy of kwargs
        lstm_kwargs = kwargs.copy()
        
        # Ensure num_cls is set correctly (can come from either output_dim or num_cls)
        if 'output_dim' in lstm_kwargs and 'num_cls' not in lstm_kwargs:
            lstm_kwargs['num_cls'] = lstm_kwargs.pop('output_dim')
        
        # Remove any parameters not accepted by NextActivityLSTM
        valid_params = ['num_cls', 'emb_dim', 'hidden_dim', 'num_layers', 'dropout', 
                        'bidirectional', 'use_attention', 'use_layer_norm', 'mem_efficient']
        
        lstm_kwargs = {k: v for k, v in lstm_kwargs.items() if k in valid_params}
        
        # Check if num_cls is present
        if 'num_cls' not in lstm_kwargs:
            raise ValueError(f"num_cls or output_dim is required for {model_type} model")
        
        # Create the model with cleaned parameters
        return NextActivityLSTM(**lstm_kwargs)

    elif model_type == 'enhanced_lstm':
        from processmine.models.sequence.lstm import EnhancedProcessRNN
        
        # Create a clean copy of kwargs
        lstm_kwargs = kwargs.copy()
        
        # Ensure num_cls is set correctly (can come from either output_dim or num_cls)
        if 'output_dim' in lstm_kwargs and 'num_cls' not in lstm_kwargs:
            lstm_kwargs['num_cls'] = lstm_kwargs.pop('output_dim')
        
        # Remove any parameters not accepted by EnhancedProcessRNN
        valid_params = ['num_cls', 'emb_dim', 'hidden_dim', 'num_layers', 'dropout',
                        'use_gru', 'use_transformer', 'num_heads', 'use_time_features',
                        'time_encoding_dim', 'mem_efficient']
        
        lstm_kwargs = {k: v for k, v in lstm_kwargs.items() if k in valid_params}
        
        # Check if num_cls is present
        if 'num_cls' not in lstm_kwargs:
            raise ValueError(f"num_cls or output_dim is required for {model_type} model")
        
        # Create the model with cleaned parameters
        return EnhancedProcessRNN(**lstm_kwargs)

    elif model_type == 'mlp':
        from processmine.models.baseline.mlp import BasicMLP

        # Extract hidden_dims as a list from hidden_dim if not provided
        if 'hidden_dims' not in model_kwargs and 'hidden_dim' in model_kwargs:
            hidden_dim = model_kwargs.pop('hidden_dim')
            num_layers = model_kwargs.pop('num_layers', 2)
            hidden_dims = [hidden_dim] * num_layers
            model_kwargs['hidden_dims'] = hidden_dims

        return BasicMLP(**model_kwargs)

    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**model_kwargs)

    elif model_type == 'xgboost':
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(**model_kwargs)
        except ImportError:
            raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")

    elif model_type == 'decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**model_kwargs)

    elif model_type == 'process_loss':
        from processmine.models.gnn.architectures import ProcessLoss
        return ProcessLoss(**model_kwargs)

    elif model_type == 'positional_encoding':
        from processmine.models.gnn.architectures import PositionalEncoding
        return PositionalEncoding(**model_kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a model type
    
    Args:
        model_type: Model type string
        
    Returns:
        Default configuration dictionary
    """
    # Default configurations for different model types
    DEFAULT_CONFIGS = {
        'gnn': {
            'hidden_dim': 64,
            'num_layers': 2,
            'heads': 4,
            'dropout': 0.5,
            'attention_type': 'basic',
            'pos_enc_dim': 16,
            'diversity_weight': 0.1,
            'pooling': 'mean',
            'predict_time': False,
            'use_batch_norm': True,
            'use_residual': True,
            'use_layer_norm': False,
            'sparse_attention': False,
            'use_checkpointing': False,
            'mem_efficient': True
        },
        'enhanced_gnn': {
            'hidden_dim': 64,
            'num_layers': 2,
            'heads': 4,
            'dropout': 0.5,
            'attention_type': 'combined',
            'pos_enc_dim': 16,
            'diversity_weight': 0.1,
            'pooling': 'mean',
            'predict_time': False,
            'use_batch_norm': True,
            'use_residual': True,
            'use_layer_norm': False,
            'use_positional_encoding': True,
            'use_diverse_attention': True,
            'sparse_attention': False,
            'use_checkpointing': False,
            'mem_efficient': True
        },
        'positional_gnn': {
            'hidden_dim': 64,
            'num_layers': 2,
            'heads': 4,
            'dropout': 0.5,
            'attention_type': 'positional',
            'pos_enc_dim': 16,
            'pooling': 'mean',
            'predict_time': False,
            'use_batch_norm': True,
            'use_residual': True,
            'use_layer_norm': False,
            'use_positional_encoding': True,
            'use_diverse_attention': False,
            'mem_efficient': True
        },
        'diverse_gnn': {
            'hidden_dim': 64,
            'num_layers': 2,
            'heads': 4,
            'dropout': 0.5,
            'attention_type': 'diverse',
            'diversity_weight': 0.1,
            'pooling': 'mean',
            'predict_time': False,
            'use_batch_norm': True,
            'use_residual': True,
            'use_layer_norm': False,
            'use_positional_encoding': False,
            'use_diverse_attention': True,
            'mem_efficient': True
        },
        'lstm': {
            'hidden_dim': 64,
            'emb_dim': 64,
            'num_layers': 1,
            'dropout': 0.3,
            'bidirectional': False,
            'use_attention': True,
            'use_layer_norm': True,
            'mem_efficient': True
        },
        'enhanced_lstm': {
            'hidden_dim': 64,
            'emb_dim': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'use_gru': False,
            'use_transformer': True,
            'num_heads': 4,
            'use_time_features': True,
            'time_encoding_dim': 8,
            'mem_efficient': True
        },
        'mlp': {
            'hidden_dims': [128, 64],
            'dropout': 0.3,
            'activation': 'relu'
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'criterion': 'gini',
            'class_weight': 'balanced',
            'n_jobs': -1
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'multi:softmax'
        },
        'decision_tree': {
            'max_depth': 10,
            'min_samples_split': 5,
            'criterion': 'gini',
            'class_weight': 'balanced'
        },
        'process_loss': {
            'task_weight': 0.5,
            'time_weight': 0.3,
            'structure_weight': 0.2
        },
        'gat_layer': {
            'layer_type': 'basic',
            'num_heads': 4,
            'feat_drop': 0.5,
            'residual': True,
            'sparse_attention': False
        }
    }
    
    return DEFAULT_CONFIGS.get(model_type, {})