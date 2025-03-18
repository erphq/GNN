"""
High-performance training utilities adapted for DGL with mixed precision training,
gradient accumulation, memory-optimized batching, and systematic CUDA management.
"""
import torch
import os
import numpy as np
import time
import logging
import gc
import dgl
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    matthews_corrcoef, confusion_matrix
)

logger = logging.getLogger(__name__)


def _get_loss_display_value(loss, gradient_accumulation_steps):
    """Safely extract loss value for display, handling various loss formats"""
    if isinstance(loss, tuple):
        # If loss is a tuple, use the first element
        if isinstance(loss[0], torch.Tensor):
            return loss[0].item() * gradient_accumulation_steps
        return loss[0] * gradient_accumulation_steps
    elif isinstance(loss, torch.Tensor):
        # If loss is a tensor, use item() to get scalar value
        return loss.item() * gradient_accumulation_steps
    else:
        # If loss is already a scalar
        return loss * gradient_accumulation_steps

class MemoryTracker:
    """Utility class to track memory usage during training"""
    
    def __init__(self, logging_interval: int = 5, device: Optional[torch.device] = None):
        """
        Initialize memory tracker
        
        Args:
            logging_interval: How often to log memory usage (in iterations)
            device: Torch device to track
        """
        self.logging_interval = logging_interval
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_cuda = self.device.type == 'cuda'
        self.current_step = 0
        self.peak_memory = 0
        self.history = []
    
    def step(self, manual_log: bool = False):
        """
        Track memory for current step
        
        Args:
            manual_log: Whether to force logging regardless of interval
        """
        self.current_step += 1
        
        if self.is_cuda:
            current = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            self.peak_memory = max(self.peak_memory, peak)
            
            self.history.append(current)
            
            if manual_log or (self.current_step % self.logging_interval == 0):
                logger.debug(f"Step {self.current_step}: {current:.1f} MB, Peak: {peak:.1f} MB")
    
    def reset_peak(self):
        """Reset peak memory stats"""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def summary(self):
        """Print memory usage summary"""
        if self.is_cuda:
            current = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            peak = self.peak_memory
            logger.info(f"Memory usage - Current: {current:.1f} MB, Peak: {peak:.1f} MB")
            return {"current_mb": current, "peak_mb": peak}
        return {"current_mb": 0, "peak_mb": 0}

def clear_memory(full_clear: bool = False):
    """
    Free memory by clearing caches and unused objects
    
    Args:
        full_clear: Whether to perform more aggressive clearing
    """
    # Python garbage collection
    gc.collect()
    
    # PyTorch CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        if full_clear:
            # Force synchronization
            torch.cuda.synchronize()
            
            # Try more aggressive approach if available
            try:
                # Driver-level cache clear (only for more recent PyTorch versions)
                torch.cuda._sleep(2000)  # Wait for pending operations
                torch.cuda.empty_cache()
            except:
                pass

def train_model(
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Optional[Any] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[Any] = None,
    device: Optional[torch.device] = None,
    epochs: int = 10,
    patience: int = 5,
    model_path: Optional[str] = None,
    callback: Optional[Callable] = None,
    use_amp: bool = False,
    clip_grad_norm: Optional[float] = None,
    lr_scheduler: Optional[Any] = None,
    memory_efficient: bool = True,
    track_memory: bool = False,
    gradient_accumulation_steps: int = 1,
    eval_every: int = 1,
    metric_for_best_model: str = 'val_loss',
    greater_is_better: bool = False,
    early_stopping_threshold: float = 0.0001
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Unified training function with advanced optimization features, adapted for DGL models
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optimizer: PyTorch optimizer (default: AdamW)
        criterion: Loss function (default: CrossEntropyLoss)
        device: Computing device
        epochs: Number of training epochs
        patience: Early stopping patience
        model_path: Path to save best model
        callback: Optional callback function after each epoch
        use_amp: Whether to use automatic mixed precision
        clip_grad_norm: Max norm for gradient clipping (None to disable)
        lr_scheduler: Learning rate scheduler (None to disable)
        memory_efficient: Whether to use memory-efficient training
        track_memory: Whether to track memory usage
        gradient_accumulation_steps: Number of steps to accumulate gradients
        eval_every: Evaluate every N epochs
        metric_for_best_model: Metric to use for determining best model
        greater_is_better: Whether higher metric value is better
        early_stopping_threshold: Minimum change to be considered improvement
        
    Returns:
        Tuple of (trained model, training metrics)
    """
    # Set up device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Memory tracker
    mem_tracker = MemoryTracker() if track_memory else None
    
    # Move model to device
    model.to(device)
    
    # Enable JIT compilation for better performance if compatible model
    try:
        if hasattr(torch, 'jit') and not memory_efficient:
            # Only attempt JIT on non-memory-efficient mode
            # as it can increase memory usage initially
            model = torch.jit.script(model)
            logger.info("JIT compilation enabled")
    except Exception as e:
        logger.debug(f"JIT compilation not applicable: {e}")
    
    # Set up optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # Set up loss function if not provided
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Set up AMP if requested and available
    scaler = None
    if use_amp and device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        logger.info("Using automatic mixed precision training")
    
    # Initialize tracking variables
    best_metric_value = float('inf') if not greater_is_better else float('-inf')
    patience_counter = 0
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'epoch_times': []
    }
    best_model_state = None
    
    # Calculate total steps for logging
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    global_step = 0
    best_step = 0
    
    # Setup for gradient accumulation
    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
    
    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        # Use progress bar if tqdm available
        train_iter = _get_progress_bar(
            train_loader, f"Epoch {epoch}/{epochs} [Train]", 
            total=len(train_loader) // gradient_accumulation_steps
        )
        
        # Clear memory before epoch
        if memory_efficient:
            clear_memory()
        
        # Reset accumulated gradients
        accumulated_loss = 0.0
        
        for batch_idx, batch_graphs in enumerate(train_iter):
            # Move batch to device with memory optimization
            batch_graphs = batch_graphs.to(device)
            
            # Get labels from the graph
            # Import our utility function to get labels
            from processmine.utils.dataloader import get_graph_targets
            labels = get_graph_targets(batch_graphs)
            
            # Calculate loss normalization factor for accumulation
            accumulation_factor = 1.0 / gradient_accumulation_steps
            
            if use_amp and scaler is not None:
                # Forward pass with AMP
                with autocast():
                    outputs = model(batch_graphs)

                    # Handle dictionary outputs
                    if isinstance(outputs, dict):
                        logits = outputs.get("task_pred", next(iter(outputs.values())))

                        # Check for diversity loss
                        diversity_loss = outputs.get("diversity_loss", 0.0)
                        diversity_weight = outputs.get("diversity_weight", 0.1)

                        # Compute task loss
                        loss = criterion(logits, labels)

                        # Add diversity loss if available
                        if torch.is_tensor(diversity_loss) and diversity_loss.numel() > 0:
                            loss = loss + diversity_loss * diversity_weight
                    else:
                        # Normal output
                        loss = criterion(outputs, labels)

                    # Check if loss is a tuple
                    if isinstance(loss, tuple):
                        # Multiply only the tensor part of the tuple by the accumulation factor
                        loss = (loss[0] * accumulation_factor, loss[1])  # Scale for accumulation
                    else:
                        loss = loss * accumulation_factor  # Scale for accumulation

                # Backward pass with gradient scaling
                scaler.scale(loss[0] if isinstance(loss, tuple) else loss).backward()

                
                # Accumulate loss for reporting
                accumulated_loss += loss.item() * gradient_accumulation_steps
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping if enabled
                    if clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    
                    # Optimizer step with scaler
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Track loss
                    batch_size = batch_graphs.batch_size
                    train_loss += accumulated_loss * batch_size
                    train_samples += batch_size
                    accumulated_loss = 0.0
                    
                    # Update progress bar
                    if hasattr(train_iter, 'set_postfix'):
                        loss_display = _get_loss_display_value(loss, gradient_accumulation_steps)
                        train_iter.set_postfix({'loss': f"{loss_display:.4f}"})
                    
                    global_step += 1
            else:
                # Standard forward and backward pass
                outputs = model(batch_graphs)
                
                # Handle dictionary outputs
                if isinstance(outputs, dict):
                    logits = outputs.get("task_pred", next(iter(outputs.values())))
                    
                    # Check for diversity loss
                    diversity_loss = outputs.get("diversity_loss", 0.0)
                    diversity_weight = outputs.get("diversity_weight", 0.1)
                    
                    # Compute task loss
                    loss = criterion(logits, labels)
                    
                    # Extract loss value if it's a tuple
                    if isinstance(loss, tuple):
                        loss_value = loss[0]
                    else:
                        loss_value = loss
                    
                    # Add diversity loss if available
                    if torch.is_tensor(diversity_loss) and diversity_loss.numel() > 0:
                        combined_loss = loss_value + diversity_loss * diversity_weight
                        
                        # Reconstruct tuple if necessary
                        if isinstance(loss, tuple):
                            loss = (combined_loss, loss[1])
                        else:
                            loss = combined_loss
                else:
                    # Normal output
                    loss = criterion(outputs, labels)
                                    
                # Check if loss is a tuple
                if isinstance(loss, tuple):
                    # Multiply only the tensor part of the tuple by the accumulation factor
                    loss = (loss[0] * accumulation_factor, loss[1])  # Scale for accumulation
                    # Accumulate loss for reporting, use loss[0].item() to get the scalar value
                    accumulated_loss += loss[0].item() * gradient_accumulation_steps
                else:
                    loss = loss * accumulation_factor  # Scale for accumulation
                    # Accumulate loss for reporting, loss is already a scalar
                    accumulated_loss += loss.item() * gradient_accumulation_steps
                
                
                
                # Update weights every gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping if enabled
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Track loss
                    batch_size = batch_graphs.batch_size
                    train_loss += accumulated_loss * batch_size
                    train_samples += batch_size
                    accumulated_loss = 0.0
                    
                    # Update progress bar
                    if hasattr(train_iter, 'set_postfix'):
                        loss_display = _get_loss_display_value(loss, gradient_accumulation_steps)
                        train_iter.set_postfix({'loss': f"{loss_display:.4f}"})
                    
                    global_step += 1
            
            # Memory tracking
            if mem_tracker is not None:
                mem_tracker.step(batch_idx % 50 == 0)  # Log every 50 batches
            
            # Aggressive memory clearing for very large models
            if memory_efficient and batch_idx % 50 == 0:
                # Clear unnecessary memory
                del loss, outputs
                if torch.cuda.is_available():
                    # Don't empty cache too often as it can slow down training
                    if batch_idx % 200 == 0:
                        torch.cuda.empty_cache()
        
        # Calculate average training loss
        avg_train_loss = train_loss / max(train_samples, 1)
        metrics_history['train_loss'].append(avg_train_loss)
        
        # Update learning rate scheduler if provided
        if lr_scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            metrics_history['learning_rates'].append(current_lr)
            
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # This type needs validation loss
                if val_loader is not None:
                    # Will update after validation
                    pass
                else:
                    lr_scheduler.step(avg_train_loss)
            else:
                # Other schedulers update per epoch
                lr_scheduler.step()
        
        # Validation phase (if validation loader provided) every eval_every epochs
        val_loss = None
        val_metrics = {}
        
        if val_loader is not None and (epoch % eval_every == 0 or epoch == epochs):
            val_loss, val_metrics = evaluate_model(
                model, val_loader, 
                criterion, device, 
                memory_efficient=memory_efficient, 
                detailed=False,
                is_during_training=True
            )
            
            # Update validation metrics history
            metrics_history['val_loss'].append(val_loss)
            
            # Add other validation metrics to history
            for k, v in val_metrics.items():
                if k not in metrics_history:
                    metrics_history[k] = []
                metrics_history[k].append(v)
            
            # Update learning rate scheduler if it's ReduceLROnPlateau
            if lr_scheduler is not None and isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_loss)
            
            # Determine metric to track for model selection
            if metric_for_best_model == 'val_loss':
                current_metric = val_loss
            else:
                current_metric = val_metrics.get(metric_for_best_model, val_loss)
            
            # Check if this is the best model
            is_better = False
            if greater_is_better:
                is_better = current_metric > best_metric_value + early_stopping_threshold
            else:
                is_better = current_metric < best_metric_value - early_stopping_threshold
            
            # Early stopping check
            if is_better:
                best_metric_value = current_metric
                patience_counter = 0
                best_step = global_step
                
                # Save best model state efficiently
                if memory_efficient:
                    # Memory-efficient model saving (avoids cloning)
                    if model_path:
                        torch.save(model.state_dict(), model_path)
                        logger.info(f"Saved best model to {model_path}")
                    best_model_state = "saved_to_disk" if model_path else None
                else:
                    # Standard approach with state_dict copy
                    best_model_state = {
                        key: value.cpu().clone() 
                        for key, value in model.state_dict().items()
                    }
                    
                    # Save to disk if path provided
                    if model_path:
                        torch.save(best_model_state, model_path)
                        logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1
                logger.info(f"No improvement for {patience_counter}/{patience} epochs")
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        # Record epoch time
        epoch_time = time.time() - epoch_start
        metrics_history['epoch_times'].append(epoch_time)
        
        # Log epoch summary
        if val_loss is not None:
            logger.info(f"Epoch {epoch}/{epochs}: train_loss={avg_train_loss:.4f}, "
                       f"val_loss={val_loss:.4f}, time={epoch_time:.2f}s")
        else:
            logger.info(f"Epoch {epoch}/{epochs}: train_loss={avg_train_loss:.4f}, "
                       f"time={epoch_time:.2f}s")
        
        # Memory usage summary
        if mem_tracker is not None:
            mem_tracker.summary()
        
        # Call callback if provided
        if callback is not None:
            callback(epoch=epoch, model=model, train_loss=avg_train_loss,
                    val_loss=val_loss, metrics=val_metrics)
    
    # Restore best model if validation was used
    if val_loader is not None and best_model_state is not None and best_model_state != "saved_to_disk":
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from step {best_step}")
    elif val_loader is not None and best_model_state == "saved_to_disk" and model_path:
        model.load_state_dict(torch.load(model_path))
        logger.info(f"Restored best model from {model_path} (step {best_step})")
    
    # Final memory cleanup
    if memory_efficient:
        clear_memory(full_clear=True)
    
    # Add memory tracking data if available
    if mem_tracker is not None:
        metrics_history['memory'] = mem_tracker.summary()
    
    # Return model and metrics
    return model, metrics_history

def evaluate_model(
    model: torch.nn.Module,
    data_loader: Any,
    criterion: Optional[Any] = None,
    device: Optional[torch.device] = None,
    detailed: bool = True,
    memory_efficient: bool = True,
    is_during_training: bool = False
) -> Union[Tuple[Dict[str, Any], np.ndarray, np.ndarray], Tuple[float, Dict[str, Any]]]:
    """
    Evaluate model on test data with memory optimization.
    Unified function for both standalone evaluation and during-training evaluation.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: Test data loader
        criterion: Loss function (optional)
        device: Computing device
        detailed: Whether to compute detailed metrics
        memory_efficient: Whether to use memory-efficient evaluation
        is_during_training: Whether this is being called during training
        
    Returns:
        If is_during_training=False:
            Tuple of (metrics_dict, predictions, true_labels)
        If is_during_training=True:
            Tuple of (validation_loss, metrics_dict)
    """
    # Set up device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize result arrays
    all_preds = []
    all_labels = []
    test_loss = 0.0 if criterion is not None else None
    test_samples = 0
    
    # Import utility function for getting graph targets
    from processmine.utils.dataloader import get_graph_targets
    
    # Clear memory before evaluation
    if memory_efficient and torch.cuda.is_available():
        clear_memory()
    
    # Get progress bar
    progress_bar = _get_progress_bar(data_loader, "Evaluating")
    
    # Evaluate without gradient tracking
    with torch.no_grad():
        for batch_graphs in progress_bar:
            # Move batch to device
            batch_graphs = batch_graphs.to(device)
            
            # Get labels from the graph
            labels = get_graph_targets(batch_graphs)
            
            # Forward pass
            outputs = model(batch_graphs)
            
            # Handle dictionary outputs
            if isinstance(outputs, dict):
                logits = outputs.get("task_pred", next(iter(outputs.values())))
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Calculate loss if criterion provided
            if criterion is not None and labels is not None:
                loss = criterion(logits, labels)
                batch_size = batch_graphs.batch_size
                test_loss += loss[0].item() * batch_size
                test_samples += batch_size
            
            # Get predicted classes
            _, preds = torch.max(logits, dim=1)
            
            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            if labels is not None:
                all_labels.extend(labels.cpu().numpy())
            
            # Free batch memory for very large models
            if memory_efficient:
                del outputs
                if criterion is not None:
                    del loss
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = _calculate_metrics(all_labels, all_preds, detailed)
    
    # Add loss if calculated
    if test_loss is not None:
        avg_test_loss = test_loss / max(test_samples, 1) if test_samples > 0 else 0.0
        metrics['test_loss'] = avg_test_loss
    
    # Log summary of results if not during training
    if not is_during_training:
        logger.info(f"Evaluation results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        if 'mcc' in metrics:
            logger.info(f"  MCC: {metrics['mcc']:.4f}")
    
    # Final memory cleanup
    if memory_efficient and torch.cuda.is_available():
        clear_memory(full_clear=True)
    
    # Return different output formats based on context
    if is_during_training:
        return avg_test_loss if 'test_loss' in metrics else 0.0, metrics
    else:
        return metrics, all_preds, all_labels

def _calculate_metrics(true_labels, predictions, detailed=True):
    """Calculate performance metrics with proper handling of edge cases"""
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'f1_macro': f1_score(true_labels, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(true_labels, predictions, average='weighted', zero_division=0),
        'precision_macro': precision_score(true_labels, predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(true_labels, predictions, average='macro', zero_division=0)
    }
    
    # Add MCC for binary and multiclass (not multilabel)
    try:
        metrics['mcc'] = matthews_corrcoef(true_labels, predictions)
    except ValueError:
        # Skip MCC for incompatible data
        pass
    
    # Add detailed metrics if requested
    if detailed:
        # Class-wise metrics (per-class F1, precision, recall)
        unique_classes = np.unique(np.concatenate([true_labels, predictions]))
        
        # Check if we have a reasonable number of classes for detailed metrics
        if len(unique_classes) <= 100:  # Limit to avoid excessive output
            class_metrics = {}
            
            # Get per-class metrics
            f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)
            precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
            recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
            
            # Compile class metrics safely
            for i, cls in enumerate(unique_classes):
                if i < len(f1_per_class):
                    class_metrics[int(cls)] = {
                        'f1': float(f1_per_class[i]),
                        'precision': float(precision_per_class[i]),
                        'recall': float(recall_per_class[i])
                    }
            
            metrics['class_metrics'] = class_metrics
        
        # Confusion matrix - limit size for very large class counts
        if len(unique_classes) <= 100:
            metrics['confusion_matrix'] = confusion_matrix(true_labels, predictions).tolist()
    
    return metrics

def _get_progress_bar(iterable, desc, total=None):
    """Get progress bar for iteration with fallback"""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, leave=False, total=total)
    except ImportError:
        # Simple fallback - just return iterable
        print(f"{desc}...")
        return iterable

def compute_class_weights(df, num_classes, method='balanced'):
    """
    Compute class weights to handle imbalanced datasets with improved vectorization
    
    Args:
        df: Process data dataframe
        num_classes: Number of classes
        method: Weight calculation method ('balanced', 'log', 'sqrt', 'effective')
        
    Returns:
        Tensor of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight as sk_compute_class_weight
    
    # Extract labels efficiently
    labels = df["next_task"].values
    
    # Get unique classes
    unique_classes = np.unique(labels)
    
    # Create weight array (default to 1.0)
    weights = np.ones(num_classes, dtype=np.float32)
    
    # Compute class counts
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    
    # Compute weights based on method
    if method == 'balanced':
        # Use sklearn's balanced method
        class_weights = sk_compute_class_weight('balanced', classes=unique_classes, y=labels)
        weights[unique_classes] = class_weights
    
    elif method == 'log':
        # Log-based weighting (less aggressive than balanced)
        valid_counts = class_counts[class_counts > 0]
        
        # Log-based inverse weighting
        for cls in unique_classes:
            count = class_counts[cls]
            weights[cls] = np.log(total_samples / max(count, 1))
        
        # Normalize weights
        weights = weights / np.min(weights[weights > 0])
    
    elif method == 'sqrt':
        # Square root based weighting (even less aggressive)
        for cls in unique_classes:
            count = class_counts[cls]
            weights[cls] = np.sqrt(total_samples / max(count, 1))
        
        # Normalize weights
        weights = weights / np.min(weights[weights > 0])
    
    elif method == 'effective':
        # Effective number of samples weighting (works well for severe imbalance)
        beta = 0.9999
        for cls in unique_classes:
            count = class_counts[cls]
            # Effective number formula: (1 - beta^n) / (1 - beta)
            effective_num = (1.0 - beta ** count) / (1.0 - beta)
            weights[cls] = 1.0 / effective_num
        
        # Normalize weights
        weights = weights / np.min(weights[weights > 0])
    
    # Convert to tensor
    return torch.tensor(weights, dtype=torch.float32)

def create_optimizer(
    model: torch.nn.Module, 
    optimizer_type: str = 'adamw',
    lr: float = 0.001,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    layer_decay: Optional[float] = None,
    exclude_bn_bias: bool = True
) -> torch.optim.Optimizer:
    """
    Create optimizer with advanced configuration options
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        lr: Learning rate
        weight_decay: Weight decay factor
        momentum: Momentum factor (SGD only)
        layer_decay: Optional layer-wise learning rate decay factor
        exclude_bn_bias: Whether to exclude batch norm and bias from weight decay
        
    Returns:
        Configured PyTorch optimizer
    """
    # Define parameter groups with optimal defaults
    if exclude_bn_bias:
        # Exclude batch norm and bias from weight decay (better generalization)
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Skip batch norm and bias terms for weight decay
            if name.endswith('.bias') or 'norm' in name or 'bn' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Create parameter groups with different weight decay
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
    elif layer_decay is not None:
        # Layer-wise learning rate decay (for better fine-tuning)
        param_groups = []
        
        # Group parameters by layer depth
        layer_groups = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Extract layer depth from name
            depth = name.count('.')
            if depth not in layer_groups:
                layer_groups[depth] = []
            layer_groups[depth].append(param)
        
        # Create parameter groups with different learning rates
        max_depth = max(layer_groups.keys())
        for depth, params in layer_groups.items():
            # Calculate layer-specific learning rate
            layer_lr = lr * (layer_decay ** (max_depth - depth))
            param_groups.append({
                'params': params,
                'lr': layer_lr,
                'weight_decay': weight_decay
            })
    else:
        # Standard single parameter group
        param_groups = model.parameters()
    
    # Create optimizer based on type
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(param_groups, lr=lr)
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(param_groups, lr=lr)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'rmsprop':
        return torch.optim.RMSprop(param_groups, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    patience: int = 10,
    factor: float = 0.1
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler with improved configuration
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('cosine', 'step', 'plateau', 'linear', 'constant', 'one_cycle')
        epochs: Total epochs
        warmup_epochs: Epochs for linear warmup
        min_lr: Minimum learning rate
        patience: Patience for ReduceLROnPlateau
        factor: Reduction factor for step and plateau schedulers
        
    Returns:
        PyTorch learning rate scheduler
    """
    if scheduler_type == 'cosine':
        # Cosine decay with warmup
        if warmup_epochs > 0:
            # Linear warmup followed by cosine decay
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=warmup_epochs
            )
            
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=min_lr
            )
            
            # Chain schedulers
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            # Just cosine decay
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=min_lr
            )
    
    elif scheduler_type == 'step':
        # Step decay
        step_size = max(epochs // 3, 1)  # Default: 3 steps over the training
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=factor
        )
    
    elif scheduler_type == 'plateau':
        # Reduce on plateau
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
    
    elif scheduler_type == 'linear':
        # Linear decay
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr,
            total_iters=epochs
        )
    
    elif scheduler_type == 'one_cycle':
        # One cycle policy (generally better for short trainings)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'] * 10,
            total_steps=epochs,
            pct_start=0.3,
            anneal_strategy='cos',
            final_div_factor=1000
        )
    
    elif scheduler_type == 'constant':
        # Constant LR
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda epoch: 1.0
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across libraries
    
    Args:
        seed: Random seed
    """
    import random
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        # For deterministic behavior on CUDA
        # Note: This can impact performance negatively
        if hasattr(torch, 'set_deterministic'):
            torch.set_deterministic(True)
        
        # These settings may impact performance - enable only if strict reproducibility needed
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # DGL deterministic operations
    dgl.random.seed(seed)
    
    logger.info(f"Random seed set to {seed}")