#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced ProcessMine example demonstrating all improvements from the enhancement plan.
This example shows:
1. Enhanced Graph Representation with PositionalGATConv 
2. Memory Management Enhancements with MemoryEfficientDataLoader
3. Normalization Reconciliation with adaptive_normalization
4. Enhanced GNN with Expressivity Improvements
5. Multi-Head Attention with Diversity Mechanism
6. Multi-Objective Loss Function
7. Comprehensive Ablation Study
"""

import os
import argparse
import torch
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger("processmine_example")

def run_example():
    """Run the complete advanced ProcessMine example"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced ProcessMine example")
    parser.add_argument("--data_path", type=str, required=True, help="Path to process data CSV file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--model", type=str, default="enhanced_gnn", help="Model type")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_ablation", action="store_true", help="Whether to run ablation study")
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU if available")
    parser.add_argument("--mem_efficient", action="store_true", help="Whether to use memory-efficient mode")
    
    args = parser.parse_args()
    
    # Set up device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Import ProcessMine modules
    from processmine.core.advanced_workflow import run_advanced_workflow
    from processmine.core.ablation_runner import run_ablation_study
    
    # Run advanced workflow
    logger.info("Running advanced workflow...")
    
    workflow_results = run_advanced_workflow(
        data_path=args.data_path,
        output_dir=output_dir / "advanced_workflow",
        model=args.model,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        mem_efficient=args.mem_efficient,
        use_positional_encoding=True,
        use_diverse_attention=True,
        use_multi_objective_loss=True,
        use_adaptive_normalization=True,
        # Additional model configuration
        hidden_dim=64,
        num_layers=2,
        heads=4,
        dropout=0.5,
        diversity_weight=0.1,
        use_residual=True,
        use_batch_norm=True,
        use_layer_norm=False,
        # Multi-objective loss weights
        task_weight=0.5,
        time_weight=0.3,
        structure_weight=0.2
    )
    
    # Print advanced workflow results
    logger.info("Advanced workflow results:")
    metrics = workflow_results.get("metrics", {})
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
    
    # Plot training history if available
    history = workflow_results.get("history", {})
    if "val_loss" in history and len(history["val_loss"]) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(history.get("train_loss", []), label="Train Loss")
        plt.plot(history.get("val_loss", []), label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.savefig(output_dir / "advanced_workflow" / "training_history.png")
    
    # Run ablation study if requested
    if args.run_ablation:
        logger.info("Running ablation study...")
        
        # Define components to test
        ablation_config = {
            "components": [
                "use_positional_encoding",
                "use_diverse_attention",
                "use_batch_norm",
                "use_residual",
                "use_layer_norm"
            ],
            "disable": True,  # Test by disabling each component
            "include_combinations": False  # Don't test combinations (would be too many)
        }
        
        # Run ablation study
        ablation_results = run_ablation_study(
            data_path=args.data_path,
            base_model=args.model,
            output_dir=output_dir / "ablation_study",
            device=device,
            epochs=args.epochs // 2,  # Use fewer epochs for ablation
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            mem_efficient=args.mem_efficient,
            ablation_config=ablation_config
        )
        
        # Print ablation study results
        logger.info("Ablation study results:")
        results = ablation_results.get("results", {})
        for exp_name, exp_results in results.items():
            if "test_acc" in exp_results:
                logger.info(f"  {exp_name}: Test Accuracy = {exp_results['test_acc']:.4f}")
    
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    run_example()
    
'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for running comprehensive ablation study on ProcessMine
with all improvement plan components.

This demonstrates how to test the impact of each enhancement:
1. Enhanced Graph Representation with PositionalGATConv 
2. Memory Management Enhancements with MemoryEfficientDataLoader
3. Normalization Reconciliation with adaptive_normalization
4. Enhanced GNN with Expressivity Improvements
5. Multi-Head Attention with Diversity Mechanism
6. Multi-Objective Loss Function
"""

import os
import argparse
import torch
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger("ablation_study_example")

def run_ablation_example():
    """Run a comprehensive ablation study for ProcessMine enhancements"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ProcessMine Ablation Study Example")
    parser.add_argument("--data_path", type=str, required=True, help="Path to process data CSV file")
    parser.add_argument("--output_dir", type=str, default="results/ablation", help="Directory to save results")
    parser.add_argument("--model", type=str, default="enhanced_gnn", help="Model type")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs for each experiment")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU if available")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--include_combinations", action="store_true", help="Test combinations of features")
    parser.add_argument("--test_mode", type=str, default="disable", choices=["disable", "enable"], 
                      help="Whether to test by disabling or enabling features")
    
    args = parser.parse_args()
    
    # Set up device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Import helper functions
    from processmine.utils.memory import log_memory_usage
    
    # Log initial memory usage
    log_memory_usage()
    
    # Import ablation runner
    from processmine.core.ablation_integration import run_comprehensive_ablation
    
    # Define improvement components to test
    components_to_test = [
        "use_positional_encoding",      # Test improved graph representation
        "use_diverse_attention",        # Test diverse attention mechanism
        "use_batch_norm",               # Test batch normalization
        "use_residual",                 # Test residual connections
        "use_adaptive_normalization",   # Test adaptive normalization
        "use_multi_objective_loss"      # Test multi-objective loss
    ]
    
    # Run ablation study
    logger.info(f"Running ablation study with {len(components_to_test)} components...")
    
    ablation_results = run_comprehensive_ablation(
        data_path=args.data_path,
        base_model=args.model,
        output_dir=output_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        mem_efficient=True,
        components_to_test=components_to_test,
        include_combinations=args.include_combinations,
        test_mode=args.test_mode,
        max_workers=args.workers if args.parallel else 0,
        # Additional model configuration
        hidden_dim=64,
        num_layers=2,
        heads=4,
        dropout=0.5,
        # Multi-objective loss weights
        task_weight=0.5,
        time_weight=0.3,
        structure_weight=0.2
    )
    
    # Print results summary
    logger.info("\n==== ABLATION STUDY RESULTS ====")
    
    if "results" in ablation_results:
        results_data = []
        baseline_acc = None
        
        # Get baseline accuracy if available
        if "baseline" in ablation_results["results"]:
            baseline_acc = ablation_results["results"]["baseline"].get("test_acc_mean", None)
            logger.info(f"Baseline accuracy: {baseline_acc:.4f}" if baseline_acc else "Baseline not available")
        
        # Extract results for each experiment
        for exp_name, exp_results in ablation_results["results"].items():
            if exp_name == "baseline":
                continue
                
            test_acc = exp_results.get("test_acc_mean", None)
            if test_acc is not None:
                diff = test_acc - baseline_acc if baseline_acc else None
                diff_str = f"{diff:+.4f}" if diff is not None else "N/A"
                
                results_data.append({
                    "experiment": exp_name,
                    "accuracy": test_acc,
                    "diff": diff,
                    "diff_str": diff_str
                })
        
        # Sort by impact (difference from baseline)
        if results_data:
            results_data.sort(key=lambda x: x["diff"] if x["diff"] is not None else 0)
            
            logger.info("\nComponent Impact (sorted by performance impact):")
            for result in results_data:
                logger.info(f"  {result['experiment']:28s}: {result['accuracy']:.4f} ({result['diff_str']})")
        
        # Generate impact plot
        if results_data and baseline_acc:
            _generate_impact_plot(results_data, baseline_acc, output_dir)
    
    logger.info("\nAblation study completed!")

def _generate_impact_plot(results_data, baseline_acc, output_dir):
    """Generate a visualization of component impacts"""
    try:
        plt.figure(figsize=(12, 6))
        
        # Extract experiment names and diffs
        experiments = [r['experiment'] for r in results_data]
        diffs = [r['diff'] for r in results_data]
        
        # Create bar chart
        bars = plt.bar(range(len(experiments)), diffs, alpha=0.7)
        
        # Color bars based on impact
        for i, v in enumerate(diffs):
            # For disable mode: negative is bad (removing hurts)
            if v < 0:
                bars[i].set_color('red')      # Red for negative impact
            else:
                bars[i].set_color('green')    # Green for positive impact
                
        # Add baseline
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, label=f'Baseline ({baseline_acc:.4f})')
        
        # Add labels and formatting
        plt.xlabel('Component (tested by disabling)')
        plt.ylabel('Change in Test Accuracy')
        plt.title('Impact of Each Component on Model Performance')
        plt.xticks(range(len(experiments)), experiments, rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels
        for i, v in enumerate(diffs):
            plt.text(i, v + (0.005 if v >= 0 else -0.005), 
                     f"{v:+.4f}", ha='center', va='bottom' if v >= 0 else 'top',
                     fontweight='bold')
        
        # Save figure
        plt.savefig(output_dir / "component_impact.png", dpi=150, bbox_inches='tight')
        logger.info(f"Component impact plot saved to {output_dir / 'component_impact.png'}")
        
    except Exception as e:
        logger.error(f"Error generating impact plot: {e}")

if __name__ == "__main__":
    run_ablation_example()'''
    
'''"""
Updates to advanced_workflow.py to improve integration with ablation study functionality.
"""

import os
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

def run_advanced_workflow_with_ablation(
    data_path: str,
    output_dir: Optional[Union[str, Path]] = None,
    model: str = "enhanced_gnn",
    run_ablation: bool = False,
    ablation_epochs: int = 10,  # Shorter epochs for ablation
    device: Optional[Union[str, torch.device]] = None,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
    mem_efficient: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run advanced workflow with optional ablation study
    
    Args:
        data_path: Path to process data CSV file
        output_dir: Directory to save results
        model: Model type ("enhanced_gnn", "gnn", etc.)
        run_ablation: Whether to run ablation study
        ablation_epochs: Number of epochs for ablation study
        device: Computing device
        epochs: Number of training epochs (for main model)
        batch_size: Batch size
        lr: Learning rate
        seed: Random seed
        mem_efficient: Whether to use memory-efficient mode
        **kwargs: Additional model parameters
        
    Returns:
        Dictionary with workflow results and optional ablation results
    """
    from processmine.core.advanced_workflow import run_advanced_workflow
    
    # Create output directory if needed
    if output_dir is not None:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Run main workflow first
    logger.info("Running advanced workflow...")
    
    workflow_results = run_advanced_workflow(
        data_path=data_path,
        output_dir=output_dir / "advanced_workflow" if output_dir else None,
        model=model,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        mem_efficient=mem_efficient,
        use_positional_encoding=kwargs.get('use_positional_encoding', True),
        use_diverse_attention=kwargs.get('use_diverse_attention', True),
        use_multi_objective_loss=kwargs.get('use_multi_objective_loss', True),
        use_adaptive_normalization=kwargs.get('use_adaptive_normalization', True),
        **kwargs
    )
    
    # Store workflow results
    results = {
        "workflow": workflow_results
    }
    
    # Run ablation study if requested
    if run_ablation:
        logger.info("Running ablation study...")
        
        # Import ablation utilities
        from processmine.core.ablation_integration import run_comprehensive_ablation
        
        # Define components to test based on the improvement plan
        components_to_test = [
            "use_positional_encoding",     # Graph position encoding
            "use_diverse_attention",       # Attention diversity mechanism
            "use_batch_norm",              # Batch normalization
            "use_residual",                # Residual connections
            "use_layer_norm",              # Layer normalization
            "use_adaptive_normalization",  # Adaptive feature normalization
            "use_multi_objective_loss"     # Multi-objective loss function
        ]
        
        # Run comprehensive ablation study
        ablation_results = run_comprehensive_ablation(
            data_path=data_path,
            base_model=model,
            output_dir=output_dir / "ablation_study" if output_dir else None,
            device=device,
            epochs=ablation_epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            mem_efficient=mem_efficient,
            components_to_test=components_to_test,
            include_combinations=kwargs.get('ablation_include_combinations', False),
            test_mode=kwargs.get('ablation_test_mode', 'disable'),
            max_workers=kwargs.get('ablation_workers', 0),
            **{k: v for k, v in kwargs.items() if not k.startswith('ablation_')}
        )
        
        # Add ablation results
        results["ablation"] = ablation_results
        
        # Print a summary of ablation findings
        if "results" in ablation_results and output_dir:
            _print_ablation_summary(ablation_results, output_dir / "ablation_summary.txt")
    
    return results

def _print_ablation_summary(ablation_results, output_path):
    """Print and save a summary of ablation study findings"""
    try:
        results = ablation_results.get("results", {})
        
        # Get baseline performance if available
        baseline_acc = None
        if "baseline" in results:
            baseline_acc = results["baseline"].get("test_acc_mean", None)
        
        # Calculate impact of each component
        component_impacts = {}
        
        for exp_name, exp_results in results.items():
            if exp_name == "baseline" or baseline_acc is None:
                continue
                
            # Extract modifications
            mods = exp_results.get("modifications", {})
            if len(mods) == 1:  # Single component test
                component = list(mods.keys())[0]
                accuracy = exp_results.get("test_acc_mean", None)
                
                if accuracy is not None:
                    impact = baseline_acc - accuracy  # Impact of removing component
                    component_impacts[component] = impact
        
        # Create summary text
        summary = ["# Ablation Study Summary", ""]
        summary.append(f"Baseline accuracy: {baseline_acc:.4f}" if baseline_acc else "Baseline not available")
        summary.append("")
        
        if component_impacts:
            summary.append("## Component Impacts (when disabled)")
            summary.append("Higher positive value = more important component")
            summary.append("")
            
            # Sort by absolute impact
            sorted_components = sorted(component_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
            
            for component, impact in sorted_components:
                impact_type = "positive" if impact > 0 else "negative"
                summary.append(f"- {component}: {impact:+.4f} ({impact_type} impact)")
            
            summary.append("")
            summary.append("## Interpretation")
            summary.append("")
            summary.append("- Positive impact: Disabling hurts performance = component is valuable")
            summary.append("- Negative impact: Disabling helps performance = component may be unnecessary or conflicting")
            summary.append("- Higher absolute value = stronger effect")
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write("\n".join(summary))
        
        logger.info(f"Ablation summary saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating ablation summary: {e}")'''        