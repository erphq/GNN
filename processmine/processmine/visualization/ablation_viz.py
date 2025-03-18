"""
Advanced visualization utilities for ablation study results.
This module helps analyze and visualize the impact of different components from the improvement plan.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging

logger = logging.getLogger(__name__)

def visualize_ablation_results(
    results: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    metrics: Optional[List[str]] = None,
    show_plots: bool = False,
    style: str = 'whitegrid'
) -> Dict[str, str]:
    """
    Create visualizations of ablation study results
    
    Args:
        results: Ablation study results dictionary
        output_dir: Directory to save visualizations
        metrics: List of metrics to visualize (default: ['test_acc', 'test_f1'])
        show_plots: Whether to display plots interactively
        style: Seaborn style for plots
        
    Returns:
        Dictionary mapping visualization types to file paths
    """
    # Set default metrics
    if metrics is None:
        metrics = ['test_acc', 'test_f1']
    
    # Create output directory if needed
    if output_dir is not None:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Set seaborn style
    sns.set_style(style)
    
    # Dictionary to track output files
    output_files = {}
    
    # Extract actual results
    try:
        ablation_results = results.get('results', {})
        
        # Skip if no results
        if not ablation_results:
            logger.warning("No ablation results to visualize")
            return output_files
        
        # Convert to dataframe for easier analysis
        data = []
        for exp_name, exp_results in ablation_results.items():
            row = {'experiment': exp_name}
            
            # Extract modifications
            mods = exp_results.get('modifications', {})
            for comp, value in mods.items():
                row[comp] = value
            
            # Extract metrics
            for metric in metrics:
                mean_key = f"{metric}_mean"
                std_key = f"{metric}_std"
                
                if mean_key in exp_results:
                    row[metric] = exp_results[mean_key]
                    if std_key in exp_results:
                        row[f"{metric}_std"] = exp_results[std_key]
            
            data.append(row)
        
        # Create dataframe
        if data:
            df = pd.DataFrame(data)
            
            # Save data for reference
            if output_dir:
                csv_path = output_dir / "ablation_results.csv"
                df.to_csv(csv_path, index=False)
                output_files['data_csv'] = str(csv_path)
        else:
            logger.warning("No data extracted from ablation results")
            return output_files
        
        # Create visualizations
        for metric in metrics:
            if metric in df.columns:
                # 1. Component impact visualization
                impact_path = _create_component_impact_plot(
                    df, metric, output_dir, show_plots, baseline_exp='baseline'
                )
                if impact_path:
                    output_files[f'{metric}_impact'] = impact_path
                
                # 2. Performance comparison bar chart
                comparison_path = _create_performance_comparison_plot(
                    df, metric, output_dir, show_plots
                )
                if comparison_path:
                    output_files[f'{metric}_comparison'] = comparison_path
                
                # 3. Component correlation heatmap
                if len(df) >= 5:  # Need enough experiments for meaningful correlation
                    correlation_path = _create_component_correlation_plot(
                        df, metric, output_dir, show_plots
                    )
                    if correlation_path:
                        output_files[f'{metric}_correlation'] = correlation_path
        
        # 4. Generate summary table
        summary_path = _create_summary_table(df, metrics, output_dir)
        if summary_path:
            output_files['summary'] = summary_path
        
        return output_files
        
    except Exception as e:
        logger.error(f"Error visualizing ablation results: {e}")
        import traceback
        traceback.print_exc()
        return output_files

def _create_component_impact_plot(
    df: pd.DataFrame,
    metric: str,
    output_dir: Optional[Path],
    show_plots: bool,
    baseline_exp: str = 'baseline'
) -> Optional[str]:
    """Create a plot showing the impact of disabling each component"""
    try:
        # Find baseline experiment
        if baseline_exp not in df['experiment'].values:
            baseline_exps = [e for e in df['experiment'] if 'baseline' in e.lower()]
            if baseline_exps:
                baseline_exp = baseline_exps[0]
            else:
                # No baseline found, use the first experiment as reference
                baseline_exp = df['experiment'].iloc[0]
        
        # Get baseline performance
        baseline_perf = df.loc[df['experiment'] == baseline_exp, metric].iloc[0]
        
        # Extract single-component experiments
        component_impacts = {}
        component_experiments = {}
        
        for _, row in df.iterrows():
            exp_name = row['experiment']
            if exp_name == baseline_exp:
                continue
            
            # Check if this is a single-component experiment
            component = None
            for col in df.columns:
                if col.startswith('use_') and col in row:
                    if not pd.isna(row[col]) and row[col] != baseline_exp:
                        if component is None:  # First component found
                            component = col
                        else:  # More than one component modified
                            component = None
                            break
            
            # If single-component experiment, calculate impact
            if component:
                perf_diff = baseline_perf - row[metric]  # Effect of disabling
                component_impacts[component] = perf_diff
                component_experiments[component] = exp_name
        
        # Skip if not enough components to visualize
        if len(component_impacts) < 2:
            logger.warning("Not enough single-component experiments to visualize impacts")
            return None
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Sort components by impact magnitude
        components = sorted(component_impacts.keys(), 
                          key=lambda x: abs(component_impacts[x]), 
                          reverse=True)
        impacts = [component_impacts[c] for c in components]
        
        # Create bar chart
        bars = plt.bar(components, impacts, alpha=0.7)
        
        # Color bars by impact
        for i, impact in enumerate(impacts):
            if impact > 0:  # Positive impact (removing hurts performance)
                bars[i].set_color('red')
            else:  # Negative impact (removing helps performance)
                bars[i].set_color('green')
        
        # Add baseline reference line
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, 
                   label=f'Baseline ({baseline_perf:.4f})')
        
        # Labels and formatting
        plt.xlabel('Component (disabled in experiment)')
        plt.ylabel(f'Impact on {metric} (baseline - experiment)')
        plt.title(f'Impact of Disabling Each Component ({metric})')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels
        for i, impact in enumerate(impacts):
            plt.text(i, impact + (0.005 if impact >= 0 else -0.005), 
                    f"{impact:+.4f}", ha='center', va='bottom' if impact >= 0 else 'top',
                    fontweight='bold')
        
        plt.tight_layout()
        
        # Save and/or show
        if output_dir:
            output_path = output_dir / f"component_impact_{metric}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return str(output_path) if output_dir else None
        
    except Exception as e:
        logger.error(f"Error creating component impact plot: {e}")
        return None

def _create_performance_comparison_plot(
    df: pd.DataFrame,
    metric: str,
    output_dir: Optional[Path],
    show_plots: bool
) -> Optional[str]:
    """Create a bar chart comparing performance across experiments"""
    try:
        # Ensure metric exists
        if metric not in df.columns:
            return None
        
        # Add error bars if available
        std_key = f"{metric}_std"
        has_error_bars = std_key in df.columns
        
        # Sort by performance
        sorted_df = df.sort_values(by=metric, ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Create bar chart with or without error bars
        if has_error_bars:
            bars = plt.bar(sorted_df['experiment'], sorted_df[metric], 
                          yerr=sorted_df[std_key], alpha=0.7,
                          error_kw={'capsize': 5, 'elinewidth': 1, 'alpha': 0.8})
        else:
            bars = plt.bar(sorted_df['experiment'], sorted_df[metric], alpha=0.7)
        
        # Highlight baseline experiment
        baseline_idx = sorted_df['experiment'].str.contains('baseline', case=False)
        for i, is_baseline in enumerate(baseline_idx):
            if is_baseline:
                bars[i].set_color('gold')
                bars[i].set_edgecolor('black')
        
        # Labels and formatting
        plt.xlabel('Experiment')
        plt.ylabel(f'{metric}')
        plt.title(f'Performance Comparison Across Experiments ({metric})')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(sorted_df[metric]):
            plt.text(i, v + 0.005, f"{v:.4f}", ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save and/or show
        if output_dir:
            output_path = output_dir / f"performance_comparison_{metric}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return str(output_path) if output_dir else None
        
    except Exception as e:
        logger.error(f"Error creating performance comparison plot: {e}")
        return None

def _create_component_correlation_plot(
    df: pd.DataFrame,
    metric: str,
    output_dir: Optional[Path],
    show_plots: bool
) -> Optional[str]:
    """Create a heatmap showing correlations between components and performance"""
    try:
        # Identify component columns
        component_cols = [col for col in df.columns if col.startswith('use_')]
        
        if not component_cols or len(component_cols) < 2:
            return None
        
        # Create correlation matrix
        corr_cols = component_cols + [metric]
        correlation = df[corr_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm",
                   mask=mask, vmin=-1, vmax=1, center=0,
                   square=True, linewidths=0.5)
        
        plt.title(f'Correlation Between Components and {metric}')
        plt.tight_layout()
        
        # Save and/or show
        if output_dir:
            output_path = output_dir / f"component_correlation_{metric}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return str(output_path) if output_dir else None
        
    except Exception as e:
        logger.error(f"Error creating component correlation plot: {e}")
        return None

def _create_summary_table(
    df: pd.DataFrame,
    metrics: List[str],
    output_dir: Optional[Path]
) -> Optional[str]:
    """Create a summary table of ablation results"""
    try:
        if not output_dir:
            return None
        
        # Extract baseline metrics
        baseline_row = df[df['experiment'].str.contains('baseline', case=False)]
        baseline_metrics = {}
        
        for metric in metrics:
            if metric in df.columns and not baseline_row.empty:
                baseline_metrics[metric] = baseline_row[metric].iloc[0]
        
        # Create a summary table
        rows = []
        
        for _, row in df.iterrows():
            exp_name = row['experiment']
            if 'baseline' in exp_name.lower():
                continue  # Skip baseline in detailed rows
                
            summary_row = {'Experiment': exp_name, 'Components Modified': []}
            
            # Identify modified components
            for col in df.columns:
                if col.startswith('use_') and not pd.isna(row[col]):
                    if baseline_row.empty or row[col] != baseline_row[col].iloc[0]:
                        component_name = col.replace('use_', '')
                        status = "Enabled" if row[col] else "Disabled"
                        summary_row['Components Modified'].append(f"{component_name}: {status}")
            
            # Format component modifications
            if summary_row['Components Modified']:
                summary_row['Components Modified'] = ', '.join(summary_row['Components Modified'])
            else:
                summary_row['Components Modified'] = "None (baseline)"
            
            # Add performance metrics
            for metric in metrics:
                if metric in df.columns:
                    value = row[metric]
                    summary_row[metric] = f"{value:.4f}"
                    
                    # Add difference from baseline
                    if metric in baseline_metrics:
                        diff = value - baseline_metrics[metric]
                        summary_row[f"{metric} vs Baseline"] = f"{diff:+.4f}"
            
            rows.append(summary_row)
        
        # Add baseline row at top
        if not baseline_row.empty:
            baseline_dict = {'Experiment': baseline_row['experiment'].iloc[0],
                             'Components Modified': "None (baseline)"}
            
            for metric in metrics:
                if metric in baseline_metrics:
                    baseline_dict[metric] = f"{baseline_metrics[metric]:.4f}"
                    baseline_dict[f"{metric} vs Baseline"] = "0.0000"
                    
            rows.insert(0, baseline_dict)
        
        # Create Markdown table
        lines = ["# Ablation Study Summary", ""]
        
        # Add baseline reference
        if baseline_metrics:
            lines.append("## Baseline Performance")
            for metric, value in baseline_metrics.items():
                lines.append(f"- {metric}: {value:.4f}")
            lines.append("")
        
        # Create table header
        columns = ['Experiment', 'Components Modified']
        for metric in metrics:
            if any(metric in row for row in rows):
                columns.append(metric)
                if f"{metric} vs Baseline" in rows[0]:
                    columns.append(f"{metric} vs Baseline")
        
        lines.append("## Detailed Results")
        lines.append("")
        
        # Add header row
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["-" * len(col) for col in columns]) + " |")
        
        # Add data rows
        for row in rows:
            line = "| "
            for col in columns:
                cell = row.get(col, "")
                line += str(cell) + " | "
            lines.append(line)
        
        # Add interpretation section
        lines.extend([
            "",
            "## Interpretation Guide",
            "",
            "- **Positive difference**: Experiment performed better than baseline",
            "- **Negative difference**: Experiment performed worse than baseline",
            "- For component disabling experiments, a negative difference indicates the component is valuable"
        ])
        
        # Write to file
        output_path = output_dir / "ablation_summary.md"
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error creating summary table: {e}")
        return None

def analyze_component_importance(
    results: Dict[str, Any],
    metric: str = 'test_acc',
    output_file: Optional[Union[str, Path]] = None
) -> Dict[str, float]:
    """
    Analyze the importance of each component based on ablation results
    
    Args:
        results: Ablation study results dictionary
        metric: Metric to use for analysis
        output_file: File to save results (optional)
        
    Returns:
        Dictionary mapping components to importance scores
    """
    try:
        ablation_results = results.get('results', {})
        
        # Skip if no results
        if not ablation_results:
            logger.warning("No ablation results to analyze")
            return {}
        
        # Find baseline experiment
        baseline_exp = None
        baseline_perf = None
        
        for exp_name, exp_results in ablation_results.items():
            if 'baseline' in exp_name.lower():
                mean_key = f"{metric}_mean"
                if mean_key in exp_results:
                    baseline_exp = exp_name
                    baseline_perf = exp_results[mean_key]
                    break
        
        if baseline_exp is None or baseline_perf is None:
            logger.warning("No baseline experiment found for comparison")
            return {}
        
        # Calculate impact of each component
        component_importance = {}
        
        for exp_name, exp_results in ablation_results.items():
            if exp_name == baseline_exp:
                continue
                
            # Extract modifications
            mods = exp_results.get('modifications', {})
            if len(mods) == 1:  # Single component test
                component = list(mods.keys())[0]
                mean_key = f"{metric}_mean"
                
                if mean_key in exp_results:
                    # Calculate impact (for disable tests: baseline - experiment)
                    # Positive impact means disabling hurts performance, component is valuable
                    impact = baseline_perf - exp_results[mean_key]
                    component_importance[component] = impact
        
        # Sort by absolute importance
        sorted_importance = {k: v for k, v in sorted(
            component_importance.items(), 
            key=lambda item: abs(item[1]),
            reverse=True
        )}
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"# Component Importance Analysis (based on {metric})\n\n")
                f.write(f"Baseline performance: {baseline_perf:.4f}\n\n")
                f.write("## Components Ranked by Importance\n\n")
                
                for component, importance in sorted_importance.items():
                    status = "Positive" if importance > 0 else "Negative"
                    f.write(f"- **{component}**: {importance:+.4f} ({status} impact)\n")
                
                f.write("\n## Interpretation\n\n")
                f.write("- Positive impact: Disabling hurts performance = component is valuable\n")
                f.write("- Negative impact: Disabling helps performance = component may be unnecessary or conflicting\n")
                f.write("- Higher absolute value = stronger effect\n")
        
        return sorted_importance
        
    except Exception as e:
        logger.error(f"Error analyzing component importance: {e}")
        return {}