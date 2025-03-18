"""
Memory-efficient visualization module with lazy loading, data sampling,
and chunked rendering for handling large process mining datasets.
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import time
import gc

logger = logging.getLogger(__name__)

# Check for optional visualization dependencies
# These imports are lazy-loaded to reduce unnecessary dependencies
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.info("Seaborn not installed. Using Matplotlib for visualizations.")

class ProcessVisualizer:
    """
    Unified visualization class for process mining with memory optimization
    and progressive rendering for large datasets
    """
    
    def __init__(
        self, 
        output_dir: Optional[str] = None,
        style: str = 'default',
        force_static: bool = False,
        memory_efficient: bool = True,
        sampling_threshold: int = 100000,
        max_plot_points: int = 50000,
        dpi: int = 120
    ):
        """
        Initialize visualizer with options for memory management
        
        Args:
            output_dir: Directory to save visualizations
            style: Visualization style ('default', 'dark', 'light')
            force_static: Whether to force static (non-interactive) visualizations
            memory_efficient: Whether to use memory optimizations for large datasets
            sampling_threshold: Threshold for data sampling in points
            max_plot_points: Maximum number of points to plot
            dpi: DPI for saved figures
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        self.style = style
        self.force_static = force_static
        self.memory_efficient = memory_efficient
        self.sampling_threshold = sampling_threshold
        self.max_plot_points = max_plot_points
        self.dpi = dpi
        
        # Set matplotlib style based on preference
        self._set_style(style)
        
        # Check for interactive visualization capabilities with lazy imports
        self.has_plotly = self._check_plotly()
        self.has_networkx = self._check_networkx()
        
        # Set flags for using interactive visualizations
        self.use_interactive = self.has_plotly and not force_static
    
    def _check_plotly(self) -> bool:
        """Lazy check for Plotly availability"""
        try:
            import importlib.util
            spec = importlib.util.find_spec('plotly')
            return spec is not None
        except ImportError:
            return False
    
    def _check_networkx(self) -> bool:
        """Lazy check for NetworkX availability"""
        try:
            import importlib.util
            spec = importlib.util.find_spec('networkx')
            return spec is not None
        except ImportError:
            return False
    
    def _set_style(self, style: str) -> None:
        """Set visualization style"""
        # Matplotlib style settings
        if style == 'dark':
            plt.style.use('dark_background')
            self.colors = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'tertiary': '#2ca02c',
                'quaternary': '#d62728',
                'highlight': '#bcbd22',
                'background': '#2d2d2d',
                'text': '#ffffff'
            }
        elif style == 'light':
            plt.style.use('seaborn-v0_8')
            self.colors = {
                'primary': '#4c72b0',
                'secondary': '#dd8452',
                'tertiary': '#55a868',
                'quaternary': '#c44e52',
                'highlight': '#ccb974',
                'background': '#ffffff',
                'text': '#000000'
            }
        else:  # default
            plt.style.use('seaborn-v0_8-whitegrid')
            self.colors = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'tertiary': '#2ca02c',
                'quaternary': '#d62728',
                'highlight': '#bcbd22',
                'background': '#f7f7f7',
                'text': '#333333'
            }
        
        # Update matplotlib rcParams for better defaults
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
    
    def _save_figure(self, fig, filename: str) -> Optional[str]:
        """
        Save figure with optimization for file size and memory
        
        Args:
            fig: Matplotlib figure object
            filename: Filename to save figure
            
        Returns:
            Path to saved figure or None
        """
        if not self.output_dir:
            plt.show()
            plt.close(fig)
            return None
        
        # Add extension if needed
        if not filename.lower().endswith(('.png', '.jpg', '.svg', '.pdf')):
            filename += '.png'
        
        # Create full path
        filepath = os.path.join(self.output_dir, filename)
        
        # Save with optimized settings
        fig.savefig(
            filepath, 
            bbox_inches='tight', 
            dpi=self.dpi,
            # Use compression for PNG to reduce file size
            #optimize=True,
            transparent=False
        )
        
        # Close figure to free memory
        plt.close(fig)
        
        logger.info(f"Saved visualization to {filepath}")
        return filepath
    
    def _sample_data(self, data: np.ndarray, max_points: Optional[int] = None) -> np.ndarray:
        """
        Sample data for visualization to reduce memory usage and improve performance
        
        Args:
            data: Data array to sample
            max_points: Maximum number of points (default: self.max_plot_points)
            
        Returns:
            Sampled data array
        """
        if max_points is None:
            max_points = self.max_plot_points
            
        # If data is smaller than threshold, return as is
        if len(data) <= self.sampling_threshold:
            return data
        
        # For larger datasets, use intelligent sampling
        if len(data) > max_points:
            # Calculate sampling ratio
            sampling_ratio = max_points / len(data)
            
            # Use stratified sampling for better representation
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                # For pandas objects, use built-in sampling
                return data.sample(n=max_points, random_state=42)
            else:
                # For numpy arrays, use random sampling
                indices = np.random.choice(
                    len(data), size=max_points, replace=False
                )
                return data[indices]
        
        return data
    
    def cycle_time_distribution(
        self, 
        durations: Union[np.ndarray, pd.Series],
        filename: str = "cycle_time_distribution",
        bins: Optional[int] = None,
        include_kde: bool = True,
        show_percentiles: bool = True
    ) -> Optional[str]:
        """
        Create cycle time distribution visualization with memory optimization
        
        Args:
            durations: Array of case durations in hours
            filename: Output filename (without extension for auto-format detection)
            bins: Number of histogram bins (auto-calculated if None)
            include_kde: Whether to include KDE curve
            show_percentiles: Whether to show percentile lines
            
        Returns:
            Path to saved figure or None
        """
        # Check for interactive visualization
        if self.use_interactive and not filename.endswith('.png'):
            return self._create_interactive_histogram(
                durations, 
                filename, 
                "Process Cycle Time Distribution",
                "Duration (hours)",
                "Number of Cases"
            )
        
        # Sample data if needed for large datasets
        if self.memory_efficient and len(durations) > self.sampling_threshold:
            logger.info(f"Sampling cycle time data from {len(durations):,} to {self.max_plot_points:,} points")
            durations = self._sample_data(durations)
        
        # Calculate statistics (use original data for accuracy)
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)
        p90 = np.percentile(durations, 90)
        p95 = np.percentile(durations, 95)
        
        # Auto-calculate bins based on data size and range
        if bins is None:
            # Freedman-Diaconis rule for optimal bin width
            q75, q25 = np.percentile(durations, [75, 25])
            iqr = q75 - q25
            bin_width = 2 * iqr / (len(durations) ** (1/3))
            data_range = np.max(durations) - np.min(durations)
            bins = max(10, min(100, int(data_range / bin_width)))
        
        # Create figure and plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram with or without KDE
        if HAS_SEABORN and include_kde:
            # Use Seaborn for better looking histogram with KDE
            import seaborn as sns
            sns.histplot(
                durations, 
                kde=True, 
                bins=bins,
                color=self.colors['primary'],
                line_kws={'linewidth': 2},
                ax=ax
            )
        else:
            # Use matplotlib for histogram
            ax.hist(
                durations, 
                bins=bins, 
                color=self.colors['primary'],
                alpha=0.7,
                edgecolor='white'
            )
        
        # Add percentile lines
        if show_percentiles:
            ax.axvline(mean_duration, color=self.colors['secondary'], linestyle='-', 
                      linewidth=2, label=f"Mean: {mean_duration:.1f}h")
            ax.axvline(median_duration, color=self.colors['tertiary'], linestyle='--', 
                      linewidth=2, label=f"Median: {median_duration:.1f}h")
            ax.axvline(p95, color=self.colors['quaternary'], linestyle='-.', 
                      linewidth=2, label=f"95th %: {p95:.1f}h")
        
        # Add statistics text box
        stats_text = (
            f"Total cases: {len(durations)}\n"
            f"Mean: {mean_duration:.2f}h\n"
            f"Median: {median_duration:.2f}h\n"
            f"Min: {np.min(durations):.2f}h\n"
            f"Max: {np.max(durations):.2f}h\n"
            f"P90: {p90:.2f}h\n"
            f"P95: {p95:.2f}h"
        )
        
        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Add styling
        ax.set_title("Process Cycle Time Distribution", fontsize=14)
        ax.set_xlabel("Duration (hours)")
        ax.set_ylabel("Number of Cases")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Save figure
        return self._save_figure(fig, filename)
    
    def _create_interactive_histogram(
        self, 
        data: Union[np.ndarray, pd.Series],
        filename: str,
        title: str,
        x_label: str,
        y_label: str
    ) -> Optional[str]:
        """
        Create interactive histogram with Plotly
        
        Args:
            data: Data for histogram
            filename: Output filename
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            
        Returns:
            Path to saved figure or None
        """
        # Import plotly only when needed
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("Plotly not available. Falling back to static visualization.")
            if not filename.endswith('.png'):
                filename += '.png'
            return self.cycle_time_distribution(data, filename)
        
        # Sample data if needed
        if self.memory_efficient and len(data) > self.sampling_threshold:
            logger.info(f"Sampling from {len(data):,} to {self.max_plot_points:,} points for histogram")
            data = self._sample_data(data)
        
        # Calculate statistics
        mean_val = np.mean(data)
        median_val = np.median(data)
        p90 = np.percentile(data, 90)
        p95 = np.percentile(data, 95)
        
        # Create figure
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=30,
            marker_color=self.colors['primary'],
            opacity=0.7,
            name="Duration"
        ))
        
        # Add lines for statistics
        fig.add_vline(x=mean_val, line_dash="solid", line_color=self.colors['secondary'],
                     annotation_text=f"Mean: {mean_val:.1f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color=self.colors['tertiary'],
                     annotation_text=f"Median: {median_val:.1f}")
        fig.add_vline(x=p95, line_dash="dot", line_color=self.colors['quaternary'],
                     annotation_text=f"P95: {p95:.1f}")
        
        # Add annotations for statistics
        stats_text = (
            f"Total: {len(data)}<br>"
            f"Mean: {mean_val:.2f}<br>"
            f"Median: {median_val:.2f}<br>"
            f"Min: {np.min(data):.2f}<br>"
            f"Max: {np.max(data):.2f}<br>"
            f"P90: {p90:.2f}<br>"
            f"P95: {p95:.2f}"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=stats_text,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            align="left"
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white" if self.style != 'dark' else "plotly_dark",
            hovermode="closest"
        )
        
        # Save figure if output directory provided
        if self.output_dir:
            # Ensure filename has .html extension
            if not filename.lower().endswith('.html'):
                filename += '.html'
            
            filepath = os.path.join(self.output_dir, filename)
            
            # Set renderer to 'browser' for interactive plots
            fig.write_html(
                filepath,
                include_plotlyjs='cdn',  # Use CDN for smaller file size
                full_html=True,
                auto_open=False
            )
            
            logger.info(f"Saved interactive visualization to {filepath}")
            return filepath
        
        return None
    
    def bottleneck_analysis(
        self, 
        bottleneck_stats: pd.DataFrame,
        significant_bottlenecks: pd.DataFrame,
        task_encoder: Any,
        filename: str = "bottleneck_analysis",
        top_n: int = 10
    ) -> Optional[str]:
        """
        Create bottleneck analysis visualization
        
        Args:
            bottleneck_stats: DataFrame with bottleneck statistics
            significant_bottlenecks: DataFrame with significant bottlenecks
            task_encoder: Task label encoder
            filename: Output filename
            top_n: Number of top bottlenecks to display
            
        Returns:
            Path to saved figure or None
        """
        # Check for empty data
        if len(significant_bottlenecks) == 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No significant bottlenecks found", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return self._save_figure(fig, filename)
        
        # Get top bottlenecks
        display_bottlenecks = significant_bottlenecks.head(min(top_n, len(significant_bottlenecks)))
        
        # Create labels
        labels = []
        for _, row in display_bottlenecks.iterrows():
            src_id, dst_id = int(row["task_id"]), int(row["next_task_id"])
            try:
                src_name = task_encoder.inverse_transform([src_id])[0]
                dst_name = task_encoder.inverse_transform([dst_id])[0]
                
                # Truncate long names
                if len(src_name) > 15:
                    src_name = src_name[:12] + "..."
                if len(dst_name) > 15:
                    dst_name = dst_name[:12] + "..."
                    
                labels.append(f"{src_name} → {dst_name}")
            except:
                labels.append(f"Task {src_id} → Task {dst_id}")
        
        # Check for interactive visualization
        if self.use_interactive and not filename.endswith('.png'):
            return self._create_interactive_bottleneck_chart(
                display_bottlenecks, labels, filename
            )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.5)))
        
        # Plot horizontal bar chart
        bars = ax.barh(
            labels, 
            display_bottlenecks["mean_hours"].values,
            color=self.colors['primary'],
            alpha=0.8
        )
        
        # Add count annotations
        for i, bar in enumerate(bars):
            count = int(display_bottlenecks.iloc[i]["count"])
            score = float(display_bottlenecks.iloc[i].get("bottleneck_score", 0))
            
            ax.text(
                bar.get_width() + 0.3, 
                bar.get_y() + bar.get_height()/2, 
                f"n={count}" + (f", score={score:.1f}" if score > 0 else ""),
                va='center',
                fontsize=9
            )
        
        # Add styling
        ax.set_title("Top Process Bottlenecks", fontsize=14)
        ax.set_xlabel("Average Wait Time (hours)")
        ax.set_ylabel("Transition")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add threshold information
        if hasattr(bottleneck_stats, 'attrs') and 'bottleneck_threshold' in bottleneck_stats.attrs:
            threshold = bottleneck_stats.attrs['bottleneck_threshold'] / 3600  # Convert to hours
            ax.axvline(
                threshold, 
                color=self.colors['secondary'], 
                linestyle='--',
                alpha=0.7,
                label=f"Threshold: {threshold:.1f}h"
            )
            ax.legend()
        
        # Ensure y-axis has enough padding
        plt.tight_layout()
        
        # Save figure
        return self._save_figure(fig, filename)
    
    def _create_interactive_bottleneck_chart(
        self,
        bottlenecks: pd.DataFrame,
        labels: List[str],
        filename: str
    ) -> Optional[str]:
        """
        Create interactive bottleneck chart with Plotly
        
        Args:
            bottlenecks: DataFrame with bottleneck data
            labels: List of bottleneck labels
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly not available. Falling back to static visualization.")
            filename += '.png'
            return self.bottleneck_analysis(
                bottlenecks, bottlenecks, None, filename, len(bottlenecks)
            )
        
        # Create figure
        fig = go.Figure()
        
        # Create hover text
        hover_text = []
        for _, row in bottlenecks.iterrows():
            hover_text.append(
                f"Count: {int(row['count'])}<br>"
                f"Mean wait: {row['mean_hours']:.2f}h<br>"
                f"Median wait: {row['median']/3600:.2f}h<br>"
                f"CV: {row.get('cv', 0):.2f}"
            )
        
        # Add horizontal bar chart
        fig.add_trace(go.Bar(
            y=labels,
            x=bottlenecks["mean_hours"].values,
            orientation='h',
            marker_color=self.colors['primary'],
            text=hover_text,
            hoverinfo="text+x",
            name="Wait Time (h)"
        ))
        
        # Add threshold line if available
        if 'bottleneck_threshold' in bottlenecks.attrs:
            threshold = bottlenecks.attrs['bottleneck_threshold'] / 3600  # Convert to hours
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color=self.colors['secondary'],
                annotation_text=f"Threshold: {threshold:.1f}h"
            )
        
        # Update layout
        fig.update_layout(
            title="Top Process Bottlenecks",
            xaxis_title="Average Wait Time (hours)",
            yaxis_title="Transition",
            template="plotly_white" if self.style != 'dark' else "plotly_dark",
            height=max(500, len(labels) * 40)  # Dynamic height based on number of items
        )
        
        # Save figure if output directory provided
        if self.output_dir:
            # Ensure filename has .html extension
            if not filename.lower().endswith('.html'):
                filename += '.html'
            
            filepath = os.path.join(self.output_dir, filename)
            
            fig.write_html(
                filepath,
                include_plotlyjs='cdn',
                full_html=True,
                auto_open=False
            )
            
            logger.info(f"Saved interactive bottleneck chart to {filepath}")
            return filepath
        
        return None
    
    def process_flow(
        self, 
        bottleneck_stats: pd.DataFrame,
        task_encoder: Any,
        significant_bottlenecks: Optional[pd.DataFrame] = None,
        filename: str = "process_flow",
        max_nodes: int = 50,
        layout: str = 'auto',
        use_dgl_sampling: bool = True  # New parameter to enable DGL sampling
    ) -> Optional[str]:
        """
        Create process flow visualization with DGL-optimized memory usage
        
        Args:
            bottleneck_stats: DataFrame with bottleneck statistics
            task_encoder: Task label encoder
            significant_bottlenecks: DataFrame with significant bottlenecks
            filename: Output filename
            max_nodes: Maximum number of nodes to display
            layout: Graph layout algorithm ('auto', 'spring', 'circular', 'kamada', 'spectral')
            use_dgl_sampling: Whether to use DGL's graph sampling features
            
        Returns:
            Path to saved figure or None
        """
        # Check NetworkX and DGL availability
        if not self.has_networkx:
            logger.error("NetworkX is required for process flow visualization")
            return None
        
        # Import required libraries
        import networkx as nx
        
        try:
            # Try to import DGL for enhanced sampling
            import dgl
            has_dgl = True
        except ImportError:
            has_dgl = False
        
        # Create graph using NetworkX (for visualization compatibility)
        G = nx.DiGraph()
        
        # Limit to most important transitions for readability
        if self.memory_efficient and len(bottleneck_stats) > 1000:
            # Sort by count to get most frequent transitions
            limited_stats = bottleneck_stats.sort_values("count", ascending=False).head(1000)
            logger.info(f"Limiting visualization to top 1000 transitions out of {len(bottleneck_stats)}")
        else:
            limited_stats = bottleneck_stats
        
        # Add edges with attributes
        for _, row in limited_stats.iterrows():
            src = int(row["task_id"])
            dst = int(row["next_task_id"])
            G.add_edge(src, dst, 
                    freq=int(row["count"]), 
                    mean_hours=row["mean_hours"],
                    weight=float(row["count"]))
        
            # Use DGL sampling for better memory efficiency if available and requested
            if has_dgl and use_dgl_sampling and len(G.nodes()) > max_nodes:
                # Convert NetworkX graph to DGL for sampling
                dgl_G = dgl.from_networkx(G, edge_attrs=['freq', 'mean_hours', 'weight'])
                
                try:
                    # Use modern DGL node sampling methods
                    if hasattr(dgl.sampling, 'select_topk'):
                        # Use topk sampling - now preferred in DGL
                        sampled_nodes = dgl.sampling.select_topk(dgl_G, max_nodes, 'weight')
                        subg = dgl_G.subgraph(sampled_nodes)
                    else:
                        # Fall back to random walk sampling
                        seeds = torch.arange(0, min(100, dgl_G.num_nodes()))
                        traces, _ = dgl.sampling.random_walk(
                            dgl_G, 
                            seeds,
                            length=5
                        )
                        # Flatten and find unique nodes
                        nodes = torch.unique(traces.flatten())
                        # Remove -1 values (which indicate invalid nodes in random walk)
                        nodes = nodes[nodes >= 0]
                        sampled_nodes = nodes[:min(len(nodes), max_nodes)]
                        subg = dgl_G.subgraph(sampled_nodes)
                    
                    # Convert back to NetworkX with proper attributes
                    G = dgl.to_networkx(subg, edge_attrs=['freq', 'mean_hours', 'weight'])
                    logger.info(f"Used DGL sampling to reduce graph from {dgl_G.num_nodes()} to {subg.num_nodes()} nodes")
                except Exception as e:
                    logger.warning(f"DGL sampling failed, falling back to NetworkX: {e}")
                    
                        
                    # Convert back to NetworkX
                    G = nx.DiGraph(dgl.to_networkx(subg, edge_attrs=['freq', 'mean_hours', 'weight']))
                    logger.info(f"Used DGL sampling to reduce graph from {dgl_G.num_nodes()} to {subg.num_nodes()} nodes")
                except Exception as e:
                    logger.warning(f"DGL sampling failed, falling back to NetworkX: {e}")
                    # Continue with NetworkX-based sampling below
        
        # Simplify graph if still too large (NetworkX fallback)
        if len(G.nodes()) > max_nodes:
            # Keep only most important nodes
            node_importance = {}
            for node in G.nodes():
                # Importance = sum of in and out edge weights
                in_weight = sum(data['weight'] for _, _, data in G.in_edges(node, data=True))
                out_weight = sum(data['weight'] for _, _, data in G.out_edges(node, data=True))
                node_importance[node] = in_weight + out_weight
            
            # Sort nodes by importance
            important_nodes = sorted(
                node_importance.keys(), 
                key=lambda x: node_importance[x], 
                reverse=True
            )[:max_nodes]
            
            # Create subgraph with important nodes
            G = G.subgraph(important_nodes)
            logger.info(f"Reduced graph from {len(node_importance)} to {len(G.nodes())} nodes")
        
        # Identify bottleneck edges
        bottleneck_edges = set()
        if significant_bottlenecks is not None:
            for _, row in significant_bottlenecks.iterrows():
                bottleneck_edges.add((int(row["task_id"]), int(row["next_task_id"])))
        
        # Check for interactive visualization
        if self.has_plotly and self.use_interactive and not filename.endswith('.png'):
            return self._create_interactive_process_flow(
                G, task_encoder, bottleneck_edges, filename
            )
        
        # Create figure with appropriate size
        figsize = (
            max(10, min(20, len(G.nodes()) / 4)), 
            max(8, min(16, len(G.nodes()) / 5))
        )
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout based on graph size and user preference
        if layout == 'auto':
            if len(G.nodes()) <= 20:
                layout = 'kamada'
            else:
                layout = 'spring'
        
        # Calculate node positions
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42, k=0.3, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:
            # Fallback to spring layout
            pos = nx.spring_layout(G, seed=42)
        
        # Calculate node sizes based on importance
        node_sizes = {}
        for node in G.nodes():
            # Size based on sum of in and out degrees with weights
            in_weight = sum(data.get('weight', 1) for _, _, data in G.in_edges(node, data=True))
            out_weight = sum(data.get('weight', 1) for _, _, data in G.out_edges(node, data=True))
            node_sizes[node] = 300 + 50 * np.sqrt(in_weight + out_weight)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=[node_sizes[n] for n in G.nodes()],
            node_color=self.colors['primary'],
            alpha=0.8,
            edgecolors='gray',
            ax=ax
        )
        
        # Draw edges with different colors and widths for bottlenecks
        for (u, v, data) in G.edges(data=True):
            if (u, v) in bottleneck_edges:
                # Bottleneck edge
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    width=3.0,
                    alpha=1.0,
                    edge_color=self.colors['secondary'],
                    connectionstyle='arc3,rad=0.1',
                    arrows=True,
                    arrowsize=15,
                    ax=ax
                )
            else:
                # Normal edge
                width = 0.5 + min(3, data.get('weight', 1) / 100)
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    width=width,
                    alpha=0.6,
                    edge_color=self.colors['primary'],
                    connectionstyle='arc3,rad=0.1',
                    arrows=True,
                    arrowsize=10,
                    ax=ax
                )
        
        # Create labels with task names
        try:
            # Try to use encoder for readable names
            labels = {n: task_encoder.inverse_transform([int(n)])[0] for n in G.nodes()}
            
            # Truncate long labels
            labels = {k: (v[:10] + "..." if len(v) > 10 else v) for k, v in labels.items()}
        except:
            # Fallback to IDs
            labels = {n: f"Task {n}" for n in G.nodes()}
            
        # Draw node labels
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=9,
            font_weight='normal',
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray", boxstyle="round,pad=0.2"),
            ax=ax
        )
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=self.colors['secondary'], lw=3, label='Bottleneck'),
            Line2D([0], [0], color=self.colors['primary'], lw=1, label='Normal Flow')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add title and info
        ax.set_title(f"Process Flow Diagram ({len(G.nodes())} activities, {len(G.edges())} transitions)", fontsize=14)
        ax.axis('off')
        
        return self._save_figure(fig, filename)

    def _create_interactive_process_flow(
    self,
    G,
    task_encoder,
    bottleneck_edges,
    filename
) -> Optional[str]:
        """
        Create interactive process flow visualization with Plotly
        
        Args:
            G: NetworkX graph
            task_encoder: Task label encoder
            bottleneck_edges: Set of bottleneck edges
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        try:
            import plotly.graph_objects as go
            import networkx as nx
        except ImportError:
            logger.warning("Required libraries not available. Falling back to static visualization.")
            filename += '.png'
            return self.process_flow(None, task_encoder, None, filename)
        
        # Calculate layout
        pos = nx.kamada_kawai_layout(G) if len(G.nodes()) <= 30 else nx.spring_layout(G, seed=42)
        
        # Create node trace
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=15,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                ),
                line_width=2
            )
        )
        
        # Color nodes by number of connections
        node_adjacencies = []
        node_text = []
        
        for node in G.nodes():
            # Fix: Use G[node] instead of G.adjacency()[node] for compatibility with newer NetworkX
            adjacencies = list(G[node])
            node_adjacencies.append(len(adjacencies))
            
            # Get node label
            try:
                label = task_encoder.inverse_transform([int(node)])[0]
            except:
                label = f"Task {node}"
            
            # Get node info
            connections = len(list(G.successors(node))) + len(list(G.predecessors(node)))
            in_degree = len(list(G.predecessors(node)))
            out_degree = len(list(G.successors(node)))
            
            node_text.append(
                f"{label}<br>"
                f"Connections: {connections}<br>"
                f"In: {in_degree}, Out: {out_degree}"
            )
        
        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text
        
        # Create edge traces (normal and bottleneck)
        edge_x, edge_y = [], []
        bottleneck_x, bottleneck_y = [], []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            if edge in bottleneck_edges:
                # Bottleneck edge
                bottleneck_x.extend([x0, x1, None])
                bottleneck_y.extend([y0, y1, None])
            else:
                # Normal edge
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color=self.colors['primary']),
            hoverinfo='none'
        )
        
        bottleneck_trace = go.Scatter(
            x=bottleneck_x, y=bottleneck_y,
            mode='lines',
            line=dict(width=3, color=self.colors['secondary']),
            hoverinfo='none'
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, bottleneck_trace, node_trace],
            layout=go.Layout(
                title=f"Process Flow Diagram ({len(G.nodes())} activities)",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800,
                template="plotly_white" if self.style != 'dark' else "plotly_dark"
            )
        )
        
        # Add annotations for labels
        for node, (x, y) in pos.items():
            try:
                label = task_encoder.inverse_transform([int(node)])[0]
                # Truncate long label
                if len(label) > 15:
                    label = label[:12] + "..."
            except:
                label = f"Task {node}"
                
            fig.add_annotation(
                x=x,
                y=y,
                text=label,
                showarrow=False,
                font=dict(size=10),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="gray",
                borderwidth=1
            )
        
        # Save figure
        if self.output_dir:
            # Ensure filename has .html extension
            if not filename.lower().endswith('.html'):
                filename += '.html'
            
            filepath = os.path.join(self.output_dir, filename)
            
            fig.write_html(
                filepath,
                include_plotlyjs='cdn',
                full_html=True,
                auto_open=False
            )
            
            logger.info(f"Saved interactive process flow to {filepath}")
            return filepath
        
        return None
    
    def transition_heatmap(
        self, 
        transitions: pd.DataFrame,
        task_encoder: Any,
        filename: str = "transition_heatmap",
        max_activities: int = 20
    ) -> Optional[str]:
        """
        Create transition probability heatmap with memory optimization
        
        Args:
            transitions: DataFrame with transition data or transition matrix
            task_encoder: Task label encoder
            filename: Output filename
            max_activities: Maximum number of activities to display
            
        Returns:
            Path to saved figure or None
        """
        # Build transition matrix if not already provided
        if isinstance(transitions, pd.DataFrame) and 'task_id' in transitions.columns:
            # This is the raw transitions dataframe, create matrix
            trans_count = pd.crosstab(
                transitions["task_id"], 
                transitions["next_task_id"],
                normalize=False
            )
            
            # Calculate probability matrix
            row_sums = trans_count.sum(axis=1)
            prob_matrix = trans_count.div(row_sums, axis=0).fillna(0)
        else:
            # Assume it's already a transition matrix
            prob_matrix = transitions
        
        # Limit matrix size for readability
        if len(prob_matrix) > max_activities:
            # Find most important activities
            activity_importance = prob_matrix.sum(axis=1) + prob_matrix.sum(axis=0)
            top_activities = activity_importance.nlargest(max_activities).index
            prob_matrix = prob_matrix.loc[top_activities, top_activities]
            logger.info(f"Limiting heatmap to {max_activities} activities out of {len(activity_importance)}")
        
        # Get task names for labels
        try:
            xlabels = [task_encoder.inverse_transform([int(c)])[0] for c in prob_matrix.columns]
            ylabels = [task_encoder.inverse_transform([int(r)])[0] for r in prob_matrix.index]
            
            # Truncate long labels
            xlabels = [l[:15] + "..." if len(l) > 15 else l for l in xlabels]
            ylabels = [l[:15] + "..." if len(l) > 15 else l for l in ylabels]
        except:
            # Fallback to IDs
            xlabels = [f"Task {c}" for c in prob_matrix.columns]
            ylabels = [f"Task {r}" for r in prob_matrix.index]
        
        # Check for interactive visualization
        if self.use_interactive and not filename.endswith('.png'):
            return self._create_interactive_heatmap(
                prob_matrix, xlabels, ylabels, filename
            )
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(
            figsize=(
                max(10, min(20, len(prob_matrix.columns) * 0.7)),
                max(8, min(16, len(prob_matrix.index) * 0.7))
            )
        )
        
        # Use Seaborn for better looking heatmap if available
        if HAS_SEABORN:
            import seaborn as sns
            sns.heatmap(
                prob_matrix,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                xticklabels=xlabels,
                yticklabels=ylabels,
                ax=ax
            )
        else:
            # Use matplotlib
            im = ax.imshow(prob_matrix, cmap="YlGnBu")
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Probability", rotation=-90, va="bottom")
            
            # Set labels
            ax.set_xticks(np.arange(len(xlabels)))
            ax.set_yticks(np.arange(len(ylabels)))
            ax.set_xticklabels(xlabels)
            ax.set_yticklabels(ylabels)
            
            # Annotate cells
            for i in range(len(ylabels)):
                for j in range(len(xlabels)):
                    text = ax.text(j, i, f"{prob_matrix.iloc[i, j]:.2f}",
                                   ha="center", va="center", color="black" if prob_matrix.iloc[i, j] < 0.5 else "white")
        
        # Rotate x-axis labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add styling
        ax.set_title("Transition Probability Heatmap", fontsize=14)
        ax.set_xlabel("Next Activity")
        ax.set_ylabel("Current Activity")
        
        # Adjust layout for rotated labels
        plt.tight_layout()
        
        # Save figure
        return self._save_figure(fig, filename)
    
    def _create_interactive_heatmap(
        self,
        matrix: pd.DataFrame,
        xlabels: List[str],
        ylabels: List[str],
        filename: str
    ) -> Optional[str]:
        """
        Create interactive heatmap with Plotly
        
        Args:
            matrix: Transition probability matrix
            xlabels: X-axis labels
            ylabels: Y-axis labels
            filename: Output filename
            
        Returns:
            Path to saved figure or None
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly not available. Falling back to static visualization.")
            filename += '.png'
            return self.transition_heatmap(matrix, None, filename)
        
        # Create figure
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=xlabels,
            y=ylabels,
            colorscale='YlGnBu',
            hoverongaps=False,
            text=matrix.values,
            hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{z:.3f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title="Transition Probability Heatmap",
            xaxis_title="Next Activity",
            yaxis_title="Current Activity",
            template="plotly_white" if self.style != 'dark' else "plotly_dark",
            height=max(500, len(ylabels) * 25),
            width=max(700, len(xlabels) * 35)
        )
        
        # Save figure
        if self.output_dir:
            # Ensure filename has .html extension
            if not filename.lower().endswith('.html'):
                filename += '.html'
            
            filepath = os.path.join(self.output_dir, filename)
            
            fig.write_html(
                filepath,
                include_plotlyjs='cdn',
                full_html=True,
                auto_open=False
            )
            
            logger.info(f"Saved interactive heatmap to {filepath}")
            return filepath
        
        return None
    
    def create_dashboard(
        self, 
        df: Optional[pd.DataFrame] = None,
        cycle_times: Optional[np.ndarray] = None,
        bottleneck_stats: Optional[pd.DataFrame] = None,
        significant_bottlenecks: Optional[pd.DataFrame] = None,
        task_encoder: Optional[Any] = None,
        case_stats: Optional[pd.DataFrame] = None,
        resource_stats: Optional[pd.DataFrame] = None,
        variant_stats: Optional[pd.DataFrame] = None,
        filename: str = "dashboard"
    ) -> Optional[str]:
        """
        Create comprehensive dashboard combining multiple visualizations
        
        Args:
            df: Process data dataframe (optional)
            cycle_times: Array of case durations in hours (optional)
            bottleneck_stats: DataFrame with bottleneck statistics (optional)
            significant_bottlenecks: DataFrame with significant bottlenecks (optional)
            task_encoder: Task label encoder (optional)
            case_stats: DataFrame with case statistics (optional)
            resource_stats: DataFrame with resource statistics (optional)
            variant_stats: DataFrame with variant statistics (optional)
            filename: Output filename
            
        Returns:
            Path to saved dashboard or None
        """
        # Check if interactive visualization is available
        if not self.has_plotly:
            logger.error("Plotly is required for dashboard creation")
            return None
        
        if not self.use_interactive:
            logger.warning("Interactive visualizations disabled. Dashboard requires interactive mode.")
            return None
        
        # Import Plotly only when needed
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("Plotly is required for dashboard creation")
            return None
        
        # Create subplot figure with appropriate layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Cycle Time Distribution", 
                "Top Bottlenecks",
                "Activity Distribution", 
                "Resource Utilization"
            ],
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12
        )
        
        # 1. Cycle time distribution
        if cycle_times is not None or (case_stats is not None and 'duration_h' in case_stats.columns):
            # Get cycle times data
            if cycle_times is None:
                cycle_times = case_stats['duration_h'].values
            
            # Sample data if needed
            if self.memory_efficient and len(cycle_times) > self.sampling_threshold:
                cycle_times = self._sample_data(cycle_times)
            
            # Calculate statistics
            mean_duration = np.mean(cycle_times)
            median_duration = np.median(cycle_times)
            p90 = np.percentile(cycle_times, 90)
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=cycle_times,
                    nbinsx=30,
                    marker_color=self.colors['primary'],
                    opacity=0.7,
                    name="Cycle Time"
                ),
                row=1, col=1
            )
            
            # Add mean line
            fig.add_vline(
                x=mean_duration,
                line_dash="solid",
                line_color=self.colors['secondary'],
                row=1, col=1,
                annotation_text=f"Mean: {mean_duration:.1f}h"
            )
            
            # Add median line
            fig.add_vline(
                x=median_duration,
                line_dash="dash",
                line_color=self.colors['tertiary'],
                row=1, col=1,
                annotation_text=f"Median: {median_duration:.1f}h"
            )
        
        # 2. Top bottlenecks
        if bottleneck_stats is not None and significant_bottlenecks is not None:
            # Get top bottlenecks
            top_bottlenecks = significant_bottlenecks.head(min(8, len(significant_bottlenecks)))
            
            if len(top_bottlenecks) > 0:
                # Create labels
                labels = []
                for _, row in top_bottlenecks.iterrows():
                    src_id, dst_id = int(row["task_id"]), int(row["next_task_id"])
                    try:
                        if task_encoder is not None:
                            src_name = task_encoder.inverse_transform([src_id])[0]
                            dst_name = task_encoder.inverse_transform([dst_id])[0]
                            
                            # Truncate long names
                            if len(src_name) > 10:
                                src_name = src_name[:7] + "..."
                            if len(dst_name) > 10:
                                dst_name = dst_name[:7] + "..."
                                
                            labels.append(f"{src_name} → {dst_name}")
                        else:
                            labels.append(f"{src_id} → {dst_id}")
                    except:
                        labels.append(f"{src_id} → {dst_id}")
                
                # Add horizontal bar chart
                fig.add_trace(
                    go.Bar(
                        y=labels,
                        x=top_bottlenecks["mean_hours"].values,
                        marker_color=self.colors['primary'],
                        orientation='h',
                        name="Wait Time"
                    ),
                    row=1, col=2
                )
        
        # 3. Activity distribution
        if df is not None:
            # Get activity counts
            activity_counts = df["task_id"].value_counts().nlargest(10)
            
            # Create labels
            try:
                if task_encoder is not None:
                    activity_labels = [task_encoder.inverse_transform([int(i)])[0] for i in activity_counts.index]
                    # Truncate long names
                    activity_labels = [l[:12] + "..." if len(l) > 12 else l for l in activity_labels]
                else:
                    activity_labels = [f"Task {i}" for i in activity_counts.index]
            except:
                activity_labels = [f"Task {i}" for i in activity_counts.index]
            
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=activity_labels,
                    y=activity_counts.values,
                    marker_color=self.colors['primary'],
                    name="Activity Count"
                ),
                row=2, col=1
            )
        
        # 4. Resource utilization
        if resource_stats is not None:
            # Get top resources
            top_resources = resource_stats.head(min(10, len(resource_stats)))
            
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=[f"R{i}" for i in top_resources.index],
                    y=top_resources["workload_percentage"].values,
                    marker_color=self.colors['primary'],
                    name="Workload %"
                ),
                row=2, col=2
            )
        elif df is not None:
            # Calculate resource counts
            resource_counts = df["resource_id"].value_counts().nlargest(10)
            
            # Add bar chart
            fig.add_trace(
                go.Bar(
                    x=[f"R{i}" for i in resource_counts.index],
                    y=resource_counts.values,
                    marker_color=self.colors['primary'],
                    name="Resource Count"
                ),
                row=2, col=2
            )
        
        # Add process statistics
        if df is not None:
            stats_text = (
                f"Cases: {df['case_id'].nunique():,}<br>"
                f"Activities: {df['task_id'].nunique():,}<br>"
                f"Resources: {df['resource_id'].nunique():,}<br>"
                f"Events: {len(df):,}"
            )
            
            # Add annotation
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                text=stats_text,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=4,
                align="center"
            )
        
        # Update layout
        fig.update_layout(
            title_text="Process Mining Dashboard",
            height=800,
            template="plotly_white" if self.style != 'dark' else "plotly_dark"
        )
        
        # Update x-axis titles
        fig.update_xaxes(title_text="Duration (hours)", row=1, col=1)
        fig.update_xaxes(title_text="Wait Time (hours)", row=1, col=2)
        fig.update_xaxes(title_text="Activity", row=2, col=1)
        fig.update_xaxes(title_text="Resource", row=2, col=2)
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Transition", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Workload %", row=2, col=2)
        
        # Save figure
        if self.output_dir:
            # Ensure filename has .html extension
            if not filename.lower().endswith('.html'):
                filename += '.html'
            
            filepath = os.path.join(self.output_dir, filename)
            
            fig.write_html(
                filepath,
                include_plotlyjs='cdn',
                full_html=True,
                auto_open=False
            )
            
            logger.info(f"Saved interactive dashboard to {filepath}")
            return filepath
        
        return None