# ProcessMine: Memory-Efficient Process Mining with DGL Integration

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)

ProcessMine is a high-performance, memory-efficient toolkit for process mining using Graph Neural Networks, LSTMs, and Reinforcement Learning. It provides advanced analytics and predictive capabilities for business process optimization, with a focus on handling large datasets efficiently. The toolkit now features full DGL (Deep Graph Library) integration for optimized graph operations.

## Key Features

- **DGL-Optimized Graph Operations**:
  - Efficient graph construction and sampling for large process logs
  - Sparse attention implementations for large-scale graphs
  - Checkpointing for reduced memory usage during backpropagation

- **Memory Optimization**: Handles large process logs with minimal memory footprint through:
  - Dynamic chunking and streaming processing
  - Vectorized operations for performance
  - Adaptive batching strategies based on available memory
  - Efficient data structures with proper type handling

- **Advanced Model Architectures**:
  - Graph Neural Networks with multiple attention mechanisms:
    - Basic attention for standard process analysis
    - Positional attention to capture sequential information
    - Diverse attention to prevent attention collapse
    - Combined attention integrating all mechanisms
  - Sequence models with LSTM and attention for next activity prediction
  - Reinforcement Learning with multi-objective reward functions

- **Comprehensive Analysis**:
  - Bottleneck detection and analysis
  - Cycle time analysis and forecasting
  - Process variant identification
  - Resource workload analysis
  - Automated conformance checking

- **Interactive Visualizations**:
  - Process flow diagrams with bottleneck highlighting
  - Sankey diagrams of process transitions
  - Attention heatmaps for model interpretability
  - Comprehensive dashboards for analysis
  - Memory-efficient rendering for large datasets

- **Accelerated Training**:
  - Mixed precision training for GPUs
  - Memory-efficient batching strategies
  - CUDA-optimized implementations
  - Gradient checkpointing for large models

- **Ablation Studies**:
  - Systematic testing of model components
  - Parallel experimentation
  - Comprehensive reporting and visualization

- **Simple Interface**:
  - Unified model interfaces with consistent APIs
  - Command-line tools for quick analysis
  - Python API for integration with existing applications

## Installation

### Quick Installation

```bash
pip install processmine
```

### Install with Optional Dependencies

```bash
# For graph neural networks
pip install "processmine[gnn]"

# For visualizations
pip install "processmine[viz]"

# For traditional machine learning models
pip install "processmine[ml]"

# Full installation with all dependencies
pip install "processmine[all]"

# Development installation
pip install "processmine[all,dev]"
```

### Installation from Source

```bash
git clone https://github.com/erp-ai/processmine.git
cd processmine
pip install -e ".[all]"
```

## Quick Start

### Command Line Usage

```bash
# Basic process analysis
processmine analyze path/to/process_log.csv

# Train a model with DGL integration
processmine train path/to/process_log.csv --model enhanced_gnn --use_edge_features

# Run an ablation study to evaluate components
processmine ablation path/to/process_log.csv --ablate_components use_positional_encoding,use_diverse_attention

# Full pipeline (analyze, train, optimize)
processmine full path/to/process_log.csv --output-dir results/my_analysis
```

### Python API

```python
from processmine import run_analysis, create_model, load_and_preprocess_data
from processmine.visualization.viz import ProcessVisualizer
from processmine.core.advanced_workflow import run_advanced_workflow

# Load and preprocess data
df, task_encoder, resource_encoder = load_and_preprocess_data("process_log.csv")

# Run analysis
analysis_results = run_analysis(df)

# Create and train a model with combined attention
model = create_model(
    model_type="enhanced_gnn",
    input_dim=len([col for col in df.columns if col.startswith("feat_")]),
    hidden_dim=64,
    output_dim=len(task_encoder.classes_),
    attention_type="combined",
    use_positional_encoding=True,
    use_diverse_attention=True
)

# Create visualizations
viz = ProcessVisualizer(output_dir="results/visualizations")
viz.process_flow(
    analysis_results["bottleneck_stats"],
    task_encoder,
    analysis_results["significant_bottlenecks"]
)

# Use advanced workflow with all enhancements
workflow_results = run_advanced_workflow(
    data_path="process_log.csv",
    output_dir="results/advanced",
    model="enhanced_gnn",
    use_positional_encoding=True,
    use_diverse_attention=True,
    use_multi_objective_loss=True,
    use_adaptive_normalization=True
)
```

## Memory Efficiency Guidelines

ProcessMine is designed to handle large process logs efficiently. Here are some tips for maximizing performance:

1. **Use chunking for large files**: When loading large process logs, set appropriate `chunk_size` in `load_and_preprocess_data`.
2. **Enable memory-efficient mode**: Use `mem_efficient=True` for models and training to reduce memory usage at the cost of slightly slower processing.
3. **Use DGL sampling for large graphs**: Enable graph sampling with `dgl_sampling='neighbor'` or `'topk'` when working with very large processes.
4. **Enable sparse attention**: For large graphs, use `sparse_attention=True` to reduce memory requirements.
5. **Use gradient checkpointing**: Enable `use_checkpointing=True` for training very deep models with limited GPU memory.
6. **Optimize batch size**: For very large graphs, use smaller batch sizes with `build_graph_data(batch_size=100)`.
7. **Use CUDA wisely**: Clear CUDA cache when needed with `clear_memory(full_clear=True)`.
8. **Leverage sampling for huge datasets**: For exploratory analysis, sample the data first to get quick insights.

## Documentation

For full documentation, visit [docs.processmine.com](https://docs.processmine.com).

- [Tutorial](https://docs.processmine.com/tutorial)
- [API Reference](https://docs.processmine.com/api)
- [Examples](https://docs.processmine.com/examples)

## Examples

### Bottleneck Analysis and Visualization

```python
import pandas as pd
from processmine.process_mining.analysis import analyze_bottlenecks
from processmine.visualization.viz import ProcessVisualizer

# Load process data
df = pd.read_csv("process_log.csv")

# Analyze bottlenecks
bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(
    df,
    freq_threshold=5,
    percentile_threshold=90.0
)

# Visualize bottlenecks
viz = ProcessVisualizer(output_dir="results")
viz.bottleneck_analysis(bottleneck_stats, significant_bottlenecks, task_encoder)
```

### Training a GNN Model with Combined Attention

```python
from processmine import create_model, load_and_preprocess_data
from processmine.data.graphs import build_graph_data
from processmine.core.training import train_model, evaluate_model
import torch

# Load and preprocess data
df, task_encoder, resource_encoder = load_and_preprocess_data("process_log.csv")

# Build graph data with edge features enabled
graphs = build_graph_data(df, enhanced=True, use_edge_features=True)

# Create enhanced GNN model with combined attention
model = create_model(
    model_type="enhanced_gnn",
    input_dim=len([col for col in df.columns if col.startswith("feat_")]),
    hidden_dim=64,
    output_dim=len(task_encoder.classes_),
    attention_type="combined",
    use_positional_encoding=True,
    use_diverse_attention=True,
    diversity_weight=0.1
)

# Train model with memory optimization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Use multi-objective loss function
from processmine.models.gnn.architectures import ProcessLoss
criterion = ProcessLoss(task_weight=0.5, time_weight=0.3, structure_weight=0.2)

model, metrics = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    epochs=20,
    use_amp=True,  # Enable mixed precision
    memory_efficient=True
)

# Evaluate model
eval_metrics = evaluate_model(model, test_loader, device)
```

### Running Ablation Studies

```python
from processmine.core.ablation_runner import run_ablation_study

# Configure and run ablation study
results = run_ablation_study(
    data_path="process_log.csv",
    base_model="enhanced_gnn",
    output_dir="results/ablation",
    ablation_config={
        "components": [
            "use_positional_encoding",
            "use_diverse_attention", 
            "use_batch_norm",
            "use_adaptive_normalization"
        ],
        "disable": True  # Test by disabling each component
    },
    epochs=15,
    parallel=True,  # Run experiments in parallel
    max_workers=4
)

# Visualize ablation results
from processmine.visualization.ablation_viz import visualize_ablation_results
visualize_ablation_results(results, output_dir="results/ablation/viz")
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ProcessMine in your research, please cite:

```
@software{processmine2025,
  author = {ERP.AI},
  title = {ProcessMine: Memory-Efficient Process Mining with Graph Neural Networks},
  year = {2025},
  url = {https://github.com/erp-ai/processmine}
}
```