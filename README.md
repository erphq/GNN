## Process Mining with Graph Neural Networks

An advanced implementation combining Graph Neural Networks, Deep Learning, and Process Mining techniques for business process analysis and prediction.

## 1. Overview

This research project implements a novel approach to process mining using Graph Neural Networks (GNN) and deep learning techniques. The framework combines state-of-the-art machine learning models with traditional process mining methods to provide comprehensive process analysis and prediction capabilities.

## 2. Authors

- **Somesh Misra** [@mathprobro](https://x.com/mathprobro)
- **Shashank Dixit** [@sphinx](https://x.com/protosphinx)
- **Research Group**: [ERP•AI](https://www.erp.ai) Research

## 3. Key Components

1. **Process Analysis**
- Advanced bottleneck detection using temporal analysis
- Conformance checking with inductive mining
- Cycle time analysis and prediction
- Transition pattern discovery
- Spectral clustering for process segmentation

2. **Machine Learning Models**
- Graph Attention Networks (GAT) for structural learning
- LSTM networks for temporal dependencies
- Reinforcement Learning for process optimization
- Custom neural architectures for process prediction

3. **Visualization Suite**
- Interactive process flow visualization
- Temporal pattern analysis
- Performance bottleneck identification
- Resource utilization patterns
- Custom process metrics

## 4. Technical Architecture

```
src/
├── input/                # input files
├── models/
│   ├── gat_model.py      # Graph Attention Network implementation
│   └── lstm_model.py     # LSTM sequence model
├── modules/
│   ├── data_preprocessing.py  # Data handling and feature engineering
│   ├── process_mining.py     # Core process mining functions
│   └── rl_optimization.py    # Reinforcement learning components
├── visualization/
│   └── process_viz.py        # Visualization toolkit
└── main.py                   # Main execution script
```

## 5. Technical Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- PM4Py
- NetworkX
- Additional dependencies in requirements.txt

## 6. Installation

1. Clone the repository:
```bash
git clone https://github.com/erphq/GNN.git
cd GNN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 7. Data Requirements

The system expects process event logs in CSV format with the following structure:
- case_id: Process instance identifier
- task_name: Activity name
- timestamp: Activity timestamp
- resource: Resource identifier
- amount: Numerical attribute (if applicable)

## 8. Usage

```bash
python main.py <input-file-path>
```

Results are stored in timestamped directories under `results/` with the following structure:
```
results/run_timestamp/
├── models/          # Trained model weights
├── visualizations/  # Generated visualizations
├── metrics/         # Performance metrics
├── analysis/        # Detailed analysis results
└── policies/        # Learned optimization policies
```

## 9. Technical Details

Graph Neural Network Architecture
- Multi-head attention mechanisms
- Dynamic graph construction
- Adaptive feature learning
- Custom loss functions for process-specific metrics

LSTM Implementation
- Bidirectional sequence modeling
- Variable-length sequence handling
- Custom embedding layer for process activities

Process Mining Components
- Inductive miner implementation
- Token-based replay
- Custom conformance checking metrics
- Advanced bottleneck detection algorithms

Reinforcement Learning
- Custom environment for process optimization
- State-action space modeling
- Policy gradient methods
- Resource allocation optimization

## 10. Contributing

We welcome contributions from the research community. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request with detailed documentation

## 11. Citation

If you use this code in your research, please cite:

```bibtex
@software{GNN_ProcessMining,
  author = {Shashank Dixit/Somesh Misra},
  title = {Process Mining with Graph Neural Networks},
  year = {2025},
  publisher = {ERP•AI},
  url = {https://github.com/erphq/GNN}
}
``` 
