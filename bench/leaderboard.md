### Predictive performance

| dataset | cases | tasks | most-common | **Markov** | LSTM acc / F1 / ΔMarkov | **GAT acc / F1 / ΔMarkov** | GAT ECE | GAT T | GAT dt MAE (h) |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| BPI 2020 Domestic Declarations | 10366 | 17 | 22.0% | **85.4%** | 81.4% / 0.270 / -3.9% | **48.6% / 0.203 / -36.7%** | 0.065 | 0.98 | 50.35 |
| Synthetic Markov (500 cases) | 500 | 8 | 23.2% | **92.0%** | 90.5% / 0.599 / -1.4% | **8.0% / 0.019 / -83.9%** | 0.070 | 1.35 | 0.57 |

### Process-mining quality (PM4Py inductive miner + token replay)

| dataset | fitness | precision | F-score |
|---|---:|---:|---:|
| BPI 2020 Domestic Declarations | 1.000 | 0.247 | 0.396 |
| Synthetic Markov (500 cases) | 1.000 | 1.000 | 1.000 |
