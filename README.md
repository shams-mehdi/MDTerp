# MDTerp

[![image](https://img.shields.io/pypi/v/MDTerp.svg)](https://pypi.python.org/pypi/MDTerp)

![MDTerp](./docs/MDTerp_logo.png)

**MDTerp** interprets black-box AI classifiers trained on molecular dynamics (MD) trajectory data. It identifies which molecular features drive transitions between metastable states by building local, interpretable surrogate models around transition-state samples.

MDTerp implements the **TERP** (Thermodynamics-inspired Explanations using Ridge regression with Perturbation) framework, which uses concepts from statistical mechanics — unfaithfulness as energy, interpretation entropy, and interpretation free energy — to automatically determine the optimal number of features needed to explain each transition.

- Free software: MIT License
- Documentation: https://shams-mehdi.github.io/MDTerp

## Key Features

- **Adaptive transition detection**: Automatically tunes probability thresholds per transition to select the most informative samples
- **Multi-CPU parallel analysis**: Analyze multiple transition-state points simultaneously across available CPU cores
- **Crash recovery**: Resume interrupted analyses from per-point checkpoints
- **Forward feature selection**: Thermodynamics-inspired approach to determine optimal feature count
- **Built-in visualizations**: Feature importance bar plots, cross-transition heatmaps, unfaithfulness curves, and per-point variability plots

## Installation

```bash
pip install MDTerp
```

## Quick Start

```python
import MDTerp

# Run MDTerp analysis
result = MDTerp.run(
    np_data=training_data,
    model_function_loc='path/to/model_script.txt',
    numeric_dict={'feature_1': [0], 'feature_2': [1]},
    save_dir='./results/',
    point_max=50,               # Target points per transition
    prob_threshold_min=0.475,   # Minimum threshold floor
    n_workers=4,                # Parallel CPU workers
)

# Visualize results
MDTerp.plot_feature_importance(
    'results/MDTerp_results_all.pkl',
    'results/MDTerp_feature_names.npy',
    transition='0_1',
)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `point_max` | 50 | Target number of points per transition |
| `prob_threshold_min` | 0.475 | Minimum probability threshold (floor) |
| `n_workers` | CPU count | Number of parallel worker processes |
| `num_samples` | 10000 | Perturbed neighborhood size |
| `cutoff` | 15 | Max features kept after initial round |
| `keep_checkpoints` | True | Keep per-point result files for crash recovery |

## Demos

- [Blackbox Model Demo](MDTerp_blackbox.ipynb)
- [SPIB NTL9 Demo](MDTerp_SPIB_ntl9.ipynb)
- [SPIB Villin Demo](MDTerp_SPIB_villin.ipynb)
- [VAMPNets Demo](MDTerp_VAMPNets.ipynb)

## Citation

If you use MDTerp, please cite:

```bibtex
@article{mehdi2024thermodynamics,
  title={Thermodynamics-inspired explanations of artificial intelligence},
  author={Mehdi, Shams and Tiwary, Pratyush},
  journal={Nature Communications},
  volume={15},
  pages={3974},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s41467-024-47258-w}
}
```

Mehdi, S., Tiwary, P. Thermodynamics-inspired explanations of artificial intelligence. *Nat Commun* **15**, 3974 (2024). https://doi.org/10.1038/s41467-024-47258-w
