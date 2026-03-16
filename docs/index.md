# Welcome to MDTerp

[![image](https://img.shields.io/pypi/v/MDTerp.svg)](https://pypi.python.org/pypi/MDTerp)
![MDTerp](MDTerp_logo.png)

**MDTerp** is a Python package for interpreting molecular dynamics (MD) trajectory metastable state classifications from machine-learning models.

- Free software: MIT License
- Documentation: [https://shams-mehdi.github.io/MDTerp](https://shams-mehdi.github.io/MDTerp)

## Overview

MDTerp implements the **TERP** framework from:

> Mehdi, S. & Tiwary, P. "Thermodynamics-inspired explanations of artificial intelligence." *Nature Communications* **15**, 3974 (2024). [https://doi.org/10.1038/s41467-024-47258-w](https://doi.org/10.1038/s41467-024-47258-w)

Given a black-box classifier that assigns MD trajectory frames to metastable states, MDTerp identifies **which molecular features are most important** for each transition between states. It does this by:

1. **Detecting transition-state samples** — frames where the classifier is uncertain between two states (adaptive per-transition thresholds)
2. **Building local surrogate models** — perturbing features around each transition-state sample and fitting weighted Ridge regression
3. **Forward feature selection** — using thermodynamics-inspired interpretation free energy to determine the optimal number of explanatory features
4. **Aggregating results** — combining per-point importance scores into transition-level summaries

## Key Features (v2.0)

- **Adaptive transition detection** with per-transition probability thresholds
- **Multi-CPU parallel analysis** with configurable worker count
- **Crash recovery** via per-point checkpoint files
- **Built-in visualizations**: importance bar plots, heatmaps, unfaithfulness curves, variability plots
- Support for **numeric**, **angular**, and **sin/cos** feature types

## Demos

- [Blackbox Model Demo](MDTerp_blackbox.ipynb)
- [SPIB Villin Demo](MDTerp_SPIB_villin.ipynb)
