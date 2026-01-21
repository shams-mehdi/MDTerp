# MDTerp


[![image](https://img.shields.io/pypi/v/MDTerp.svg)](https://pypi.python.org/pypi/MDTerp)
![MDTerp](./docs/MDTerp_logo.png)


**A python project for interpreting molecular dynamics trajectory metastable state classifications from machine-learning models**

MDTerp uses thermodynamics-inspired explanations to provide feature importance for black-box AI models trained on molecular dynamics data. Based on the methodology from:

> **"Thermodynamics-inspired explanations of artificial intelligence"**
> Shams Mehdi and Pratyush Tiwary
> *Nature Communications* (2023)

-   Free software: MIT License
-   Documentation: https://shams-mehdi.github.io/MDTerp


## Features

-   Scans blackbox AI classifier models trained on MD data for feature importance
-   Multi-CPU parallel processing for faster analysis
-   Automatic checkpointing and resume functionality for long-running analyses
-   Adaptive hyperparameter tuning (prob_threshold auto-tuning)
-   Comprehensive visualization utilities for analyzing results
-   Statistical analysis tools for comparing transitions and extracting insights
  
## Installation

To install MDTerp, run this command in your terminal:

```
pip install MDTerp
```

This is the preferred method to install MDTerp, as it will always install the most recent stable release (https://pypi.org/project/MDTerp/).


## Demos

-   [Notebook 1](docs/examples/MDTerp_demo1.ipynb)
-   [Notebook 2](docs/examples/MDTerp_demo2.ipynb)
