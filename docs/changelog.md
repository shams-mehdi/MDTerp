# Changelog

## v2.0.0 - 03/16/2026

**New Features**:

-   Adaptive per-transition probability threshold tuning to select the most informative transition-state samples
-   Multi-CPU parallel analysis via configurable `n_workers` parameter
-   Crash recovery with per-point checkpoint files; interrupted runs resume automatically
-   Forward feature selection using thermodynamics-inspired interpretation free energy
-   Built-in visualization functions: `plot_feature_importance`, `plot_importance_heatmap`, `plot_unfaithfulness_curve`, `plot_point_variability`
-   `transition_summary` and `dominant_feature` result analysis helpers
-   Support for numeric, angular, and sin/cos feature types
-   Configurable `prob_threshold_min` floor for probability thresholds
-   `keep_checkpoints` parameter to control per-point result file retention

**Improvements**:

-   Rewritten README with quick-start example, parameter table, and citation block
-   Expanded documentation with detailed usage guide, model script format, and visualization reference
-   Added CITATION.cff for standardized citation metadata

## v1.0.0 - 07/31/2025

**Improvement**:

-   MDTerp published
