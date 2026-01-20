# Changelog

## v1.5.0 - 01/20/2026

**Major Features**:

-   **Multi-CPU parallel processing**: New `n_jobs` parameter enables parallel analysis across multiple CPU cores for significant speedup
-   **Checkpointing and resume**: Robust checkpoint system allows seamless resumption of interrupted analyses without data loss
-   **Auto-tuning of prob_threshold**: Adaptive algorithm automatically tunes `prob_threshold` based on `point_max` when set to `None`
-   **Visualization utilities**: Comprehensive visualization module with 5 plotting functions for analyzing results
-   **Analysis helpers**: Statistical analysis tools for comparing transitions and extracting insights
-   **Comprehensive code refactoring**: Improved code organization, documentation, and maintainability throughout the codebase

**New Modules**:

-   `MDTerp.visualization`: Feature importance plots, heatmaps, transition comparisons, and automated report generation
-   `MDTerp.analysis`: Statistical analysis, transition comparison, consensus feature identification, and CSV export

**API Changes**:

-   Added `n_jobs` parameter to `MDTerp.run()` (default: None, uses all CPUs)
-   Added `resume` parameter to `MDTerp.run()` (default: True)
-   Made `prob_threshold` optional in `MDTerp.run()` (default: None triggers auto-tuning)
-   New `CheckpointManager` class exported for advanced checkpoint management
-   Exported `transition_summary` and `dominant_feature` utility functions
-   Exported `visualization` and `analysis` modules for result analysis

**Improvements**:

-   Enhanced docstrings with examples and paper citations
-   Added type hints throughout codebase
-   Better variable naming for improved readability
-   Modular architecture with focused helper methods
-   Comprehensive logging of parallel execution and progress
-   Atomic file writes prevent checkpoint corruption
-   Worker-specific temporary directories prevent file conflicts

**Performance**:

-   Near-linear speedup with number of CPU cores
-   Minimal overhead from checkpointing
-   Efficient work distribution for parallel execution

**Backward Compatibility**:

-   All existing code continues to work unchanged
-   New parameters are optional with sensible defaults

## v1.0.0 - 07/31/2025

**Improvement**:

-   MDTerp published
