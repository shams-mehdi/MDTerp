"""
MDTerp.base -- Main MDTerp pipeline orchestrator.

Coordinates transition-state detection, parallel point analysis,
crash recovery, and result aggregation using the TERP framework.

Reference:
    Mehdi, S. & Tiwary, P. "Thermodynamics-inspired explanations of
    artificial intelligence." Nature Communications (2024).
"""
import numpy as np
import os

from MDTerp.utils import log_maker, input_summary, select_transition_points
from MDTerp.parallel import run_parallel as _run_parallel
from MDTerp.checkpoint import (
    save_run_config,
    scan_completed_points,
    aggregate_results,
)


class run:
    """
    Main class for implementing MDTerp analysis.

    MDTerp interprets black-box AI classifiers trained on molecular dynamics
    data by identifying transition-state samples and computing local feature
    importance using the TERP framework (Thermodynamics-inspired Explanations
    using Ridge regression with Perturbation).

    Attributes:
        results: Dictionary mapping sample indices to [transition, importance].
        feature_names: Array of feature name strings.
        points: Dictionary of selected transition-state points.
        thresholds: Dictionary of effective per-transition probability thresholds.
        config: Dictionary of all run parameters.
    """

    def __init__(
        self,
        np_data: np.ndarray,
        model_function_loc: str,
        numeric_dict: dict = None,
        angle_dict: dict = None,
        sin_cos_dict: dict = None,
        save_dir: str = './results/',
        point_max: int = 50,
        prob_threshold_min: float = 0.475,
        num_samples: int = 10000,
        cutoff: int = 15,
        seed: int = 0,
        unfaithfulness_threshold: float = 0.01,
        periodicity_upper: float = np.pi,
        periodicity_lower: float = -np.pi,
        alpha: float = 1.0,
        save_all: bool = False,
        n_workers: int = None,
        keep_checkpoints: bool = True,
        use_all_cutoff_features: bool = False,
    ) -> None:
        """
        Configure and execute the MDTerp analysis pipeline.

        Args:
            np_data: 2D training data array (samples x features).
            model_function_loc: Path to file defining load_model() and
                run_model() functions for the black-box classifier.
            numeric_dict: Feature name -> [column_index] for numeric features.
            angle_dict: Feature name -> [column_index] for angular features.
            sin_cos_dict: Feature name -> [sin_index, cos_index] for sin/cos pairs.
            save_dir: Directory to save all results (default: './results/').
            point_max: Target number of points per transition (default: 50).
            prob_threshold_min: Minimum probability threshold for transition
                detection (default: 0.475). Applied as a floor per transition.
            num_samples: Number of perturbed neighborhood samples (default: 10000).
            cutoff: Maximum features kept after initial round (default: 15).
            seed: Random seed (default: 0).
            unfaithfulness_threshold: Stopping criterion for forward feature
                selection (default: 0.01).
            periodicity_upper: Upper bound for angular periodicity (default: pi).
            periodicity_lower: Lower bound for angular periodicity (default: -pi).
            alpha: Ridge regression L2 penalty (default: 1.0).
            save_all: Keep intermediate DATA directories (default: False).
            n_workers: Number of parallel worker processes. Defaults to the
                number of available CPUs.
            keep_checkpoints: Keep per-point result files after aggregation
                (default: True).
            use_all_cutoff_features: If True (default), retain all cutoff features
                in per-point importance instead of pruning via interpretation
                entropy. This is useful because a feature may be irrelevant for
                one sample but relevant for another within the same transition
                ensemble. Final transition-level importance is obtained by
                averaging across samples.
        """
        if numeric_dict is None:
            numeric_dict = {}
        if angle_dict is None:
            angle_dict = {}
        if sin_cos_dict is None:
            sin_cos_dict = {}
        if n_workers is None:
            n_workers = os.cpu_count() or 1

        self.config = {
            'model_function_loc': model_function_loc,
            'save_dir': save_dir,
            'point_max': point_max,
            'prob_threshold_min': prob_threshold_min,
            'num_samples': num_samples,
            'cutoff': cutoff,
            'seed': seed,
            'unfaithfulness_threshold': unfaithfulness_threshold,
            'periodicity_upper': periodicity_upper,
            'periodicity_lower': periodicity_lower,
            'alpha': alpha,
            'save_all': save_all,
            'n_workers': n_workers,
            'keep_checkpoints': keep_checkpoints,
            'use_all_cutoff_features': use_all_cutoff_features,
        }

        self.results = None
        self.feature_names = None
        self.points = None
        self.thresholds = None

        self._execute(
            np_data, model_function_loc, numeric_dict, angle_dict,
            sin_cos_dict, save_dir, point_max, prob_threshold_min,
            num_samples, cutoff, seed, unfaithfulness_threshold,
            periodicity_upper, periodicity_lower, alpha, save_all,
            n_workers, keep_checkpoints, use_all_cutoff_features,
        )

    def _execute(
        self, np_data, model_function_loc, numeric_dict, angle_dict,
        sin_cos_dict, save_dir, point_max, prob_threshold_min,
        num_samples, cutoff, seed, unfaithfulness_threshold,
        periodicity_upper, periodicity_lower, alpha, save_all,
        n_workers, keep_checkpoints, use_all_cutoff_features,
    ):
        """Internal pipeline execution."""
        os.makedirs(save_dir, exist_ok=True)
        logger = log_maker(save_dir)
        input_summary(logger, numeric_dict, angle_dict, sin_cos_dict, save_dir, np_data)

        # Load model for transition detection
        logger.info('Loading black-box model from file >>> ' + model_function_loc)
        with open(model_function_loc, 'r') as f:
            func_code = f.read()
        local_ns = {}
        exec(func_code, globals(), local_ns)
        model = local_ns["load_model"]()
        logger.info("Model loaded!")

        # Detect transition states with adaptive thresholds
        state_probabilities = local_ns["run_model"](model, np_data)
        np.random.seed(seed)
        points, thresholds = select_transition_points(
            state_probabilities, point_max, prob_threshold_min,
        )
        self.points = points
        self.thresholds = thresholds

        n_transitions = len(points)
        logger.info(f"Number of state transitions detected >>> {n_transitions}")
        for trans, thresh in thresholds.items():
            n_pts = len(points[trans])
            logger.info(
                f"  Transition {trans}: {n_pts} points, "
                f"effective threshold = {thresh:.6f}"
            )
        if n_transitions == 0:
            logger.info("No transition detected. Check hyperparameters!")
            raise ValueError("No transition detected. Check hyperparameters!")
        logger.info(f"Max features per point (cutoff) >>> {cutoff}")
        logger.info(
            f"use_all_cutoff_features >>> {use_all_cutoff_features}"
        )
        logger.info(100 * '-')

        # Save run config for resume validation
        save_run_config(save_dir, {
            **self.config,
            'points': {k: v.tolist() for k, v in points.items()},
        })

        # Check for previously completed points (crash recovery)
        completed = scan_completed_points(save_dir)
        total_points = sum(len(v) for v in points.values())
        if completed:
            logger.info(
                f"Resuming: found {len(completed)}/{total_points} completed "
                f"results, {total_points - len(completed)} remaining"
            )

        # Build work queue, skipping completed points
        work_items = []
        for transition in points:
            for point_index in range(len(points[transition])):
                if (transition, point_index) in completed:
                    continue
                work_items.append({
                    'save_dir': save_dir,
                    'transition': transition,
                    'point_index': point_index,
                    'sample_index': int(points[transition][point_index]),
                    'training_data': np_data,
                    'numeric_dict': numeric_dict,
                    'angle_dict': angle_dict,
                    'sin_cos_dict': sin_cos_dict,
                    'seed': seed,
                    'num_samples': num_samples,
                    'cutoff': cutoff,
                    'unfaithfulness_threshold': unfaithfulness_threshold,
                    'periodicity_upper': periodicity_upper,
                    'periodicity_lower': periodicity_lower,
                    'alpha': alpha,
                    'save_all': save_all,
                    'use_all_cutoff_features': use_all_cutoff_features,
                })

        if not work_items:
            logger.info("All points already completed. Aggregating results.")
        else:
            logger.info(
                f"Analyzing {len(work_items)} points using {n_workers} workers..."
            )

            # Run analysis — model stays in main process (GPU-safe),
            # CPU work is parallelized across workers
            run_model_fn = local_ns["run_model"]
            results = _run_parallel(work_items, run_model_fn, model, n_workers)

            # Log results
            for r in results:
                if r['status'] == 'completed':
                    logger.info(
                        f"Completed {r['transition']} point {r['point_index']}: "
                        f"round 1 features = {r['n_round1_features']}, "
                        f"final features = {r['n_final_features']}"
                    )
                else:
                    logger.error(
                        f"Failed {r['transition']} point {r['point_index']}: "
                        f"{r['error']}"
                    )

        # Derive feature names from dictionaries
        feature_names = (
            list(numeric_dict.keys())
            + list(angle_dict.keys())
            + list(sin_cos_dict.keys())
        )

        # Aggregate all per-point results
        self.feature_names = np.array(feature_names)
        self.results = aggregate_results(
            save_dir, self.feature_names, keep_checkpoints,
        )

        logger.info(
            "Feature names saved at >>> "
            + os.path.join(save_dir, 'MDTerp_feature_names.npy')
        )
        logger.info(
            "All results saved at >>> "
            + os.path.join(save_dir, 'MDTerp_results_all.pkl')
        )
        logger.info("Completed!")

        # Clean up logger
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
