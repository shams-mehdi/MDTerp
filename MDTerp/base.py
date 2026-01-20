"""
MDTerp.base.py – Main MDTerp module for interpreting ML models on MD trajectories.

This module implements the MDTerp algorithm which uses thermodynamics-inspired
principles to explain black-box machine learning models trained on molecular
dynamics data. It identifies transition states and computes feature importance
using local linear interpretable models.

Reference:
    "Thermodynamics-inspired explanations of artificial intelligence"
    Shams Mehdi and Pratyush Tiwary, Nature Communications
"""
import numpy as np
import os
import shutil
import pickle
import multiprocessing as mp
from typing import Dict, Optional, Tuple
from logging import Logger
from functools import partial

from MDTerp.neighborhood import generate_neighborhood
from MDTerp.utils import log_maker, input_summary, picker_fn, make_result
from MDTerp.init_analysis import init_model
from MDTerp.final_analysis import final_model
from MDTerp.checkpoint import CheckpointManager


def _analyze_sample_worker(
    sample_index: int,
    transition_name: str,
    np_data: np.ndarray,
    model_function_loc: str,
    numeric_dict: Dict,
    angle_dict: Dict,
    sin_cos_dict: Dict,
    save_dir_base: str,
    seed: int,
    num_samples: int,
    cutoff: int,
    unfaithfulness_threshold: float,
    periodicity_upper: float,
    periodicity_lower: float,
    alpha: float,
    worker_id: int
) -> Tuple[int, str, list, list, int, int]:
    """
    Worker function for parallel sample analysis.

    This function is designed to be pickle-able for multiprocessing.
    Each worker gets its own temporary directory to avoid file conflicts.

    Args:
        sample_index: Index of sample to analyze
        transition_name: Name of transition
        np_data: Full training data array
        model_function_loc: Path to model file
        numeric_dict: Numeric feature dictionary
        angle_dict: Angular feature dictionary
        sin_cos_dict: Sin/cos feature dictionary
        save_dir_base: Base directory for results
        seed: Random seed
        num_samples: Number of perturbed samples
        cutoff: Feature cutoff for stage 1
        unfaithfulness_threshold: Threshold for forward selection
        periodicity_upper: Upper periodicity bound
        periodicity_lower: Lower periodicity bound
        alpha: Ridge regression alpha
        worker_id: Unique worker identifier

    Returns:
        Tuple of (sample_index, transition_name, importance, feature_names,
                 n_selected, n_final)
    """
    # Create worker-specific temporary directory to avoid conflicts
    worker_save_dir = os.path.join(save_dir_base, f'tmp_worker_{worker_id}_{sample_index}')
    os.makedirs(worker_save_dir, exist_ok=True)

    try:
        # Load model
        with open(model_function_loc, 'r') as file:
            func_code = file.read()
        local_ns = {}
        exec(func_code, globals(), local_ns)
        model = local_ns["load_model"]()
        run_model_fn = local_ns["run_model"]

        # Stage 1: Generate full neighborhood and run initial screening
        feature_type_indices, feature_names = generate_neighborhood(
            worker_save_dir, numeric_dict, angle_dict, sin_cos_dict,
            np_data, sample_index, seed, num_samples,
            np.array([]), periodicity_upper, periodicity_lower
        )

        neighborhood_data = np.load(worker_save_dir + '/DATA/make_prediction.npy')
        state_probs_stage1 = run_model_fn(model, neighborhood_data)
        terp_data_stage1 = np.load(worker_save_dir + '/DATA/TERP_dat.npy')

        selected_features = init_model(
            terp_data_stage1, state_probs_stage1, cutoff,
            feature_type_indices, seed, alpha
        )

        # Stage 2: Generate refined neighborhood and compute final importance
        generate_neighborhood(
            worker_save_dir, numeric_dict, angle_dict, sin_cos_dict,
            np_data, sample_index, seed, num_samples,
            selected_features, periodicity_upper, periodicity_lower
        )

        neighborhood_data_refined = np.load(worker_save_dir + '/DATA_2/make_prediction.npy')
        state_probs_stage2 = run_model_fn(model, neighborhood_data_refined)
        terp_data_stage2 = np.load(worker_save_dir + '/DATA_2/TERP_dat.npy')

        raw_importance = final_model(
            terp_data_stage2, state_probs_stage2, unfaithfulness_threshold,
            feature_type_indices, selected_features, seed
        )

        importance = make_result(feature_type_indices, feature_names, raw_importance)

        n_selected = len(selected_features)
        n_final = np.nonzero(importance)[0].shape[0]

        return (sample_index, transition_name, importance, feature_names, n_selected, n_final)

    finally:
        # Cleanup worker temporary directory
        if os.path.exists(worker_save_dir):
            shutil.rmtree(worker_save_dir)


class run:
    """
    Main class for implementing MDTerp feature importance analysis.

    This class performs feature importance analysis for black-box ML models
    trained on molecular dynamics trajectories. It identifies transition states
    between metastable states and computes feature importance using local
    linear surrogate models with thermodynamics-inspired metrics.
    """

    def __init__(
        self,
        np_data: np.ndarray,
        model_function_loc: str,
        numeric_dict: Dict = {},
        angle_dict: Dict = {},
        sin_cos_dict: Dict = {},
        save_dir: str = './results/',
        prob_threshold: Optional[float] = None,
        point_max: int = 50,
        num_samples: int = 10000,
        cutoff: int = 15,
        seed: int = 0,
        unfaithfulness_threshold: float = 0.01,
        periodicity_upper: float = np.pi,
        periodicity_lower: float = -np.pi,
        alpha: float = 1.0,
        resume: bool = True,
        n_jobs: Optional[int] = None
    ) -> None:
        """
        Initialize and run MDTerp feature importance analysis.

        Args:
            np_data: Training data array (samples × features) used to train the
                black-box model. This is the molecular dynamics trajectory data.
            model_function_loc: Path to Python file containing two required functions:
                - load_model(): Returns the trained black-box model (no arguments)
                - run_model(model, data): Returns metastable state probabilities
                See https://shams-mehdi.github.io/MDTerp/docs/examples/ for examples.
            numeric_dict: Feature specification for non-periodic numeric features.
                Keys: feature names, Values: [column_index] in np_data
            angle_dict: Feature specification for angular features in [-π, π].
                Keys: feature names, Values: [column_index] in np_data
            sin_cos_dict: Feature specification for angular features (sin/cos encoding).
                Keys: feature names, Values: [sin_index, cos_index] in np_data
            save_dir: Directory path for saving MDTerp results and logs.
            prob_threshold: Threshold for transition state identification. A sample
                is in a transition state if its probability exceeds this threshold
                for two different metastable states. If None, will be auto-tuned
                based on point_max. Recommended: close to but less than 0.50.
            point_max: Maximum number of transition samples to analyze per transition.
                Higher values increase accuracy but also computation time. Samples
                are uniformly sampled if more are available.
            num_samples: Number of perturbed samples in each neighborhood. Rule of
                thumb: proportional to sqrt(num_features).
            cutoff: Maximum features retained after initial screening for forward
                feature selection. Reduces compute time for high-dimensional data.
            seed: Random seed for reproducibility.
            unfaithfulness_threshold: Lower limit on unfaithfulness (U) for forward
                feature selection termination. Lower values demand better fidelity.
            periodicity_upper: Upper bound for angular feature periodicity.
            periodicity_lower: Lower bound for angular feature periodicity.
            alpha: L2 regularization parameter for Ridge regression.
            resume: If True, resume from checkpoint if available. If False,
                start fresh analysis (overwriting any existing checkpoint).
            n_jobs: Number of parallel processes to use for analysis. If None,
                uses all available CPUs. Set to 1 for serial execution. Set to
                -1 to use all CPUs minus one.

        Returns:
            None. Results are saved to save_dir:
                - MDTerp_feature_names.npy: Feature names array
                - MDTerp_results_all.pkl: Complete results dictionary
                - MDTerp_summary.log: Execution log

        Raises:
            ValueError: If no transitions are detected or feature dictionaries
                don't match input data dimensions.
        """
        # Store instance variables
        self.np_data = np_data
        self.model_function_loc = model_function_loc
        self.numeric_dict = numeric_dict
        self.angle_dict = angle_dict
        self.sin_cos_dict = sin_cos_dict
        self.save_dir = save_dir
        self.point_max = point_max
        self.num_samples = num_samples
        self.cutoff = cutoff
        self.seed = seed
        self.unfaithfulness_threshold = unfaithfulness_threshold
        self.periodicity_upper = periodicity_upper
        self.periodicity_lower = periodicity_lower
        self.alpha = alpha
        self.resume = resume

        # Determine number of parallel jobs
        if n_jobs is None:
            self.n_jobs = mp.cpu_count()
        elif n_jobs == -1:
            self.n_jobs = max(1, mp.cpu_count() - 1)
        else:
            self.n_jobs = max(1, n_jobs)

        # Initialize results directory and logger
        os.makedirs(save_dir, exist_ok=True)
        self.logger = log_maker(save_dir)
        input_summary(self.logger, numeric_dict, angle_dict, sin_cos_dict, save_dir, np_data)
        self.logger.info(f"Parallel processing: using {self.n_jobs} worker(s)")

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(save_dir)
        if not resume:
            self.checkpoint_manager.clear_checkpoint()
            self.logger.info("Starting fresh analysis (resume=False)")
        else:
            self.logger.info("Checkpoint system enabled - will resume if previous run found")

        # Load the black-box model
        self.model, self.run_model_fn = self._load_blackbox_model()

        # Auto-tune prob_threshold if not provided
        if prob_threshold is None:
            self.prob_threshold = self._autotune_prob_threshold()
        else:
            self.prob_threshold = prob_threshold

        # Identify transition states
        self.transitions_dict = self._identify_transitions()

        # Analyze all transition points
        self.results, self.feature_names = self._analyze_all_transitions()

        # Save results
        self._save_results()

        # Cleanup
        self._cleanup()

    def _load_blackbox_model(self) -> Tuple:
        """
        Load the black-box model from the specified file.

        Returns:
            Tuple of (model, run_model_function)

        Raises:
            FileNotFoundError: If model file doesn't exist
            KeyError: If required functions are missing
        """
        self.logger.info(f'Loading blackbox model from file >>> {self.model_function_loc}')

        with open(self.model_function_loc, 'r') as file:
            func_code = file.read()

        local_ns = {}
        exec(func_code, globals(), local_ns)

        if "load_model" not in local_ns or "run_model" not in local_ns:
            raise KeyError("Model file must contain 'load_model()' and 'run_model()' functions")

        model = local_ns["load_model"]()
        run_model_fn = local_ns["run_model"]

        self.logger.info("Model loaded successfully!")
        return model, run_model_fn

    def _autotune_prob_threshold(self) -> float:
        """
        Automatically tune prob_threshold to yield approximately point_max samples.

        This adaptive algorithm starts with a threshold close to 0.50 and
        iteratively adjusts it to find enough transition samples without
        requiring too many.

        Returns:
            Optimized probability threshold value
        """
        self.logger.info("Auto-tuning prob_threshold based on point_max...")

        # Get initial state probabilities
        state_probs = self.run_model_fn(self.model, self.np_data)

        # Try different thresholds starting from high (0.49) down to low (0.35)
        threshold_candidates = np.linspace(0.49, 0.35, 15)

        best_threshold = 0.48  # Default fallback
        best_diff = float('inf')

        for threshold in threshold_candidates:
            # Count transition points at this threshold
            transition_count = 0
            for i in range(state_probs.shape[0]):
                sorted_probs = np.sort(state_probs[i, :])[::-1][:2]
                if (sorted_probs[0] >= threshold) and (sorted_probs[1] >= threshold):
                    transition_count += 1

            # Target total samples ≈ point_max (allowing for multiple transitions)
            # Aim for 2-5x point_max total to have good coverage
            target_total = self.point_max * 3
            diff = abs(transition_count - target_total)

            if diff < best_diff and transition_count >= self.point_max:
                best_diff = diff
                best_threshold = threshold

        self.logger.info(f"Auto-tuned prob_threshold >>> {best_threshold:.4f}")
        return best_threshold

    def _identify_transitions(self) -> Dict:
        """
        Identify transition states in the dataset.

        Returns:
            Dictionary mapping transition names to sample indices

        Raises:
            ValueError: If no transitions are detected
        """
        state_probabilities = self.run_model_fn(self.model, self.np_data)
        transitions_dict = picker_fn(state_probabilities, self.prob_threshold, self.point_max)

        num_transitions = len(transitions_dict.keys())
        self.logger.info(f"Number of state transitions detected >>> {num_transitions}")
        self.logger.info(f"Probability threshold, maximum points per transition >>> "
                        f"{self.prob_threshold}, {self.point_max}")

        if num_transitions == 0:
            self.logger.info("No transition detected. Check hyperparameters!")
            raise ValueError("No transition detected. Check hyperparameters!")

        self.logger.info(100 * '-')
        return transitions_dict

    def _analyze_single_point(
        self,
        sample_index: int,
        transition_name: str
    ) -> Tuple[int, str, list, list, int, int]:
        """
        Analyze a single transition point for feature importance (serial mode).

        This is a wrapper around the worker function for serial execution.

        Args:
            sample_index: Index of the sample in np_data
            transition_name: Name of the transition (e.g., "0_1")

        Returns:
            Tuple of (sample_index, transition_name, importance, feature_names,
                     n_selected, n_final)
        """
        return _analyze_sample_worker(
            sample_index=sample_index,
            transition_name=transition_name,
            np_data=self.np_data,
            model_function_loc=self.model_function_loc,
            numeric_dict=self.numeric_dict,
            angle_dict=self.angle_dict,
            sin_cos_dict=self.sin_cos_dict,
            save_dir_base=self.save_dir,
            seed=self.seed,
            num_samples=self.num_samples,
            cutoff=self.cutoff,
            unfaithfulness_threshold=self.unfaithfulness_threshold,
            periodicity_upper=self.periodicity_upper,
            periodicity_lower=self.periodicity_lower,
            alpha=self.alpha,
            worker_id=0  # Use worker_id 0 for serial execution
        )

    def _analyze_all_transitions(self) -> Tuple[Dict, list]:
        """
        Analyze all identified transitions for feature importance.

        Supports resumption from checkpoints and parallel processing.
        Uses multiprocessing when n_jobs > 1, otherwise runs serially.

        Returns:
            Tuple of (results_dict, feature_names_list)
            - results_dict: Maps sample indices to [transition_name, importance_list]
            - feature_names_list: List of feature names (same for all analyses)
        """
        # Load existing results and determine remaining work
        importance_results, feature_names, completed_samples = \
            self.checkpoint_manager.load_checkpoint()

        if completed_samples:
            self.logger.info(f"Resuming from checkpoint: {len(completed_samples)} samples already completed")

        # Determine remaining work
        remaining_transitions = self.checkpoint_manager.get_remaining_work(self.transitions_dict)

        if not remaining_transitions:
            self.logger.info("All transitions already completed!")
            return importance_results, feature_names

        # Build list of all samples to analyze
        all_work_items = []
        for transition_name, sample_indices in remaining_transitions.items():
            for sample_index in sample_indices:
                all_work_items.append((sample_index, transition_name))

        total_remaining = len(all_work_items)
        self.logger.info(f"Total samples to analyze: {total_remaining}")

        if self.n_jobs == 1:
            # Serial execution
            self.logger.info("Running in serial mode")
            results_iter = self._analyze_serial(all_work_items)
        else:
            # Parallel execution
            self.logger.info(f"Running in parallel mode with {self.n_jobs} workers")
            results_iter = self._analyze_parallel(all_work_items)

        # Process results as they complete
        for idx, (sample_index, trans_name, importance, feat_names, n_selected, n_final) in enumerate(results_iter):
            # Store feature names from first analysis
            if feature_names is None:
                feature_names = feat_names

            importance_results[sample_index] = [trans_name, importance]
            completed_samples.add(sample_index)

            # Save checkpoint after each sample
            self.checkpoint_manager.save_checkpoint(
                importance_results,
                feature_names,
                completed_samples
            )

            self.logger.info(
                f"Completed {idx + 1}/{total_remaining} samples | "
                f"Transition {trans_name} | "
                f"Stage 1 features: {n_selected}, Stage 2 features: {n_final}"
            )

        self.logger.info(100 * '_')
        return importance_results, feature_names

    def _analyze_serial(self, work_items):
        """
        Analyze samples serially (one at a time).

        Args:
            work_items: List of (sample_index, transition_name) tuples

        Yields:
            Analysis results for each sample
        """
        for sample_index, transition_name in work_items:
            result = self._analyze_single_point(sample_index, transition_name)
            yield result

    def _analyze_parallel(self, work_items):
        """
        Analyze samples in parallel using multiprocessing.

        Args:
            work_items: List of (sample_index, transition_name) tuples

        Yields:
            Analysis results as they complete
        """
        # Create worker function with fixed parameters
        worker_fn = partial(
            _analyze_sample_worker,
            np_data=self.np_data,
            model_function_loc=self.model_function_loc,
            numeric_dict=self.numeric_dict,
            angle_dict=self.angle_dict,
            sin_cos_dict=self.sin_cos_dict,
            save_dir_base=self.save_dir,
            seed=self.seed,
            num_samples=self.num_samples,
            cutoff=self.cutoff,
            unfaithfulness_threshold=self.unfaithfulness_threshold,
            periodicity_upper=self.periodicity_upper,
            periodicity_lower=self.periodicity_lower,
            alpha=self.alpha
        )

        # Create pool and process samples
        with mp.Pool(processes=self.n_jobs) as pool:
            # Prepare arguments for each work item
            worker_args = [
                (sample_idx, trans_name, worker_id)
                for worker_id, (sample_idx, trans_name) in enumerate(work_items)
            ]

            # Use starmap to unpack arguments
            results = pool.starmap(
                lambda sample_idx, trans_name, worker_id: worker_fn(
                    sample_idx, trans_name, worker_id
                ),
                worker_args
            )

            # Yield results
            for result in results:
                yield result

    def _save_results(self) -> None:
        """
        Finalize and verify saved results.

        Results are already incrementally saved via checkpoints, so this
        method primarily serves to verify completeness and log final paths.
        """
        # Paths (already saved via checkpoints)
        feature_names_path = os.path.join(self.save_dir, 'MDTerp_feature_names.npy')
        results_path = os.path.join(self.save_dir, 'MDTerp_results_all.pkl')

        # Verify files exist
        if not os.path.exists(feature_names_path):
            self.logger.warning("Feature names file not found - saving now")
            np.save(feature_names_path, self.feature_names)

        if not os.path.exists(results_path):
            self.logger.warning("Results file not found - saving now")
            with open(results_path, 'wb') as f:
                pickle.dump(self.results, f)

        self.logger.info(f"Feature names saved at >>> {feature_names_path}")
        self.logger.info(f"All results saved at >>> {results_path}")
        self.logger.info(f"Checkpoint file at >>> {self.checkpoint_manager.checkpoint_file}")

    def _cleanup(self) -> None:
        """Clean up temporary directories and close logger."""
        # Remove temporary data directories
        temp_dirs = [
            os.path.join(self.save_dir, 'DATA'),
            os.path.join(self.save_dir, 'DATA_2')
        ]

        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        # Clean up any worker temporary directories (in case of incomplete cleanup)
        try:
            for item in os.listdir(self.save_dir):
                if item.startswith('tmp_worker_'):
                    worker_dir = os.path.join(self.save_dir, item)
                    if os.path.isdir(worker_dir):
                        shutil.rmtree(worker_dir)
                        self.logger.info(f"Cleaned up orphaned worker directory: {item}")
        except Exception as e:
            self.logger.warning(f"Could not clean up worker directories: {e}")

        self.logger.info("Completed!!!")

        # Flush and close logger handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)



