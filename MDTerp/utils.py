"""
MDTerp.utils.py – Auxiliary utility functions for MDTerp package.
"""
import logging
from logging import Logger
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict
import pickle
import os

def log_maker(save_dir: str) -> Logger:
    """
    Function for creating a logger detailing MDTerp operations.

    Args:
        save_dir (str): Location to save MDTerp results.

    Returns:
        Logger: Logger object created using Python's built-in logging module.
    """
    fmt = '%(asctime)s %(name)-15s %(levelname)-8s %(message)s'
    datefmt = '%m-%d-%y %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    logger = logging.getLogger('MDTerp')
    logger.setLevel(logging.INFO)
    # Clear any existing handlers from previous runs
    logger.handlers.clear()

    file_handler = logging.FileHandler(
        os.path.join(save_dir, 'MDTerp_summary.log'), mode='w'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info(100*'-')
    logger.info('Starting MDTerp...')
    logger.info(100*'-')

    return logger

def input_summary(logger: Logger, numeric_dict: dict, angle_dict: dict, sin_cos_dict: dict, save_dir: str, np_data: np.ndarray) -> None:
    """
    Function for summarizing user-provided input data in Python Logger.

    Args:
        logger (Logger): Logger object created using Python's built-in logging module.
        numeric_dict (dict): Python dictionary, each key represents the name of a numeric feature (non-periodic). Values should be lists with a single element using the index of the corresponding numpy array in np_data.
        angle_dict (dict): Python dictionary, each key represents the name of an angular feature in [-pi, pi]. Values should be lists with a single element with the index of the corresponding numpy array in np_data.
        sin_cos_dict (dict): Python dictionary, each key represents the name of an angular feature. Values should be lists with two elements using the sine, cosine indices of the corresponding numpy array in np_data.
        save_dir (str): Location to save MDTerp results.
        np_data (np.ndarray): Numpy 2D array containing training data for the black-box model. Samples along rows and features along columns.
        
    Returns:
        None
    """
    logger.info('MDTerp result location >>> ' + save_dir )
    logger.info('Defined numeric features >>> ' + str(len(list(numeric_dict.keys()))) )
    logger.info('Defined angle features >>> ' + str(len(list(angle_dict.keys()))) )
    logger.info('Defined sin_cos features >>> ' + str(len(list(sin_cos_dict.keys()))) )
    logger.info('Number of samples in blackbox model training data >>> ' + str(np_data.shape[0]) )
    logger.info('Number of columns in blackbox model training data >>> ' + str(np_data.shape[1]) )
    if np_data.shape[1] != len(list(numeric_dict.keys())) + len(list(angle_dict.keys())) + len(list(sin_cos_dict.keys()))*2:
        logger.error('Assertion failure between provided feature dictionaries and input data!')
        raise ValueError('Assertion failure between provided feature dictionaries and input data!')
    
    logger.info(100*'-')

def select_transition_points(
    prob: np.ndarray,
    point_max: int = 50,
    prob_threshold_min: float = 0.475,
) -> Tuple[dict, dict]:
    """
    Select transition-state samples with adaptive per-transition thresholds.

    For each transition, selects up to point_max samples closest to the
    decision boundary (highest min-of-top-2 probabilities). The effective
    threshold is determined automatically per transition, with
    prob_threshold_min as the absolute floor.

    Args:
        prob: 2D array of state probabilities (samples x states). Rows sum to 1.
        point_max: Target number of points per transition (default: 50).
        prob_threshold_min: Minimum allowed probability threshold (default: 0.475).

    Returns:
        points: Dict mapping transition keys (e.g., "0_1") to arrays of
            selected sample indices.
        thresholds: Dict mapping transition keys to the effective threshold
            used for that transition.
    """
    import warnings

    candidates = defaultdict(list)
    for i in range(prob.shape[0]):
        top2_indices = np.argsort(prob[i, :])[::-1][:2]
        top2_values = prob[i, top2_indices]
        min_top2 = top2_values[1]

        if min_top2 >= prob_threshold_min:
            key = str(np.sort(top2_indices)[0]) + '_' + str(np.sort(top2_indices)[1])
            candidates[key].append((i, min_top2))

    points = {}
    thresholds = {}

    for transition, cands in candidates.items():
        cands_sorted = sorted(cands, key=lambda x: x[1], reverse=True)

        if len(cands_sorted) <= point_max:
            selected = [c[0] for c in cands_sorted]
            effective_threshold = cands_sorted[-1][1] if cands_sorted else prob_threshold_min

            if len(cands_sorted) < point_max:
                warnings.warn(
                    f"Transition {transition}: only {len(cands_sorted)} points found "
                    f"at minimum threshold {prob_threshold_min} "
                    f"(requested {point_max})"
                )
        else:
            selected = [c[0] for c in cands_sorted[:point_max]]
            effective_threshold = cands_sorted[point_max - 1][1]

        points[transition] = np.array(selected)
        thresholds[transition] = effective_threshold

    return points, thresholds

def transition_summary(all_result_loc: str, importance_coverage: float = 0.8) -> dict:
    """
    Function summarizing MDTerp results for all the transitions present in the dataset.

    Args:
        all_results_loc (str): Location to save MDTerp results.
        importance_coverage (float): For a specific transition, sets a cutoff for the sum of the most important features in descending order.
        
    Returns:
        dict: Dictionary with keys representing detected transitions. E.g., key '3_8' means transition between index 3 and index 8 according to the prob array. Values are lists representing mean and standard deviations of the feature importance using the length of the list equaling the number of features in the provided dataset for that transition.
    """
    with open(all_result_loc, 'rb') as f:
        loaded_dict = pickle.load(f)  
    # Save all the unique transitions
    transitions = []
    for ii in loaded_dict:
        transitions.append(loaded_dict[ii][0])
    # Save summary results for each transition
    summary_imp = {}
    for ii in np.unique(transitions):
        summary_imp[ii] = []
    for ii in loaded_dict:
        summary_imp[loaded_dict[ii][0]].append(loaded_dict[ii][1])
    for ii in summary_imp:
        tmp_a = np.mean(summary_imp[ii], axis = 0)
        # Normalize results for the transition
        normalization = np.sum(tmp_a)
        tmp_a = tmp_a/normalization
        tmp_b = np.std(summary_imp[ii], axis = 0)/normalization

        trim_args = np.argsort(tmp_a)[::-1]
        trim_vals = np.sort(tmp_a)[::-1]
        # Discard irrelevant features for each transition, based on the importance_coverage hyperparameter
        cutoff_k = 0
        current_coverage = 0
        while current_coverage < importance_coverage:
          try:  
            current_coverage += trim_vals[cutoff_k]
            cutoff_k += 1
          except:
            break

        tmp_a[trim_args[cutoff_k:]] = 0
        tmp_b[trim_args[cutoff_k:]] = 0
        
        summary_imp[ii] = [tmp_a, tmp_b]

    return summary_imp
    
def dominant_feature(all_result_loc: str, n: int = 0) -> dict:
    """
    Get the n-th most important feature for each analyzed sample.

    Args:
        all_result_loc: Path to the MDTerp_results_all.pkl file.
        n: Rank of the dominant feature to extract (0 = most important).

    Returns:
        Dict mapping sample index to the feature index of the n-th most
        important feature.
    """
    with open(all_result_loc, 'rb') as f:
        loaded_dict = pickle.load(f)  

    for ii in loaded_dict:
        tmp_c = loaded_dict[ii][1]
        loaded_dict[ii] = np.argsort(tmp_c)[::-1][n]

    return loaded_dict

def make_result(
    feature_type_indices: list,
    feature_names: list,
    importance: np.ndarray,
) -> list:
    """
    Map importance values from the analysis feature space back to the
    original feature space, combining sin/cos pairs.

    Args:
        feature_type_indices: List of [numeric_indices, angle_indices,
            sin_indices, cos_indices] arrays.
        feature_names: List of feature names in order.
        importance: Raw importance array from the final model.

    Returns:
        List of importance values aligned with feature_names ordering.
    """
    result = []
    for i in range(feature_type_indices[0].shape[0]):
        result.append(importance[feature_type_indices[0][i]])
    for i in range(feature_type_indices[1].shape[0]):
        result.append(importance[feature_type_indices[1][i]])
    for i in range(feature_type_indices[2].shape[0]):
        result.append(
            importance[feature_type_indices[2][i]]
            + importance[feature_type_indices[3][i]]
        )
    return result