"""
MDTerp.utils.py – Auxiliary utility functions for MDTerp package.

This module provides helper functions for logging, input validation,
transition state detection, and result summarization.
"""
import logging
from logging import Logger
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
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
    datefmt='%m-%d-%y %H:%M:%S'
    logging.basicConfig(level=logging.INFO,format=fmt,datefmt=datefmt,filename=save_dir+'/MDTerp_summary.log',filemode='w')
    logger = logging.getLogger('initialization')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt,datefmt=datefmt)
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

def picker_fn(
    state_probabilities: np.ndarray,
    probability_threshold: float,
    max_points_per_transition: int
) -> Dict[str, np.ndarray]:
    """
    Identify and sample transition state points from model predictions.

    A sample is considered to be in a transition state if its predicted
    probabilities for two different metastable states both exceed the
    threshold. This indicates uncertainty between states.

    Args:
        state_probabilities: Array of shape (n_samples, n_states) containing
            predicted probabilities for each state. Each row should sum to 1.
        probability_threshold: Minimum probability for a state to be considered.
            A sample is in a transition if its top 2 probabilities both exceed
            this threshold. Typical values: 0.40-0.49.
        max_points_per_transition: Maximum number of samples to select per
            unique transition. If more samples exist, they are uniformly sampled
            without replacement.

    Returns:
        Dictionary mapping transition names to sample indices. Keys are strings
        like "0_1" (transition between states 0 and 1). Values are numpy arrays
        of sample indices undergoing that transition.

    Example:
        >>> probs = np.array([[0.8, 0.2], [0.45, 0.45], [0.3, 0.7]])
        >>> picker_fn(probs, 0.40, 10)
        {'0_1': array([1])}  # Only middle sample is transitioning
    """
    transitions_dict = defaultdict(list)

    # Scan all samples for transition states
    for sample_idx in range(state_probabilities.shape[0]):
        # Get the top 2 states and their probabilities
        top_2_state_indices = np.argsort(state_probabilities[sample_idx, :])[::-1][:2]
        top_2_probabilities = np.sort(state_probabilities[sample_idx, :])[::-1][:2]

        # Check if both top probabilities exceed threshold (transition state)
        if (top_2_probabilities[0] >= probability_threshold and
            top_2_probabilities[1] >= probability_threshold):

            # Create transition key (ensure consistent ordering)
            sorted_states = np.sort(top_2_state_indices)
            transition_key = f"{sorted_states[0]}_{sorted_states[1]}"

            transitions_dict[transition_key].append(sample_idx)

    # Sample max_points_per_transition from each transition
    for transition_key in transitions_dict.keys():
        available_samples = transitions_dict[transition_key]
        n_samples_to_select = min(max_points_per_transition, len(available_samples))

        transitions_dict[transition_key] = np.random.choice(
            available_samples,
            size=n_samples_to_select,
            replace=False
        )

    return dict(transitions_dict)

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
    Function summarizing MDTerp results for all the transitions present in the dataset.

    Args:
        all_results_loc (str): Location to save MDTerp results.
        importance_coverage (float): For a specific transition, sets a cutoff for the sum of the most important features in descending order.
        
    Returns:
        dict: Dictionary with keys representing detected transitions. E.g., key '3_8' means transition between index 3 and index 8 according to the prob array. Values are lists representing feature importance using the length of the list equaling the number of features in the provided dataset.
    """
    with open(all_result_loc, 'rb') as f:
        loaded_dict = pickle.load(f)  

    for ii in loaded_dict:
        tmp_c = loaded_dict[ii][1]
        loaded_dict[ii] = np.argsort(tmp_c)[::-1][n]

    return loaded_dict

def make_result(
    feature_type_indices: List[np.ndarray],
    feature_names: List[str],
    raw_importance: np.ndarray
) -> List[float]:
    """
    Convert raw feature importance to human-readable format.

    Combines sin/cos feature importances for angular features and
    organizes results by feature type (numeric, angle, sin_cos).

    Args:
        feature_type_indices: List of 4 arrays [numeric_idx, angle_idx, sin_idx, cos_idx]
            containing column indices for each feature type.
        feature_names: List of feature names in output order.
        raw_importance: Array of raw importance values from final_model.

    Returns:
        List of importance values ordered by feature_names. For sin/cos
        features, importances are summed.
    """
    numeric_indices, angle_indices, sin_indices, cos_indices = feature_type_indices

    importance_list = []

    # Add numeric feature importances
    for idx in numeric_indices:
        importance_list.append(raw_importance[idx])

    # Add angle feature importances
    for idx in angle_indices:
        importance_list.append(raw_importance[idx])

    # Add sin_cos feature importances (sum sin and cos components)
    for sin_idx, cos_idx in zip(sin_indices, cos_indices):
        combined_importance = raw_importance[sin_idx] + raw_importance[cos_idx]
        importance_list.append(combined_importance)

    return importance_list