"""
MDTerp.utils.py – Auxiliary utility functions for MDTerp package.
"""
import logging
from logging import Logger
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

def picker_fn(prob: np.ndarray, threshold: float, point_max: int) -> dict:
    """
    Function for picking points at the transition state ensemble. Uses provided data and metastable state probability from the black-box model.

    Args:
        prob (np.ndarray): Numpy 2D array containing metastable state prediction probabilities from the black-box model. Rows represent samples, and the number of columns represents the number of states. Each row should sum to 1.
        threshold (float): Threshold for identifying if a sample belongs to a transition state predicted by the black-box model. If the metastable state probability > threshold for two different classes for a specific sample, it's suitable for analysis.
        point_max (int): If too many suitable points exist for a specific transition (e.g., transition between metastable state 3 and 8), point_max sets the maximum number of points chosen for analysis. Points are chosen from a uniform distribution.
        
    Returns:
        dict: Dictionary with keys representing detected transitions. E.g., key '3_8' means transition between index 3 and index 8 according to the prob array. Values represent chosen samples/rows in the provided dataset undergoing this transition.
    """
    transition_dict = defaultdict(list)
    for i in range(prob.shape[0]):
        sorted_ind = np.sort(np.argsort(prob[i, :])[::-1][:2])
        sorted_val = np.sort(prob[i, :])[::-1][:2]
        if (sorted_val[0]>=threshold) and (sorted_val[1]>=threshold):
            transition_dict[str(sorted_ind[0]) + '_' + str(sorted_ind[1])].append(i)
    for i in transition_dict.keys():
        transition_dict[i] = np.random.choice(transition_dict[i], size = min(point_max, len(transition_dict[i])), replace = False)
    
    return transition_dict

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

def make_result(given_indices, indices_names, importance):
    tmp = []
    for i in range(given_indices[0].shape[0]):
        tmp.append(importance[given_indices[0][i]])
        
    for i in range(given_indices[1].shape[0]):
        tmp.append(importance[given_indices[1][i]])

    for i in range(given_indices[2].shape[0]):
        tmp.append((importance[given_indices[2][i]] + importance[given_indices[3][i]]))
    return tmp