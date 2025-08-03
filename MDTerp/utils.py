"""
MDTerp.utils.py – Auxiliary utility functions for MDTerp package.
"""
import logging
from logging import Logger
import numpy as np
from collections import defaultdict
import pickle
import os
import matplotlib.pyplot as plt

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
        numeric_dict (dict): Python dictionary, each key represents the name of a numeric feature (non-periodic). Values should be lists with a single element with the index of the corresponding numpy array in np_data.
        angle_dict (dict): Python dictionary, each key represents the name of an angular feature in [-pi, pi]. Values should be lists with a single element with the index of the corresponding numpy array in np_data.
        sin_cos_dict (dict): Python dictionary, each key represents the name of an angular feature. Values should be lists with two elements with the sine, cosine indices respectively of the corresponding numpy array in np_data.
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
    Function for picking points at the transition state ensemble. Uses provided data and metastable state probability from black-box.

    Args:
        prob (np.ndarray): Numpy 2D array containing metastable state prediction probabilities from the black-box model. Rows represent samples and number of columns equal to number of states. Each row should sum to 1.
        threshold (float): Threshold for identifying if a sample belongs to transition state predicted by the black-box model. If metastable state probability > threshold for two different classes for a specific sample, it's suitable for analysis.
        point_max (int): If too many suitable points exist for a specific transition (e.g., transition between metastable state 3 and 8), point_max sets maximum number of points chosen for analysis. Points chosen from a uniform distribution.
        
    Returns:
        dict : Dictionary with keys representing detected transitions. E.g., key '3_8' means transition between index 3 and index 8 in according to the prob array. Values represent chosen samples/rows in the provided dataset which undergo this transition.
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

def summary(feature_names_loc: str, all_result_loc: str, save_fig_dir: str, top_k: int = 10, fs: int = 12, dpi: int = 300) -> dict:
    """
    Function summarizing MDTerp results for all the transitions present in the dataset.

    Args:
        feature_names_loc (str): Location of the saved combined (all) feature names.
        all_results_loc (str): Location to save MDTerp results.
        save_fig_dir (str): Location to save MDTerp results figures.
        top_k (int): Number of top ranked features to show in summary figures.
        fs (int): Fontsize of the labels in generated figures.
        dpi (int): DPI of the generated figures
        
    Returns:
        dict : Dictionary with keys representing detected transitions. E.g., key '3_8' means transition between index 3 and index 8 in according to the prob array. Values are lists representing feature importance with length of the list equaling number of features in the provided dataset.
    """
    feature_names = np.load(feature_names_loc)
    os.makedirs(save_fig_dir, exist_ok = True)
    with open(all_result_loc, 'rb') as f:
        loaded_dict = pickle.load(f)  
    transitions = []
    for ii in loaded_dict:
        transitions.append(loaded_dict[ii][0])
    summary_imp = {}
    for ii in np.unique(transitions):
        summary_imp[ii] = []
    for ii in loaded_dict:
        summary_imp[loaded_dict[ii][0]].append(loaded_dict[ii][1])
    for ii in summary_imp:
        summary_imp[ii] = np.mean(summary_imp[ii], axis = 0)
        tmp_vals = summary_imp[ii]
        trim_args = np.argsort(tmp_vals)[::-1][:top_k]
        trim_vals = np.sort(tmp_vals)[::-1][:top_k]
        fig, ax = plt.subplots(figsize = (8,8))
        ax.barh(np.arange(trim_vals.shape[0]), trim_vals)
        ax.set_title('Importance coverage: ' + str(int(100*np.sum(trim_vals)/np.sum(tmp_vals))) + '%', fontsize = fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.tick_params(axis='both', which='minor', labelsize=int(fs/2))
        ax.set_yticks(np.arange(trim_args.shape[0]))
        ax.set_yticklabels(np.array(feature_names)[trim_args])
        fig.tight_layout()
        fig.savefig(save_fig_dir + '/' + ii + '.png', dpi = dpi, transparent = True)
        
    return summary_imp
    


    
