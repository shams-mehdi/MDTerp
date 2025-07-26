import logging
import numpy as np
from collections import defaultdict
import pickle
import os
import matplotlib.pyplot as plt

def log_maker(save_dir):
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

def input_summary(logger, numeric_dict, angle_dict, sin_cos_dict, save_dir, np_data):
    logger.info('MDTerp result location >>> ' + save_dir )
    logger.info('Defined numeric features >>> ' + str(len(list(numeric_dict.keys()))) )
    logger.info('Defined angle features >>> ' + str(len(list(angle_dict.keys()))) )
    logger.info('Defined sin_cos features >>> ' + str(len(list(sin_cos_dict.keys()))) )
    logger.info('Number of samples in blackbox model training data >>> ' + str(np_data.shape[0]) )
    logger.info('Number of columns in blackbox model training data >>> ' + str(np_data.shape[1]) )

    if np_data.shape[1] != len(list(numeric_dict.keys())) + len(list(angle_dict.keys())) + len(list(sin_cos_dict.keys()))//2:
        logger.error('Assertion failure between provided feature dictionaries and input data!')
        raise ValueError('Assertion failure between provided feature dictionaries and input data!')

    logger.info(100*'-')

def picker_fn(prob, threshold, point_max):
    transition_dict = defaultdict(list)
    for i in range(prob.shape[0]):
        sorted_ind = np.sort(np.argsort(prob[i, :])[::-1][:2])
        sorted_val = np.sort(prob[i, :])[::-1][:2]
        if (sorted_val[0]>=threshold) and (sorted_val[1]>=threshold):
            transition_dict[str(sorted_ind[0]) + '_' + str(sorted_ind[1])].append(i)
    for i in transition_dict.keys():
        transition_dict[i] = np.random.choice(transition_dict[i], size = min(point_max, len(transition_dict[i])), replace = False)
    
    return transition_dict

def summary(feature_names_loc, all_result_loc, save_fig_dir, top_k = 10, fs = 12, dpi = 300):
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
    


    