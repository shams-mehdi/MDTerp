"""Main module."""
import numpy as np
import os
import shutil
import pickle

from neighborhood import generate_neighborhood
from utils import log_maker, input_summary, picker_fn
from init_analysis import init_model
from final_analysis import final_model

class run:
    def __init__(self, np_data, model_function_loc, numeric_dict = {}, angle_dict = {}, sin_cos_dict = {}, save_dir = './results/', prob_threshold = 0.48, point_max = 20, num_samples = 5000, cutoff = 25, seed = 0, unfaithfulness_threshold = 0.01):

        # Initializing necessities
        os.makedirs(save_dir, exist_ok = True)
        tmp_dir = save_dir + 'tmp/'
        os.makedirs(tmp_dir, exist_ok = True)
        logger = log_maker(save_dir)
        input_summary(logger, numeric_dict, angle_dict, sin_cos_dict, save_dir, np_data)

        # Load Model
        logger.info('Loading blackbox model >>> ')
        with open(model_function_loc, 'r') as file:
            func_code = file.read()
        local_ns = {}
        exec(func_code, globals(), local_ns)
        model_loc, model = local_ns["load_model"]()
        logger.info("Model loaded from location >>> " + model_loc)

        # Identify transition states for given/training dataset
        state_probabilities = local_ns["run_model"](model, np_data)
        points = picker_fn(state_probabilities, prob_threshold, point_max)
        logger.info("Number of state transitions detected >>> " + str(len(list(points.keys()))))
        logger.info("Probability threshold, maximum number of points per transition >>> " + str(prob_threshold) + ", " + str(point_max) )
        logger.info(100*'-')
        
        # Loop over all the transitions
        importance_master = {}
        for transition in points:
            logger.info("Starting transition >>> " + transition)
            for point in range(len(points[transition])):
                index = points[transition][point]
                given_indices, indices_names = generate_neighborhood(tmp_dir, numeric_dict, angle_dict, sin_cos_dict, np_data, index, seed, num_samples, selected_features = False)
                state_probabilities2 = local_ns["run_model"](model, np.load(tmp_dir + 'DATA/make_prediction.npy'))
                TERP_dat = np.load(tmp_dir + 'DATA/TERP_dat.npy')
                selected_features = init_model(tmp_dir, TERP_dat, state_probabilities2, cutoff, given_indices, seed)

                generate_neighborhood(tmp_dir, numeric_dict, angle_dict, sin_cos_dict, np_data, index, seed, num_samples, selected_features)
                state_probabilities3 = local_ns["run_model"](model, np.load(tmp_dir + 'DATA_2/make_prediction.npy'))
                TERP_dat = np.load(tmp_dir + 'DATA_2/TERP_dat.npy')
                importance = final_model(tmp_dir, TERP_dat, state_probabilities3, unfaithfulness_threshold, given_indices, selected_features, seed)

                importance_master[index] = [transition, importance]
                logger.info("Generated " + str(point + 1) + "/" + str(len(points[transition])) + " results!")
            logger.info(100*'_')

        np.save(save_dir + 'MDTerp_feature_names.npy', indices_names)
        with open(save_dir + 'MDTerp_results_all.pkl', 'wb') as f:
            pickle.dump(importance_master, f)

        logger.info("Feature names saved at >>> " + save_dir + 'MDTerp_feature_names.npy')
        logger.info("All results saved at >>> " + save_dir + 'MDTerp_results_all.pkl')
        
        
        shutil.rmtree(tmp_dir)
        logger.info("Completed!!!")
        
        # Flush and close logger
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)



