"""
MDTerp.base.py – Main MDTerp module.
"""
import numpy as np
import os
import shutil
import pickle

from MDTerp.neighborhood import generate_neighborhood
from MDTerp.utils import log_maker, input_summary, picker_fn, make_result
from MDTerp.init_analysis import init_model
from MDTerp.final_analysis import final_model

class run:
    """
    MDTerp.base.run - Main class for implementing MDTERP.
    """
    def __init__(self, np_data: np.ndarray, model_function_loc: str, numeric_dict: dict = {}, angle_dict: dict = {}, sin_cos_dict:dict = {}, save_dir: str = './results/', prob_threshold: float = 0.48, point_max: int = 50, num_samples: int = 10000, cutoff: int = 15, seed: int = 0, unfaithfulness_threshold: float = 0.01, periodicity_upper: float = np.pi, periodicity_lower: float = -np.pi, alpha: float = 1.0) -> None:
        """
        Constructor for the MDTerp.base.run class.
        
        Args:
            np_data (np.ndarray): Black-box training data.
            model_function_loc (str): Location of a human-readable file containing two functions called 'load_model()', and 'run_model()'. 'load_model' must not take any arguments and should return the black-box model. 'run_model' must be a function that takes two arguments: the model and data, and returns metastable state probabilities. Go to https://shams-mehdi.github.io/MDTerp/docs/examples/ for example files.
            numeric_dict (dict): Python dictionary, each key represents the name of a numeric feature (non-periodic). Values should be lists with a single element with the index of the corresponding numpy array in np_data.
            angle_dict (dict): Python dictionary, each key represents the name of an angular feature in [-pi, pi]. Values should be lists with a single element with the index of the corresponding numpy array in np_data.
            sin_cos_dict (dict): Python dictionary, each key represents the name of an angular feature. Values should be lists with two elements using the sine, cosine indices of the corresponding numpy array in np_data.
            save_dir (str): Location to save MDTerp results.
            prob_threshold (float): Threshold for identifying if a sample belongs to a transition state predicted by the black-box model. If metastable state probability > threshold for two different classes for a specific sample, it's suitable for analysis (Default: 0.48).
            point_max (int): If too many suitable samples exist for a specific transition (e.g., transition between metastable state 3 and 8), point_max sets the maximum number of points chosen for further analysis. Points are chosen from a uniform distribution (Default: 50).
            num_samples (int): Size of the perturbed neighborhood (Default: 10000). Ad hoc rule: should be proportional to the square root of the number of features.
            cutoff (int): Maximum number of features kept for the final round of MDTerp and forward feature selection (use to improve compute time: when too many features are in the dataset and a priori it is known it is unlikely that more features than set by cutoff will be relevant).
            seed (int): Random seed.
            unf_threshold (float): Hyperparameter that sets a lower limit on surrogate model unfaithafulness (U). Forward feature selection ends when unfaithfulness reaches lower than this threshold.
            periodicity_upper (float): Sets periodicity of the angular features (Default: numpy.pi).
            periodicity_lower (float): Sets periodicity of the angular features (Default: -numpy.py).
            alpha (float): L2 norm of Ridge regression (Default: 1.0).
        Returns:
            None
        """
        # Initialization
        os.makedirs(save_dir, exist_ok = True)
        logger = log_maker(save_dir)
        input_summary(logger, numeric_dict, angle_dict, sin_cos_dict, save_dir, np_data)

        # Load black-box model
        logger.info('Loading blackbox model from file >>> ' + model_function_loc)
        with open(model_function_loc, 'r') as file:
            func_code = file.read()
        local_ns = {}
        exec(func_code, globals(), local_ns)
        model = local_ns["load_model"]()
        logger.info("Model loaded!")

        # Identify transition states for given/training dataset
        state_probabilities = local_ns["run_model"](model, np_data)
        points = picker_fn(state_probabilities, prob_threshold, point_max)
        logger.info("Number of state transitions detected >>> " + str(len(list(points.keys()))))
        logger.info("Probability threshold, maximum number of points per transition >>> " + str(prob_threshold) + ", " + str(point_max) )
        if len(list(points.keys())) == 0:
            logger.info("No transition detected. Check hyperparamters!")
            raise ValueError("No transition detected. Check hyperparameters!")
        logger.info(100*'-')
        
        # Loop over all the transitions
        importance_master = {}
        for transition in points:
            logger.info("Starting transition >>> " + transition)
            for point in range(len(points[transition])):
                index = points[transition][point]
                feature_type_indices, indices_names = generate_neighborhood(save_dir, numeric_dict, angle_dict, sin_cos_dict, np_data, index, seed, num_samples, np.array([]), periodicity_upper, periodicity_lower)
                state_probabilities2 = local_ns["run_model"](model, np.load(save_dir + 'DATA/make_prediction.npy'))
                TERP_dat = np.load(save_dir + 'DATA/TERP_dat.npy')
                selected_features = init_model(TERP_dat, state_probabilities2, cutoff, feature_type_indices, seed, alpha)

                generate_neighborhood(save_dir, numeric_dict, angle_dict, sin_cos_dict, np_data, index, seed, num_samples, selected_features, periodicity_upper, periodicity_lower)
                state_probabilities3 = local_ns["run_model"](model, np.load(save_dir + 'DATA_2/make_prediction.npy'))
                TERP_dat = np.load(save_dir + 'DATA_2/TERP_dat.npy')
                importance_0 = final_model(TERP_dat, state_probabilities3, unfaithfulness_threshold, feature_type_indices, selected_features, seed)
                importance = make_result(feature_type_indices, indices_names, importance_0)
                importance_master[index] = [transition, importance]
                logger.info("Completed generating " + str(point + 1) + "/" + str(len(points[transition])) + " results!" + " First round features kept >>> " + str(len(selected_features)) + ", Final round features kept >>> " + str(np.nonzero(importance)[0].shape[0]))
            logger.info(100*'_')

        np.save(save_dir + 'MDTerp_feature_names.npy', indices_names)
        with open(save_dir + 'MDTerp_results_all.pkl', 'wb') as f:
            pickle.dump(importance_master, f)

        logger.info("Feature names saved at >>> " + save_dir + 'MDTerp_feature_names.npy')
        logger.info("All results saved at >>> " + save_dir + 'MDTerp_results_all.pkl')
        
        
        shutil.rmtree(save_dir + 'DATA')
        shutil.rmtree(save_dir + 'DATA_2')
        logger.info("Completed!!!")
        
        # Flush and close logger
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)



