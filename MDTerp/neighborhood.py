"""
MDTerp.neighborhood.py – Function for generating perturbed neighborhood samples.
"""
import numpy as np
import os
import copy
import scipy.stats as sst
from typing import Union, List
import numpy as np

def generate_neighborhood(save_dir: str, numeric_dict: dict, angle_dict: dict, sin_cos_dict: dict, np_dat: np.ndarray, index: int, seed: int, num_samples: int, perturbation_max: float, selected_features: Union[bool, List[int]] = False):
    """
    Function for creating a logger detailing MDTerp operations.

    Args:
        save_dir (str): Location to save MDTerp results.
        numeric_dict (dict): Python dictionary, each key represents the name of a numeric feature (non-periodic). Values should be lists with a single element with the index of the corresponding numpy array in np_data.
        angle_dict (dict): Python dictionary, each key represents the name of an angular feature in [-pi, pi]. Values should be lists with a single element with the index of the corresponding numpy array in np_data.
        sin_cos_dict (dict): Python dictionary, each key represents the name of an angular feature. Values should be lists with two elements with the sine, cosine indices respectively of the corresponding numpy array in np_data.
        np_dat (np.ndarray): Numpy 2D array containing training data for the black-box model. Samples along rows and features along columns.
        index (int): Row/sample of the provided dataset using np_dat to analyze.
        seed (int): Random seed.
        num_samples (int): Size of the generated perturbed neighborhood.
        selected_features: If False (Default), perturbs all the features/columns. Otherwise, List of integers represent subset of features to perturb.
        perturbation_max (float): Hyperparameter that sets a maximum on the perturbation. Perturbation is calculated by taking standard deviation of the dataset for a feature and multiplying with value drawn from a normal distribution ~N(0,1). However, sometimes this can result in very high perturbation if the standard deviation of the feature is unreasonably high. To solve this, a percentage (chosen by this hyperparameter) of the range of that feature is set as the maximum factor to be multiplied to the values drawn from ~N(0,1).
    Returns:
        list: List of np.ndarray indicating indices of numeric, angular, sin_cos features respectively.
        list: List of the combined name of the features.
    """
    np.random.seed(seed)
    if selected_features == False:
        save_directory = save_dir + 'DATA'
        os.makedirs(save_directory, exist_ok = True)
        
        numeric_indices = []
        angle_indices = []
        sin_cos_indices = []

        indices_names = []

        for i in numeric_dict:
                numeric_indices.append(numeric_dict[i])
                indices_names.append(i)
                assert numeric_dict[i][0] in np.arange(np_dat.shape[1]), 'Invalid numeric index'
        for i in angle_dict:
                angle_indices.append(angle_dict[i])
                indices_names.append(i)
                assert angle_dict[i][0] in np.arange(np_dat.shape[1]), 'Invalid angle index'
        for i in sin_cos_dict:
                sin_cos_indices.append(sin_cos_dict[i])
                indices_names.append(i)
                assert sin_cos_dict[i][0] in np.arange(np_dat.shape[1]), 'Invalid sin index'
                assert sin_cos_dict[i][1] in np.arange(np_dat.shape[1]), 'Invalid cos index'
    
        numeric_indices = np.array(numeric_indices).flatten()
        angle_indices = np.array(angle_indices).flatten()
        sin_cos_indices = np.array(sin_cos_indices)
    
    else:
      save_directory = save_dir + 'DATA_2'
      os.makedirs(save_directory, exist_ok = True)
      
      numeric_indices = np.array(selected_features[0])
      angle_indices = np.array(selected_features[1])
      sin_cos_indices = np.array(selected_features[2])

      indices_names = 'Dummy'

    make_pred = np.ones((num_samples, np_dat.shape[1]))*np_dat[index,:]
    TERP_dat = np.zeros((num_samples, 1))

    if numeric_indices.shape[0]>0:
          input_numeric = np_dat[:, numeric_indices]
          numeric = copy.deepcopy(input_numeric)

          std_numeric = []
          for i in range(input_numeric.shape[1]):
            std_numeric.append(min(np.std(input_numeric[:,i]), perturbation_max*(np.max(input_numeric[:,i]) - np.min(input_numeric[:,i]))))
        
          make_prediction_numeric = np.zeros((num_samples, input_numeric.shape[1]))
          TERP_numeric = np.zeros((num_samples, input_numeric.shape[1]))
        
          perturb_numeric = np.random.randint(0, 2, num_samples * input_numeric.shape[1]).reshape((num_samples, input_numeric.shape[1]))
          perturb_numeric[0,:] = 1
            
          for i in range(num_samples):
            for j in range(input_numeric.shape[1]):
              if perturb_numeric[i,j] == 1:
                make_prediction_numeric[i,j] = input_numeric[index,j]
              elif perturb_numeric[i,j] == 0:
                rand_data = np.random.normal(0, 1)
                make_prediction_numeric[i,j] = input_numeric[index,j] + std_numeric[j]*rand_data
                TERP_numeric[i,j] = rand_data
        
          make_pred[:, numeric_indices] = make_prediction_numeric
          TERP_dat = np.column_stack((TERP_dat, TERP_numeric))

    if angle_indices.shape[0]>0:
      periodic = np_dat[:, angle_indices]
      assert np.all(periodic<=np.pi+0.001) and np.all(periodic>-np.pi-0.001), 'Provide periodic data in appropriate domain...'
      input_periodic = copy.deepcopy(periodic)

      std_periodic = []
      for i in range(input_periodic.shape[1]):
        std_periodic.append(min(sst.circstd(input_periodic[:,i], high = np.pi, low = -np.pi), perturbation_max*(np.max(input_periodic[:,i]) - np.min(input_periodic[:,i]))))

      make_prediction_periodic = np.zeros((num_samples, input_periodic.shape[1]))
      TERP_periodic = np.zeros((num_samples, input_periodic.shape[1]))

      perturb_periodic = np.random.randint(0, 2, num_samples * input_periodic.shape[1]).reshape((num_samples, input_periodic.shape[1]))
      perturb_periodic[0,:] = 1

      for i in range(num_samples):
        for j in range(input_periodic.shape[1]):
          if perturb_periodic[i,j] == 1:
            make_prediction_periodic[i,j] = input_periodic[index,j]
          elif perturb_periodic[i,j] == 0:
            rand_data = np.random.normal(0, 1)
            make_prediction_periodic[i,j] = input_periodic[index,j] + std_periodic[j]*rand_data
            TERP_periodic[i,j] = rand_data
            if make_prediction_periodic[i,j] < -np.pi or make_prediction_periodic[i,j] > np.pi:
              make_prediction_periodic[i,j] = np.arctan2(np.sin(make_prediction_periodic[i,j]), np.cos(make_prediction_periodic[i,j]))


      make_pred[:, angle_indices] = make_prediction_periodic      
      TERP_dat = np.column_stack((TERP_dat, TERP_periodic))

    if sin_cos_indices.shape[0]>0:
      sin = np_dat[:, sin_cos_indices[:, 0]]
      assert np.all(sin>=-1) and np.all(sin<=1), 'Provide sin data in domain [-1,1]'
      input_sin = copy.deepcopy(sin)
        
      cos = np_dat[:, sin_cos_indices[:, 1]]
      assert np.all(cos>=-1) and np.all(cos<=1), 'Provide cosine data in domain [-1,1]'
      input_cos = copy.deepcopy(cos)

      std_sin_cos = []
      input_sin_cos = np.zeros((input_sin.shape[0], input_sin.shape[1]))
      for i in range(input_sin.shape[1]):
        input_sin_cos[:,i] = np.arctan2(input_sin[:,i], input_cos[:,i])
        std_sin_cos.append(min(sst.circstd(input_sin_cos[:,i], high = np.pi, low = -np.pi), perturbation_max*(np.max(input_sin_cos[:,i]) - np.min(input_sin_cos[:,i]))))

      make_prediction_sin = np.zeros((num_samples, input_sin_cos.shape[1]))
      make_prediction_cos = np.zeros((num_samples, input_sin_cos.shape[1]))
      TERP_sin_cos = np.zeros((num_samples, input_sin_cos.shape[1]))

      perturb_sin_cos = np.random.randint(0, 2, num_samples * input_sin_cos.shape[1]).reshape((num_samples, input_sin_cos.shape[1]))
      perturb_sin_cos[0,:] = 1

      for i in range(num_samples):
        for j in range(input_sin_cos.shape[1]):
          if perturb_sin_cos[i,j] == 1:
            make_prediction_sin[i,j] = np.sin(input_sin_cos[index,j])
            make_prediction_cos[i,j] = np.cos(input_sin_cos[index,j])

          elif perturb_sin_cos[i,j] == 0:
            rand_data = np.random.normal(0, 1)
            make_prediction_sin[i,j] = np.sin(input_sin_cos[index,j] + std_sin_cos[j]*rand_data)
            make_prediction_cos[i,j] = np.cos(input_sin_cos[index,j] + std_sin_cos[j]*rand_data)
            TERP_sin_cos[i,j] = rand_data


      make_pred[:, sin_cos_indices[:, 0]] = make_prediction_sin
      make_pred[:, sin_cos_indices[:, 1]] = make_prediction_cos

      TERP_dat = np.column_stack((TERP_dat, TERP_sin_cos))

    np.save(save_directory + '/make_prediction.npy', make_pred)  
    np.save(save_directory + '/TERP_dat.npy', TERP_dat[:,1:])

    return [numeric_indices, angle_indices, sin_cos_indices], indices_names
