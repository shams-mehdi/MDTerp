"""
MDTerp.neighborhood.py – Function for generating perturbed neighborhood samples.
"""
import numpy as np
import os
import copy
import scipy.stats as sst
from typing import Union, List
import numpy as np

def perturbation(data: np.ndarray, std: np.ndarray, num_samples: float, index: int, seed: int):
      """
      Function for generating perturbed samples.
    
      Args:
        data (np.ndarray): Numpy 2D array containing training data for the black-box model. Samples along rows and features along columns.
        std (np.ndarray): Numpy 1D array containing the standard deviation of features in data.
        num_samples (int): Size of the generated perturbed neighborhood.
        index (int): Row/sample of the provided dataset using np_dat to analyze.
        seed (int): Random seed.
      Returns:
        np.ndarray: Perturbed samples to be passed to the black-box model to fetch state probabilities.
        np.ndarray: Perturbed samples for constructing linear models.
      """
      make_prediction_data = np.zeros((num_samples, data.shape[1]))
      TERP_data = np.zeros((num_samples, data.shape[1]))
    
      perturb = np.random.randint(0, 2, num_samples * data.shape[1]).reshape((num_samples, data.shape[1]))
      perturb[0,:] = 1
      
      np.random.seed(seed)
      
      for i in range(num_samples):
        for j in range(data.shape[1]):
          if perturb[i,j] == 1:
            make_prediction_data[i,j] = data[index,j]
          elif perturb[i,j] == 0:
            rand_data = np.random.normal(0, 1)
            make_prediction_data[i,j] = data[index,j] + std[j]*rand_data
            TERP_data[i,j] = rand_data

      return make_prediction_data, TERP_data


def generate_neighborhood(save_dir: str, numeric_dict: dict, angle_dict: dict, sin_cos_dict: dict, np_dat: np.ndarray, index: int, seed: int, num_samples: int, selected_features: np.array, periodicity_upper: float = np.pi, periodicity_lower: float = -np.pi):
    """
    Function for creating a logger detailing MDTerp operations.

    Args:
        save_dir (str): Location to save MDTerp results.
        numeric_dict (dict): Python dictionary, each key represents the name of a numeric feature (non-periodic). Values should be lists with a single element using the index of the corresponding numpy array in np_data.
        angle_dict (dict): Python dictionary, each key represents the name of an angular feature in [-pi, pi]. Values should be lists with a single element using the index of the corresponding numpy array in np_data.
        sin_cos_dict (dict): Python dictionary, each key represents the name of an angular feature. Values should be lists with two elements representing the sine, cosine indices of the corresponding numpy array in np_data.
        np_dat (np.ndarray): Numpy 2D array containing training data for the black-box model. Samples along rows and features along columns.
        index (int): Row/sample of the provided dataset using np_dat to analyze.
        seed (int): Random seed.
        num_samples (int): Size of the generated perturbed neighborhood.
        selected_features: If an empty array (Default), perturbs all the features/columns. Otherwise, an array of integers representing the subset of features to perturb.
        periodicity_upper (float): Sets periodicity of the angular features (Default: numpy.pi).
        periodicity_lower (float): Sets periodicity of the angular features (Default: -numpy.py).
    Returns:
        list: List of np.ndarray indicating indices of numeric, angular, sin_cos features respectively.
        list: List of the combined names of the features.
    """
    
    numeric_indices = []
    angle_indices = []
    sin_indices = []
    cos_indices = []

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
            sin_indices.append(sin_cos_dict[i][0])
            cos_indices.append(sin_cos_dict[i][1])
            indices_names.append(i)
            assert sin_cos_dict[i][0] in np.arange(np_dat.shape[1]), 'Invalid sin index'
            assert sin_cos_dict[i][1] in np.arange(np_dat.shape[1]), 'Invalid cos index'

    numeric_indices = np.array(numeric_indices).flatten()
    angle_indices = np.array(angle_indices).flatten()
    sin_indices = np.array(sin_indices).flatten()
    cos_indices = np.array(cos_indices).flatten()
    
    std_master = []
    for i in range(np_dat.shape[1]):
        if i not in angle_indices:
            std_master.append(np.std(np_dat[:,i]))
        else:
            std_master.append(sst.circstd(np_dat[:,i], high = np.pi, low = -np.pi))

    std_master = np.array(std_master).flatten()
    if selected_features.shape[0]==0:
        save_directory = save_dir + 'DATA'
        os.makedirs(save_directory, exist_ok = True)
        make_prediction_data, TERP_data = perturbation(np_dat, std_master, num_samples, index, seed)
    else:
        save_directory = save_dir + 'DATA_2'
        os.makedirs(save_directory, exist_ok = True)
        make_prediction_trimmed, TERP_data = perturbation(np_dat[:, selected_features], std_master[selected_features], num_samples, index, seed)
        make_prediction_data = np.ones((num_samples, np_dat.shape[1]))*np_dat[index,:]
        make_prediction_data[:, selected_features] = make_prediction_trimmed

    np.save(save_directory + '/make_prediction.npy', make_prediction_data)  
    np.save(save_directory + '/TERP_dat.npy', TERP_data)

    return [numeric_indices, angle_indices, sin_indices, cos_indices], indices_names
