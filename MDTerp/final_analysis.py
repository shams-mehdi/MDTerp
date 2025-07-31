"""
MDTerp.final_analysis.py – Final MDTerp round for implementing forward feature selection and attributing feature importance.
"""
import numpy as np
import os
import sklearn.metrics as met
import copy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.linear_model import Ridge
from typing import Tuple

def similarity_kernel(data: np.ndarray, kernel_width: float = 1.0) -> np.ndarray:
    """
    Function for computing similarity∈[0,1] of a perturbed sample with respect to the original sample using LDA transformed distance.

    Args:
        data (np.ndarray): LDA transformed data.
        kernel_width (float): Width of the similarity kernel (Default: 1.0).

    Returns:
        np.ndarray: Similarity∈[0,1] of neighborhood.
    """
    distances = met.pairwise_distances(data,data[0].reshape(1, -1),metric='euclidean').ravel()
    return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

def SGDreg(data: np.ndarray, labels: np.ndarray, seed: int, alpha: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Function for implementing linear regression using stochastic gradient descent.

    Args:
        data (np.ndarray): Numpy 2D array containing similarity weighted training data for the black-box model. Samples along rows and features along columns.
        labels (np.ndarray): Numpy array containing metastable state prediction probabilities for a perturbed neighborhood corresponding to a specific sample. Includes the state for which the original sample has the highest probability.
        seed (int): Random seed.
        alpha (float): L2 norm of Ridge regression (Default: 1.0).
        
    Returns:
        np.ndarray: Numpy array with coefficients of all the features of the fitted linear model.
        float: Intercept of the fitted linear model.
    """
    clf = Ridge(alpha, random_state = seed, solver = 'saga')
    clf.fit(data,labels.ravel())
    coefficients = clf.coef_
    intercept = clf.intercept_
    return coefficients, intercept

def interp(coef_array: np.ndarray) -> float:
  """
    Function for computing interpretation entropy of the coefficients of a fitted linear model.

    Args:
        coef_array (np.ndarray): Numpy array with coefficients of all the features of the fitted linear model.
        
    Returns:
        float: Interpretation entropy of the coefficients of a fitted linear model.
  """
  a = np.absolute(coef_array)/np.sum(np.absolute(coef_array))
  t = 0
  for i in range(a.shape[0]):
    if a[i]==0:
      continue
    else:
      t += a[i]*np.log(a[i])
  return -t/np.log(coef_array.shape[0])

def unfaithfulness_calc(k: int, N: int, data: np.ndarray, predict_proba: np.ndarray, best_parameters_master: list, labels: np.ndarray, best_interp_master: list, best_parameters_converted: list, best_unfaithfulness_master: list, tot_feat: int, all_features: np.ndarray, seed: int) -> None:
  """
    Function for implementing linear regression using stochastic gradient descent.

    Args:
        k (int): Number of features for building local linear model.
        N (int): Number of features selected for detailed analysis.
        data (np.ndarray): Numpy 2D array containing similarity weighted training data for the black-box model. Samples along rows and features along columns.
        predict_proba (np.ndarray): Numpy array containing metastable state prediction probabilities for a perturbed neighborhood corresponding to a specific sample. Includes the state for which the original sample has the highest probability.
        best_parameters_master (list): List of lists that saves the best fit coefficients for linear models built using k=1, .., N features.
        best_interp_master (list): List that saves the interpretation entropy for the linear models built using k=1,...,N features.
        best_parameters_converted (list): List of lists that saves the best fit coefficients for linear models built using k=1, .., N features, and imputes the discarded features in initial MDTerp round with 0 importance to preserve feature ID.
        best_unfaithfulness_master (list): List that saves the unfaithfulness of the best fit linear models built using k=1, ..., N features.
        tot_feat (int): Total number of features in the dataset including both discarded features and features under analysis.
        all_features (np.ndarray): Indices of the features selected for detailed MDTerp analysis.
        seed (int): Random seed.
        
    Returns:
        None
  """ 
  models = []
  TERP_SGD_parameters = []
  TERP_SGD_unfaithfulness = []
  TERP_SGD_interp = []
  if k == 1:
    inherited_nonzero = np.array([],dtype=int)
    inherited_zero = np.arange(N)

  elif k > 1:
    inherited_nonzero = np.nonzero(best_parameters_master[k-2][:-1])[0]
    inherited_zero = np.where(best_parameters_master[k-2][:-1] == 0)[0]

  for i in range(N-k+1):
    models.append(np.append(inherited_nonzero, inherited_zero[i]))
    result_a, result_b = SGDreg(data[:,models[i]], labels, seed)
    parameters = np.zeros((N+1))
    parameters[models[i]] = result_a
    parameters[-1] = result_b
    TERP_SGD_parameters.append(parameters)
    residual = np.corrcoef(labels[:,0],(np.column_stack((data, np.ones((data.shape[0]))))@parameters[:]).reshape(-1,1)[:,0])[0,1]
    TERP_SGD_unfaithfulness.append(1-np.absolute(residual))
    TERP_SGD_interp.append(interp(TERP_SGD_parameters[-1][:-1]))
    TERP_SGD_IFE = np.array(TERP_SGD_unfaithfulness)

  best_model = np.argsort(TERP_SGD_IFE)[0]
  best_parameters_master.append(TERP_SGD_parameters[best_model])
  best_interp_master.append(TERP_SGD_interp[best_model])

  temp_coef_1 = TERP_SGD_parameters[best_model][:-1]
  temp_coef_2 = np.zeros((tot_feat))
  temp_coef_2[all_features] = copy.deepcopy(temp_coef_1)
  best_parameters_converted.append(temp_coef_2)
  best_unfaithfulness_master.append(TERP_SGD_unfaithfulness[best_model])

  surrogate_pred = data@TERP_SGD_parameters[best_model][:-1]

def zeta(U: np.ndarray,S: np.ndarray,theta: float) -> np.ndarray:
  """
    Function for computing interpretation free energy.

    Args:
        U (np.ndarray): Numpy array with unfaithfulness of the best models for number of features, k = 1, ..., N.
        S (np.ndarray): Numpy array with interpretation entropy of the best models for number of features, k = 1, ..., N.
        theta (float): Temperature of the fitted linear model.
        
    Returns:
        np.ndarray: Interpretation free energy.
  """
  return U + theta*S

def charac_theta(d_U: np.ndarray,d_S: np.ndarray) -> np.ndarray:
  """
    Function for computing change in unfaithfulness per unit change in interpretation entropy as number of features used to build linear model increases by 1.

    Args:
        d_U (np.ndarray): Change in unfaithfulness with increasing features in linear models.
        d_S (np.ndarray): Change in interpretation entropy with increasing features in linear models.
        
    Returns:
        np.ndarray: Change in unfaithfulness per unit change in interpretation entropy.
  """
  return -d_U/d_S
    
def final_model(neighborhood_data: np.ndarray, pred_proba: np.ndarray, unf_threshold: float, given_indices: np.ndarray, selected_features: np.ndarray, seed:int) -> np.ndarray:
    """
    Function for computing final feature importance by implementing forward feature selection.

    Args:
        neighborhood_data (np.ndarray): Perturbed data generated by MDTerp.neighborhood.py.
        pred_proba (np.ndarray): Metastable state probabilities obtained from the black-box.
        unf_threshold (float): Hyperparameter setting a lower limit on unfaithafulness. Forward feature selection ends when unfaithfulness reaches lower than this threshold.
        given_indices (np.ndarray): Indices of the features to perform final round of MDTerp on.
        seed (int): Random seed.
        
    Returns:
        np.ndarray: Normalized feature importance.
    """
    tot_feat = given_indices[0].shape[0] + given_indices[1].shape[0] + given_indices[2].shape[0]
    
    all_features = []
    for i in range(len(selected_features[0])):
        all_features.append(selected_features[0][i])
    for i in range(len(selected_features[1])):
        all_features.append(selected_features[1][i])  
    for i in range(len(selected_features[2])):
        all_features.append(given_indices[0].shape[0] + given_indices[1].shape[0] + np.where(selected_features[2][i][0] == given_indices[2])[0][0])  

    k_max = neighborhood_data.shape[1]

    explain_class = np.argmax(pred_proba[0,:])

    target = pred_proba[:,explain_class]

    threshold, upper, lower = 0.5, 1, 0
    target_binarized = np.where(target>threshold, upper, lower)

    clf = lda()
    clf.fit(neighborhood_data,target_binarized)
    projected_data = clf.transform(neighborhood_data)
    weights = similarity_kernel(projected_data.reshape(-1,1), 1)

    predict_proba = pred_proba[:,explain_class]
    data = neighborhood_data*(weights**0.5).reshape(-1,1)
    labels = target.reshape(-1,1)*(weights.reshape(-1,1)**0.5)

    best_parameters_master = []
    best_parameters_converted = []
    best_unfaithfulness_master = []
    best_interp_master = []
    
    N = data.shape[1]
    k_array = np.arange(1,k_max+1)
    
    for k in k_array:
      unfaithfulness_calc(k, N, data, predict_proba, best_parameters_master, labels, best_interp_master, best_parameters_converted, best_unfaithfulness_master, tot_feat, all_features, seed)


    optimal_k = 1
    
    
    if N<=3:
      for i in range(1,N):
        prime_model = -1
        if best_unfaithfulness_master[i]<=best_unfaithfulness_master[i-1] - unf_threshold:
          prime_model = copy.deepcopy(i)-1
          continue
        else:
          break
    
    else:
      charac_theta_mast = []
    
      d_U_lst = []
      d_S_lst = []
      for i in range(1, len(all_features)):
        d_U_lst.append(best_unfaithfulness_master[i] - best_unfaithfulness_master[i-1])
        d_S_lst.append(best_interp_master[i] - best_interp_master[i-1])
    
      for i in range(len(all_features)-1):
        charac_theta_mast.append(charac_theta(d_U_lst[i], d_S_lst[i]))
      
      range_theta_mast = []
      for i in range(1,len(charac_theta_mast)):
        range_theta_mast.append(np.array(charac_theta_mast)[i]-np.array(charac_theta_mast)[i-1])
    
      prime_model = np.argmin(np.array(range_theta_mast))
    
    return np.absolute(np.array(best_parameters_converted)[prime_model+1])/np.sum(np.absolute(np.array(best_parameters_converted)[prime_model+1]))
