import numpy as np
import os
import sklearn.metrics as met
import copy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.linear_model import Ridge

def similarity_kernel(data, kernel_width):
  distances = met.pairwise_distances(data,data[0].reshape(1, -1),metric='euclidean').ravel()
  return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

def SGDreg(predict_proba, data, labels, seed):
  clf = Ridge(alpha=1.0, random_state = seed, solver = 'saga')
  clf.fit(data,labels.ravel())
  coefficients = clf.coef_
  intercept = clf.intercept_
  return coefficients, intercept

def interp(coef_array):
  a = np.absolute(coef_array)/np.sum(np.absolute(coef_array))
  t = 0
  for i in range(a.shape[0]):
    if a[i]==0:
      continue
    else:
      t += a[i]*np.log(a[i])
  return -t/np.log(coef_array.shape[0])

def unfaithfulness_calc(k, N, data, predict_proba, best_parameters_master, labels, best_interp_master, best_parameters_converted, best_unfaithfulness_master, tot_feat, all_features, seed):
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
    result_a, result_b = SGDreg(predict_proba, data[:,models[i]], labels, seed)
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

def zeta(U,S,theta):
  return U + theta*S

def charac_theta(d_U,d_S):
  return -d_U/d_S
    
def final_model(save_dir, neighborhood_data, pred_proba, unf_threshold, given_indices, selected_features, seed):
    tot_feat = given_indices[0].shape[0] + given_indices[1].shape[0] + given_indices[2].shape[0]
    
    all_features = []
    for i in range(len(selected_features[0])):
        all_features.append(selected_features[0][i])
    for i in range(len(selected_features[1])):
        all_features.append(selected_features[1][i])  
    for i in range(len(selected_features[2])):
        all_features.append(given_indices[0].shape[0] + given_indices[1].shape[0] + np.where(selected_features[2][i][0] == given_indices[2])[0][0])  

    
    results_directory = save_dir + 'TERP_results_2'
    os.makedirs(results_directory, exist_ok = True)
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