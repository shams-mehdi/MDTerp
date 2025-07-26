import numpy as np
import os
import sklearn.metrics as met
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

def selection(coefficients, threshold):
  coefficients_abs = np.absolute(coefficients)
  selected_features = []
  coverage = 0
  for i in range(coefficients_abs.shape[0]):
    if i==threshold:
      break
    coverage = coverage+np.sort(coefficients_abs)[::-1][i]/np.sum(coefficients_abs)
    selected_features.append(np.argsort(coefficients_abs)[::-1][i])
  return selected_features    
    
def init_model(save_dir, neighborhood_data, pred_proba, cutoff, given_indices, seed):
    results_directory = save_dir + 'TERP_results'
    os.makedirs(results_directory, exist_ok = True)

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
    
    coefficients_selection, intercept_selection = SGDreg(predict_proba, data, labels, seed)
    coefficients_selection = coefficients_selection/np.sum(np.absolute(coefficients_selection))
    selected_features = selection(coefficients_selection, cutoff)

    selected_features_lst = [[], [], []]
    
    for i in range(len(selected_features)):
        if selected_features[i]<given_indices[0].shape[0]:
            selected_features_lst[0].append(given_indices[0][selected_features[i]])
        else:
            if selected_features[i]<given_indices[0].shape[0] + given_indices[1].shape[0]:
                selected_features_lst[1].append(given_indices[1][selected_features[i] - given_indices[0].shape[0]])
            else:
                selected_features_lst[2].append(given_indices[2][selected_features[i] - given_indices[0].shape[0] - given_indices[1].shape[0], :])

    return selected_features_lst