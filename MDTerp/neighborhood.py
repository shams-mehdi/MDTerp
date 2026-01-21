"""
MDTerp.neighborhood.py – Neighborhood generation for local interpretability.

This module generates perturbed samples around a target point to build
local linear surrogate models for interpreting black-box model predictions.
Uses Gaussian perturbations with feature-specific standard deviations.
"""
import numpy as np
import os
import scipy.stats as sst
from typing import Tuple, List, Dict


def perturbation(
    data: np.ndarray,
    std_devs: np.ndarray,
    num_samples: int,
    sample_index: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate perturbed neighborhood samples for local interpretability.

    Creates synthetic samples by randomly perturbing features around a target
    sample. Each feature is independently chosen to be either kept at its
    original value or perturbed by Gaussian noise scaled by that feature's
    standard deviation.

    Args:
        data: Training data array of shape (n_samples, n_features).
        std_devs: Standard deviations for each feature, shape (n_features,).
            Used to scale perturbations appropriately for each feature.
        num_samples: Number of perturbed samples to generate.
        sample_index: Index of the target sample to perturb around.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (perturbed_predictions_data, terp_model_data):
        - perturbed_predictions_data: Array (num_samples, n_features) with
          perturbed samples in original feature space for black-box predictions.
        - terp_model_data: Array (num_samples, n_features) with normalized
          perturbations (in units of std) for training surrogate model.
    """
    n_features = data.shape[1]

    # Initialize output arrays
    perturbed_predictions = np.zeros((num_samples, n_features))
    terp_input = np.zeros((num_samples, n_features))

    # Generate random binary mask: 1 = keep original, 0 = perturb
    # First sample always uses original values (all 1s)
    np.random.seed(seed)
    perturbation_mask = np.random.randint(0, 2, num_samples * n_features)
    perturbation_mask = perturbation_mask.reshape((num_samples, n_features))
    perturbation_mask[0, :] = 1

    np.random.seed(seed)

    # Generate perturbed samples
    for sample_idx in range(num_samples):
        for feat_idx in range(n_features):
            if perturbation_mask[sample_idx, feat_idx] == 1:
                # Keep original feature value
                perturbed_predictions[sample_idx, feat_idx] = data[sample_index, feat_idx]
                # TERP input is 0 (no perturbation)
            else:
                # Perturb with scaled Gaussian noise
                noise = np.random.normal(0, 1)
                perturbed_predictions[sample_idx, feat_idx] = \
                    data[sample_index, feat_idx] + std_devs[feat_idx] * noise
                terp_input[sample_idx, feat_idx] = noise

    return perturbed_predictions, terp_input


def generate_neighborhood(
    save_dir: str,
    numeric_dict: Dict,
    angle_dict: Dict,
    sin_cos_dict: Dict,
    np_dat: np.ndarray,
    sample_index: int,
    seed: int,
    num_samples: int,
    selected_features: np.ndarray,
    periodicity_upper: float = np.pi,
    periodicity_lower: float = -np.pi
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generate perturbed neighborhood and organize feature metadata.

    Creates a perturbed neighborhood around a target sample, handling different
    feature types appropriately (numeric, angular, sin/cos encoded). Computes
    feature-specific standard deviations (circular std for angles) and saves
    the perturbed data for black-box prediction and surrogate modeling.

    Args:
        save_dir: Directory for saving intermediate results.
        numeric_dict: Non-periodic numeric features. Keys: feature names,
            Values: [column_index] in np_dat.
        angle_dict: Angular features in [-π, π]. Keys: feature names,
            Values: [column_index] in np_dat.
        sin_cos_dict: Angular features with sin/cos encoding. Keys: feature names,
            Values: [sin_column_index, cos_column_index] in np_dat.
        np_dat: Training data array of shape (n_samples, n_features).
        sample_index: Index of target sample to analyze.
        seed: Random seed for reproducibility.
        num_samples: Number of perturbed samples to generate.
        selected_features: If empty array, perturbs all features (stage 1).
            Otherwise, array of feature indices to perturb (stage 2).
        periodicity_upper: Upper bound for angular periodicity.
        periodicity_lower: Lower bound for angular periodicity.

    Returns:
        Tuple of (feature_type_indices, feature_names):
        - feature_type_indices: List [numeric_idx, angle_idx, sin_idx, cos_idx]
        - feature_names: Ordered list of feature names

    Saves:
        - {save_dir}/DATA/make_prediction.npy (stage 1) or
          {save_dir}/DATA_2/make_prediction.npy (stage 2): Perturbed samples
        - {save_dir}/DATA/TERP_dat.npy or DATA_2/TERP_dat.npy: Normalized perturbations
    """
    # Parse feature dictionaries and build indices
    numeric_indices_list = []
    angle_indices_list = []
    sin_indices_list = []
    cos_indices_list = []
    feature_names = []

    # Extract numeric features
    for feature_name in numeric_dict:
        col_idx = numeric_dict[feature_name][0]
        assert col_idx in np.arange(np_dat.shape[1]), \
            f'Invalid numeric index {col_idx} for feature {feature_name}'
        numeric_indices_list.append(col_idx)
        feature_names.append(feature_name)

    # Extract angle features
    for feature_name in angle_dict:
        col_idx = angle_dict[feature_name][0]
        assert col_idx in np.arange(np_dat.shape[1]), \
            f'Invalid angle index {col_idx} for feature {feature_name}'
        angle_indices_list.append(col_idx)
        feature_names.append(feature_name)

    # Extract sin/cos features
    for feature_name in sin_cos_dict:
        sin_idx = sin_cos_dict[feature_name][0]
        cos_idx = sin_cos_dict[feature_name][1]
        assert sin_idx in np.arange(np_dat.shape[1]), \
            f'Invalid sin index {sin_idx} for feature {feature_name}'
        assert cos_idx in np.arange(np_dat.shape[1]), \
            f'Invalid cos index {cos_idx} for feature {feature_name}'
        sin_indices_list.append(sin_idx)
        cos_indices_list.append(cos_idx)
        feature_names.append(feature_name)

    # Convert to numpy arrays
    numeric_indices = np.array(numeric_indices_list, dtype=int).flatten()
    angle_indices = np.array(angle_indices_list, dtype=int).flatten()
    sin_indices = np.array(sin_indices_list, dtype=int).flatten()
    cos_indices = np.array(cos_indices_list, dtype=int).flatten()

    # Compute feature-specific standard deviations
    std_devs = []
    for col_idx in range(np_dat.shape[1]):
        if col_idx in angle_indices:
            # Use circular standard deviation for angular features
            std = sst.circstd(np_dat[:, col_idx], high=periodicity_upper, low=periodicity_lower)
        else:
            # Use regular standard deviation for numeric features
            std = np.std(np_dat[:, col_idx])
        std_devs.append(std)
    std_devs = np.array(std_devs)

    # Generate perturbations (stage-dependent)
    if selected_features.shape[0] == 0:
        # Stage 1: Perturb all features
        output_dir = os.path.join(save_dir, 'DATA')
        os.makedirs(output_dir, exist_ok=True)

        perturbed_data, terp_input = perturbation(
            np_dat, std_devs, num_samples, sample_index, seed
        )

    else:
        # Stage 2: Perturb only selected features
        output_dir = os.path.join(save_dir, 'DATA_2')
        os.makedirs(output_dir, exist_ok=True)

        # Perturb selected features only
        selected_data = np_dat[:, selected_features]
        selected_stds = std_devs[selected_features]

        perturbed_selected, terp_input = perturbation(
            selected_data, selected_stds, num_samples, sample_index, seed
        )

        # Build full perturbed array (non-selected features stay at original values)
        perturbed_data = np.tile(np_dat[sample_index, :], (num_samples, 1))
        perturbed_data[:, selected_features] = perturbed_selected

    # Save outputs
    np.save(os.path.join(output_dir, 'make_prediction.npy'), perturbed_data)
    np.save(os.path.join(output_dir, 'TERP_dat.npy'), terp_input)

    feature_type_indices = [numeric_indices, angle_indices, sin_indices, cos_indices]
    return feature_type_indices, feature_names
