"""
MDTerp.models -- Shared linear model utilities for MDTerp analysis rounds.
"""
import numpy as np
import sklearn.metrics as met
from sklearn.linear_model import Ridge
from typing import Tuple


def similarity_kernel(data: np.ndarray, kernel_width: float = 1.0) -> np.ndarray:
    """
    Compute similarity in [0,1] of perturbed samples relative to the original
    sample using LDA-transformed Euclidean distance.

    Args:
        data: LDA-transformed data. First row is the original sample.
        kernel_width: Width of the exponential kernel (default: 1.0).

    Returns:
        Array of similarity weights in [0,1] for each sample.
    """
    distances = met.pairwise_distances(
        data, data[0].reshape(1, -1), metric='euclidean'
    ).ravel()
    return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))


def ridge_regression(
    data: np.ndarray,
    labels: np.ndarray,
    seed: int,
    alpha: float = 1.0
) -> Tuple[np.ndarray, float]:
    """
    Fit a Ridge regression model on similarity-weighted data.

    Args:
        data: 2D array of similarity-weighted features (samples x features).
        labels: 1D or 2D array of similarity-weighted target probabilities.
        seed: Random seed for solver reproducibility.
        alpha: L2 regularization strength (default: 1.0).

    Returns:
        coefficients: Feature coefficients from the fitted model.
        intercept: Intercept term of the fitted model.
    """
    clf = Ridge(alpha, random_state=seed, solver='saga')
    clf.fit(data, labels.ravel())
    return clf.coef_, clf.intercept_
