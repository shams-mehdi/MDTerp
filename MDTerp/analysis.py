"""
MDTerp.analysis.py – Additional analysis utilities for MDTerp results.

This module provides helper functions for analyzing MDTerp feature importance
results, including statistical analysis, feature ranking, and comparison utilities.

Reference:
    "Thermodynamics-inspired explanations of artificial intelligence"
    Shams Mehdi and Pratyush Tiwary, Nature Communications
"""
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import os


def get_top_features(
    results_file: str,
    feature_names_file: str,
    transition: Optional[str] = None,
    n: int = 10,
    normalize: bool = True
) -> List[Tuple[str, float, float]]:
    """
    Get top N most important features for a transition or across all transitions.

    Args:
        results_file: Path to MDTerp_results_all.pkl file
        feature_names_file: Path to MDTerp_feature_names.npy file
        transition: Specific transition (e.g., "0_1"). If None, averages across all.
        n: Number of top features to return
        normalize: Whether to normalize importance values

    Returns:
        List of tuples (feature_name, mean_importance, std_importance)
        sorted by mean importance in descending order
    """
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    feature_names = np.load(feature_names_file, allow_pickle=True)

    # Organize by transition
    transition_results = defaultdict(list)
    for sample_idx, (trans_name, importance) in results.items():
        transition_results[trans_name].append(importance)

    # Calculate statistics
    if transition is not None:
        if transition not in transition_results:
            raise ValueError(f"Transition {transition} not found in results")
        importances = np.array(transition_results[transition])
    else:
        # Combine all transitions
        all_importances = []
        for trans_name, imps in transition_results.items():
            all_importances.extend(imps)
        importances = np.array(all_importances)

    mean_importance = np.mean(importances, axis=0)
    std_importance = np.std(importances, axis=0)

    if normalize and np.sum(mean_importance) > 0:
        norm = np.sum(mean_importance)
        mean_importance = mean_importance / norm
        std_importance = std_importance / norm

    # Get top N
    top_indices = np.argsort(mean_importance)[::-1][:n]
    top_features = [
        (feature_names[i], mean_importance[i], std_importance[i])
        for i in top_indices
    ]

    return top_features


def compare_transitions(
    results_file: str,
    feature_names_file: str,
    transition1: str,
    transition2: str,
    top_n: int = 20
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Compare feature importance between two transitions.

    Args:
        results_file: Path to MDTerp_results_all.pkl file
        feature_names_file: Path to MDTerp_feature_names.npy file
        transition1: First transition name (e.g., "0_1")
        transition2: Second transition name (e.g., "1_2")
        top_n: Number of top features to analyze

    Returns:
        Dictionary with keys:
        - 'shared': Features important in both (name, avg_importance)
        - 'unique_to_1': Features unique to transition1 (name, importance)
        - 'unique_to_2': Features unique to transition2 (name, importance)
        - 'difference': Features with largest difference (name, diff)
    """
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    feature_names = np.load(feature_names_file, allow_pickle=True)

    # Get importance for each transition
    trans1_importances = []
    trans2_importances = []

    for sample_idx, (trans_name, importance) in results.items():
        if trans_name == transition1:
            trans1_importances.append(importance)
        elif trans_name == transition2:
            trans2_importances.append(importance)

    if not trans1_importances:
        raise ValueError(f"No samples found for transition {transition1}")
    if not trans2_importances:
        raise ValueError(f"No samples found for transition {transition2}")

    mean1 = np.mean(trans1_importances, axis=0)
    mean2 = np.mean(trans2_importances, axis=0)

    # Normalize
    if np.sum(mean1) > 0:
        mean1 = mean1 / np.sum(mean1)
    if np.sum(mean2) > 0:
        mean2 = mean2 / np.sum(mean2)

    # Get top features for each
    top1_indices = set(np.argsort(mean1)[::-1][:top_n])
    top2_indices = set(np.argsort(mean2)[::-1][:top_n])

    # Find shared and unique
    shared_indices = top1_indices & top2_indices
    unique1_indices = top1_indices - top2_indices
    unique2_indices = top2_indices - top1_indices

    # Calculate differences
    diff = np.abs(mean1 - mean2)
    diff_indices = np.argsort(diff)[::-1][:top_n]

    result = {
        'shared': [(feature_names[i], (mean1[i] + mean2[i]) / 2) for i in shared_indices],
        'unique_to_1': [(feature_names[i], mean1[i]) for i in unique1_indices],
        'unique_to_2': [(feature_names[i], mean2[i]) for i in unique2_indices],
        'difference': [(feature_names[i], diff[i]) for i in diff_indices]
    }

    # Sort each list by importance/difference
    result['shared'].sort(key=lambda x: x[1], reverse=True)
    result['unique_to_1'].sort(key=lambda x: x[1], reverse=True)
    result['unique_to_2'].sort(key=lambda x: x[1], reverse=True)
    result['difference'].sort(key=lambda x: x[1], reverse=True)

    return result


def get_feature_statistics(
    results_file: str,
    feature_names_file: str,
    feature_name: Optional[str] = None,
    feature_idx: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Get detailed statistics for a specific feature across all transitions.

    Args:
        results_file: Path to MDTerp_results_all.pkl file
        feature_names_file: Path to MDTerp_feature_names.npy file
        feature_name: Name of feature to analyze (either this or feature_idx)
        feature_idx: Index of feature to analyze (either this or feature_name)

    Returns:
        Dictionary mapping transition names to statistics dict containing:
        - 'mean': Mean importance
        - 'std': Standard deviation
        - 'min': Minimum importance
        - 'max': Maximum importance
        - 'median': Median importance
        - 'n_samples': Number of samples
    """
    # Load data
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    feature_names = np.load(feature_names_file, allow_pickle=True)

    # Determine feature index
    if feature_idx is None and feature_name is None:
        raise ValueError("Either feature_idx or feature_name must be provided")

    if feature_name is not None:
        try:
            feature_idx = list(feature_names).index(feature_name)
        except ValueError:
            raise ValueError(f"Feature {feature_name} not found")

    # Collect values by transition
    transition_values = defaultdict(list)
    for sample_idx, (trans_name, importance) in results.items():
        transition_values[trans_name].append(importance[feature_idx])

    # Calculate statistics for each transition
    stats = {}
    for trans_name, values in transition_values.items():
        values = np.array(values)
        stats[trans_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'n_samples': len(values)
        }

    return stats


def get_transition_summary(
    results_file: str,
    feature_names_file: str
) -> Dict[str, Dict]:
    """
    Get summary statistics for all transitions.

    Args:
        results_file: Path to MDTerp_results_all.pkl file
        feature_names_file: Path to MDTerp_feature_names.npy file

    Returns:
        Dictionary mapping transition names to summary info:
        - 'n_samples': Number of samples for this transition
        - 'top_features': List of (feature_name, importance) tuples (top 5)
        - 'mean_entropy': Average entropy of importance distribution
        - 'n_important_features': Number of features with importance > threshold
    """
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    feature_names = np.load(feature_names_file, allow_pickle=True)

    # Organize by transition
    transition_results = defaultdict(list)
    for sample_idx, (trans_name, importance) in results.items():
        transition_results[trans_name].append(importance)

    summary = {}
    for trans_name, importances in transition_results.items():
        importances = np.array(importances)
        mean_imp = np.mean(importances, axis=0)

        # Normalize
        if np.sum(mean_imp) > 0:
            mean_imp_norm = mean_imp / np.sum(mean_imp)
        else:
            mean_imp_norm = mean_imp

        # Get top features
        top_indices = np.argsort(mean_imp_norm)[::-1][:5]
        top_features = [(feature_names[i], mean_imp_norm[i]) for i in top_indices]

        # Calculate entropy
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy = -np.sum(mean_imp_norm * np.log(mean_imp_norm + eps))

        # Count important features (>1% of total importance)
        n_important = np.sum(mean_imp_norm > 0.01)

        summary[trans_name] = {
            'n_samples': len(importances),
            'top_features': top_features,
            'mean_entropy': float(entropy),
            'n_important_features': int(n_important)
        }

    return summary


def identify_consensus_features(
    results_file: str,
    feature_names_file: str,
    threshold: float = 0.5,
    top_n_per_transition: int = 10
) -> Dict[str, int]:
    """
    Identify features that are consistently important across multiple transitions.

    Args:
        results_file: Path to MDTerp_results_all.pkl file
        feature_names_file: Path to MDTerp_feature_names.npy file
        threshold: Fraction of transitions a feature must be important in
        top_n_per_transition: Consider top N features per transition

    Returns:
        Dictionary mapping feature names to count of transitions where
        they appear in top N
    """
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    feature_names = np.load(feature_names_file, allow_pickle=True)

    # Organize by transition
    transition_results = defaultdict(list)
    for sample_idx, (trans_name, importance) in results.items():
        transition_results[trans_name].append(importance)

    # Count appearances in top N
    feature_counts = defaultdict(int)
    n_transitions = len(transition_results)

    for trans_name, importances in transition_results.items():
        mean_imp = np.mean(importances, axis=0)
        top_indices = np.argsort(mean_imp)[::-1][:top_n_per_transition]

        for idx in top_indices:
            feature_counts[feature_names[idx]] += 1

    # Filter by threshold
    min_appearances = int(threshold * n_transitions)
    consensus_features = {
        feat: count for feat, count in feature_counts.items()
        if count >= min_appearances
    }

    # Sort by count
    consensus_features = dict(sorted(consensus_features.items(),
                                    key=lambda x: x[1], reverse=True))

    return consensus_features


def export_results_to_csv(
    results_file: str,
    feature_names_file: str,
    output_file: str,
    include_raw_data: bool = False
) -> None:
    """
    Export MDTerp results to CSV format for external analysis.

    Args:
        results_file: Path to MDTerp_results_all.pkl file
        feature_names_file: Path to MDTerp_feature_names.npy file
        output_file: Path to output CSV file
        include_raw_data: If True, includes per-sample data; if False,
            only includes summary statistics per transition

    Returns:
        None. Writes CSV file to output_file.
    """
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    feature_names = np.load(feature_names_file, allow_pickle=True)

    with open(output_file, 'w') as f:
        if include_raw_data:
            # Header
            header = ['sample_idx', 'transition'] + list(feature_names)
            f.write(','.join(header) + '\n')

            # Write per-sample data
            for sample_idx in sorted(results.keys()):
                trans_name, importance = results[sample_idx]
                row = [str(sample_idx), trans_name] + [f"{val:.6f}" for val in importance]
                f.write(','.join(row) + '\n')
        else:
            # Summary statistics per transition
            transition_results = defaultdict(list)
            for sample_idx, (trans_name, importance) in results.items():
                transition_results[trans_name].append(importance)

            # Header
            header = ['transition', 'n_samples'] + \
                    [f"{name}_mean" for name in feature_names] + \
                    [f"{name}_std" for name in feature_names]
            f.write(','.join(header) + '\n')

            # Write summary data
            for trans_name in sorted(transition_results.keys()):
                importances = np.array(transition_results[trans_name])
                mean_imp = np.mean(importances, axis=0)
                std_imp = np.std(importances, axis=0)

                row = [trans_name, str(len(importances))] + \
                      [f"{val:.6f}" for val in mean_imp] + \
                      [f"{val:.6f}" for val in std_imp]
                f.write(','.join(row) + '\n')

    print(f"Results exported to {output_file}")
