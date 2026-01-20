"""
MDTerp.visualization.py – Visualization utilities for MDTerp analysis results.

This module provides functions for visualizing feature importance, transition
states, and other analysis results from MDTerp. It includes plotting functions
for individual transitions, comparative analyses across transitions, and
summary visualizations.

Reference:
    "Thermodynamics-inspired explanations of artificial intelligence"
    Shams Mehdi and Pratyush Tiwary, Nature Communications
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from typing import Dict, List, Optional, Tuple, Union
import os


def plot_feature_importance(
    results_file: str,
    feature_names_file: str,
    transition: Optional[str] = None,
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show_std: bool = True,
    importance_coverage: float = 0.8
) -> plt.Figure:
    """
    Plot feature importance for a specific transition or all transitions.

    Args:
        results_file: Path to MDTerp_results_all.pkl file
        feature_names_file: Path to MDTerp_feature_names.npy file
        transition: Specific transition to plot (e.g., "0_1"). If None, plots
            mean importance across all transitions.
        top_n: Number of top features to display
        figsize: Figure size (width, height)
        save_path: If provided, saves figure to this path
        show_std: If True, shows error bars with standard deviation
        importance_coverage: Coverage threshold for filtering features (0-1)

    Returns:
        matplotlib Figure object
    """
    # Load data
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    feature_names = np.load(feature_names_file, allow_pickle=True)

    # Organize results by transition
    transition_results = {}
    for sample_idx, (trans_name, importance) in results.items():
        if trans_name not in transition_results:
            transition_results[trans_name] = []
        transition_results[trans_name].append(importance)

    # Calculate mean and std for each transition
    transition_stats = {}
    for trans_name, importances in transition_results.items():
        mean_imp = np.mean(importances, axis=0)
        std_imp = np.std(importances, axis=0)

        # Normalize
        norm = np.sum(mean_imp)
        if norm > 0:
            mean_imp = mean_imp / norm
            std_imp = std_imp / norm

        transition_stats[trans_name] = (mean_imp, std_imp)

    # Select data to plot
    if transition is not None:
        if transition not in transition_stats:
            raise ValueError(f"Transition {transition} not found in results")
        mean_importance, std_importance = transition_stats[transition]
        title = f"Feature Importance for Transition {transition}"
    else:
        # Average across all transitions
        all_means = [stats[0] for stats in transition_stats.values()]
        mean_importance = np.mean(all_means, axis=0)
        std_importance = np.std(all_means, axis=0)
        title = "Average Feature Importance Across All Transitions"

    # Get top N features
    top_indices = np.argsort(mean_importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = mean_importance[top_indices]
    top_std = std_importance[top_indices]

    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(top_features))

    if show_std:
        ax.bar(x_pos, top_importance, yerr=top_std, capsize=5, alpha=0.7,
               color='steelblue', edgecolor='black', linewidth=1.2)
    else:
        ax.bar(x_pos, top_importance, alpha=0.7, color='steelblue',
               edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Importance', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_features, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_transition_heatmap(
    results_file: str,
    feature_names_file: str,
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    cmap: str = 'YlOrRd'
) -> plt.Figure:
    """
    Create a heatmap showing feature importance across all transitions.

    Args:
        results_file: Path to MDTerp_results_all.pkl file
        feature_names_file: Path to MDTerp_feature_names.npy file
        top_n: Number of top features to display
        figsize: Figure size (width, height)
        save_path: If provided, saves figure to this path
        cmap: Colormap name for heatmap

    Returns:
        matplotlib Figure object
    """
    # Load data
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    feature_names = np.load(feature_names_file, allow_pickle=True)

    # Organize results by transition
    transition_results = {}
    for sample_idx, (trans_name, importance) in results.items():
        if trans_name not in transition_results:
            transition_results[trans_name] = []
        transition_results[trans_name].append(importance)

    # Calculate mean importance for each transition
    transition_means = {}
    for trans_name, importances in transition_results.items():
        mean_imp = np.mean(importances, axis=0)
        # Normalize
        norm = np.sum(mean_imp)
        if norm > 0:
            mean_imp = mean_imp / norm
        transition_means[trans_name] = mean_imp

    # Get overall top N features across all transitions
    all_importances = np.array(list(transition_means.values()))
    mean_across_transitions = np.mean(all_importances, axis=0)
    top_indices = np.argsort(mean_across_transitions)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]

    # Build heatmap matrix
    transitions_sorted = sorted(transition_means.keys())
    heatmap_data = np.zeros((len(transitions_sorted), len(top_features)))

    for i, trans_name in enumerate(transitions_sorted):
        for j, feat_idx in enumerate(top_indices):
            heatmap_data[i, j] = transition_means[trans_name][feat_idx]

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', interpolation='nearest')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(top_features)))
    ax.set_yticks(np.arange(len(transitions_sorted)))
    ax.set_xticklabels(top_features, rotation=45, ha='right')
    ax.set_yticklabels(transitions_sorted)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Importance', rotation=270, labelpad=20, fontsize=12)

    # Add labels
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Transitions', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Heatmap Across Transitions', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_transition_comparison(
    results_file: str,
    feature_names_file: str,
    transitions: List[str],
    top_n: int = 10,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare feature importance across multiple specific transitions.

    Args:
        results_file: Path to MDTerp_results_all.pkl file
        feature_names_file: Path to MDTerp_feature_names.npy file
        transitions: List of transition names to compare (e.g., ["0_1", "1_2"])
        top_n: Number of top features to display
        figsize: Figure size (width, height)
        save_path: If provided, saves figure to this path

    Returns:
        matplotlib Figure object
    """
    # Load data
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    feature_names = np.load(feature_names_file, allow_pickle=True)

    # Organize results by transition
    transition_results = {}
    for sample_idx, (trans_name, importance) in results.items():
        if trans_name not in transition_results:
            transition_results[trans_name] = []
        transition_results[trans_name].append(importance)

    # Calculate mean importance for requested transitions
    transition_means = {}
    for trans_name in transitions:
        if trans_name not in transition_results:
            raise ValueError(f"Transition {trans_name} not found in results")

        mean_imp = np.mean(transition_results[trans_name], axis=0)
        # Normalize
        norm = np.sum(mean_imp)
        if norm > 0:
            mean_imp = mean_imp / norm
        transition_means[trans_name] = mean_imp

    # Get top features across all requested transitions
    all_importances = np.array(list(transition_means.values()))
    mean_across = np.mean(all_importances, axis=0)
    top_indices = np.argsort(mean_across)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]

    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(top_features))
    width = 0.8 / len(transitions)

    colors = plt.cm.Set2(np.linspace(0, 1, len(transitions)))

    for i, trans_name in enumerate(transitions):
        importance_vals = [transition_means[trans_name][idx] for idx in top_indices]
        offset = (i - len(transitions)/2 + 0.5) * width
        ax.bar(x + offset, importance_vals, width, label=f'Transition {trans_name}',
               color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Importance', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Comparison Across Transitions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_sample_importance_distribution(
    results_file: str,
    feature_names_file: str,
    transition: str,
    feature_idx: Optional[int] = None,
    feature_name: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of importance values across samples for a specific feature.

    Args:
        results_file: Path to MDTerp_results_all.pkl file
        feature_names_file: Path to MDTerp_feature_names.npy file
        transition: Transition name (e.g., "0_1")
        feature_idx: Index of feature to plot (either this or feature_name required)
        feature_name: Name of feature to plot (either this or feature_idx required)
        figsize: Figure size (width, height)
        save_path: If provided, saves figure to this path

    Returns:
        matplotlib Figure object
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
            raise ValueError(f"Feature {feature_name} not found in feature names")

    # Collect importance values for this feature and transition
    importance_values = []
    for sample_idx, (trans_name, importance) in results.items():
        if trans_name == transition:
            importance_values.append(importance[feature_idx])

    if len(importance_values) == 0:
        raise ValueError(f"No samples found for transition {transition}")

    importance_values = np.array(importance_values)

    # Create histogram
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(importance_values, bins=20, alpha=0.7, color='steelblue',
            edgecolor='black', linewidth=1.2)

    # Add mean line
    mean_val = np.mean(importance_values)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.4f}')

    ax.set_xlabel('Importance Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Importance Distribution for {feature_names[feature_idx]} '
                f'(Transition {transition})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_summary_report(
    results_dir: str,
    output_dir: Optional[str] = None,
    top_n: int = 15
) -> None:
    """
    Create a comprehensive PDF report with multiple visualization plots.

    Args:
        results_dir: Directory containing MDTerp results
        output_dir: Directory to save visualizations (defaults to results_dir/visualizations)
        top_n: Number of top features to display in plots

    Returns:
        None. Saves multiple PNG files to output_dir.
    """
    results_file = os.path.join(results_dir, 'MDTerp_results_all.pkl')
    feature_names_file = os.path.join(results_dir, 'MDTerp_feature_names.npy')

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    if not os.path.exists(feature_names_file):
        raise FileNotFoundError(f"Feature names file not found: {feature_names_file}")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating summary visualizations in {output_dir}...")

    # Load data to get transitions
    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    # Get unique transitions
    transitions = set()
    for sample_idx, (trans_name, importance) in results.items():
        transitions.add(trans_name)
    transitions = sorted(list(transitions))

    print(f"Found {len(transitions)} transitions: {transitions}")

    # 1. Overall importance plot
    print("Creating overall feature importance plot...")
    fig1 = plot_feature_importance(
        results_file, feature_names_file,
        transition=None, top_n=top_n,
        save_path=os.path.join(output_dir, 'overall_importance.png')
    )
    plt.close(fig1)

    # 2. Per-transition importance plots
    for trans in transitions:
        print(f"Creating importance plot for transition {trans}...")
        fig = plot_feature_importance(
            results_file, feature_names_file,
            transition=trans, top_n=top_n,
            save_path=os.path.join(output_dir, f'importance_transition_{trans}.png')
        )
        plt.close(fig)

    # 3. Heatmap
    if len(transitions) > 1:
        print("Creating transition heatmap...")
        fig3 = plot_transition_heatmap(
            results_file, feature_names_file,
            top_n=top_n,
            save_path=os.path.join(output_dir, 'transition_heatmap.png')
        )
        plt.close(fig3)

    # 4. Comparison plot (if multiple transitions)
    if len(transitions) >= 2:
        print("Creating transition comparison plot...")
        fig4 = plot_transition_comparison(
            results_file, feature_names_file,
            transitions=transitions[:4],  # Compare up to 4 transitions
            top_n=min(10, top_n),
            save_path=os.path.join(output_dir, 'transition_comparison.png')
        )
        plt.close(fig4)

    print(f"Summary report complete! Visualizations saved to {output_dir}")
