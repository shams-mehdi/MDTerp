"""
MDTerp.visualization -- Post-analysis plotting functions for MDTerp results.

Provides publication-quality visualizations of feature importance across
transitions, unfaithfulness curves, and per-point variability analysis.
"""
import numpy as np
import pickle
import os
import glob
import matplotlib.pyplot as plt
from typing import Optional
from MDTerp.utils import transition_summary


def plot_feature_importance(
    result_path: str,
    feature_names_path: str,
    transition: str,
    show_std: bool = False,
    top_n: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """
    Plot mean feature importance as a horizontal bar chart for a transition.

    Args:
        result_path: Path to MDTerp_results_all.pkl.
        feature_names_path: Path to MDTerp_feature_names.npy.
        transition: Transition key (e.g., "0_1").
        show_std: Show standard deviation error bars (default: False).
        top_n: Only show the top N features. None shows all non-zero.
        save_path: If provided, save the figure to this path.
        figsize: Figure size as (width, height) tuple.

    Returns:
        matplotlib Figure object.
    """
    feature_names = np.load(feature_names_path, allow_pickle=True)
    summary = transition_summary(result_path, importance_coverage=1.0)

    if transition not in summary:
        raise ValueError(
            f"Transition '{transition}' not found. "
            f"Available: {list(summary.keys())}"
        )

    mean_imp = summary[transition][0]
    std_imp = summary[transition][1]

    nonzero_mask = mean_imp > 0
    ordered_indices = np.argsort(mean_imp)[::-1]
    ordered_indices = ordered_indices[nonzero_mask[ordered_indices]]

    if top_n is not None:
        ordered_indices = ordered_indices[:top_n]

    ordered_mean = mean_imp[ordered_indices]
    ordered_std = std_imp[ordered_indices]
    ordered_names = feature_names[ordered_indices]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(ordered_mean))

    if show_std:
        ax.barh(y_pos, ordered_mean, xerr=ordered_std, capsize=4,
                color='steelblue', edgecolor='black', linewidth=0.5)
    else:
        ax.barh(y_pos, ordered_mean,
                color='steelblue', edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_names, fontsize=12)
    ax.set_xlabel('Feature Importance', fontsize=14)
    ax.set_title(
        f'Feature Importance for Transition {transition}\n'
        f'Coverage: {int(100 * np.sum(ordered_mean))}%',
        fontsize=16,
    )
    ax.invert_yaxis()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_importance_heatmap(
    result_path: str,
    feature_names_path: str,
    importance_coverage: float = 0.8,
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Plot a heatmap of feature importance across all transitions.

    Rows are features, columns are transitions, color intensity represents
    mean importance. Only features with non-zero importance in at least one
    transition are shown.

    Args:
        result_path: Path to MDTerp_results_all.pkl.
        feature_names_path: Path to MDTerp_feature_names.npy.
        importance_coverage: Filter features by cumulative importance
            per transition (default: 0.8).
        save_path: If provided, save the figure to this path.
        figsize: Figure size. Auto-scaled if None.

    Returns:
        matplotlib Figure object.
    """
    feature_names = np.load(feature_names_path, allow_pickle=True)
    summary = transition_summary(result_path, importance_coverage=importance_coverage)

    transitions = sorted(summary.keys())
    n_features = len(feature_names)

    imp_matrix = np.zeros((n_features, len(transitions)))
    for j, trans in enumerate(transitions):
        imp_matrix[:, j] = summary[trans][0]

    active_mask = np.any(imp_matrix > 0, axis=1)
    active_indices = np.where(active_mask)[0]
    imp_matrix = imp_matrix[active_indices, :]
    active_names = feature_names[active_indices]

    if figsize is None:
        figsize = (max(6, len(transitions) * 2), max(6, len(active_names) * 0.4))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(imp_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_xticks(np.arange(len(transitions)))
    ax.set_xticklabels(transitions, fontsize=12, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(active_names)))
    ax.set_yticklabels(active_names, fontsize=10)
    ax.set_xlabel('Transition', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    ax.set_title('Feature Importance Across Transitions', fontsize=16)

    for i in range(imp_matrix.shape[0]):
        for j in range(imp_matrix.shape[1]):
            val = imp_matrix[i, j]
            if val > 0.01:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color='black' if val < 0.5 else 'white')

    fig.colorbar(im, ax=ax, label='Importance', shrink=0.8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_unfaithfulness_curve(
    result_dir: str,
    transition: str,
    point_index: int,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Plot the unfaithfulness vs number of features curve for a single point.

    Shows how surrogate model quality improves as more features are included
    in the linear explanation, visualizing the TERP free energy trade-off.

    Args:
        result_dir: Directory containing per-point result files.
        transition: Transition key (e.g., "0_1").
        point_index: Point index within the transition.
        save_path: If provided, save the figure to this path.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    filename = f"{transition}_point{point_index}_result.npz"
    filepath = os.path.join(result_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Result file not found: {filepath}. "
            f"Ensure keep_checkpoints=True when running MDTerp."
        )

    data = np.load(filepath, allow_pickle=True)
    unfaithfulness = data['unfaithfulness_all']

    fig, ax = plt.subplots(figsize=figsize)
    k_values = np.arange(1, len(unfaithfulness) + 1)

    ax.plot(k_values, unfaithfulness, 'o-', color='steelblue',
            linewidth=2, markersize=6)
    ax.set_xlabel('Number of Features (k)', fontsize=14)
    ax.set_ylabel('Unfaithfulness (1 - |r|)', fontsize=14)
    ax.set_title(
        f'Unfaithfulness Curve\n'
        f'Transition {transition}, Point {point_index}',
        fontsize=16,
    )
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_point_variability(
    result_path: str,
    feature_names_path: str,
    transition: str,
    top_n: int = 5,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot per-point importance variability for the top features in a transition.

    Shows a strip plot where each dot represents one point's importance value
    for a given feature, revealing how consistent the explanations are.

    Args:
        result_path: Path to MDTerp_results_all.pkl.
        feature_names_path: Path to MDTerp_feature_names.npy.
        transition: Transition key (e.g., "0_1").
        top_n: Number of top features to show (default: 5).
        save_path: If provided, save the figure to this path.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    feature_names = np.load(feature_names_path, allow_pickle=True)

    with open(result_path, 'rb') as f:
        all_results = pickle.load(f)

    importances = []
    for sample_idx, (trans, imp) in all_results.items():
        if trans == transition:
            importances.append(np.array(imp))

    if not importances:
        raise ValueError(
            f"No results found for transition '{transition}'. "
            f"Available: {list(set(v[0] for v in all_results.values()))}"
        )

    imp_array = np.array(importances)
    mean_imp = np.mean(imp_array, axis=0)
    top_indices = np.argsort(mean_imp)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(top_n)

    for i, feat_idx in enumerate(top_indices):
        values = imp_array[:, feat_idx]
        jitter = np.random.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(
            positions[i] + jitter, values,
            alpha=0.6, s=40, color='steelblue', edgecolors='black',
            linewidth=0.5,
        )
        ax.scatter(positions[i], mean_imp[feat_idx],
                   marker='D', s=80, color='red', zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(feature_names[top_indices], fontsize=12, rotation=30, ha='right')
    ax.set_ylabel('Feature Importance', fontsize=14)
    ax.set_title(
        f'Per-Point Importance Variability\n'
        f'Transition {transition} (red = mean)',
        fontsize=16,
    )
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
