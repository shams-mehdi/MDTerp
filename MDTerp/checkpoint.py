"""
MDTerp.checkpoint -- Per-point result persistence and crash recovery.

Saves each analyzed point as an individual .npz file so that interrupted runs
can resume from where they left off.
"""
import numpy as np
import json
import os
import glob
from typing import Dict, Set, Tuple
import pickle


def save_point_result(
    save_dir: str,
    transition: str,
    point_index: int,
    sample_index: int,
    importance: np.ndarray,
    importance_all: np.ndarray,
    unfaithfulness_all: np.ndarray,
    selected_features: np.ndarray,
) -> str:
    """
    Save a single point's analysis result to disk.

    Args:
        save_dir: Directory to save results.
        transition: Transition key string (e.g., "0_1").
        point_index: Index of the point within this transition.
        sample_index: Original row index in the training data.
        importance: Final importance vector for all features.
        importance_all: Importance vectors for all k values.
        unfaithfulness_all: Unfaithfulness values for all k values.
        selected_features: Feature indices kept after round 1.

    Returns:
        Path to the saved result file.
    """
    filename = f"{transition}_point{point_index}_result.npz"
    filepath = os.path.join(save_dir, filename)
    np.savez(
        filepath,
        sample_index=sample_index,
        transition=transition,
        importance=np.array(importance),
        importance_all=importance_all,
        unfaithfulness_all=unfaithfulness_all,
        selected_features=selected_features,
    )
    return filepath


def scan_completed_points(save_dir: str) -> Set[Tuple[str, int]]:
    """
    Scan the result directory for previously completed point results.

    Args:
        save_dir: Directory containing result files.

    Returns:
        Set of (transition, point_index) tuples already completed.
    """
    completed = set()
    pattern = os.path.join(save_dir, "*_point*_result.npz")
    for filepath in glob.glob(pattern):
        basename = os.path.basename(filepath)
        name = basename.replace("_result.npz", "")
        point_pos = name.rfind("_point")
        if point_pos == -1:
            continue
        transition = name[:point_pos]
        try:
            point_index = int(name[point_pos + 6:])
            # Validate file is readable
            with np.load(filepath, allow_pickle=False) as _:
                pass
            completed.add((transition, point_index))
        except (ValueError, Exception):
            continue
    return completed


def save_run_config(save_dir: str, config: dict) -> str:
    """
    Save run configuration for resume validation.

    Args:
        save_dir: Directory to save config.
        config: Dictionary of all run parameters.

    Returns:
        Path to the saved config file.
    """
    filepath = os.path.join(save_dir, "run_config.json")
    serializable = {}
    for k, v in config.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            serializable[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serializable[k] = float(v)
        else:
            serializable[k] = v
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    return filepath


def load_run_config(save_dir: str) -> dict:
    """
    Load a previously saved run configuration.

    Args:
        save_dir: Directory containing the config file.

    Returns:
        Configuration dictionary, or empty dict if not found.
    """
    filepath = os.path.join(save_dir, "run_config.json")
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)


def aggregate_results(
    save_dir: str,
    feature_names: np.ndarray,
    keep_checkpoints: bool = True,
) -> dict:
    """
    Aggregate all per-point result files into the final results dictionary.

    Args:
        save_dir: Directory containing per-point result files.
        feature_names: Array of feature names.
        keep_checkpoints: Whether to keep individual result files (default: True).

    Returns:
        Dictionary mapping sample_index -> [transition, importance_vector].
    """
    importance_results = {}
    pattern = os.path.join(save_dir, "*_point*_result.npz")
    for filepath in glob.glob(pattern):
        try:
            with np.load(filepath, allow_pickle=True) as data:
                sample_index = int(data['sample_index'])
                transition = str(data['transition'])
                importance = np.array(data['importance'])
                importance_results[sample_index] = [transition, importance]
        except Exception:
            continue

    result_path = os.path.join(save_dir, 'MDTerp_results_all.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(importance_results, f)

    names_path = os.path.join(save_dir, 'MDTerp_feature_names.npy')
    np.save(names_path, feature_names)

    if not keep_checkpoints:
        for filepath in glob.glob(pattern):
            os.remove(filepath)

    return importance_results
