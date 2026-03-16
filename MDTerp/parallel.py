"""
MDTerp.parallel -- Multi-CPU parallel analysis of transition-state points.

Separates CPU-bound work (neighborhood generation, Ridge regression) from
GPU-bound work (black-box model inference) so that:
- The model is loaded ONCE in the main process
- GPU inference runs serially in the main process
- CPU work is parallelized across worker processes

This avoids CUDA context issues with multiprocessing and prevents redundant
model loading across workers.
"""
import numpy as np
import os
import shutil
import multiprocessing as mp
from typing import List, Callable, Any

from MDTerp.neighborhood import generate_neighborhood
from MDTerp.init_analysis import init_model
from MDTerp.final_analysis import final_model
from MDTerp.utils import make_result
from MDTerp.checkpoint import save_point_result


# ---------------------------------------------------------------------------
# Phase 1: Generate round-1 neighborhood (CPU only, parallelizable)
# ---------------------------------------------------------------------------

def _phase1_generate_neighborhood(args: dict) -> dict:
    """
    Generate the round-1 perturbed neighborhood for a single point.
    CPU-only: no model inference.

    Returns dict with feature_type_indices, feature_names, prefix, and
    paths to the generated .npy files.
    """
    transition = args['transition']
    point_index = args['point_index']
    sample_index = args['sample_index']
    save_dir = args['save_dir']
    prefix = os.path.join(save_dir, f"{transition}_{point_index}_")

    result = {
        'transition': transition,
        'point_index': point_index,
        'sample_index': sample_index,
        'prefix': prefix,
        'status': 'failed',
        'error': None,
    }

    try:
        feature_type_indices, feature_names = generate_neighborhood(
            prefix,
            args['numeric_dict'], args['angle_dict'], args['sin_cos_dict'],
            args['training_data'], sample_index, args['seed'],
            args['num_samples'], np.array([]),
            args['periodicity_upper'], args['periodicity_lower'],
        )
        result['feature_type_indices'] = feature_type_indices
        result['feature_names'] = feature_names
        result['prediction_input_path'] = prefix + 'DATA/make_prediction.npy'
        result['perturbation_data_path'] = prefix + 'DATA/TERP_dat.npy'
        result['status'] = 'completed'
    except Exception as e:
        result['error'] = str(e)

    return result


# ---------------------------------------------------------------------------
# Phase 2: Initial feature selection + round-2 neighborhood (CPU only)
# ---------------------------------------------------------------------------

def _phase2_init_and_neighborhood(args: dict) -> dict:
    """
    Run initial feature selection on round-1 results, then generate
    round-2 neighborhood with selected features only. CPU-only.

    Expects args to include round-1 state_probs from GPU inference.
    """
    result = {
        'transition': args['transition'],
        'point_index': args['point_index'],
        'sample_index': args['sample_index'],
        'prefix': args['prefix'],
        'feature_type_indices': args['feature_type_indices'],
        'feature_names': args['feature_names'],
        'status': 'failed',
        'error': None,
    }

    try:
        perturbation_data = np.load(args['perturbation_data_path'])
        state_probs = args['state_probs_round1']

        selected_features = init_model(
            perturbation_data, state_probs,
            args['cutoff'], args['feature_type_indices'],
            args['seed'], args['alpha'],
        )

        prefix = args['prefix']
        generate_neighborhood(
            prefix,
            args['numeric_dict'], args['angle_dict'], args['sin_cos_dict'],
            args['training_data'], args['sample_index'], args['seed'],
            args['num_samples'], selected_features,
            args['periodicity_upper'], args['periodicity_lower'],
        )

        result['selected_features'] = selected_features
        result['prediction_input_path_r2'] = prefix + 'DATA_2/make_prediction.npy'
        result['perturbation_data_path_r2'] = prefix + 'DATA_2/TERP_dat.npy'
        result['status'] = 'completed'
    except Exception as e:
        result['error'] = str(e)

    return result


# ---------------------------------------------------------------------------
# Phase 3: Forward feature selection + save (CPU only)
# ---------------------------------------------------------------------------

def _phase3_final_model_and_save(args: dict) -> dict:
    """
    Run forward feature selection on round-2 results and save the
    per-point checkpoint. CPU-only.

    Expects args to include round-2 state_probs from GPU inference.
    """
    result = {
        'transition': args['transition'],
        'point_index': args['point_index'],
        'sample_index': args['sample_index'],
        'status': 'failed',
        'error': None,
        'feature_names': args['feature_names'],
    }

    try:
        perturbation_data = np.load(args['perturbation_data_path_r2'])
        state_probs = args['state_probs_round2']

        importance_0, importance_all, unfaithfulness_all = final_model(
            perturbation_data, state_probs,
            args['unfaithfulness_threshold'], args['feature_type_indices'],
            args['selected_features'], args['seed'],
        )

        importance = make_result(
            args['feature_type_indices'], args['feature_names'], importance_0,
        )

        save_point_result(
            args['save_dir'], args['transition'], args['point_index'],
            args['sample_index'], np.array(importance), importance_all,
            unfaithfulness_all, args['selected_features'],
        )

        # Clean intermediate directories
        if not args.get('save_all', False):
            prefix = args['prefix']
            for suffix in ['DATA', 'DATA_2']:
                dirpath = prefix + suffix
                if os.path.isdir(dirpath):
                    shutil.rmtree(dirpath)

        result['status'] = 'completed'
        result['n_round1_features'] = len(args['selected_features'])
        result['n_final_features'] = int(np.count_nonzero(importance))
    except Exception as e:
        result['error'] = str(e)

    return result


# ---------------------------------------------------------------------------
# Orchestrator: runs the 3-phase pipeline
# ---------------------------------------------------------------------------

def _map_parallel_or_serial(func, items: list, n_workers: int) -> list:
    """Map a function over items, using multiprocessing if n_workers > 1."""
    if n_workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]

    with mp.Pool(processes=n_workers) as pool:
        return pool.map(func, items)


def run_parallel(
    work_items: List[dict],
    run_model_fn: Callable,
    model: Any,
    n_workers: int,
) -> List[dict]:
    """
    Process multiple points using a phased CPU/GPU pipeline.

    The model stays in the main process for GPU inference. CPU-heavy work
    (neighborhood generation, Ridge regression) is parallelized across
    worker processes.

    Pipeline per batch of points:
        Phase 1 (CPU, parallel): Generate round-1 neighborhoods
        GPU (serial):            Run model on all round-1 neighborhoods
        Phase 2 (CPU, parallel): Initial feature selection + round-2 neighborhoods
        GPU (serial):            Run model on all round-2 neighborhoods
        Phase 3 (CPU, parallel): Forward feature selection + save results

    Args:
        work_items: List of argument dicts for each point.
        run_model_fn: The run_model() function from the user's script.
        model: The loaded black-box model object.
        n_workers: Number of CPU worker processes.

    Returns:
        List of result dicts with status info per point.
    """
    # === Phase 1: Generate round-1 neighborhoods (CPU, parallel) ===
    phase1_results = _map_parallel_or_serial(
        _phase1_generate_neighborhood, work_items, n_workers,
    )

    # === GPU: Run model on all round-1 neighborhoods (serial, main process) ===
    for p1 in phase1_results:
        if p1['status'] != 'completed':
            continue
        prediction_input = np.load(p1['prediction_input_path'])
        p1['state_probs_round1'] = run_model_fn(model, prediction_input)

    # === Phase 2: Init model + round-2 neighborhoods (CPU, parallel) ===
    phase2_inputs = []
    for p1, wi in zip(phase1_results, work_items):
        if p1['status'] != 'completed':
            continue
        phase2_inputs.append({
            **wi,
            'prefix': p1['prefix'],
            'feature_type_indices': p1['feature_type_indices'],
            'feature_names': p1['feature_names'],
            'perturbation_data_path': p1['perturbation_data_path'],
            'state_probs_round1': p1['state_probs_round1'],
        })

    phase2_results = _map_parallel_or_serial(
        _phase2_init_and_neighborhood, phase2_inputs, n_workers,
    )

    # === GPU: Run model on all round-2 neighborhoods (serial, main process) ===
    for p2 in phase2_results:
        if p2['status'] != 'completed':
            continue
        prediction_input = np.load(p2['prediction_input_path_r2'])
        p2['state_probs_round2'] = run_model_fn(model, prediction_input)

    # === Phase 3: Final model + save (CPU, parallel) ===
    phase3_inputs = []
    for p2, wi in zip(phase2_results, phase2_inputs):
        if p2['status'] != 'completed':
            continue
        phase3_inputs.append({
            **wi,
            'prefix': p2['prefix'],
            'selected_features': p2['selected_features'],
            'perturbation_data_path_r2': p2['perturbation_data_path_r2'],
            'state_probs_round2': p2['state_probs_round2'],
        })

    phase3_results = _map_parallel_or_serial(
        _phase3_final_model_and_save, phase3_inputs, n_workers,
    )

    # === Build final results, including failed points from any phase ===
    final_results = []

    # Track which points made it through each phase
    phase2_idx = 0
    phase3_idx = 0

    for i, p1 in enumerate(phase1_results):
        if p1['status'] != 'completed':
            final_results.append({
                'transition': p1['transition'],
                'point_index': p1['point_index'],
                'sample_index': p1['sample_index'],
                'status': 'failed',
                'error': f"Phase 1 failed: {p1['error']}",
                'feature_names': None,
            })
            continue

        p2 = phase2_results[phase2_idx]
        phase2_idx += 1

        if p2['status'] != 'completed':
            final_results.append({
                'transition': p2['transition'],
                'point_index': p2['point_index'],
                'sample_index': p2['sample_index'],
                'status': 'failed',
                'error': f"Phase 2 failed: {p2['error']}",
                'feature_names': p2.get('feature_names'),
            })
            continue

        p3 = phase3_results[phase3_idx]
        phase3_idx += 1

        final_results.append(p3)

    return final_results
