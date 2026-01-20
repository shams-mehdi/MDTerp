"""
MDTerp.checkpoint.py – Checkpoint and resume functionality.

This module provides checkpointing capabilities for MDTerp analyses,
allowing runs to be resumed after crashes or interruptions. Checkpoints
track completed transitions and individual sample analyses.
"""
import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime


class CheckpointManager:
    """
    Manages checkpointing and resumption of MDTerp analyses.

    Tracks completed analyses at both transition and sample levels,
    enabling efficient resumption of interrupted runs without data loss
    or redundant computation.
    """

    def __init__(self, save_dir: str):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory for saving results and checkpoints.
        """
        self.save_dir = save_dir
        self.checkpoint_file = os.path.join(save_dir, 'mdterp_checkpoint.json')
        self.results_file = os.path.join(save_dir, 'MDTerp_results_all.pkl')
        self.feature_names_file = os.path.join(save_dir, 'MDTerp_feature_names.npy')

    def load_checkpoint(self) -> Tuple[Dict, Optional[List[str]], Set[int]]:
        """
        Load existing checkpoint data if available.

        Returns:
            Tuple of (results_dict, feature_names, completed_samples):
            - results_dict: Previously computed results
            - feature_names: Feature names (None if not yet computed)
            - completed_samples: Set of sample indices already analyzed

        Raises:
            None. Returns empty state if no checkpoint exists.
        """
        results = {}
        feature_names = None
        completed_samples = set()

        # Load checkpoint metadata
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)

                completed_samples = set(checkpoint_data.get('completed_samples', []))
                print(f"Found checkpoint with {len(completed_samples)} completed samples")

            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load checkpoint file: {e}")
                print("Starting fresh analysis")

        # Load existing results
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'rb') as f:
                    results = pickle.load(f)
                print(f"Loaded {len(results)} existing results")
            except (pickle.PickleError, IOError) as e:
                print(f"Warning: Could not load results file: {e}")

        # Load feature names
        if os.path.exists(self.feature_names_file):
            try:
                feature_names = np.load(self.feature_names_file, allow_pickle=True)
                print(f"Loaded {len(feature_names)} feature names")
            except (IOError, ValueError) as e:
                print(f"Warning: Could not load feature names: {e}")

        return results, feature_names, completed_samples

    def save_checkpoint(
        self,
        results: Dict,
        feature_names: List[str],
        completed_samples: Set[int],
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save current analysis state to checkpoint.

        Args:
            results: Dictionary of computed results (sample_idx -> [transition, importance])
            feature_names: List of feature names
            completed_samples: Set of completed sample indices
            metadata: Optional additional metadata to save

        Saves:
            - Checkpoint JSON with completed sample tracking
            - Results pickle file
            - Feature names array
        """
        # Save checkpoint metadata
        checkpoint_data = {
            'completed_samples': list(completed_samples),
            'total_samples': len(results),
            'last_updated': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        try:
            # Use atomic write (write to temp, then rename)
            temp_checkpoint = self.checkpoint_file + '.tmp'
            with open(temp_checkpoint, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            os.replace(temp_checkpoint, self.checkpoint_file)

        except IOError as e:
            print(f"Warning: Could not save checkpoint: {e}")

        # Save results
        if results:
            try:
                temp_results = self.results_file + '.tmp'
                with open(temp_results, 'wb') as f:
                    pickle.dump(results, f)
                os.replace(temp_results, self.results_file)
            except IOError as e:
                print(f"Warning: Could not save results: {e}")

        # Save feature names
        if feature_names is not None:
            try:
                temp_features = self.feature_names_file + '.tmp'
                np.save(temp_features, feature_names)
                os.replace(temp_features, self.feature_names_file)
            except IOError as e:
                print(f"Warning: Could not save feature names: {e}")

    def save_incremental_result(
        self,
        sample_index: int,
        transition_name: str,
        importance: List[float],
        feature_names: Optional[List[str]] = None
    ) -> None:
        """
        Save a single analysis result incrementally.

        This allows saving results as they're computed, reducing risk of
        data loss from crashes. More efficient than saving all results
        every time.

        Args:
            sample_index: Index of analyzed sample
            transition_name: Name of transition (e.g., "0_1")
            importance: Feature importance values
            feature_names: Feature names (only needed on first call)
        """
        # Load current state
        results, stored_feature_names, completed_samples = self.load_checkpoint()

        # Update with new result
        results[sample_index] = [transition_name, importance]
        completed_samples.add(sample_index)

        # Use provided feature names if this is the first result
        if feature_names is not None and stored_feature_names is None:
            stored_feature_names = feature_names

        # Save updated state
        self.save_checkpoint(results, stored_feature_names, completed_samples)

    def get_remaining_work(
        self,
        all_transitions: Dict[str, np.ndarray]
    ) -> Dict[str, List[int]]:
        """
        Determine which samples still need analysis.

        Args:
            all_transitions: Dictionary mapping transition names to sample arrays

        Returns:
            Dictionary mapping transition names to lists of unanalyzed sample indices
        """
        _, _, completed_samples = self.load_checkpoint()

        remaining_work = {}

        for transition_name, sample_indices in all_transitions.items():
            # Filter out completed samples
            remaining_indices = [
                idx for idx in sample_indices
                if idx not in completed_samples
            ]

            if remaining_indices:
                remaining_work[transition_name] = remaining_indices

        return remaining_work

    def clear_checkpoint(self) -> None:
        """
        Remove checkpoint file to start fresh analysis.

        Does not delete results files, only the checkpoint metadata.
        """
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                print("Checkpoint cleared")
            except IOError as e:
                print(f"Warning: Could not remove checkpoint: {e}")
