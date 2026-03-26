"""
Comprehensive tests for MDTerp v2.0 modules.

Tests cover:
- models.py: similarity_kernel, ridge_regression
- checkpoint.py: save/load/scan/aggregate results
- utils.py: select_transition_points, transition_summary, dominant_feature, make_result
- visualization.py: all 4 plot functions
- base.py: full pipeline integration test
"""
import unittest
import numpy as np
import os
import shutil
import tempfile
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for tests
import matplotlib.pyplot as plt


class TestModels(unittest.TestCase):
    """Tests for MDTerp.models."""

    def test_similarity_kernel_identity(self):
        """First sample should have similarity 1.0 (distance 0 from itself)."""
        from MDTerp.models import similarity_kernel
        data = np.array([[0.0], [1.0], [2.0], [5.0]])
        weights = similarity_kernel(data, kernel_width=1.0)
        self.assertAlmostEqual(weights[0], 1.0, places=10)

    def test_similarity_kernel_decreasing(self):
        """Similarity should decrease with distance."""
        from MDTerp.models import similarity_kernel
        data = np.array([[0.0], [0.5], [1.0], [3.0]])
        weights = similarity_kernel(data, kernel_width=1.0)
        for i in range(len(weights) - 1):
            self.assertGreaterEqual(weights[i], weights[i + 1])

    def test_similarity_kernel_range(self):
        """All weights should be in [0, 1]."""
        from MDTerp.models import similarity_kernel
        np.random.seed(42)
        data = np.random.randn(100, 1)
        weights = similarity_kernel(data, kernel_width=1.0)
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 1))

    def test_ridge_regression_basic(self):
        """Ridge regression should fit a simple linear relationship."""
        from MDTerp.models import ridge_regression
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_coef = np.array([2.0, -1.0, 0.5])
        y = X @ true_coef + 0.1 * np.random.randn(100)
        coef, intercept = ridge_regression(X, y, seed=42, alpha=0.01)
        np.testing.assert_array_almost_equal(coef, true_coef, decimal=1)

    def test_ridge_regression_returns_correct_shapes(self):
        """Coefficients should have shape (n_features,)."""
        from MDTerp.models import ridge_regression
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        coef, intercept = ridge_regression(X, y, seed=0)
        self.assertEqual(coef.shape, (5,))
        self.assertIsInstance(intercept, (float, np.floating))


class TestCheckpoint(unittest.TestCase):
    """Tests for MDTerp.checkpoint."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_and_load_point_result(self):
        """Saved point result should be loadable and contain correct data."""
        from MDTerp.checkpoint import save_point_result
        importance = np.array([0.1, 0.5, 0.3, 0.1])
        importance_all = np.random.rand(3, 4)
        unfaithfulness_all = np.array([0.5, 0.2, 0.05])
        selected_features = np.array([0, 2, 3])

        path = save_point_result(
            self.test_dir, "0_1", 3, 42,
            importance, importance_all, unfaithfulness_all, selected_features,
        )
        self.assertTrue(os.path.exists(path))

        data = np.load(path, allow_pickle=True)
        self.assertEqual(int(data['sample_index']), 42)
        self.assertEqual(str(data['transition']), "0_1")
        np.testing.assert_array_almost_equal(data['importance'], importance)

    def test_scan_completed_points(self):
        """Scan should find all valid result files."""
        from MDTerp.checkpoint import save_point_result, scan_completed_points
        importance = np.array([0.5, 0.5])

        save_point_result(self.test_dir, "0_1", 0, 10,
                          importance, np.array([[0.5, 0.5]]),
                          np.array([0.1]), np.array([0]))
        save_point_result(self.test_dir, "0_1", 1, 20,
                          importance, np.array([[0.5, 0.5]]),
                          np.array([0.1]), np.array([0]))
        save_point_result(self.test_dir, "2_3", 0, 30,
                          importance, np.array([[0.5, 0.5]]),
                          np.array([0.1]), np.array([0]))

        completed = scan_completed_points(self.test_dir)
        self.assertEqual(len(completed), 3)
        self.assertIn(("0_1", 0), completed)
        self.assertIn(("0_1", 1), completed)
        self.assertIn(("2_3", 0), completed)

    def test_scan_empty_dir(self):
        """Scan on empty directory should return empty set."""
        from MDTerp.checkpoint import scan_completed_points
        completed = scan_completed_points(self.test_dir)
        self.assertEqual(len(completed), 0)

    def test_save_and_load_run_config(self):
        """Config should roundtrip through save/load."""
        from MDTerp.checkpoint import save_run_config, load_run_config
        config = {
            'point_max': 50,
            'prob_threshold_min': 0.475,
            'seed': 42,
            'save_dir': self.test_dir,
        }
        save_run_config(self.test_dir, config)
        loaded = load_run_config(self.test_dir)
        self.assertEqual(loaded['point_max'], 50)
        self.assertAlmostEqual(loaded['prob_threshold_min'], 0.475)

    def test_load_run_config_missing(self):
        """Loading from nonexistent dir should return empty dict."""
        from MDTerp.checkpoint import load_run_config
        loaded = load_run_config(os.path.join(self.test_dir, "nonexistent"))
        self.assertEqual(loaded, {})

    def test_aggregate_results(self):
        """Aggregation should combine all point files into pickle and npy."""
        from MDTerp.checkpoint import save_point_result, aggregate_results
        importance1 = np.array([0.8, 0.2])
        importance2 = np.array([0.3, 0.7])

        save_point_result(self.test_dir, "0_1", 0, 10,
                          importance1, np.array([[0.8, 0.2]]),
                          np.array([0.1]), np.array([0]))
        save_point_result(self.test_dir, "0_1", 1, 20,
                          importance2, np.array([[0.3, 0.7]]),
                          np.array([0.2]), np.array([1]))

        feature_names = np.array(["feat_a", "feat_b"])
        results = aggregate_results(self.test_dir, feature_names, keep_checkpoints=True)

        self.assertIn(10, results)
        self.assertIn(20, results)
        self.assertEqual(results[10][0], "0_1")
        np.testing.assert_array_almost_equal(results[10][1], importance1)

        # Check files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'MDTerp_results_all.pkl')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'MDTerp_feature_names.npy')))

    def test_aggregate_cleanup(self):
        """With keep_checkpoints=False, individual files should be removed."""
        from MDTerp.checkpoint import save_point_result, aggregate_results
        save_point_result(self.test_dir, "0_1", 0, 10,
                          np.array([1.0]), np.array([[1.0]]),
                          np.array([0.1]), np.array([0]))

        aggregate_results(self.test_dir, np.array(["f"]), keep_checkpoints=False)
        # npz files should be gone
        import glob
        npz_files = glob.glob(os.path.join(self.test_dir, "*_result.npz"))
        self.assertEqual(len(npz_files), 0)


class TestAdaptiveThreshold(unittest.TestCase):
    """Tests for select_transition_points in utils.py."""

    def test_basic_selection(self):
        """Should select samples near transition boundary."""
        from MDTerp.utils import select_transition_points
        # 10 samples, 2 states
        prob = np.zeros((10, 2))
        # Make 5 samples near the boundary
        for i in range(5):
            prob[i, 0] = 0.50 - i * 0.005
            prob[i, 1] = 0.50 + i * 0.005
        # Make 5 samples clearly in state 1
        for i in range(5, 10):
            prob[i, 0] = 0.1
            prob[i, 1] = 0.9

        points, thresholds = select_transition_points(prob, point_max=3, prob_threshold_min=0.475)
        self.assertIn("0_1", points)
        self.assertEqual(len(points["0_1"]), 3)
        # Should select the 3 closest to boundary (indices 0, 1, 2)
        selected = set(points["0_1"].tolist())
        self.assertTrue(selected.issubset({0, 1, 2, 3, 4}))

    def test_adaptive_threshold_per_transition(self):
        """Different transitions should get different effective thresholds."""
        from MDTerp.utils import select_transition_points
        prob = np.zeros((20, 3))
        # Transition 0_1: many points near boundary
        for i in range(10):
            prob[i, 0] = 0.50 - i * 0.002
            prob[i, 1] = 0.50 + i * 0.002
        # Transition 1_2: fewer points near boundary
        for i in range(10, 15):
            prob[i, 1] = 0.50 - (i - 10) * 0.005
            prob[i, 2] = 0.50 + (i - 10) * 0.005
        # Rest clearly in state 2
        for i in range(15, 20):
            prob[i, 2] = 0.95
            prob[i, 0] = 0.05

        points, thresholds = select_transition_points(prob, point_max=5, prob_threshold_min=0.475)
        # Both transitions should be found
        self.assertIn("0_1", points)
        self.assertIn("1_2", points)

    def test_no_transitions_raises(self):
        """When no samples are near boundary, should return empty dict."""
        from MDTerp.utils import select_transition_points
        prob = np.array([[0.9, 0.1], [0.05, 0.95], [0.85, 0.15]])
        points, thresholds = select_transition_points(prob, point_max=5, prob_threshold_min=0.475)
        self.assertEqual(len(points), 0)

    def test_warning_when_sparse(self):
        """Should warn when fewer than point_max samples available."""
        from MDTerp.utils import select_transition_points
        prob = np.zeros((5, 2))
        prob[0] = [0.50, 0.50]
        prob[1] = [0.49, 0.51]
        for i in range(2, 5):
            prob[i] = [0.1, 0.9]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            points, thresholds = select_transition_points(prob, point_max=10, prob_threshold_min=0.475)
            # Should have issued a warning about too few points
            warning_msgs = [str(x.message) for x in w]
            self.assertTrue(any("only" in msg for msg in warning_msgs))

    def test_returns_two_dicts(self):
        """Should return (points_dict, thresholds_dict) tuple."""
        from MDTerp.utils import select_transition_points
        prob = np.array([[0.50, 0.50], [0.49, 0.51]])
        points, thresholds = select_transition_points(prob, point_max=5, prob_threshold_min=0.475)
        self.assertIsInstance(points, dict)
        self.assertIsInstance(thresholds, dict)


class TestMakeResult(unittest.TestCase):
    """Tests for make_result in utils.py."""

    def test_combines_sin_cos(self):
        """Sin/cos importance should be summed."""
        from MDTerp.utils import make_result
        feature_type_indices = [
            np.array([0]),      # numeric
            np.array([1]),      # angle
            np.array([2]),      # sin
            np.array([3]),      # cos
        ]
        feature_names = ["num", "ang", "sincos"]
        importance = np.array([0.1, 0.2, 0.3, 0.4])
        result = make_result(feature_type_indices, feature_names, importance)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.1)  # numeric
        self.assertAlmostEqual(result[1], 0.2)  # angle
        self.assertAlmostEqual(result[2], 0.7)  # sin + cos


class TestTransitionSummary(unittest.TestCase):
    """Tests for transition_summary and dominant_feature."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Create mock results
        results = {
            0: ["0_1", np.array([0.8, 0.1, 0.1])],
            1: ["0_1", np.array([0.6, 0.3, 0.1])],
            2: ["1_2", np.array([0.1, 0.5, 0.4])],
        }
        self.pkl_path = os.path.join(self.test_dir, 'MDTerp_results_all.pkl')
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(results, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_transition_summary_returns_all_transitions(self):
        from MDTerp.utils import transition_summary
        summary = transition_summary(self.pkl_path, importance_coverage=1.0)
        self.assertIn("0_1", summary)
        self.assertIn("1_2", summary)

    def test_transition_summary_mean_std(self):
        from MDTerp.utils import transition_summary
        summary = transition_summary(self.pkl_path, importance_coverage=1.0)
        # 0_1 has 2 samples
        mean, std = summary["0_1"]
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(std.shape, (3,))
        # Mean should be normalized (sum to ~1 for full coverage)
        self.assertAlmostEqual(np.sum(mean), 1.0, places=5)

    def test_dominant_feature(self):
        from MDTerp.utils import dominant_feature
        dom = dominant_feature(self.pkl_path, n=0)
        # Sample 0: [0.8, 0.1, 0.1] -> dominant feature is 0
        self.assertEqual(dom[0], 0)
        # Sample 2: [0.1, 0.5, 0.4] -> dominant feature is 1
        self.assertEqual(dom[2], 1)


class TestVisualization(unittest.TestCase):
    """Tests for MDTerp.visualization plot functions."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Create mock results
        results = {
            0: ["0_1", np.array([0.8, 0.1, 0.1])],
            1: ["0_1", np.array([0.6, 0.3, 0.1])],
            2: ["0_1", np.array([0.7, 0.2, 0.1])],
        }
        self.pkl_path = os.path.join(self.test_dir, 'MDTerp_results_all.pkl')
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(results, f)
        self.names_path = os.path.join(self.test_dir, 'MDTerp_feature_names.npy')
        np.save(self.names_path, np.array(["feat_a", "feat_b", "feat_c"]))

    def tearDown(self):
        plt.close('all')
        shutil.rmtree(self.test_dir)

    def test_plot_feature_importance_returns_figure(self):
        from MDTerp.visualization import plot_feature_importance
        fig = plot_feature_importance(self.pkl_path, self.names_path, "0_1")
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_feature_importance_with_std(self):
        from MDTerp.visualization import plot_feature_importance
        fig = plot_feature_importance(self.pkl_path, self.names_path, "0_1", show_std=True)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_feature_importance_save(self):
        from MDTerp.visualization import plot_feature_importance
        save_path = os.path.join(self.test_dir, "test_plot.png")
        fig = plot_feature_importance(self.pkl_path, self.names_path, "0_1", save_path=save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_plot_feature_importance_invalid_transition(self):
        from MDTerp.visualization import plot_feature_importance
        with self.assertRaises(ValueError):
            plot_feature_importance(self.pkl_path, self.names_path, "9_9")

    def test_plot_importance_heatmap(self):
        from MDTerp.visualization import plot_importance_heatmap
        fig = plot_importance_heatmap(self.pkl_path, self.names_path)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_unfaithfulness_curve(self):
        from MDTerp.visualization import plot_unfaithfulness_curve
        # Create a mock point result file
        np.savez(
            os.path.join(self.test_dir, "0_1_point0_result.npz"),
            sample_index=0,
            transition="0_1",
            importance=np.array([0.8, 0.2]),
            importance_all=np.random.rand(3, 2),
            unfaithfulness_all=np.array([0.5, 0.2, 0.05]),
            selected_features=np.array([0]),
        )
        fig = plot_unfaithfulness_curve(self.test_dir, "0_1", 0)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_unfaithfulness_curve_missing_file(self):
        from MDTerp.visualization import plot_unfaithfulness_curve
        with self.assertRaises(FileNotFoundError):
            plot_unfaithfulness_curve(self.test_dir, "9_9", 0)

    def test_plot_point_variability(self):
        from MDTerp.visualization import plot_point_variability
        fig = plot_point_variability(self.pkl_path, self.names_path, "0_1", top_n=2)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_point_variability_invalid_transition(self):
        from MDTerp.visualization import plot_point_variability
        with self.assertRaises(ValueError):
            plot_point_variability(self.pkl_path, self.names_path, "9_9")


class TestDisplayNames(unittest.TestCase):
    """Tests for display_names parameter in visualization functions."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        results = {
            0: ["0_1", np.array([0.8, 0.1, 0.1])],
            1: ["0_1", np.array([0.6, 0.3, 0.1])],
            2: ["0_1", np.array([0.7, 0.2, 0.1])],
        }
        self.pkl_path = os.path.join(self.test_dir, 'MDTerp_results_all.pkl')
        with open(self.pkl_path, 'wb') as f:
            pickle.dump(results, f)
        self.names_path = os.path.join(self.test_dir, 'MDTerp_feature_names.npy')
        np.save(self.names_path, np.array(["feat_a", "feat_b", "feat_c"]))
        self.display_names = {
            "feat_a": r"$\phi_1$",
            "feat_b": r"$\psi_2$",
            "feat_c": r"$\chi_3$",
        }

    def tearDown(self):
        plt.close('all')
        shutil.rmtree(self.test_dir)

    def test_feature_importance_display_names(self):
        from MDTerp.visualization import plot_feature_importance
        fig = plot_feature_importance(
            self.pkl_path, self.names_path, "0_1",
            display_names=self.display_names,
        )
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertTrue(any(r"$\phi_1$" in l for l in labels))

    def test_heatmap_display_names(self):
        from MDTerp.visualization import plot_importance_heatmap
        fig = plot_importance_heatmap(
            self.pkl_path, self.names_path,
            display_names=self.display_names,
        )
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertTrue(any(r"$\phi_1$" in l for l in labels))

    def test_point_variability_display_names(self):
        from MDTerp.visualization import plot_point_variability
        fig = plot_point_variability(
            self.pkl_path, self.names_path, "0_1", top_n=2,
            display_names=self.display_names,
        )
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertTrue(any("$" in l for l in labels))

    def test_display_names_none_uses_originals(self):
        from MDTerp.visualization import plot_feature_importance
        fig = plot_feature_importance(
            self.pkl_path, self.names_path, "0_1",
            display_names=None,
        )
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertTrue(any("feat_a" in l for l in labels))

    def test_partial_display_names(self):
        """Features not in dict should keep original names."""
        from MDTerp.visualization import plot_feature_importance
        partial = {"feat_a": r"$\phi_1$"}
        fig = plot_feature_importance(
            self.pkl_path, self.names_path, "0_1",
            display_names=partial,
        )
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertTrue(any(r"$\phi_1$" in l for l in labels))
        self.assertTrue(any("feat_b" in l for l in labels))


class TestUseAllCutoffFeatures(unittest.TestCase):
    """Tests for use_all_cutoff_features parameter in final_model."""

    def test_use_all_cutoff_returns_all_features(self):
        """With use_all_cutoff_features=True, importance should have nonzero
        values for all selected features."""
        from MDTerp.final_analysis import final_model
        np.random.seed(42)
        n_samples = 500
        n_features = 5

        # Create synthetic neighborhood data
        neighborhood_data = np.random.randn(n_samples, n_features)
        # Create synthetic probabilities with clear class separation
        pred_proba = np.zeros((n_samples, 2))
        pred_proba[:, 0] = 0.5 + 0.3 * np.tanh(neighborhood_data[:, 0] + neighborhood_data[:, 1])
        pred_proba[:, 1] = 1 - pred_proba[:, 0]

        feature_type_indices = [
            np.arange(n_features),  # numeric
            np.array([], dtype=int),  # angle
            np.array([], dtype=int),  # sin
            np.array([], dtype=int),  # cos
        ]
        selected_features = np.arange(n_features)

        imp_all, _, _ = final_model(
            neighborhood_data, pred_proba, 0.01,
            feature_type_indices, selected_features, seed=42,
            use_all_cutoff_features=True,
        )
        # All selected features should have importance assigned (last model uses all)
        self.assertEqual(imp_all.shape[0], n_features)
        # At least some features should be nonzero
        self.assertGreater(np.count_nonzero(imp_all), 0)

    def test_use_all_cutoff_vs_entropy(self):
        """use_all_cutoff_features=True should generally retain more features
        than entropy-based selection."""
        from MDTerp.final_analysis import final_model
        np.random.seed(42)
        n_samples = 500
        n_features = 5

        neighborhood_data = np.random.randn(n_samples, n_features)
        pred_proba = np.zeros((n_samples, 2))
        pred_proba[:, 0] = 0.5 + 0.3 * np.tanh(neighborhood_data[:, 0])
        pred_proba[:, 1] = 1 - pred_proba[:, 0]

        feature_type_indices = [
            np.arange(n_features),
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=int),
        ]
        selected_features = np.arange(n_features)

        imp_all, _, _ = final_model(
            neighborhood_data, pred_proba, 0.01,
            feature_type_indices, selected_features, seed=42,
            use_all_cutoff_features=True,
        )
        imp_entropy, _, _ = final_model(
            neighborhood_data, pred_proba, 0.01,
            feature_type_indices, selected_features, seed=42,
            use_all_cutoff_features=False,
        )
        # use_all should have >= nonzero features compared to entropy-based
        self.assertGreaterEqual(
            np.count_nonzero(imp_all),
            np.count_nonzero(imp_entropy),
        )


class TestInterpretationEntropy(unittest.TestCase):
    """Tests for interpretation_entropy in final_analysis.py."""

    def test_uniform_coefficients_max_entropy(self):
        """Uniform coefficients should give maximum entropy (1.0)."""
        from MDTerp.final_analysis import interpretation_entropy
        coef = np.array([1.0, 1.0, 1.0, 1.0])
        entropy = interpretation_entropy(coef)
        self.assertAlmostEqual(entropy, 1.0, places=5)

    def test_single_nonzero_min_entropy(self):
        """Single non-zero coefficient should give entropy close to 0."""
        from MDTerp.final_analysis import interpretation_entropy
        coef = np.array([1.0, 0.0, 0.0, 0.0])
        entropy = interpretation_entropy(coef)
        self.assertAlmostEqual(entropy, 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
