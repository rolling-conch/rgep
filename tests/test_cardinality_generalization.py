import unittest
from itertools import combinations
import numpy as np
from cardinality_data import simulate_cardinality
from rssnet_data import simulate, normalize_input


class CardinalityTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.pure = [normalize_input(rng.uniform(.01, 1, (4, 32)))[0][..., 0] for _ in range(7)]

    def test_all_combinations_and_weight_law(self):
        for k in (1, 3):
            data = simulate_cardinality(self.pure, 8, 42, k)
            self.assertEqual(set(map(tuple, data['classes'])), set(combinations(range(7), k)))
            self.assertTrue(np.all(data['truth'].sum(1) == k))
            self.assertTrue(np.all(data['weights'] >= .1))
            np.testing.assert_allclose(data['weights'].sum(1), 1, atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(data['x'][..., 0], axis=1), 1, atol=1e-6)
            self.assertTrue(np.all(data['x'] >= 0))
            self.assertTrue(np.all(data['source_rows'] < 4))
            self.assertTrue(np.all((data['snr_db'] >= 10) & (data['snr_db'] <= 30)))

    def test_determinism_and_binary_untouched(self):
        binary = simulate(self.pure, 3, 55)
        a = simulate_cardinality(self.pure, 3, 55, 3)
        b = simulate_cardinality(self.pure, 3, 55, 3)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        for key, value in binary.items():
            np.testing.assert_array_equal(value, simulate(self.pure, 3, 55)[key])

    def test_invalid_cardinality(self):
        for k in (0, 2, 4):
            with self.assertRaises(ValueError):
                simulate_cardinality(self.pure, 3, 42, k)

    def test_count_correct_is_not_exact_set_correct(self):
        from evaluate_cardinality_generalization import statistics
        truth = np.array([[1, 0, 0], [0, 1, 1]], bool)
        predicted = np.array([[0, 1, 0], [0, 1, 1]], bool)
        result = statistics(truth, predicted)
        self.assertEqual(result['count_accuracy'], 1.)
        self.assertEqual(result['exact_set_accuracy'], .5)

    def test_frozen_predictions_match_report(self):
        import json
        from pathlib import Path
        from evaluate_cardinality_generalization import statistics
        out = Path(__file__).resolve().parents[1] / 'run/Cardinality_Frozen_CPU_20261006'
        if not (out / 'results.json').exists():
            self.skipTest('Formal frozen evaluation has not completed')
        result = json.loads((out / 'results.json').read_text(encoding='utf-8'))
        self.assertTrue(result['frozen_files_unchanged'])
        self.assertEqual(result['protocol']['bic_selection']['calibrated_BIC']['penalty'], 32.)
        for kind, data in result['datasets'].items():
            with np.load(out / (kind + '_labels.npz')) as z:
                self.assertTrue(np.all(z['truth'].sum(1) == data['true_count']))
                for method, metrics in data['methods'].items():
                    self.assertEqual(statistics(z['truth'], z[method]), metrics)


if __name__ == '__main__':
    unittest.main()
