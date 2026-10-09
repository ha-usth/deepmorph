"""Small paired-data and metric checks; no GPU/model checkpoint needed."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmark_degraded_test_set import condition_list, evaluate_condition, inventory, relative_scores


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.conditions = condition_list(['gaussian_noise'])
        for condition, _, _ in self.conditions:
            folder = self.root / 'droso_small' / condition
            folder.mkdir(parents=True)
            # Two byte-identical images MUST remain two paired observations.
            for name, count in [('a', 15), ('b', 15), ('incomplete', 14)]:
                Image.new('RGB', (30, 40), 'gray').save(folder / f'{name}.png')
                np.savetxt(folder / f'{name}.txt', np.zeros((count, 2)))

    def test_duplicates_retained_and_incomplete_excluded_consistently(self):
        names, gt, sizes, excluded, _, total = inventory(self.root, 'droso_small', self.conditions)
        self.assertEqual(names, ['a.png', 'b.png'])
        self.assertEqual(total, 3)
        self.assertEqual(gt.shape, (2, 15, 2))
        self.assertEqual(sizes.tolist(), [[30, 40], [30, 40]])
        self.assertEqual(excluded[0]['image'], 'incomplete.png')

    def test_changed_annotation_rejected(self):
        target = self.root / 'droso_small/gaussian_noise/level_2/a.txt'
        target.write_text('1 2\n' * 15)
        with self.assertRaisesRegex(ValueError, 'Ground truth changed'):
            inventory(self.root, 'droso_small', self.conditions)

    def test_missing_image_rejected(self):
        (self.root / 'droso_small/gaussian_noise/level_1/a.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Image list differs'):
            inventory(self.root, 'droso_small', self.conditions)

    def test_metrics_archive_and_relative_changes(self):
        fake = types.ModuleType('stage3_predict')
        # Known 3-4-5 triangle: 5 pixel error and 10% of a 30x40 diagonal.
        fake.predict_one_image = lambda *args: np.tile([3., 4.], (15, 1))
        names, gt, sizes, _, _, _ = inventory(self.root, 'droso_small', self.conditions)
        archive = self.root / 'scores.npz'
        with patch.dict(sys.modules, {'stage3_predict': fake}):
            row = evaluate_condition(None, {}, 'cpu', self.root, names, gt, sizes, archive)
        self.assertEqual(row['mre_px'], 5.)
        self.assertEqual(row['nme_percent'], 10.)
        with np.load(archive, allow_pickle=False) as data:
            self.assertEqual(data['image_names'].tolist(), names)
            np.testing.assert_allclose(data['errors_px'], 5)
            np.testing.assert_allclose(data['predictions'] - data['ground_truth'], [[[3., 4.]] * 15] * 2)
        rows = [dict(mre_px=5., nme_percent=10.), dict(mre_px=7., nme_percent=14.)]
        relative_scores(rows)
        self.assertEqual(rows[1]['delta_mre_px'], 2.)
        self.assertEqual(rows[1]['relative_mre_increase_percent'], 40.)
        self.assertEqual(rows[1]['delta_nme_percentage_points'], 4.)
        rows[0]['mre_px'] = 0.
        relative_scores(rows)
        self.assertIsNone(rows[1]['relative_mre_increase_percent'])
        json.dumps(rows, allow_nan=False)

    def test_prediction_failure_does_not_create_archive(self):
        fake = types.ModuleType('stage3_predict')
        fake.predict_one_image = lambda *args: np.full((15, 2), np.nan)
        archive = self.root / 'bad.npz'
        with patch.dict(sys.modules, {'stage3_predict': fake}):
            with self.assertRaisesRegex(ValueError, 'Invalid model prediction'):
                evaluate_condition(None, {}, 'cpu', self.root, ['a.png'],
                                   np.zeros((1, 15, 2)), np.array([[30, 40]]), archive)
        self.assertFalse(archive.exists())


if __name__ == '__main__':
    unittest.main()
