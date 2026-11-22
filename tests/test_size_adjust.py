"""Calibration rule of step 32 (size-adjusted resolution); no data needed."""
import importlib.util
import os
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("s32", os.path.join(ROOT, "scripts", "32_size_adjusted_resolution.py"))
s32 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s32)


class TestCalibration(unittest.TestCase):
    def test_critical_value_rejects_at_most_alpha(self):
        rng = np.random.default_rng(0)
        for n in (20, 199, 1800):
            p = rng.random(n) ** 2  # an over-rejecting test: p-values pile up near 0
            c = s32.critical(p, 0.05)
            self.assertLessEqual((p < c).mean(), 0.05)
            self.assertGreater((p <= c).mean(), 0.05 - 1.0 / n - 1e-12)  # and not needlessly strict

    def test_uniform_null_keeps_critical_near_alpha(self):
        p = (np.arange(1000) + 0.5) / 1000
        self.assertAlmostEqual(s32.critical(p, 0.05), 0.0505, places=6)

    def test_resolution_rule(self):
        gaps = np.array([0.02, 0.05, 0.10, 0.12, 0.13])
        power = np.array([0.1, 0.5, 0.9, 0.85, 0.95])
        self.assertEqual(s32.resolution(gaps, power), (0.10, 3))
        self.assertEqual(s32.resolution(gaps, np.zeros(5)), (None, 0))


if __name__ == "__main__":
    unittest.main()
