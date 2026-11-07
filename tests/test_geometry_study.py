"""Selection must prefer geometry and reject coverage loss or one-seed luck."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from run_geometry_study import confirmed, passes_gates, preserve_incomplete, quality, run_one


def measurement(hit=.7, coverage=.99, near=.5, far=.3, mae=.09, fscore=.51):
    return {"geometry": {"depth": {"within_5cm_fraction": hit, "coverage": coverage,
                                   "within_2cm_fraction": near, "mae_m": mae},
                         "depth_bands": {"2-4m": {"mae_m": far}},
                         "cloud": {"thresholds": {"0.02": {"fscore": fscore}}}},
            "appearance": {"aligned": {"psnr": 21.}}}


class SelectionTests(unittest.TestCase):
    def test_aligned_psnr_cannot_override_geometry(self):
        reference = measurement()
        sharp_but_wrong = measurement(hit=.65)
        sharp_but_wrong["appearance"]["aligned"]["psnr"] = 30.
        self.assertGreater(quality(reference), quality(sharp_but_wrong))

    def test_discarding_difficult_pixels_does_not_pass(self):
        self.assertFalse(passes_gates(measurement(hit=.8, coverage=.95), measurement()))
        self.assertFalse(passes_gates(measurement(hit=.8, near=.48), measurement()))
        self.assertFalse(passes_gates(measurement(hit=.8, far=.35), measurement()))

    def test_single_lucky_seed_is_not_confirmation(self):
        baseline = {s: measurement() for s in (42, 43, 44)}
        lucky = {42: measurement(hit=.9), 43: measurement(hit=.69), 44: measurement(hit=.69)}
        self.assertFalse(confirmed(lucky, baseline))
        repeatable = {42: measurement(hit=.72), 43: measurement(hit=.72), 44: measurement(hit=.699)}
        self.assertTrue(confirmed(repeatable, baseline))
        repeatable[43]["geometry"]["depth"]["coverage"] = .8
        self.assertFalse(confirmed(repeatable, baseline))

    def test_missing_surface_is_failed_not_an_exception(self):
        missing = measurement(coverage=0., hit=0., mae=None, far=None, fscore=0.)
        self.assertFalse(passes_gates(missing, measurement()))
        self.assertFalse(passes_gates(measurement(), missing))
        self.assertLess(quality(missing), quality(measurement()))
        self.assertFalse(confirmed({s: missing for s in (42, 43, 44)},
                                   {s: measurement() for s in (42, 43, 44)}))

    def test_interrupted_output_is_preserved_for_fresh_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"training"
            path.mkdir()
            (path/"trace.jsonl").write_text("unfinished measurement")
            saved = preserve_incomplete(path, "result.json")
            self.assertFalse(path.exists())
            self.assertEqual((saved/"trace.jsonl").read_text(), "unfinished measurement")

    def test_cached_measurement_cannot_be_relabelled_to_another_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"measurements").mkdir()
            candidate = {"name": "candidate", "initialization": "retained"}
            (root/"measurements/candidate-s42.json").write_text(json.dumps(
                {"seed": 43, "candidate": candidate, "config": {}}))
            with self.assertRaises(ValueError):
                run_one(root, "candidate", candidate, 42, {"runs": {}})


if __name__ == "__main__":
    unittest.main()
