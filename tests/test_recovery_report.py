import hashlib
import json
import unittest

from recovery.report import ARMS, compare


def triplet(seed=42, psnr_dense=(25., 25., 25., 25.), psnr_regular=(23., 23., 23., 23.)):
    comparison = {"data_hash": "test", "metric": "alex-v0.1", "test_ids": [str(i) for i in range(4)]}
    fingerprint = hashlib.sha256(json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    runs = []
    for k, arm in enumerate(ARMS):
        frames = [dict(frame=str(i), psnr=(psnr_dense[i], 20., psnr_regular[i])[k],
                       ssim=(.9, .7, .8)[k], lpips=(.1, .3, .25)[k]) for i in range(4)]
        means = {key: sum(f[key] for f in frames) / 4 for key in ("psnr", "ssim", "lpips")}
        runs.append(dict(scene="lego", protocol=arm, seed=seed, comparison_hash=fingerprint,
                         comparison=comparison, frames=frames, mean=means, train_views=100 if k == 0 else 8))
    return runs


class RecoveryReportTests(unittest.TestCase):
    def test_paired_identities_and_negative_and_single_seed_scope(self):
        for regular, pct in ((20., 0.), (25., 100.), (19., -20.), (27., 140.)):
            result = compare(triplet(psnr_regular=(regular,)*4), resamples=200)["scenes"][0]
            self.assertIn("one seed", result["uncertainty"])
            self.assertEqual(result["paired"]["psnr"]["recovery_pct_ci95"], [pct, pct])
            self.assertAlmostEqual(result["paired"]["lpips"]["recovery_pct"], 25.)

    def test_psnr_gap_gate_for_both_metrics(self):
        for dense in ((22.99,)*4, (2., 32., 32., 32.)):
            result = compare(triplet(psnr_dense=dense), resamples=3000)["scenes"][0]["paired"]
            self.assertIsNone(result["psnr"]["recovery_pct"])
            self.assertIsNone(result["lpips"]["recovery_pct"])
        result = compare(triplet(psnr_dense=(23.,)*4), resamples=200)["scenes"][0]["paired"]
        self.assertEqual(result["psnr"]["recovery_pct"], 100)

    def test_two_seeds_and_joint_resampling(self):
        runs = triplet() + triplet(seed=43)
        for run in runs:
            for j, frame in enumerate(run["frames"]):
                # View difficulty and seed shifts cancel in paired differences.
                frame["psnr"] += 10*j + run["seed"]
            run["mean"]["psnr"] += 15 + run["seed"]
        scene = compare(runs, resamples=500)["scenes"][0]
        self.assertEqual(scene["seeds"], [42, 43])
        self.assertEqual(scene["paired"]["psnr"]["gap_ci95"], [5., 5.])
        self.assertEqual(scene["paired"]["psnr"]["recovery_pct_ci95"], [60., 60.])
        runs[-1]["frames"].reverse()
        with self.assertRaisesRegex(ValueError, "unmatched"):
            compare(runs)

    def test_rejects_missing_and_corrupted(self):
        runs = triplet()
        with self.assertRaisesRegex(ValueError, "missing"):
            compare(runs[:2])
        runs[0]["mean"]["psnr"] += 1
        with self.assertRaisesRegex(ValueError, "mean differs"):
            compare(runs)
        runs[0]["comparison"]["data_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            compare(runs)

    def test_three_equally_truncated_runs_still_fail(self):
        runs = triplet()
        for run in runs:
            run["frames"].pop()
        with self.assertRaisesRegex(ValueError, "frozen evaluation order"):
            compare(runs)


if __name__ == "__main__":
    unittest.main()
