"""Compare complete matched recovery runs by scene, with joint paired bootstrap intervals."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("dense-vanilla", "sparse-vanilla", "sparse-regularized")
METRICS = ("psnr", "ssim", "lpips")


def _samples(run):
    config = json.dumps(run["comparison"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(config.encode()).hexdigest() != run["comparison_hash"]:
        raise ValueError("Comparison fingerprint differs from its recorded settings")
    ids = tuple(f["frame"] for f in run["frames"])
    if not ids or len(ids) != len(set(ids)) or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("Expected unique ordered test frame IDs")
    if list(ids) != run["comparison"].get("test_ids"):
        raise ValueError("Measured test frame IDs are unmatched with the frozen evaluation order")
    samples = np.array([[f[key] for key in METRICS] for f in run["frames"]], dtype=np.float64)
    means = np.array([run["mean"][key] for key in METRICS])
    if not np.isfinite(samples).all() or not np.allclose(means, samples.mean(axis=0), rtol=0, atol=1e-7):
        raise ValueError("Nonfinite metric or recorded mean differs from the per-frame measurements")
    return ids, samples


def _ci(values):
    return np.percentile(values, [2.5, 97.5]).tolist()


def _paired_draws(values, count, rng):
    # [seed, view, arm, metric]. Every sampled index applies to all three arms.
    n_seeds, n_views = values.shape[:2]
    samples = np.empty((count, len(ARMS), len(METRICS)))
    for first in range(0, count, 128):
        batch = min(128, count - first)
        seeds = rng.integers(n_seeds, size=(batch, n_seeds, 1))
        views = rng.integers(n_views, size=(batch, n_seeds, n_views))
        samples[first:first+batch] = values[seeds, views].mean(axis=(1, 2))
    return samples


def _scene_report(scene, group, count, seed):
    seeds = sorted({key[0] for key in group})
    missing = [(s, a) for s in seeds for a in ARMS if (s, a) not in group]
    if missing:
        raise ValueError(f"{scene} is missing matched runs: {missing}")
    first = group[seeds[0], ARMS[0]]
    ids, _ = _samples(first)
    values = np.empty((len(seeds), len(ids), len(ARMS), len(METRICS)))
    train_counts = {}
    seed_rows = []
    for j, s in enumerate(seeds):
        for k, arm in enumerate(ARMS):
            run = group[s, arm]
            frame_ids, scores = _samples(run)
            if first["comparison_hash"] != run["comparison_hash"] or ids != frame_ids:
                raise ValueError(f"{scene}: unmatched settings, targets or ordered test frame IDs")
            views = run["train_views"]
            if (train_counts.setdefault(arm, views) != views or
                    (arm == "dense-vanilla" and views <= 8) or (arm != "dense-vanilla" and views != 8)):
                raise ValueError(f"{scene}: invalid number of training views")
            values[j, :, k, :] = scores
            seed_rows.append(dict(seed=s, arm=arm, **dict(zip(METRICS, scores.mean(axis=0).tolist()))))
    center = values.mean(axis=(0, 1))
    boot = _paired_draws(values, count, np.random.default_rng(seed))
    arms = {arm: {metric: dict(mean=float(center[k, q]), ci95=_ci(boot[:, k, q]))
                  for q, metric in enumerate(METRICS)} for k, arm in enumerate(ARMS)}
    differences = {}
    for metric, sign in (("psnr", 1), ("lpips", -1)):
        idx = METRICS.index(metric)
        dense, sparse, reg = center[:, idx]
        gap, gain = sign*(dense-sparse), sign*(reg-sparse)
        bgap = sign*(boot[:, 0, idx] - boot[:, 1, idx])
        bgain = sign*(boot[:, 2, idx] - boot[:, 1, idx])
        differences[metric] = dict(gap=float(gap), gap_ci95=_ci(bgap), gain=float(gain), gain_ci95=_ci(bgain),
                                   recovery_pct=None, recovery_pct_ci95=None)
        psnr = differences["psnr"]
        if psnr["gap"] < 3.0 or psnr["gap_ci95"][0] <= 0:
            reason = "PSNR dense-to-sparse gap below 3 dB or its paired 95% interval includes zero."
        elif metric == "lpips" and (gap <= 0 or differences[metric]["gap_ci95"][0] <= 0):
            reason = "LPIPS denominator is reversed or its paired 95% interval includes zero."
        elif np.any(bgap == 0):
            reason = "Paired bootstrap contains a zero denominator."
        else:
            reason = None
            differences[metric]["recovery_pct"] = float(100 * gain / gap)
            differences[metric]["recovery_pct_ci95"] = _ci(100 * bgain / bgap)
        differences[metric]["withheld_reason"] = reason
    return dict(scene=scene, comparison_hash=first["comparison_hash"], comparison=first["comparison"],
                train_views=train_counts, test_frames=list(ids), seeds=seeds, arms=arms, per_seed=seed_rows,
                paired=differences, uncertainty=("paired test views conditional on this one seed"
                if len(seeds) == 1 else "paired bootstrap of seeds, then test views within each resampled seed"))


def compare(records, *, resamples=10_000, bootstrap_seed=42):
    if resamples < 1:
        raise ValueError("Need at least one bootstrap resample")
    groups = {}
    for run in records:
        if run["protocol"] not in ARMS:
            raise ValueError(f"Unknown arm {run['protocol']}")
        group = groups.setdefault(run["scene"], {})
        key = (run["seed"], run["protocol"])
        if key in group:
            raise ValueError(f"Duplicate run for {run['scene']}, {key}")
        group[key] = run
    if not groups:
        raise ValueError("No recovery evaluation records supplied")
    return dict(bootstrap_resamples=resamples, bootstrap_seed=bootstrap_seed, confidence=0.95,
                scenes=[_scene_report(name, runs, resamples, bootstrap_seed) for name, runs in sorted(groups.items())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="recovery metrics.json files, run folders, or recovery output roots")
    parser.add_argument("--output", type=Path, required=True, help="output report JSON path")
    args = parser.parse_args()
    try:
        files = set()
        for path in args.runs:
            candidates = [path] if path.is_file() else list(path.glob("**/eval/metrics.json"))
            if not candidates:
                raise ValueError(f"No eval/metrics.json under {path}")
            files.update(p.resolve() for p in candidates)
        result = compare([json.loads(p.read_text(encoding="utf-8")) for p in sorted(files)])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for scene in result["scenes"]:
            print(f"{scene['scene']} ({len(scene['test_frames'])} views), {scene['uncertainty']}")
            for arm, metrics in scene["arms"].items():
                print(arm, " ".join(f"{k}={v['mean']:.4f}" for k, v in metrics.items()))
            print(json.dumps(scene["paired"], indent=2))
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"Cannot compare runs: {exc}\n")


if __name__ == "__main__":
    main()
