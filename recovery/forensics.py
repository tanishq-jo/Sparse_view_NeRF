"""Probe the archived Lego checkpoints using local, source-matched code on CPU."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from sparse_nerf import model as legacy
from sparse_nerf.data import SceneDataset
from sparse_nerf.evaluate import image_scores, lpips_model


ROOT = Path(__file__).resolve().parents[1]
RUNS = "cvision/runs/benchmark/lego/seed-42"
ARMS = ("dense-vanilla", "sparse-vanilla", "sparse-regularized")


def sample_rays(manifest, downscale):
    """Use identical, stratified pixels across fields, independent of predictions."""
    generator = torch.Generator().manual_seed(42)
    origins, directions, targets, foreground = [], [], [], []
    cameras = []
    for split in ("sparse", "test"):
        scene = SceneDataset(manifest, split, downscale)
        indices = range(len(scene)) if split == "sparse" else (0, len(scene) - 1)
        for index in indices:
            frame = scene.frame_id(index)
            o, d, target = scene.frame(index)
            center = d[scene.height // 2, scene.width // 2]
            position = scene.poses[index, :3, 3]
            cameras.append({
                "frame": frame,
                "center_cosine_to_origin": torch.nn.functional.cosine_similarity(
                    center, -position, dim=0).item(),
                "z_closest_to_origin": float(-torch.dot(position, center) / torch.dot(center, center)),
            })
            with Image.open(scene.source / frame) as image:
                alpha = image.getchannel("A").resize((scene.width, scene.height), Image.Resampling.LANCZOS)
                alpha = torch.from_numpy(np.array(alpha))
            masks = [(True, (alpha >= 250) & (target.mean(-1) < 0.8)), (False, alpha == 0)]
            for is_foreground, mask in masks:
                pixels = torch.nonzero(mask)
                if len(pixels) < 4:
                    raise ValueError(f"{frame} has too few eligible probe pixels")
                row, col = pixels[torch.randperm(len(pixels), generator=generator)[:4]].T
                origins.append(o[row, col])
                directions.append(d[row, col])
                targets.append(target[row, col])
                foreground.extend([is_foreground] * len(row))
    rays = (torch.cat(origins), torch.cat(directions), torch.cat(targets), torch.tensor(foreground))
    return rays, cameras


def probe_field(archive, arm, rays, manifest_digest, downscale):
    prefix = f"{RUNS}/{arm}"
    metadata = json.loads(archive.read(f"{prefix}/run.json"))
    state = torch.load(io.BytesIO(archive.read(f"{prefix}/checkpoint.pt")),
                       map_location="cpu", weights_only=True)
    if (metadata["protocol"] != arm or metadata["manifest_hash"] != manifest_digest
            or state["run_id"] != metadata["run_id"] or metadata["downscale"] != downscale
            or state["step"] != metadata["training_config"]["steps"]):
        raise ValueError(f"Inconsistent archived configuration or checkpoint for {arm}")

    renderer = legacy.NeRFRenderer(legacy.ModelConfig(**metadata["training_config"]["model"]))
    renderer.load_state_dict(state["model"])
    renderer.eval()
    outputs, logits = [], []
    composite = legacy._composite

    def record_outputs(rgb, density, depths, ray_lengths, white_background):
        color, weights, background = composite(rgb, density, depths, ray_lengths, white_background)
        outputs.append((density.detach(), weights.detach().sum(-1), color.detach()))
        return color, weights, background

    def record_logits(module, inputs, output):
        logits.append(output.detach())

    for field in (renderer.coarse, renderer.fine):
        field.density.register_forward_hook(record_logits)
    origins, directions, targets, fg = rays
    with patch.object(legacy, "_composite", record_outputs):
        rendered = renderer(origins, directions, metadata["near"], metadata["far"], progress=1.0,
                            randomized=False, regularized=(arm == "sparse-regularized"),
                            white_background=metadata["white_background"])
    loss = sum((rendered[key][fg] - targets[fg]).square().mean() for key in ("rgb", "rgb_coarse"))
    loss.backward()
    result = {"step": state["step"], "hardware_at_training": metadata["hardware"]}
    for name, field, raw, (sigma, opacity, rgb) in zip(
        ("coarse", "fine"), (renderer.coarse, renderer.fine), logits, outputs, strict=True
    ):
        result[name] = {
            "logit_min": raw.min().item(), "logit_max": raw.max().item(),
            "positive_sigma_fraction": (sigma > 0).float().mean().item(),
            "sigma_max": sigma.max().item(), "fg_opacity_mean": opacity[fg].mean().item(),
            "bg_opacity_mean": opacity[~fg].mean().item(),
            "fg_rgb_mean": rgb[fg].mean(0).tolist(),
            "fg_rgb_mse": (rgb[fg] - targets[fg]).square().mean().item(),
            "fg_density_weight_grad_norm": field.density.weight.grad.norm().item(),
            "fg_color_weight_grad_norm": field.color.weight.grad.norm().item(),
        }
    return result


def white_scores(manifest, downscale, archived_frames, use_lpips):
    """Compare all 200 archived fine-render scores with an actual white tensor."""
    scene = SceneDataset(manifest, "test", downscale)
    if len(scene) != len(archived_frames):
        raise ValueError("Held-out frame counts differ")
    perceptual = lpips_model() if use_lpips else None
    scores = {"psnr": []}
    if use_lpips:
        scores["lpips"] = []
    with torch.inference_mode():
        for index, archived in enumerate(archived_frames):
            if scene.frame_id(index) != archived["frame"]:
                raise ValueError("Held-out frame identities or order differ")
            target = scene.image(index)
            if use_lpips:
                psnr, distance = image_scores(torch.ones_like(target), target, perceptual, "cpu")
                scores["lpips"].append(distance)
            else:
                mse = (1.0 - target.double()).square().mean().item()
                psnr = -10 * math.log10(max(1e-12, mse))
            scores["psnr"].append(psnr)
    return {key: {
        "mean": mean(values), "max_abs_delta_vs_archived_sparse": max(
            abs(value - frame[key]) for value, frame in zip(values, archived_frames, strict=True))
    } for key, values in scores.items()}


def inspect(archive, manifest, use_lpips):
    # The ZIP also contains a Git checkout. Always import installed local code.
    for name in ("model", "data", "evaluate"):
        relative = f"sparse_nerf/{name}.py"
        archived = archive.read(f"cvision/{relative}").decode().splitlines()
        if archived != (ROOT / relative).read_text(encoding="utf-8").splitlines():
            raise ValueError(f"Local {relative} differs from the archived implementation")
    archived_manifest = archive.read("cvision/manifests/lego.json")
    if json.loads(archived_manifest) != json.loads(manifest.read_bytes()):
        raise ValueError("Local and archived Lego manifests differ")
    manifest_digest = hashlib.sha256(archived_manifest).hexdigest()
    downscale = json.loads(archive.read(f"{RUNS}/dense-vanilla/run.json"))["downscale"]
    rays, cameras = sample_rays(manifest, downscale)
    metrics = {arm: json.loads(archive.read(f"{RUNS}/{arm}/metrics.json")) for arm in ARMS}
    return {
        "archived_commit": archive.read("cvision/.git/refs/heads/main").decode().strip(),
        "archived_manifest_sha256": manifest_digest, "rays": len(rays[0]),
        "foreground_rays": rays[3].sum().item(), "fg_target_rgb_mean": rays[2][rays[3]].mean(0).tolist(),
        "cameras": cameras,
        "archived_mean": {arm: metric["mean"] for arm, metric in metrics.items()},
        "retrospective_testskip8": {
            arm: {key: mean(frame[key] for frame in metric["frames"][::8]) for key in ("psnr", "lpips")}
            for arm, metric in metrics.items()
        },
        "sparse_arms_all_frames_identical": metrics[ARMS[1]]["frames"] == metrics[ARMS[2]]["frames"],
        "checkpoints": {arm: probe_field(archive, arm, rays, manifest_digest, downscale) for arm in ARMS},
        "white_baseline": white_scores(manifest, downscale, metrics[ARMS[2]]["frames"], use_lpips),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/lego.json")
    parser.add_argument("--lpips", action="store_true", help="Also score all white views with pretrained AlexNet")
    parser.add_argument("--output", type=Path, help="Optional new JSON file for the evidence")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with ZipFile(args.archive) as archive:
        report = inspect(archive, args.manifest, args.lpips)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
