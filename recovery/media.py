"""Score a completed recovery checkpoint or resume rendering its matched video."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from tempfile import NamedTemporaryFile
import time

import numpy as np
from PIL import Image
import torch
from torch.nn.functional import avg_pool2d

from sparse_nerf.data import SceneDataset
from sparse_nerf.evaluate import image_scores, lpips_model
from sparse_nerf.run import select_device
from sparse_nerf.video import encoder_executable


def digest(value):
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def write_json(path, value):
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        name = Path(tmp.name)
        try:
            json.dump(value, tmp, indent=2, allow_nan=False)
            tmp.write("\n")
        except BaseException:
            tmp.close()
            name.unlink(missing_ok=True)
            raise
    try:
        name.replace(path)
    finally:
        name.unlink(missing_ok=True)


def rgb_uint8(image):
    return (image.clamp(0, 1) * 255).round().byte().cpu().numpy()


def model_fingerprint(model):
    checksum = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        checksum.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
        checksum.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


@torch.inference_mode()
def render_maps(model, dataset, rays, regularized, device, chunk):
    """Return fine RGB, camera-axis depth and opacity, in row-major image order."""
    if chunk < 1:
        raise ValueError("chunk must be positive")
    origins, directions = (part.reshape(-1, 3) for part in rays)
    rgb = torch.empty_like(origins, device="cpu")
    depth = torch.empty(len(origins), dtype=torch.float32)
    opacity = torch.empty_like(depth)
    for first in range(0, len(origins), chunk):
        section = slice(first, first + chunk)
        result = model(origins[section].to(device), directions[section].to(device),
                       dataset.near, dataset.far, progress=1.0, regularized=regularized,
                       randomized=False, white_background=dataset.white_background)
        rgb[section] = result["rgb"].cpu()
        depth[section] = result["depth"].reshape(-1).cpu()
        opacity[section] = result["opacity"].reshape(-1).cpu()
    if not all(torch.isfinite(t).all() for t in (rgb, depth, opacity)):
        raise ValueError("Nonfinite prediction in RGB, depth or opacity")
    return (rgb.view(dataset.height, dataset.width, 3),
            depth.view(dataset.height, dataset.width), opacity.view(dataset.height, dataset.width))


def ssim(prediction, target):
    """Mean valid-window SSIM, matching skimage RGB defaults for data_range=1."""
    def window(t):
        return avg_pool2d(t, 7, stride=1)

    x, y = (t.permute(2, 0, 1).unsqueeze(0).double() for t in (prediction, target))
    mx, my = window(x), window(y)
    # skimage uses sample covariance by default and crops the filter border.
    vx = (window(x.square()) - mx.square()) * (49 / 48)
    vy = (window(y.square()) - my.square()) * (49 / 48)
    covariance = (window(x * y) - mx * my) * (49 / 48)
    luminance = (2 * mx * my + 0.01**2) / (mx.square() + my.square() + 0.01**2)
    structure = (2 * covariance + 0.03**2) / (vx + vy + 0.03**2)
    return (luminance * structure).mean().item()


def psnr(prediction, target):
    mse = (prediction.double() - target.double()).square().mean().item()
    return -10 * math.log10(max(mse, 1e-12))


def alpha_mask(dataset, i):
    if dataset.manifest["format"] != "blender":
        return None
    with Image.open(dataset.source / dataset.frame_id(i)) as source:
        alpha = source.getchannel("A").resize((dataset.width, dataset.height), Image.Resampling.LANCZOS)
        mask = torch.from_numpy(np.array(alpha)) >= 128
    if not mask.any():
        raise ValueError(f"No alpha foreground in {dataset.frame_id(i)}")
    return mask


def comparison_settings(dataset, meta):
    return {
        "version": 1, "data_hash": meta["data_hash"], "training_config": meta["training_config"],
        "test_ids": [dataset.frame_id(i) for i in range(len(dataset))],
        "width": dataset.width, "height": dataset.height, "near": dataset.near, "far": dataset.far,
        "white_background": dataset.white_background, "render": "fine deterministic fp32",
        "psnr": "sRGB [0,1], mean per-view dB, MSE floor 1e-12",
        "ssim": "sRGB [0,1], RGB mean, 7x7 uniform valid windows, sample covariance, K1=.01 K2=.03",
        "lpips": "AlexNet v0.1, lpips 0.1.4, sRGB mapped to [-1,1]",
        "foreground": "Blender alpha resized with Lanczos, >=128/255; unavailable for RGB captures",
        "depth": "camera-axis z, residual transmittance at far, PNG16 black=near white=far",
    }


@torch.inference_mode()
def evaluate(run, manifest, model, meta, device, chunk, perceptual=None):
    dataset = SceneDataset(manifest, "test", meta["downscale"])
    if min(dataset.height, dataset.width) < 31:
        raise ValueError("AlexNet LPIPS requires image dimensions >=31 pixels")
    output = run / "eval"
    output.mkdir(parents=True, exist_ok=True)
    perceptual = (perceptual if perceptual is not None else lpips_model()).to(device).eval()
    settings = comparison_settings(dataset, meta)
    settings["lpips_weights_sha256"] = model_fingerprint(perceptual)
    regularized = meta["protocol"] == "sparse-regularized"
    representatives = {0, len(dataset) // 2, len(dataset) - 1}
    frames = []
    targets_hash = hashlib.sha256()
    started = time.perf_counter()
    for i in range(len(dataset)):
        origins, directions, target = dataset.frame(i)
        targets_hash.update(rgb_uint8(target).tobytes())
        rgb, depth, opacity = render_maps(model, dataset, (origins, directions), regularized, device, chunk)
        full_psnr, lpips = image_scores(rgb, target, perceptual, device)
        record = dict(frame=dataset.frame_id(i), psnr=full_psnr, ssim=ssim(rgb, target), lpips=lpips,
                      white_psnr=psnr(torch.ones_like(target), target))
        mask = alpha_mask(dataset, i)
        if mask is not None:
            record.update(foreground_psnr=psnr(rgb[mask], target[mask]),
                          foreground_opacity=opacity[mask].mean().item(),
                          white_foreground_psnr=psnr(torch.ones_like(target[mask]), target[mask]))
        if i in representatives:
            prefix = f"view-{i:03d}"
            record["artifacts"] = {name: f"{prefix}-{name}.png" for name in ("rgb", "target", "depth", "opacity")}
            scaled = (depth - dataset.near) / (dataset.far - dataset.near)
            images = dict(rgb=rgb_uint8(rgb), target=rgb_uint8(target), opacity=rgb_uint8(opacity),
                          depth=(scaled.clamp(0, 1) * 65535).round().numpy().astype(np.uint16))
            for key, pixels in images.items():
                Image.fromarray(pixels).save(output / record["artifacts"][key])
        frames.append(record)
        print(f"[test {i+1}/{len(dataset)}] {record['frame']}  PSNR {full_psnr:.3f}  "
              f"SSIM {record['ssim']:.4f}  LPIPS-Alex {lpips:.4f}", flush=True)
    heldout_seconds = time.perf_counter() - started
    settings["target_hash"] = targets_hash.hexdigest()
    metrics = {"psnr", "ssim", "lpips", "white_psnr", "foreground_psnr",
               "foreground_opacity", "white_foreground_psnr"} & frames[0].keys()
    mean = {key: statistics.mean(frame[key] for frame in frames) for key in sorted(metrics)}
    split = "dense" if meta["protocol"] == "dense-vanilla" else "sparse"
    result = {key: meta[key] for key in ("scene", "seed", "protocol", "run_id")}
    result.update(version=1, comparison=settings, comparison_hash=digest(settings), frames=frames, mean=mean,
                  model_sha256=model_fingerprint(model),
                  test_seconds=heldout_seconds, train_views=len(dataset.manifest["splits"][split]))
    if split == "sparse":
        training = SceneDataset(manifest, "sparse", meta["downscale"])
        scores = []
        for i in range(len(training)):
            origin, direction, target = training.frame(i)
            prediction, _, _ = render_maps(model, training, (origin, direction), regularized, device, chunk)
            score = psnr(prediction, target)
            scores.append({"frame": training.frame_id(i), "psnr": score})
            print(f"[train {i+1}/{len(training)}] PSNR {score:.3f}", flush=True)
        mean_train = statistics.mean(row["psnr"] for row in scores)
        result["training_views"] = {"frames": scores, "mean_psnr": mean_train, "at_least_30_db": mean_train >= 30}
    elif dataset.manifest["format"] == "blender" and meta["scene"].lower() == "lego":
        result["dense_lego_reference"] = {
            "range_db": [30.54, 34.54], "inside_range": 30.54 <= mean["psnr"] <= 34.54,
            "note": "Cross-protocol sanity check. The published full-resolution reference used different settings.",
        }
    write_json(output / "metrics.json", result)
    print(f"Saved {output / 'metrics.json'}", flush=True)
    return result


@torch.inference_mode()
def video(run, manifest, model, meta, device, chunk, frames=120, fps=24):
    if min(frames, fps, chunk) < 1:
        raise ValueError("frames, fps and chunk must be positive")
    dataset = SceneDataset(manifest, "test", meta["downscale"])
    trajectory = dict(data_hash=meta["data_hash"], downscale=meta["downscale"], frames=frames, fps=fps,
                      path="blender-orbit-v1" if dataset.manifest["format"] == "blender" else "capture-interpolation-v1")
    trajectory_hash = digest(trajectory)
    output = run / "video" / trajectory_hash[:12]
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    descriptor = dict(run_id=meta["run_id"], model_sha256=model_fingerprint(model),
                      trajectory=trajectory, trajectory_hash=trajectory_hash)
    description_path = output / "trajectory.json"
    if description_path.exists() and json.loads(description_path.read_text(encoding="utf-8")) != descriptor:
        raise ValueError("Existing video frames belong to a different checkpoint or trajectory")
    if not description_path.exists() and any(frame_dir.glob("*.png")):
        raise ValueError("Restore the trajectory.json that belongs to these saved video frames")
    write_json(description_path, descriptor)
    for i in range(frames):
        path = frame_dir / f"{i:05d}.png"
        if path.exists():
            try:
                with Image.open(path) as cached:
                    if cached.size != (dataset.width, dataset.height) or cached.mode != "RGB":
                        raise ValueError("mismatched frame dimensions")
                    cached.verify()
                continue
            except (OSError, ValueError, SyntaxError):
                print(f"Re-rendering damaged frame {i}", flush=True)
        rgb, _, _ = render_maps(model, dataset, dataset.video_rays(i, frames),
                                meta["protocol"] == "sparse-regularized", device, chunk)
        with NamedTemporaryFile(dir=frame_dir, suffix=".png", delete=False) as tmp:
            pending = Path(tmp.name)
        try:
            Image.fromarray(rgb_uint8(rgb)).save(pending)
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
        print(f"[video {i+1}/{frames}] {path.name}", flush=True)
    movie = output / "novel.mp4"
    with NamedTemporaryFile(dir=output, suffix=".mp4", delete=False) as tmp:
        pending = Path(tmp.name)
    try:
        subprocess.run([encoder_executable(), "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
                        "-i", str(frame_dir / "%05d.png"), "-frames:v", str(frames), "-an", "-c:v", "libx264",
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-crf", "18",
                        "-movflags", "+faststart", str(pending)], check=True)
        pending.replace(movie)
    finally:
        pending.unlink(missing_ok=True)
    print(f"Saved {movie}", flush=True)
    return movie


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    evaluate_cli = commands.add_parser("eval", help="score final predictions and export representative images")
    video_cli = commands.add_parser("video", help="resume frame rendering and encode the matched video")
    for command in (evaluate_cli, video_cli):
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--device", default="auto")
        command.add_argument("--chunk", type=int, default=1024)
    video_cli.add_argument("--frames", type=int, default=120)
    video_cli.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    from .training import load_final
    device = select_device(args.device)
    torch.set_float32_matmul_precision("highest")
    model, meta = load_final(args.run, args.manifest, device)
    arguments = (args.run, args.manifest, model.eval(), meta, device, args.chunk)
    if args.stage == "eval":
        evaluate(*arguments)
    else:
        video(*arguments, frames=args.frames, fps=args.fps)


if __name__ == "__main__":
    main()
