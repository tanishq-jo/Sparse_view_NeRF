"""Train one recovery arm, or run the short one-camera Lego diagnostic."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
from uuid import uuid4

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from sparse_nerf.data import SceneDataset, image_downscale
from sparse_nerf.model import ModelConfig
from sparse_nerf.run import select_device
from sparse_nerf.train import PROTOCOLS, training_lock
from recovery.data import verify_manifest
from recovery.health import camera_check, dead_field, diagnostics, fixed_rays
from recovery.model import Renderer


BACKEND = "recovery-pytorch-finite-softplus-v1"
PRESETS = {
    "main": dict(steps=50000, batch_size=512),
    # Same architecture, ray count and image scale as main. Only one training camera.
    "diagnostic": dict(steps=256, batch_size=512),
}


def implementation_hash():
    """Catch code drift across notebook snapshots and tolerate Git line-ending conversions."""
    root = Path(__file__).resolve().parents[1]
    files = ("recovery/training.py", "recovery/model.py", "recovery/data.py", "recovery/health.py",
             "sparse_nerf/model.py", "sparse_nerf/data.py", "sparse_nerf/prepare.py")
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode() + b"\0" + (root / name).read_text(encoding="utf-8").encode() + b"\0")
    return digest.hexdigest()


def settings(preset, camera):
    return dict(backend=BACKEND, implementation_hash=implementation_hash(), preset=preset, **PRESETS[preset],
                model=asdict(ModelConfig()), near_samples=20, occ_weight=.01,
                lr=.0005, lr_final=.00005, downscale=image_downscale(camera, None),
                optimizer="Adam", sampling=("50% object, 50% background on one training view"
                                             if preset == "diagnostic" else "uniform pixels"),
                precision="fp32-highest")


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def fingerprint(meta):
    keys = ("data_hash", "protocol", "seed", "scene", "training_config", "run_id")
    return hashlib.sha256(json.dumps({k: meta[k] for k in keys}, sort_keys=True).encode()).hexdigest()


def checkpoint(run, model, optimizer, step, meta, device):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                 fingerprint=fingerprint(meta), cpu_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
    path = run / "checkpoint.pt.tmp"
    torch.save(state, path)
    path.replace(run / "checkpoint.pt")


def load_state(run, meta):
    state = torch.load(run / "checkpoint.pt", weights_only=True, map_location="cpu")
    if state["fingerprint"] != fingerprint(meta) or not 0 <= state["step"] <= meta["training_config"]["steps"]:
        raise ValueError("Checkpoint and run metadata disagree")
    return state


def make_model(cfg, device):
    torch.set_float32_matmul_precision("highest")
    return Renderer(ModelConfig(**cfg["model"]), cfg["near_samples"]).to(device)


def diagnostic_pixel_pools(dataset):
    """Let the short one-view fit exercise both the object and its background."""
    with Image.open(dataset.source / dataset.frame_id(0)) as image:
        alpha = image.getchannel("A").resize((dataset.width, dataset.height), Image.Resampling.BOX)
    foreground = torch.from_numpy(np.array(alpha)).flatten() >= 128
    pools = (foreground.nonzero().flatten(), (~foreground).nonzero().flatten())
    if any(len(p) == 0 for p in pools):
        raise ValueError("One-camera diagnostic needs transparent background and visible object pixels")
    return pools


def load_final(run, manifest, device):
    run = Path(run)
    data = verify_manifest(Path(manifest))
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    cfg = meta["training_config"]
    if (cfg["backend"] != BACKEND or cfg["implementation_hash"] != implementation_hash()
            or meta["data_hash"] != data["data_hash"]):
        raise ValueError("Dataset or backend differs from this run")
    state = load_state(run, meta)
    if cfg["preset"] != "main" or state["step"] != cfg["steps"]:
        raise ValueError("Final evaluation requires a completed main run")
    model = make_model(cfg, device)
    model.load_state_dict(state["model"])
    return model.eval(), meta


@torch.no_grad()
def preview(run, step, model, dataset, i, progress, regularized, device):
    target = dataset.image(i)
    rows = torch.linspace(0, dataset.height - 1, 48).round().long()
    cols = torch.linspace(0, dataset.width - 1, 48).round().long()
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    o, d = dataset.rays(i, rr.flatten(), cc.flatten())
    colors, opacities = [], []
    for start in range(0, len(o), 128):
        out = model(o[start:start+128].to(device), d[start:start+128].to(device),
                    dataset.near, dataset.far, progress=progress, regularized=regularized,
                    white_background=dataset.white_background)
        colors.append(out["rgb"].cpu())
        opacities.append(out["opacity"].cpu())
    pred = torch.cat(colors).view(48, 48, 3)
    opacity = torch.cat(opacities).view(48, 48, 1).expand_as(pred)
    strip = torch.cat((target[rr, cc], pred, opacity), dim=1)
    Image.fromarray(np.asarray((strip.clamp(0, 1)*255).round().byte())).save(run / f"train-{step:06}.png")


def train(run, manifest, protocol, seed, device, cfg, *, resume=False, stop_at=None, save_every=500):
    run, manifest, device = Path(run), Path(manifest), torch.device(device)
    data = verify_manifest(manifest)
    diagnostic = cfg["preset"] == "diagnostic"
    if diagnostic and (data["name"] != "lego" or protocol != "dense-vanilla"):
        raise ValueError("The diagnostic uses one Lego training camera from the dense split")
    steps = cfg["steps"]
    stop = steps if stop_at is None else stop_at
    if not 1 <= stop <= steps or save_every < 1:
        raise ValueError("Invalid stage end or save interval")
    split = "dense" if protocol == "dense-vanilla" else "sparse"
    dataset = SceneDataset(manifest, split, cfg["downscale"])
    view_ids = [0] if diagnostic else list(range(len(dataset)))
    common = dict(scene=data["name"], data_hash=data["data_hash"], training_config=cfg,
                  protocol=protocol, seed=seed, train_views=len(view_ids), downscale=cfg["downscale"])
    regularized = protocol == "sparse-regularized"
    checks = camera_check(dataset)
    with training_lock(run):
        meta_file = run / "run.json"
        if resume:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if any(meta.get(k) != v for k, v in common.items()) or meta["device_type"] != device.type:
                raise ValueError("Resume requires the same data, arm, device type and frozen settings")
        else:
            if any(p.name != ".train.lock" for p in run.iterdir()):
                raise FileExistsError(f"Run directory contains files: {run}. Use --resume for a checkpoint.")
            meta = dict(**common, run_id=uuid4().hex, device_type=device.type,
                        hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
                        torch_version=str(torch.__version__), cuda_runtime=torch.version.cuda,
                        camera_check=checks, diagnostic_view=dataset.frame_id(0) if diagnostic else None)
            write_json(meta_file, meta)
        torch.manual_seed(seed)
        model = make_model(cfg, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        start = 0
        if resume:
            restored = load_state(run, meta)
            model.load_state_dict(restored["model"])
            optimizer.load_state_dict(restored["optimizer"])
            torch.set_rng_state(restored["cpu_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(restored["cuda_rng"], device)
            start = restored["step"]
        else:
            checkpoint(run, model, optimizer, 0, meta, device)
        if start >= stop:
            if diagnostic and stop == steps:
                gate = json.loads((run / "gate.json").read_text(encoding="utf-8"))
                if not gate["passed"]:
                    raise RuntimeError("Diagnostic failed; inspect gate.json")
            print(f"Already at step {start}")
            return
        images = torch.stack([dataset.image_uint8(i) for i in view_ids])
        pools = diagnostic_pixel_pools(dataset) if diagnostic else None
        monitor_ids = view_ids if len(view_ids) <= 8 else view_ids[::max(1, len(view_ids)//8)][:8]
        rays = fixed_rays(dataset, monitor_ids, device)
        def inspect(step):
            p = max(0, step-1) / max(1, steps-1)
            info = diagnostics(model, dataset, rays, p, regularized)
            info.update(step=step)
            write_json(run / "health.json", info)
            print(f"step {step} | fine opacity {info['fine']['region_opacity']:.4f} | "
                  f"region PSNR {info['fine']['region_psnr']:.2f} | "
                  f"white {info['white_region_psnr']:.2f}", flush=True)
            if dead_field(info):
                raise RuntimeError("Dead density or color gradient on fixed training rays; inspect health.json")
            return info
        baseline_path = run / "initial-health.json"
        initial = inspect(start)
        if start == 0:
            write_json(baseline_path, initial)
        original = json.loads(baseline_path.read_text(encoding="utf-8")) if diagnostic else None
        start_time = time.perf_counter()
        for step in range(start+1, stop+1):
            progress = (step-1) / max(1, steps-1)
            for group in optimizer.param_groups:
                group["lr"] = cfg["lr"] * (cfg["lr_final"] / cfg["lr"]) ** progress
            n = cfg["batch_size"]
            if diagnostic:
                ix = torch.zeros(n, dtype=torch.long)
                pixels = torch.cat([pool[torch.randint(len(pool), (size,))]
                                    for pool, size in zip(pools, (n//2, n-n//2))])
                r, c = pixels // dataset.width, pixels % dataset.width
            else:
                ix = torch.randint(len(view_ids), (n,))
                r = torch.randint(dataset.height, (n,))
                c = torch.randint(dataset.width, (n,))
            origins, directions = dataset.rays(ix, r, c)
            target = images[ix, r, c].to(device, dtype=torch.float32) / 255.
            out = model(origins.to(device), directions.to(device), dataset.near, dataset.far,
                        progress=progress, regularized=regularized, randomized=True,
                        white_background=dataset.white_background)
            fine = F.mse_loss(out["rgb"], target)
            coarse = F.mse_loss(out["rgb_coarse"], target)
            loss = fine + coarse
            if regularized:
                loss = loss + cfg["occ_weight"] * out["near_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step in (1, 32, 128) or step % save_every == 0 or step == stop:
                info = inspect(step)
                elapsed = time.perf_counter() - start_time
                rate = (step-start) / elapsed
                record = dict(**info, elapsed_s=elapsed, steps_per_s=rate, fine_mse=fine.item(),
                              coarse_mse=coarse.item(), near_loss=out["near_loss"].item())
                with (run / "training.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                print(f"{rate:.3f} steps/s | remaining training {(steps-step)/rate/3600:.3f} h", flush=True)
                if step == stop:
                    preview(run, step, model, dataset, view_ids[0], progress, regularized, device)
                if diagnostic and step == steps:
                    fg = info["fine"]
                    passed = (fg["region_psnr"] > info["white_region_psnr"] + 3
                              and fg["region_mse"] < original["fine"]["region_mse"]
                              and fg["region_opacity"] > .02)
                    write_json(run / "gate.json", dict(passed=passed, step=step, steps_per_s=rate,
                               health=info, initial_region_mse=original["fine"]["region_mse"]))
                if step % save_every == 0 or step == stop:
                    checkpoint(run, model, optimizer, step, meta, device)
                if diagnostic and step == steps and not passed:
                    raise RuntimeError("Diagnostic did not pass; send gate.json and the preview")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--protocol", choices=PROTOCOLS, default="dense-vanilla")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--stop-at", type=int, help="Absolute checkpoint step; preserves the full-run schedule")
    args = parser.parse_args()
    cfg = settings(args.preset, json.loads(args.manifest.read_text(encoding="utf-8"))["camera"])
    train(args.run, args.manifest, args.protocol, args.seed, select_device(args.device), cfg,
          resume=args.resume, stop_at=args.stop_at, save_every=args.save_every)


if __name__ == "__main__":
    main()
