"""Paste this whole file into a new Kaggle cell to export surviving dense Lego weights.

Use the original session, or attach the saved Stage 3 notebook output to a fresh
notebook. A CPU session is sufficient. No Stage 0 kit or network access needed.
"""

import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import torch


RUN = "runs/lego/dense-vanilla/seed-42"
RUN_ID = "08ff8d2fceda43759554aa6470054853"
FINGERPRINT = "4779927a3d27bf16070f9227dcec388a71ebc81658ebfa453b5aa8f5cae9ca13"
DATA_HASH = "bdc22418467720a3dbf5555318532caf17c250a05433f8553ae58372f11e89e2"
STEP2000_HASH = "4babbd7b23849a9d98c578753bb273dc6b943b9e2382dd9955ee271ae23b3a8e"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect_checkpoint(read):
    meta = json.loads(read(f"{RUN}/run.json"))
    keys = ("data_hash", "protocol", "seed", "scene", "training_config", "run_id")
    identity = json.dumps({k: meta[k] for k in keys}, sort_keys=True).encode()
    if sha(identity) != FINGERPRINT:
        raise ValueError("Different run or training configuration")
    manifest = json.loads(read("manifests/lego.json"))
    portable = {k: v for k, v in manifest.items() if k not in {"source", "data_hash"}}
    if sha(json.dumps(portable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) != DATA_HASH:
        raise ValueError("Lego manifest differs from reviewed run")
    if manifest.get("data_hash") != DATA_HASH:
        raise ValueError("Recorded manifest digest differs")
    raw = read(f"{RUN}/checkpoint.pt")
    state = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    step = state["step"]
    if state["fingerprint"] != FINGERPRINT or type(step) is not int or not 2000 <= step <= 50_000:
        raise ValueError("Invalid checkpoint fingerprint or step")
    slots = state["optimizer"]["state"]
    if len(slots) != 48 or any(float(v["step"]) != step for v in slots.values()):
        raise ValueError("Adam counters disagree with checkpoint")
    tensors = list(state["model"].values()) + [
        v[key] for v in slots.values() for key in ("exp_avg", "exp_avg_sq")
    ]
    if not state["model"] or not all(t.isfinite().all().item() for t in tensors):
        raise ValueError("Missing weights or nonfinite model/Adam state")
    if any(state[k] is None or not state[k].numel() for k in ("cpu_rng", "cuda_rng")):
        raise ValueError("Incomplete RNG state")
    return step, sha(raw)


def inspect_zip(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP failed CRC")
        step, digest = inspect_checkpoint(archive.read)
        record = json.loads(archive.read("stage3-review.json"))
        if (record["run_id"], record["checkpoint_step"], record["checkpoint_sha256"],
                record["source_checkpoint_sha256"]) != (RUN_ID, step, digest, STEP2000_HASH):
            raise ValueError("Stage 3 record differs from checkpoint")
        return step, digest


def salvage(roots, output, already_saved=20_000):
    """Find the highest verified state beyond the operator's saved checkpoint."""
    checked = []
    for root in roots:
        if not root.is_dir():
            continue
        files = set(root.rglob("lego-dense-vanilla-s42-stage3-*.zip"))
        files.update(root.rglob("lego-dense-vanilla-s42-step50000.zip"))
        files.update(root.rglob(f"{RUN}/checkpoint.pt"))
        for p in sorted(files):
            try:
                if p.suffix == ".zip":
                    step, digest = inspect_zip(p)
                else:
                    run_root = p.parents[4]
                    step, digest = inspect_checkpoint(lambda name: (run_root / name).read_bytes())
                print(f"Verified step {step:,} at {p}", flush=True)
                if step > already_saved:
                    checked.append((step, p.suffix == ".zip", p, digest))
            except Exception as exc:
                print(f"Skipping {p}: {type(exc).__name__}: {exc}", flush=True)
    if not checked:
        print(f"No verified checkpoint newer than {already_saved:,} in these mounts.")
        return None
    step, is_zip, path, digest = max(checked)
    output.mkdir(parents=True, exist_ok=True)
    # Each invocation gets its own destination; every source stays intact.
    folder = Path(tempfile.mkdtemp(prefix="salvaged-lego-", dir=output))
    tag = "step50000" if step == 50_000 else f"stage3-{step:05d}"
    final = folder / f"lego-dense-vanilla-s42-{tag}.zip"
    if is_zip:
        shutil.copyfile(path, final)
    else:
        run_root = path.parents[4]
        metadata = {"run_id": RUN_ID, "source_checkpoint_sha256": STEP2000_HASH,
                    "checkpoint_sha256": digest, "checkpoint_step": step,
                    "complete": step == 50_000, "planned_stop_reached": step == 50_000,
                    "salvaged_from_loose_checkpoint": True}
        names = ["manifests/lego.json", "stage3-hardware.json", "dense-finish.log"]
        names += [f"{RUN}/{n}" for n in ("checkpoint.pt", "run.json", "health.json",
                                         "initial-health.json", "training.jsonl")]
        names += [p.relative_to(run_root).as_posix() for p in path.parent.glob("train-*.png")]
        with zipfile.ZipFile(final, "x", compression=zipfile.ZIP_STORED) as archive:
            for name in names:
                if (run_root / name).is_file():
                    archive.write(run_root / name, name)
            archive.writestr("stage3-review.json", json.dumps(metadata, indent=2))
    if inspect_zip(final) != (step, digest):
        raise RuntimeError("Export verification failed. Rerun after the checkpoint write finishes.")
    return final


def show_download(path):
    from IPython.display import HTML, display

    payload = path.read_bytes()
    print(f"Export verified: {path}\n{len(payload):,} bytes\nSHA-256: {sha(payload)}", flush=True)
    print("Download from the notebook's Output files, or click the embedded download below.")
    encoded = base64.b64encode(payload).decode("ascii")
    # The link carries the bytes directly, independent of Jupyter's file-serving URL.
    display(HTML(f'<a download="{path.name}" href="data:application/zip;base64,{encoded}">'
                 f'Download {path.name} ({len(payload) / 1e6:.1f} MB)</a>'))


if __name__ == "__main__":
    file = salvage([Path("/kaggle/working"), Path("/kaggle/input")], Path("/kaggle/working"))
    if file is not None:
        show_download(file)
