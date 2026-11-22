# Kaggle training and checkpoint handoff

**The current assignment is Stage 0 only.** Import [kaggle_stage0.ipynb](kaggle_stage0.ipynb), attach the private Stage 0 input, select T4 ×2 and run its three code cells. Send `stage0-review.zip` even if the diagnostic exits early. The notebook fits a single Lego training camera for 256 updates on GPU 0. It records both fields' density, opacity, foreground RGB and gradients, a white-image control, and a target/prediction/opacity strip. The pass condition requires foreground PSNR more than 3 dB above white, lower foreground MSE than initialization, and mean foreground opacity above 0.02. Both fields must keep finite, nonzero density and color gradients during monitoring.

The previous PyTorch implementation took 5.78–6.23 steps/s over its first 100 updates on a T4. That implies roughly 41–45 seconds of updates for 256 steps on that implementation, plus setup, preview and checkpoint time. The recovery version has different integration and density code. Use the rate recorded by this notebook for the next estimate. [Archived evidence](FORENSICS.md)

## The first dense checkpoint, after Stage 0 review

Use a fresh Kaggle notebook with the **same version** of the Stage 0 input. Run just the Stage 0 notebook's first setup cell to locate `repo` and the GPUs. Keep the first `stage0-review.zip` available separately. The following cells are a reference for the operator running a single approved job.

```python
from pathlib import Path
import json, shutil, subprocess, sys, time, zipfile

work = Path("/kaggle/working/recovery-main")
scene, arm, seed = "lego", "dense-vanilla", 42
manifest = work / "manifests" / f"{scene}.json"
run = work / "runs" / scene / arm / f"seed-{seed}"
source = repo / "data/blender/lego"

def call(module, *args):
    subprocess.run([sys.executable, "-u", "-m", module, *map(str, args)],
                   cwd=repo, check=True)

def fit(end, resume=False):
    call("recovery.training", "--preset", "main", "--protocol", arm, "--seed", seed,
         "--manifest", manifest, "--run", run, "--device", "cuda:0",
         "--save-every", 500, "--stop-at", end, *(["--resume"] if resume else []))

def snapshot(tag):
    archive = work.parent / f"{scene}-{arm}-s{seed}-{tag}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        for p in [manifest, *sorted(run.rglob("*"))]:
            if p.is_file() and not p.name.startswith(".") and not p.name.endswith(".tmp"):
                z.write(p, p.relative_to(work))
    print(archive, archive.stat().st_size, "bytes")
```

For a new dense Lego run, execute the next cell once.

```python
call("recovery.data", "--scene", scene, "--source", source, "--output", manifest)
try:
    fit(500)
finally:
    snapshot("first-stop")
```

This starts a new 100-view main run with a 50,000-step schedule. Stage 0 was a one-view diagnostic. At the 500-update stop the main run has its own model, Adam state, RNG state and checkpoint. Send `health.json`, `initial-health.json`, `training.jsonl`, `run.json`, and the `train-000500.png` strip. Download the complete snapshot ZIP from the Output pane before releasing the session. Inspect both fields' foreground signal, center-ray bounds and the fine RGB against the target. The Stage 0 `steps_per_s` gives an initial estimate of `500 / steps_per_s` seconds for these updates. Replace it with the measured 500-step main rate.

Once that checkpoint passes review, the same session can run `fit(50000, resume=True)` and `snapshot("trained")`. Estimate remaining training hours as `(50000 - checkpoint_step) / measured_steps_per_s / 3600`. Choose any earlier absolute `end` when the available session time is shorter. Allow time to save the resulting snapshot and download it. A live file in `/kaggle/working` still needs a saved copy outside the active session.

## Resume or evaluate in a fresh session

Attach the same code and image inputs and the **entire** prior snapshot. Run the setup cell to locate `repo`, then rerun the Python definitions above with the same scene, arm and seed. Continue with the restore cell below. Choose the source scene directory on the new mount. For a zipped checkpoint that Kaggle exposes as a file, first expand your own downloaded snapshot in a temporary directory. If Kaggle already exposes its contents as a directory, point `saved` to the directory containing `manifests/` and `runs/`.

```python
import tempfile

saved = Path("/kaggle/input/YOUR-SNAPSHOT")      # set to the folder or .zip on this mount
if saved.is_file():
    destination = Path(tempfile.mkdtemp(prefix="nerf-snapshot-"))
    with zipfile.ZipFile(saved) as z:
        assert all((destination / n).resolve().is_relative_to(destination) for n in z.namelist())
        z.extractall(destination)
    saved = destination

call("recovery.data", "--verify", saved / "manifests" / f"{scene}.json",
     "--source", source, "--output", manifest)
run.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(saved / run.relative_to(work), run)  # fresh writable destination
print(json.loads((run / "run.json").read_text())["training_config"])
```

Resume a partial training checkpoint with `fit(end, resume=True)` and snapshot after the planned stop. For a completed 50,000-step checkpoint, proceed directly to evaluation in its own stage. Training verifies identities and reloads the saved optimizer and RNG. Keep the same Torch/CUDA environment where practical and record the notebook environment and T4 model for each session.

## Score, review, then render

Evaluation uses PyTorch, a matching torchvision, the `lpips==0.1.4` package, and pretrained AlexNet v0.1 weights. Stage 0 requires only the training dependencies already present in a standard PyTorch notebook. For the evaluation session, enable notebook Internet for the following initial setup or provide preinstalled packages and a populated Torch cache as inputs.

```python
import os

subprocess.run([sys.executable, "-m", "pip", "install", "lpips==0.1.4"], check=True)
os.environ["TORCH_HOME"] = str(work / "torch-cache")  # set to a populated cache if offline
subprocess.run([sys.executable, "-c",
                "from sparse_nerf.evaluate import lpips_model; lpips_model()"],
               cwd=repo, check=True)
started = time.monotonic()
call("recovery.media", "eval", "--run", run, "--manifest", manifest,
     "--device", "cuda:0", "--chunk", 1024)
print("Total scoring seconds", time.monotonic() - started)
snapshot("scored")
```

Send `eval/metrics.json` plus the three held-out RGB/target/depth/opacity image sets. Synthetic scenes use 25 views. Each sparse run also scores all eight training images for the proposal's 30 dB training-fit check. Evaluation reports mean per-view PSNR, 7×7-window SSIM and LPIPS-AlexNet v0.1. Synthetic alpha supplies foreground PSNR. For dense Lego, review the proposal's 30.54–34.54 dB cross-resolution reference window alongside image structure, PSNR against white, foreground opacity, depth and LPIPS. The old dense checkpoint scored 28.947 dB on these 25 test IDs. If the new baseline misses the proposal window, record the miss and investigate the training-camera fit and configuration before launching additional arms. Preserve the held-out set for reporting and avoid selecting settings on its scores.

After review, render **separately** with the complete run directory writable and an FFmpeg binary supporting `libx264`. Kaggle's system `ffmpeg` can be checked with `shutil.which("ffmpeg")`; the repository also supports `imageio-ffmpeg`.

```python
started = time.monotonic()
try:
    call("recovery.media", "video", "--run", run, "--manifest", manifest,
         "--device", "cuda:0", "--chunk", 1024, "--frames", 120, "--fps", 24)
finally:
    print("Video wall seconds", time.monotonic() - started)
    snapshot("video")
```

Save the MP4, trajectory JSON and its PNG frame cache. A restored snapshot with the trajectory and partial frames resumes rendering with the same command. The old renderer averaged about 22 seconds per 400 × 400 frame, or about 44 minutes per 120-frame video. Replace that estimate with the current renderer's measured time per frame. Keep the path, frame count, FPS and resolution identical across each scene's three arms.

## The remaining jobs and the time budget

| Scene | Dense vanilla | Sparse vanilla | Sparse regularised | Held-out |
| --- | ---: | ---: | ---: | ---: |
| Lego | 100 | 8 | 8 | 25 |
| Chair | 100 | 8 | 8 | 25 |
| Personal capture | all registered non-test | 8 | 8 | 12 by default |

Start each new arm in its own empty run directory by changing `scene` and `arm` in the setup cell. Use the exact strings `dense-vanilla`, `sparse-vanilla` and `sparse-regularized`. The seed is 42 for all nine runs. Keep one common backend and a fixed 50k × 512-ray training budget. Uniform training pixels, Adam, the learning-rate schedule and deterministic evaluation are shared. The synthetic inputs are 400 × 400 with white backgrounds and 2/6 bounds. The personal scene uses a fixed longest edge at most 400 pixels, original RGB photographs with a black residual background, and bounds from registered geometry. The sparse regularised arm adds the frequency mask and fine near-camera density penalty. The baseline arms keep both terms disabled. [Backend and reference configs](REFERENCES.md)

The offline Stage 0 input supplies Lego. For Chair, attach the unchanged local `data/blender/chair` and the frozen `scratch/recovery-validation/chair.json` as private inputs. For the personal scene, attach the accepted processed capture and frozen personal manifest. In the setup cell use the mounted scene root for `source` and rebind with `call("recovery.data", "--verify", previous_manifest, "--source", source, "--output", manifest)`. The code confirms every source image hash on the new mount. [Capture and registration](CAPTURE.md)

Use one device per run. A T4 ×2 selection exposes two GPUs, while `cuda:0` selects one for this command. If time requires concurrent jobs after review, give the second approved job `cuda:1` and a separate run directory. The old 5.8 updates/s implies about 2.4 active GPU-hours per trained arm or 21.6 for nine. Nine old-speed videos would add 6.6 hours. Scoring the old 200 views took 73 minutes, so a 25-view test pass at that throughput takes about nine minutes, with additional training-view scoring on sparse arms. These are planning figures. Recalculate against measured main-stage speeds, actual capture resolution and available account quota. Budget for both provisioned GPUs when estimating exposure to the 60–90 GPU-hour cap. Schedule dense Lego first, then each sparse Lego arm, Chair, and the registered capture. Reserve time for scored outputs and checkpoint transfers before adding repeats or stretch runs.

Once all three scored arms for a scene are present, attach or copy their `eval/metrics.json` files and compute the matched report. This can run on CPU.

```python
call("recovery.report", "--runs", work / "runs" / scene,
     "--output", work / f"{scene}-comparison.json")
```

Report the per-seed, per-view means and paired 95% intervals. A single seed quantifies view uncertainty conditional on that training run. Recovery percentages require a measured dense-minus-sparse PSNR gap of at least 3 dB with a positive interval lower bound. LPIPS also requires its own positive, separated denominator. Report the signed gains and reasons for withheld ratios. The 102-job matrix, RegNeRF, extra seeds and parameter sweeps follow the nine requested comparisons only if time remains.
