# Kaggle training and checkpoint handoff

## Stage 14. Finish and score both Chair baselines

1. Import [the Stage 14 notebook](kaggle_stage14_chair_vanilla_finish.ipynb). Attach the original Chair kit, `chair-s42-stage13-review.zip`, and the Stage 4 offline LPIPS kit. Choose **GPU T4 ×2** and **Run All**. Setup should restore dense vanilla and eight-view vanilla at **2,000** and verify the official 25 test targets.
2. Dense vanilla measured 5.38 updates/s and sparse vanilla 6.50 updates/s in the last Stage 13 segment. The slower T4 projects to **3 h 46 min** of optimisation for the remaining 73,000 updates. Allow **4–4½ hours total** with checkpoints and scoring, approximately 8–9 allocated GPU-hours. Check the actual rates in the notebook. Each 10k milestone checks gradients and foreground learning before continuing.
3. Save **`chair-vanilla-pair-s42-stage14-step50000.zip`** and **`chair-vanilla-pair-s42-stage14-step75000.zip`**. On interruption, attach the latest successful Stage 14 snapshot alongside the two kits and run all cells. The progress message reports the actual restored step for each arm.
4. The separate scoring cell produces PSNR, SSIM, LPIPS-Alex v0.1 and representative RGB, depth and opacity for each arm. It verifies the test pixels and the per-view white control. It also scores all eight training images for sparse vanilla. Send **`chair-vanilla-pair-s42-stage14-review.zip`** after both reports finish, or the latest ZIP and traceback on failure. Video export follows scoring.

## Stage 13. Continue Chair vanilla and test the initial frequency mask

1. Import [the Stage 13 notebook](kaggle_stage13_chair_continue.ipynb). Attach the original Chair dataset (`chair-input.zip` or expanded) and **`chair-s42-stage11-review.zip`** (or expanded). Select **GPU T4 ×2** and **Run All**. Setup verifies the two vanilla checkpoints at 500 and the regularised checkpoint at zero.
2. The two vanilla workers resume on separate T4s. Each writes a checkpoint at 1,000 and continues to 2,000 if the foreground check passes. Download **`chair-s42-stage13-vanilla2000.zip`** when it appears. The 1,000-update backup is also available in Output.
3. Two eight-view diagnostic copies start from the same regularised initial weights and RNGs. Both use the 0.01 near-camera penalty. **First-band** exposes the lowest sine/cosine pair at initialisation and progressively opens the remaining bands. **Near-only** exposes all bands from initialisation. Each records any dead-field warning and ends at 500 with a checkpoint, preview, gradients and two fixed-ray probes. These are separately identified post-failure experiments. The original Chair and Lego settings retain their own run identities.
4. The measured vanilla rates of 5.91 and 6.26 updates/s project to 4.2 minutes for 1,500 further updates in parallel. The Stage 12 rates of 6.30–6.82 updates/s project to about 80 seconds for the second pair. Allow **8–15 minutes total** with startup and exports. The printed per-worker training ETA uses the 50k prefix, while the notebook stops at the short milestones.
5. A foreground pass requires finite live coarse and fine gradients, fine foreground PSNR over 3 dB above its white control, lower fine foreground MSE than its own initialisation and foreground opacity above 0.02. Return **`chair-s42-stage13-review.zip`** for either diagnostic outcome. If a worker errors, send the diagnostics ZIP and traceback. In a fresh session, attach the latest completed Stage 13 ZIP alongside the original two inputs and select Run All.

## Stage 12. Replay the Chair foreground failure

1. Import [the Stage 12 notebook](kaggle_stage12_chair_probe.ipynb). Attach the **same Chair dataset** used in Stage 11 and the returned **`chair-s42-stage11-review.zip`** as a second private input. Both ZIP and expanded inputs work. Select **GPU T4 ×2** and **Run All**.
2. Setup checks the exact Stage 11 receipt, member hashes, starting checkpoint and eight cameras. Expect the full diagnostic copy on one GPU and the frequency-only copy on the other. Both restore the regularised step-zero state. The second copy changes only the near-camera coefficient, from 0.01 to zero. The two vanilla checkpoints at 500 remain available for their next resume.
3. Save **`chair-s42-stage12-step128.zip`** when its link appears. Each diagnostic copy retains its actual step-128 weights, target/prediction/opacity preview and fixed-ray measurements. Both record the original dead-field warning and continue under observation to a hard limit of 500. The frozen loss, optimiser, sampling, frequency ramp and finite-value checks apply throughout. At each stop the notebook checks that both CPU and CUDA RNG streams still agree across copies.
4. A successful diagnostic prints whether the full copy reproduced the white signature at 128 and whether each copy passed the foreground check at 500. Either outcome is a result. At the short Stage 11 regularised rate of 5.35 steps/s, 500 updates project to about 94 seconds per GPU in parallel. Allow **4–8 minutes wall time** including imports, probes and archives. Send **`chair-s42-stage12-review.zip`**, or the **diagnostics ZIP and traceback** if a worker errors. For a fresh session, reattach the latest completed Stage 12 snapshot with the original two inputs. Review this evidence before changing the main protocol or launching a long run.

The original 128-ray foreground check at 500 requires fine PSNR above the initial 13.506 dB (and more than 3 dB above white), opacity above 0.02 and live coarse and fine heads. A full replay that recovers after reproducing the stop would show that the failure was transient in that replay. Recovery only with the near penalty disabled would implicate that term in this controlled run. If both copies stay white, use the saturation, opacity and separate RGB/near gradient measurements to narrow the next test. A replay that differs at 128 must be reported as such.

## Stage 11. Check the three Chair runs

1. Import [the Stage 11 notebook](kaggle_stage11_chair_health.ipynb). Upload the prepared private `recovery-output/chair-input.zip` as a Kaggle dataset and attach it, or reuse it if already uploaded. Either the ZIP or its expanded contents work. Select **GPU T4 ×2** and **Run All**.
2. Expect verification of 300 Chair source images, 100 dense training views, the published eight sparse views and 25 held-out cameras at 400 × 400. All models start from seed 42. Dense vanilla and eight-view vanilla start on separate T4s, then the eight-view regularised arm uses a free T4. Each run keeps its own weights, Adam state and RNG states.
3. The fixed foreground rays give white-control PSNRs of about 6.68 dB for dense and 6.37 dB for both sparse arms. At step 500, each arm's fine foreground PSNR must exceed its own control by more than 3 dB, foreground MSE must improve from initialisation and foreground opacity must exceed 0.02. Both passes need finite, nonzero density and colour gradients. A failure stops the stage. When all three pass, download `chair-s42-stage11-step500.zip` as a backup and let the notebook continue to step 2,000. Training time at the measured Lego rates of 5.4–6.2 updates/s projects to 11–13 minutes across two waves. Allow **15–20 minutes wall time** with setup and export.
4. Send **`chair-s42-stage11-review.zip`**, including on a failed health check, along with any traceback. The ZIP includes per-arm checkpoints, initial and latest diagnostics, previews, logs, exact split and schedule. In a fresh session, add the latest Stage 11 ZIP to the Chair kit and rerun. Completed checkpoints are reused. The next long run depends on this review.

## Stage 10. Export the three matched Lego videos

1. Import [the Stage 10 notebook](kaggle_stage10_lego_videos.ipynb). Attach the original Stage 0 Lego kit, `lego-dense-vanilla-s42-stage7-review.zip` and `lego-sparse-pair-s42-stage9-review.zip`. Select **GPU T4 ×2** and **Run All**. Expanded inputs also work. Expect three verified model identities at 75,000 updates and a shared 120-frame orbit.
2. Each T4 handles a disjoint set of camera/model pairs. Frames print with their arm, index and seconds per frame. Measured held-out scoring took 20.95 s/view for dense, 23.23 for sparse vanilla and 20.22 for sparse regularised. At those rates, 360 frames on two T4s project to about 64 minutes of computation. Budget **65–80 minutes wall time**, roughly 2.2–2.7 provisioned GPU-hours. The notebook prints a revised ETA from actual orbit frames.
3. Save **`lego-s42-stage10-cache.zip`** from Output during rendering. The notebook refreshes this portable snapshot about every 30 new frames and displays an embedded backup around halfway. If interrupted, attach the newest saved cache alongside the original three inputs in a fresh session. Verify that the setup reports the expected reusable frames before rendering resumes.
4. Send **`lego-s42-stage10-videos.zip`** after all three MP4s report 120 decoded frames. The archive includes the 400 × 400 H.264 movies, the lossless PNG frames, the shared orbit settings, checkpoint identities and payload hashes. Each clip is five seconds at 24 fps. If a cell stops, send the latest cache and traceback.

## Stage 9. Finish the sparse Lego pair and score it

1. Import [the Stage 9 notebook](kaggle_stage9_lego_sparse_finish_eval.ipynb). Attach the original Stage 0 Lego data/code kit, `lego-sparse-pair-s42-stage8-review.zip`, and the Stage 4 offline LPIPS kit. Select **GPU T4 ×2** and **Run All**. The restore cell should print **2,000** for each arm. The LPIPS probe should report an identical-image distance of zero.
2. Training resumes one arm per T4. Both finish their original 50,000-update decay and carry their own weights, Adam slots and RNG states through 25,000 more updates at 0.00005. The regularised arm keeps its near-camera penalty through both phases. Its frequency ramp finishes during the first phase. The first portable pair ZIP appears at 10,000, about 23 minutes of training from the restored pair. Checkpoints follow at 20k, 30k, 40k, 50k, 60k, 70k and 75k. Save the 50k and 75k ZIPs outside the running session. If an Output link fails, run `download(outputs / "lego-sparse-pair-s42-stage9-step75000.zip")` for an embedded copy, substituting the milestone you need.
3. At the concurrent rates measured in Stage 8, the remaining 73,000 updates take **3 h 33 min** for vanilla and **3 h 18 min** for regularised. The jobs overlap. Stage 7 measured 20.95 s per test camera, implying about **12 minutes** for 25 held-out and eight training images per arm if that rate holds. Allow **about four hours wall time** including setup and exports, or roughly eight provisioned T4-hours. A successful session prints 25 test rows and eight training rows per arm and saves a review ZIP.
4. Send **`lego-sparse-pair-s42-stage9-review.zip`**. It contains both final checkpoints, per-view PSNR, SSIM, AlexNet LPIPS v0.1, the full-image training-fit checks, RGB, targets, depth, opacity and the training logs. If a training check fails, send the diagnostics ZIP and traceback. If scoring is interrupted, save the `eval-interrupted.zip` if available. In a fresh T4 ×2 session attach the latest Stage 9 milestone or scoring snapshot with the two original kits and run all cells. A 75k snapshot goes straight to scoring. Completed scores in an interrupted scoring snapshot are reused. Video rendering has its own stage.

## Stage 8. Start the two sparse Lego arms on T4 ×2

1. Import [kaggle_stage8_lego_sparse_pair.ipynb](kaggle_stage8_lego_sparse_pair.ipynb) and attach the original private Stage 0 Lego kit. Select **GPU T4 ×2** and run all cells. The two independent processes each initialize a fresh model at seed 42. Each worker sees one GPU and maintains its own checkpoint and random number generator states.
2. Both arms train on cameras 26, 86, 2, 55, 75, 93, 16 and 73 with the same batch size, split, image scale, backgrounds and two-phase schedule as the 75,000-update comparison. The regularised arm also uses frequency annealing and the near-camera density penalty. This notebook stops after 2,000 prefix updates. The trainer's remaining-time display refers to the full 50,000-update first phase.
3. A pass at step 500 saves a portable pair snapshot and starts the second segment. At each review the fine field must improve on its initial foreground MSE, beat the white control by more than 3 dB and show foreground opacity above 0.02. Both fields must retain density and color gradients. Inspect the target, RGB and opacity strips. These early checks diagnose a white render. The proposal's C3 check uses full training views after fitting.
4. The final 15,000 dense updates averaged 5.92 updates/s. At that single-GPU rate, 2,000 updates would take about **5 min 38 s** on each GPU. The two jobs overlap. Setup, checkpoints and previews add time, and simultaneous sparse throughput remains to be measured. Send **`lego-sparse-pair-s42-stage8-review.zip`** even if a check fails. Also send any traceback. If interrupted, attach the latest Stage 8 ZIP along with the Stage 0 kit and rerun in a fresh notebook. The export records each arm's saved update and state independently.

## Stage 7. Finish and evaluate dense Lego in one Kaggle session

1. Import [kaggle_stage7_dense_finish_eval.ipynb](kaggle_stage7_dense_finish_eval.ipynb). Attach the original Stage 0 dataset, `lego-dense-vanilla-s42-stage6-review.zip` and the Stage 4 offline LPIPS kit. Select **GPU T4 ×2** and **Run All**. Only the first GPU is used for this dependent sequence.
2. Expect the 60,000 checkpoint to verify and the offline LPIPS probe to pass. Training holds the agreed learning rate at 0.00005 and saves portable ZIPs at 65,000, 70,000 and 75,000. At the Stage 6 rate of 6.205 updates/s the final 15,000 updates take about **40 minutes**.
3. The scoring cell then reports PSNR, SSIM and AlexNet LPIPS for all 25 held-out views. Stage 4 took **8 minutes 20 seconds** for these images. Allow roughly **50–55 minutes** for this session, excluding input uploads. Save the 75,000-step checkpoint when its link appears. Send **`lego-dense-vanilla-s42-stage7-review.zip`**, which also contains the final weights, per-view metrics, 12 example images and provenance. Send it whatever the C4 outcome. If interrupted, attach the latest Stage 7 ZIP alongside the data and LPIPS kits and run the notebook again. A completed training snapshot skips the optimizer updates.

### Priorities for the submission week

The decision after Stage 7 was to freeze the 75,000-update schedule and report the C4 miss. Run independent sparse jobs on separate T4s after checking for the former white-render failure. Move personal-scene capture and registration forward during synthetic runs. Reserve the last two days for missing outputs, matched videos, analysis and writing. Budget depends on remaining Kaggle quota and registration success. [Capture and registration guide](CAPTURE.md)

## Stage 6. Continue dense Lego from 50,000 to 60,000 updates

1. Import [kaggle_stage6_dense_continue.ipynb](kaggle_stage6_dense_continue.ipynb) in a fresh Kaggle session. Attach the original Stage 0 kit and `lego-dense-vanilla-s42-step50000.zip`. Those two inputs contain the data, renderer, parent weights, Adam state and both random number generator states.
2. Select **GPU T4 ×2**, then **Run All**. The new run starts at 50,000 on one T4. It holds the learning rate at the parent's final 0.00005 and retains the same ray sampler, architecture and training split. The trainer prints progress against the 75,000-update candidate budget. This particular notebook stops at **60,000**. At 5.285 updates/s, allow **31 min 32 s for training** plus about **3 min 4 s** for the eight training images and time for setup and downloads.
3. Save the checkpoint ZIP at 55,000 while the session is live. At 60,000 there is a second checkpoint link followed by eight training-image comparisons against Stage 5. Expect finite losses and gradients and a visible Lego foreground in the training preview. Send **`lego-dense-vanilla-s42-stage6-review.zip`**, including if the measured improvement is small or negative. If interrupted, save the newest `lego-dense-vanilla-s42-extension-*.zip` and attach it with the original two inputs in a fresh session. If a cell fails, send its traceback and latest exported checkpoint.

The 75,000 endpoint fixes the candidate budget before any new held-out evaluation. Record the extension as a protocol amendment made after the original C4 miss. Its first 50,000 updates use the original exponential decay and frequency schedule. Its final 25,000 use the constant terminal learning rate. At the measured dense rate, this would cost about 3.94 active GPU-hours per training run, or 35.5 across nine arms before evaluation and videos. The current 60,000 checkpoint is for a training-only decision about spending the final segment. A scientific comparison using this candidate requires all three arms on each scene to follow the same two-phase budget, including the original first 50,000 updates. The 50,000-step results remain a separate protocol.

## Stage 5. Diagnose the dense Lego training fit

1. In a fresh Kaggle notebook import [kaggle_stage5_dense_fit.ipynb](kaggle_stage5_dense_fit.ipynb). Attach the original private Stage 0 data/code kit, `lego-dense-vanilla-s42-stage3-20000.zip` and `lego-dense-vanilla-s42-step50000.zip`. Either ZIPs or their expanded Kaggle datasets work.
2. Select **GPU T4 ×2** and run all cells. Expect both checkpoint identities to verify. Each checkpoint renders eight 400 × 400 training images in the published eight-camera order. Stage 4 took 500.15 seconds for 25 images on one T4, implying **5 minutes 20 seconds** for these 16 renders, plus setup and export. Each progress line names a training frame and prints its PSNR and time.
3. Send **`/kaggle/working/lego-dense-vanilla-s42-train-fit.zip`**. Use the embedded download if the Output link fails. The ZIP holds paired per-camera PSNR changes, foreground/background opacity, opaque-foreground depth summaries and three sets of target and checkpoint images. For each camera it also reports what fraction of image pixels touch the target silhouette and what fraction of total squared RGB error those pixels account for. Review these alongside full-image and object PSNR and any depth values concentrated near a bound. This stage measures training fit and makes zero optimizer updates. Sparse runs await the dense-baseline decision.

## Stage 4. Score dense Lego on the held-out cameras

1. Import the [Stage 4 evaluation notebook](kaggle_stage4_dense_eval.ipynb) into a fresh Kaggle notebook. Attach the same private Stage 0 dataset and add `lego-dense-vanilla-s42-step50000.zip` and `stage4-lpips.zip` as private inputs. Keep the original checkpoint ZIP on your computer.
2. Select **GPU T4 ×2** and run the cells in order. The evaluator uses one T4, the restored step-50,000 weights, and the pretrained AlexNet LPIPS metric. Expect 25 test cameras at 400 × 400. The old T4 dense evaluation took 73 minutes for 200 images, which scales to **about 9 minutes for 25**, plus setup and export. The Stage 4 notebook measures the recovery renderer's own time. [Historical timing](FORENSICS.md)
3. Download **`/kaggle/working/lego-dense-vanilla-s42-eval.zip`** and send it for review. Expect per-view scores, mean PSNR, SSIM, AlexNet LPIPS and three examples with target, predicted RGB, depth and opacity. Review the predicted Lego against the targets and the PSNR against a white image. The proposal's 30.54–34.54 dB dense range uses a full-resolution reference. A result outside that range should be recorded and reviewed before allocating time to sparse fits. If a cell fails, send the traceback and the last view printed.

The 244 MB private LPIPS kit supplies the pinned wheel and pretrained AlexNet backbone. To rebuild it, download the `lpips==0.1.4` wheel with `python -m pip download --no-deps --only-binary=:all: lpips==0.1.4`. Run the following command from the repository root with the paths to the downloaded wheel and model. The builder checks both hashes. [Official AlexNet weights](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth)

```sh
python -m recovery.metric_kit --wheel lpips-0.1.4-py3-none-any.whl --weights alexnet-owt-7be5be79.pth --output recovery-output/stage4-lpips.zip
```

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

The [Stage 2 notebook](kaggle_stage2_dense_resume.ipynb) restored the reviewed step-500 checkpoint and continued the run to step 2,000. For later segments, estimate remaining training hours as `(50000 - checkpoint_step) / measured_steps_per_s / 3600`. Choose an earlier absolute `end` when the available session time is shorter. Allow time to save the resulting snapshot and download it. A live file in `/kaggle/working` still needs a saved copy outside the active session.

The [Stage 3 notebook](kaggle_stage3_dense_finish.ipynb) completed dense Lego training from step 2,000. If a Stage 3 download link expires, the [checkpoint export script](kaggle_salvage.py) can package surviving files from the original session or a mounted copy of its outputs. Stage 4 scored the verified final ZIP.

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

Start each new arm in its own run directory. Use the exact strings `dense-vanilla`, `sparse-vanilla` and `sparse-regularized`. The seed is 42 for all nine runs. The matched budget is 75,000 updates of 512 rays each. The first 50,000 decay from 0.0005 to 0.00005 and the last 25,000 hold at 0.00005. Uniform training pixels, Adam, the renderer and deterministic evaluation are shared. The synthetic inputs are 400 × 400 with white backgrounds and 2/6 bounds. The personal scene uses a fixed longest edge at most 400 pixels, original RGB photographs with a black residual background, and bounds from registered geometry. The sparse regularised arm adds the frequency mask and fine near-camera density penalty. [Backend and reference configs](REFERENCES.md)

The offline Stage 0 input supplies Lego. The prepared private input `recovery-output/chair-input.zip` contains the original Chair images, poses, tracked reference manifest and code. To reconstruct the portable Chair manifest from the extracted input, run `python -m recovery.data --scene chair --source data/blender/chair --output /kaggle/working/chair.json` from the kit root. Rebuild the kit locally with `python -m recovery.operator_kit --scene chair --output recovery-output/chair-input.zip` using a new output path when the original exists. For the personal scene, attach the accepted processed capture and frozen personal manifest and rebind it using `recovery.data --verify` with the new source root. [Capture and registration](CAPTURE.md)

Use one device and one run directory per arm. Stage 8 measured 5.71 and 6.14 updates/s while the two Lego arms trained concurrently. A full 75,000-update fit at those rates would use 3.4–3.7 active GPU-hours. Stage 7 measured 8 min 44 s to score 25 held-out Lego images. Measure video timing and throughput for the other scenes as they run. Include both provisioned GPUs when estimating exposure to the 60–90 GPU-hour cap. Reserve time for scored outputs and checkpoint transfers before adding repeats or stretch runs.

Once all three scored arms for a scene are present, attach or copy their `eval/metrics.json` files and compute the matched report. This can run on CPU.

```python
call("recovery.report", "--runs", work / "runs" / scene,
     "--output", work / f"{scene}-comparison.json")
```

Report the per-seed, per-view means and paired 95% intervals. A single seed quantifies view uncertainty conditional on that training run. Recovery percentages require a measured dense-minus-sparse PSNR gap of at least 3 dB with a positive interval lower bound. LPIPS also requires its own positive, separated denominator. Report the signed gains and reasons for withheld ratios. The 102-job matrix, RegNeRF, extra seeds and parameter sweeps follow the nine requested comparisons only if time remains.
