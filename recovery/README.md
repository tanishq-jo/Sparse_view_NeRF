# Sparse-view NeRF recovery

The three Lego runs have verified metrics, representative RGB and depth images, and matched 120-frame videos. Dense vanilla reached **29.645 dB**, eight-view vanilla **25.618 dB** and eight-view regularised **25.353 dB**. The dense reference missed C4 by 0.895 dB. On Chair, both vanilla arms passed the 500-update training check. The full regulariser reproduced the white render at 128. With and without its near-camera penalty, the fine model still predicted white on the foreground at 500. Stage 13 advances the two baselines and tests the initial visibility of Fourier bands. [Lego measurements and videos](../results/recovery/lego/seed-42/README.md) · [Chair replay measurements](../results/recovery/chair/seed-42/stage12-probe.json) · [Stage 13 notebook](kaggle_stage13_chair_continue.ipynb) · [Operator guide](OPERATIONS.md)

Stage 0 uses one Lego training camera, the main model architecture, 512 rays per update, and 256 updates on one GPU. It checks whether both the coarse and fine fields receive gradients and whether the fine field learns a foreground signal. Held-out images are reserved for scoring. The old Kaggle archive provides a working dense control and two failed sparse controls. [Checkpoint investigation](FORENSICS.md)

## Stage 0 on Kaggle

1. Upload **stage0-input.zip** as a private Kaggle dataset and attach it to a new notebook. The ZIP contains the validated local Lego images, pose metadata, and the code. The notebook accepts an extracted dataset or the intact ZIP. The Kaggle run needs the preinstalled PyTorch, NumPy and Pillow. Internet access is optional for this stage.
2. Import [kaggle_stage0.ipynb](kaggle_stage0.ipynb), select GPU T4 ×2, and run the three code cells in order. The first records the GPU model and PyTorch version. The second verifies the image bytes, calibrations, exact train/test identities and camera split. Expect **100 dense / 8 sparse / 25 test**, with 300 source images checked. The third runs only the one-camera diagnostic on device 0.
3. Watch the reported steps/s. The archived T4 completed comparable training updates at **5.80–5.87 steps/s**. At that rate 256 updates would take **about 44 seconds** of training. Setup, imports, data verification, checkpoint writes and preview rendering take additional time. The current run prints its own timing. Use its measured rate when deciding on the first dense training window.
4. Download **/kaggle/working/stage0-review.zip** from Output after the diagnostic. Send the ZIP for review, including on failure. It contains the manifest, hardware info, log, checkpoint, initial/final diagnostics and, when the final update completes, the gate decision and image strip. If setup or data verification stops earlier, send the cell traceback and hardware line.

The strip shows ground truth, prediction and accumulated opacity. Inspect the object in the middle and right panels. Both fields should have finite nonzero density-head and color-head gradient norms on foreground rays. The diagnostic draws equal numbers of object and background pixels from this one camera. The automated learning check requires final fine foreground MSE below initialization, PSNR more than 3 dB above the all-white control on the same pixels, and foreground opacity above 0.02. This is a training-view check. Dense Lego geometry and held-out quality require a separate run.

To rebuild the input in the original workspace, run this from its repository root with the project Python. Choose a new ZIP filename if an earlier build already exists.

```powershell
.\.venv\Scripts\python.exe -m recovery.operator_kit --output recovery-output/stage0-input.zip
```

## Experiment and evidence

| Deliverable | Where to find it |
| --- | --- |
| Archived checkpoints, white-control replay and CPU pilot provenance | [Forensics](FORENSICS.md) |
| Backend choice, upstream Blender settings, split and regularizer equations | [Numerical references](REFERENCES.md) |
| Local checks, measured limits and remaining GPU work | [Validation record](VALIDATION.md) |
| Stage boundaries, checkpoint restore, held-out scores, videos and paired reporting | [Training operator guide](OPERATIONS.md) |
| Photographing, registering and freezing a personal scene | [Capture guide](CAPTURE.md) |

The main matrix has nine runs at seed 42: dense vanilla, eight-view vanilla and eight-view regularised on each of Lego, Chair and the registered personal scene. Every arm uses the same recovery PyTorch renderer, positive softplus density, finite ray integration, optimizer, ray budget and final evaluation procedure. The regularised arm adds frequency annealing and near-camera fine-density regularisation. This is an adaptation of the authors' released Blender config. RegNeRF, the 102-job matrix and additional seeds or sweeps are stretch work.

The measured 100-view dense Lego checkpoint supplies the within-backend reference. Its C4 failure and the budget amendment belong in the report. A single seed supports a paired interval over test views conditional on that seed. A dense-to-sparse PSNR gap below 3 dB, or a gap interval containing zero, leaves recovery unreported.
