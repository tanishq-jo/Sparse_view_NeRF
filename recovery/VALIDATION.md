# Recovery validation record

## Data and source

The starting commit was `71d53facf53437407fddfd47b60145b1f04b73ba`. The local tests ran on Windows, Python 3.13.15 and PyTorch 2.11.0+cpu. The evidence below concerns the recovery code, the supplied archive and the original local images. Scientific scores for the recovery backend require its own completed main checkpoints.

| Check | Local result |
| --- | --- |
| Blender data | Decoded all 600 RGBA images. Checked per-image SHA-256, calibration, ordered poses and paths against the tracked manifests, disjoint train/test images, nested sparse cameras and published test order. |
| Portable identities | Lego `bdc22418467720a3dbf5555318532caf17c250a05433f8553ae58372f11e89e2`; Chair `9c0224042d8d8060a2684ad70e6edfdf74f7b5acbd159aaa90142581c4e561e4`. Mount paths can change after re-verification. |
| Camera alignment | The center rays for all 100 dense Lego training cameras passed within 0.0052 scene units of the origin at 400². Their closest camera-axis depth was 4.0311, within [2, 6]. |
| Old training failure | Replayed all three final checkpoints on identical foreground/background probes. Both sparse fine fields predicted white with zero opacity and zero head gradients. Their coarse fields and the dense fine field had nonzero gradients. Independently scored white on all 200 test images. [Details and replay command](FORENSICS.md) |
| Recovery field and regularisation | Checked finite integration analytically, positive gradients for negative density logits, frequency masking and equality when fully open. Verified the near penalty value and gradient, including normalization across all fine samples. |
| Training resume | Restored a partial CPU run using only the checkpoint and run metadata into a different directory with remounted verified images. Subsequent model parameters exactly matched uninterrupted training. Incomplete scoring, changed images and mismatched arms were rejected. |
| Personal capture tooling | Tested seeded, disjoint splits and the quality gates on fixtures, including failure cases. The project still needs the operator's photos and a real COLMAP registration. |
| Metrics and comparison | Tested SSIM against an independent window calculation, renderer chunk order, paired bootstrap alignment, recovery identities, the denominator gates and rejection of missing or mismatched frames. |

All **28 tests passed** with `python -m unittest discover -s tests -v`, including the 14 legacy tests.

## Real Lego CPU checks

The one-camera development check used 100 × 100 pixels, a width-64 four-layer field, 12 coarse plus 12 importance samples, and 96 rays per update for 256 steps. Both trials used seed 42 and the same fixed 16 foreground probe pixels on training frame 0. The all-white foreground control scored 3.53 dB. Their initial fine foreground MSE was 0.10109.

| Diagnostic batch | Final foreground MSE | Foreground PSNR | Foreground opacity | Learning check |
| --- | ---: | ---: | ---: | --- |
| Uniform pixels | 0.18352 | 7.36 dB | 0.671 | Failed the improvement over initialization |
| Half foreground, half background | 0.03259 | 14.87 dB | 0.819 | Passed |

The balanced trial retained nonzero fine-head gradient norms of 0.02346 for density and 0.02840 for color. Its coarse foreground PSNR was 15.01 dB. The preview shows a blurred object silhouette and substantial opacity on the background. This establishes that the small model responded to foreground supervision. The full-width Kaggle diagnostic uses the same balanced sampling rule and pass criteria. All main arms use uniform pixels.

Three separate two-step width-16 CPU fits at 40 × 40 exercised checkpoint loading, the final evaluator and report commands. Each scored the same 25 real test views with PSNR, SSIM and pretrained AlexNet LPIPS. Both sparse arms additionally scored their eight training views. Every arm exported three sets of target, RGB, 16-bit depth and opacity PNGs and a matched three-frame H.264 smoke-test video. All three stored identical evaluation, trajectory and LPIPS-weight fingerprints. The video stage replaced a corrupted frame byte-for-byte while preserving an intact cached frame. The paired report correctly withheld both recovery percentages. These miniature fits test the pipeline only.

The 127,351,536-byte Stage 0 input ZIP passed CRC and membership checks for all 325 entries. Its SHA-256 is `40cefc59fe1b53ef2c65cfae4027fc8b433a6ed36a06c0d61621b90e82bb53b8`. The notebook was checked against both zipped and expanded input layouts, including the actual freeze cell in a temporary mount and creation of a readable review ZIP.

## Kaggle Stage 0 result

The operator's full-width diagnostic passed on one Tesla T4 with PyTorch 2.11.0+cu130. The review ZIP passed CRC checks. The step-256 checkpoint matched the run metadata and contained the model, Adam and CUDA RNG states. The source fingerprint matched the published recovery code and the local input kit. The frozen Lego data fingerprint matched the local validation. A CPU replay of the uploaded weights on the same local training pixels reproduced both foreground PSNRs within 0.00002 dB and the foreground/background opacity means within 0.0000001.

| Fixed training probe | Coarse field | Fine field |
| --- | ---: | ---: |
| Initial foreground PSNR | 7.33 dB | 7.49 dB |
| Final foreground PSNR | 16.66 dB | 17.11 dB |
| Final foreground opacity | 0.974 | 0.930 |
| Final background opacity | 0.999 | 0.452 |
| Final density-head gradient norm | 0.1140 | 0.1308 |
| Final color-head gradient norm | 0.0946 | 0.0701 |

On 16 fixed foreground pixels of one training view, the white control scored 2.22 dB. Fine foreground MSE fell from 0.17824 to 0.01944. The preview contains a recognizable, blurred Lego silhouette. Residual density on background rays, especially in the coarse field, merits inspection once more cameras constrain geometry.

The trainer logged 42.94 seconds for 256 updates, or 5.9623 updates/s. Its timer starts after initialization and data caching. This rate projects about 84 seconds for the first 500 dense updates, plus startup, image caching, preview and export. Actual dense throughput and peak GPU memory have yet to be measured. The Stage 0 review ZIP SHA-256 is `92f26aeddffd8c26092cb8eceb2bfa8562d45d1bced8af3c57712dff58fdc63e`.

## Remaining experimental stages

1. Review the first 500 updates of the 100-view dense Lego run and verify a checkpoint handoff to a fresh Kaggle session. Then measure final held-out performance on the 25-view protocol. The proposal's dense Lego sanity window is 30.54–34.54 dB around the original full-resolution 32.54 dB reference. The different evaluation protocol limits that comparison.
2. Complete matched held-out metrics, RGB/depth evidence and videos for the nine main fits. Record the eight-view training PSNR fit check, budget and actual throughput. Compute scene-specific recovery only for supported gaps. The archived dense score and the two failed sparse scores describe the previous backend.
3. Register the personal capture, check the camera frustums and reprojection statistics, then freeze its pixels, poses, split and depth bounds before training. [Capture instructions](CAPTURE.md)
