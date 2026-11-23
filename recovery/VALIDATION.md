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

The trainer logged 42.94 seconds for 256 updates, or 5.9623 updates/s. Its timer starts after initialization and data caching. This rate projected about 84 seconds for the first 500 dense updates. The Stage 0 review ZIP SHA-256 is `92f26aeddffd8c26092cb8eceb2bfa8562d45d1bced8af3c57712dff58fdc63e`.

## Kaggle dense Lego at step 500

The first 500 updates used 100 training cameras, 400 × 400 images, uniform pixel sampling, seed 42, and the scheduled learning rate for the 50,000-step dense vanilla run. The training loop measured **81.68 seconds**, or **6.1214 updates/s**, on one Tesla T4. The notebook recorded 94.7 seconds through training and preview, before writing the handoff ZIP.

| Fixed training probe across eight cameras | Coarse field | Fine field |
| --- | ---: | ---: |
| Foreground PSNR at initialization | 9.37 dB | 9.49 dB |
| Foreground PSNR at step 500 | 14.70 dB | 14.51 dB |
| Foreground opacity at step 500 | 0.860 | 0.855 |
| Background opacity at step 500 | 0.095 | 0.034 |
| Density-head gradient norm at step 500 | 0.03089 | 0.02359 |
| Color-head gradient norm at step 500 | 0.05076 | 0.03185 |

These statistics use 128 fixed foreground and 128 fixed background rays in total. The white control scored 4.41 dB on the foreground rays. The preview shows the object's outline and yellow body against a white background, with fine detail still blurred. Held-out PSNR, SSIM and LPIPS remain unmeasured for this run.

The ZIP passed CRC checks, source and data fingerprints matched, and all 48 Adam parameter states contained finite moments at step 500. Both CPU and CUDA RNG states were present. A CPU replay of the uploaded weights reproduced both foreground PSNRs within 0.00003 dB and the reported opacity means within 0.0000001. Run ID `08ff8d2fceda43759554aa6470054853`; checkpoint SHA-256 `0730a3442af623680fa2fe705a5e1dabbe613231238f3840ccc075cb9df66e5b`.

The Stage 2 restore cell was exercised locally using the uploaded snapshot, the original Lego images and a new writable directory. It verified the data and code, loaded the expected run at step 500 and preserved the checkpoint bytes. Local fixtures also exercised the notebook's artifact export after a completed or interrupted training process.

## Kaggle dense Lego at step 2,000

The CUDA resume log starts at step 500 and reproduces the earlier fine probe at 14.51 dB. The new snapshot identifies the exact source checkpoint. It preserves the run metadata, the entire earlier training log and the step-500 preview. The model weights and all 48 Adam moment pairs are finite. All optimizer counters reached 2,000, and both RNG states are present.

| Training probe on the same eight cameras | Step 500 | Step 2,000 |
| --- | ---: | ---: |
| Coarse foreground PSNR | 14.70 dB | 15.63 dB |
| Fine foreground PSNR | 14.51 dB | 17.12 dB |
| Fine foreground opacity | 0.8550 | 0.9340 |
| Fine background opacity | 0.03416 | 0.00668 |

Both fields retained density and color gradients. The preview shows the yellow body and dark lower chassis, with a distinct silhouette against the background. Replaying the uploaded checkpoint on CPU reproduced the coarse and fine foreground PSNRs within 0.00001 dB and foreground/background opacity means within 0.000001. These are training-view diagnostics. Held-out PSNR, SSIM and AlexNet LPIPS are measured after training.

The 1,500 updates took 232.60 seconds, **6.4489 updates/s** on one T4. The cell recorded 246.76 seconds through training and preview. At that rate the remaining 48,000 updates take about **2 hours 4 minutes**. The step-2,000 checkpoint SHA-256 is `4babbd7b23849a9d98c578753bb273dc6b943b9e2382dd9955ee271ae23b3a8e`.

The Stage 3 notebook restored these checkpoint bytes into a fresh local working directory and verified all 300 Lego source images. Its snapshot logic was exercised with simulated progress through the four intermediate milestones, a custom stop, an interruption and final export.

## Dense Lego at 50,000 updates

The recovered Stage 3 ZIP passed CRC verification and identifies run `08ff8d2fceda43759554aa6470054853`. The final log contains the complete saved history through step 20,000. The run metadata, training manifest and earlier previews match that snapshot byte for byte. The ZIP SHA-256 is `a24c4b5820242093f1fa9d54ab953762bc1e878c46c19234b63e9e750bda4b52`. The checkpoint SHA-256 is `a77230d4e17b3769445cf05e26b2bda7f094bf078418ff7d6366d906e19da8be`. All 48 Adam counters reached 50,000. The model weights and optimizer moments are finite. The last training segment logged 5.285 updates/s on one T4.

| Fixed training rays | Step 20,000 | Step 50,000 |
| --- | ---: | ---: |
| Fine foreground PSNR | 22.67 dB | 25.25 dB |
| White control on the same foreground | 4.41 dB | 4.41 dB |
| Fine foreground opacity | 0.99879 | 0.99977 |
| Fine background opacity | 0.00046 | 0.00079 |

A CPU replay of the final checkpoint using the local Lego images reproduced the stored training diagnostics for both fields. The fine preview resolves the yellow body, dark chassis and silhouette. The coarse field still has high background opacity (0.7334). The fine field supplies the outputs for held-out scoring. The recovered archive contains training evidence only.

The local 400 × 400 loader gives **9.492596 dB** mean per-view PSNR for a white prediction across the 25 frozen test cameras. The Stage 4 LPIPS ZIP passed its member, SHA-256 and CRC checks. A fresh offline install on CPU produced a pretrained AlexNet v0.1 fingerprint of `913e6288d265ca661f26fe0426b840bc69bafcf9fcda0fb2c99586891562ea9e`. LPIPS was zero on identical inputs and 0.577986 on the changed test input.

The Stage 4 notebook restored the actual final checkpoint with the original Stage 0 kit, verified all 300 image files and checked the 100/8/25 split in a local CPU session. Both intact ZIPs and expanded Kaggle-style inputs passed, including a mount containing an older step-20,000 review. A scoring fixture exercised progress timing, same-session reuse and export. The downloaded bytes matched the 15-member review ZIP, including its 12 PNGs. These local checks took 71 seconds.

## Dense Lego held-out result and C4 review

| Measurement | Stage 4 result |
| --- | ---: |
| Mean per-view PSNR | 29.108364 dB |
| Mean 7 × 7 SSIM | 0.933727 |
| Mean AlexNet LPIPS v0.1 | 0.046302 |
| White-control PSNR | 9.492596 dB |
| Mean foreground PSNR / white foreground PSNR | 24.318390 / 4.162358 dB |
| Mean foreground opacity | 0.993866 |
| Per-view PSNR range | 26.9923 dB (`r_8`) to 30.7608 dB (`r_16`) |
| Time on one Tesla T4 | 500.153 s for 25 views, 20.006 s/view |

Evaluation ZIP SHA-256 `5969cea504dad05e20f37b6c82339c9b3a8e57641eb46a0248577f0f1fba008c`. Recomputed the archive CRC, each of 14 payload hashes, metric means and individual white scores. The 25 target images match the local source pipeline and the three exported targets match pixel for pixel. The predicted body, wheels and baseplate are visible in all three examples. The studs and track textures are softer than the targets. The silhouettes have defined edges and the depth maps distinguish the loader from the baseplate. These images provide qualitative depth evidence.

LPIPS 0.1.4 called torchvision's deprecated `pretrained` argument. Torchvision 0.26.0+cu130 used AlexNet ImageNet V1. The complete LPIPS network fingerprint matched the offline CPU reference exactly. The identical-input distance was zero and the changed-input distance was 0.57798612. The archived held-out score is valid.

**C4 missed by 1.431636 dB.** The proposal cites 32.54 dB at 800 × 800 and sets a 30.54–34.54 dB window. Stage 4 used the frozen 400 × 400 protocol. All 25 views beat the white control by at least 16.75 dB. The source of the remaining shortfall is unresolved. Keep this result and protocol when reporting the experiment.

An independent check used all 100 raw training poses and reconstructed 800 center and corner rays from the published horizontal field of view. At 400 × 400 the focal length is 555.555516 pixels. Maximum direction-component error was below 7.81 × 10⁻⁸ and ray origins matched exactly. Scene scale was one and the center-ray depth was about 4.031 between the frozen bounds of 2 and 6. Independent white compositing and downsampling matched all 16 million training RGB pixels exactly. Checking the scene center alone cannot establish that every foreground surface is inside the bounds. Stage 5 measures predicted depth on opaque foreground pixels.

Using the actual step-50,000 weights, a separate CPU check sampled 512 distinct pixels uniformly from each of the 100 training views at 400 × 400 (51,200 rays, NumPy PCG64 seed 20261123). The mean of the 100 per-camera sampled PSNRs was **30.02283 dB**, with a **9.40639 dB** white control and **25.11448 dB** on sampled foreground. Mean foreground opacity was 0.994865. The sampling index SHA-256 was `5f2ee24966fc255e5cc3214b55855f66d6a006996f74c1ed50daac422d2c1201`. This estimates the final dense model's training fit. C3 applies to a model trained on eight views and requires scoring its complete training images.

Locally, the Stage 5 notebook resolved both real checkpoint ZIPs, verified the original data and training identities and selected the expected eight training filenames at 400 × 400. Expanded review folders resolved to the same two steps. A 20 × 20 CPU fixture exercised all 16 render calls with the actual weights, all paired summaries, the 22-member ZIP and reuse of a completed result on a second invocation. The fixture validates the code path. The full-image 400 × 400 paired comparison is the pending T4 stage.

## Remaining experimental stages

1. Compare the 20,000- and 50,000-step checkpoints on identical training images. Review the paired fit and opaque-foreground depth before deciding on the dense baseline and the shared training budget. [Stage 5 notebook](kaggle_stage5_dense_fit.ipynb)
2. Complete matched held-out metrics, RGB/depth evidence and videos for the nine main fits. Record the eight-view training PSNR fit check, budget and actual throughput. Compute scene-specific recovery only for supported gaps. The archived dense score and the two failed sparse scores describe the previous backend.
3. Register the personal capture, check the camera frustums and reprojection statistics, then freeze its pixels, poses, split and depth bounds before training. [Capture instructions](CAPTURE.md)
