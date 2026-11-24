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

Locally, the Stage 5 notebook resolved both real checkpoint ZIPs, verified the original data and training identities and selected the expected eight training filenames at 400 × 400. Expanded review folders resolved to the same two steps. A 20 × 20 CPU fixture exercised all 16 render calls with the actual weights, all paired summaries, the 22-member ZIP and reuse of a completed result on a second invocation.

## Dense Lego paired training fit on Kaggle

The returned Stage 5 archive SHA-256 is `080bb06e3e28f1a77942e8af84ed85c06dd574a83753fc18e4ad371a6b9a2350`. Its 22 members passed CRC. The step identities match the reviewed dense checkpoints. All 21 PNG hashes matched their records, and the three saved RGB targets matched the locally composited source images pixel for pixel. All eight paired deltas and means reproduced from the per-camera rows.

| Eight full training images at 400 × 400 | Step 20,000 | Step 50,000 |
| --- | ---: | ---: |
| Mean RGB PSNR | 27.6288 dB | 30.1145 dB |
| Mean foreground PSNR | 22.7684 dB | 25.1187 dB |
| Mean foreground opacity | 0.99257 | 0.99444 |
| Mean clear-background opacity | 0.000732 | 0.000648 |
| Mean fraction of RGB squared error in the silhouette band | 21.72% | 19.01% |

All eight images improved, by 2.293 to 2.716 dB each. The silhouette band occupied 2.49% of the pixels on average. Most RGB error remained elsewhere. The paired previews show sharper baseplate and track details, with fine studs still blurred. At both checkpoints all eight cameras reported zero qualifying opaque-foreground expected depths in the outer 5% of the [2, 6] interval. This check concerns rendered expected depth and has no ground-truth depth supervision. The 16 renders took 364.57 seconds on one Tesla T4, including 183.96 seconds for the final eight.

The evidence supports testing a longer training budget. It establishes neither the cause of the held-out shortfall nor a C4 pass. Stage 6 starts a separate child of the 50,000 checkpoint and measures the change on these same eight training views.

## Stage 6 local checks

The notebook restored the original Stage 0 kit and the real step-50,000 checkpoint. After forking, the child had a distinct fingerprint. Every model and Adam tensor, the step and both RNG states matched the parent exactly. ZIP and expanded-folder restores into fresh working directories passed. A CPU fixture using eight rays performed one real optimizer update on the restored weights at 0.00005 and advanced all 48 Adam counters to 50,001. A 20 × 20 fixture exercised the eight paired renders, JSON export, archive integrity and embedded download. The original parent checkpoint retained its recorded SHA-256. The actual 512-ray CUDA continuation and 400 × 400 step-60,000 training scores require the Kaggle run.

## Dense Lego at 60,000 updates

The returned Stage 6 archive SHA-256 is `23367e0ddaf677bf98b19d79384af53214fab23b2881ec3facb72fc7fa9b8671`. All eight payload hashes, archive CRC, run fingerprint and the original 50,000-step lineage verified. The 60,000 checkpoint SHA-256 is `8d60767ce8eded78af78421ee0a0daecf789a71763d11eb0c84c8a071cfd1ba1`. All 48 Adam counters reached 60,000 with finite weights and moments, both RNG states present, and learning rate 0.00005.

| Eight full training images | 50,000 | 60,000 | Paired change |
| --- | ---: | ---: | ---: |
| Mean RGB PSNR | 30.1145 dB | 30.3700 dB | +0.2556 dB |
| Mean foreground PSNR | 25.1187 dB | 25.4112 dB | +0.2925 dB |

Recomputed the means and all paired differences against the saved Stage 5 results. All eight RGB improvements were positive, ranging from 0.1845 to 0.3903 dB. The fine field retained nonzero density and color gradients. Its fixed training-ray foreground PSNR moved from 25.25 to 24.98 dB versus 4.41 dB for white, with 0.9998 mean foreground opacity. The 10,000 optimizer updates took 1,611.49 s in two segments (6.205 updates/s). Rendering the eight training views took 156.12 s on a T4. These measurements justify finishing the preselected budget and leave the 75,000 held-out outcome open.

The prepared Chair kit passed its ZIP member and CRC checks. All 300 image hashes match the audited scene and the renderer, trainer and scoring modules match the frozen Lego kit. Its portable data hash is `9c0224042d8d8060a2684ad70e6edfdf74f7b5acbd159aaa90142581c4e561e4`.

Stage 7 local checks restored the real 60,000 checkpoint byte for byte and loaded the pinned offline AlexNet model on CPU. Fixtures with simulated step counters exercised all three milestone exports, resuming a completed fit in a fresh directory, reuse of an expanded scored snapshot, and the final ZIP's payload hashes and embedded download.

## Dense Lego at 75,000 updates and decision after C4

| Held-out mean, 25 published test cameras at 400 × 400 | 50,000 steps | 75,000 steps |
| --- | ---: | ---: |
| PSNR | 29.108364 dB | 29.645220 dB |
| 7 × 7 SSIM | 0.933727 | 0.940038 |
| AlexNet LPIPS v0.1 | 0.046302 | 0.040947 |
| Foreground PSNR | 24.318390 dB | 24.822375 dB |
| White-control PSNR | 9.492596 dB | 9.492596 dB |

The continuation improved PSNR, SSIM and LPIPS on every paired test camera. Individual PSNR gains ranged from 0.281 to 0.737 dB. The final view scores ranged from 27.576 to 31.333 dB. The geometry and silhouettes of the loader are visible from all three example angles. Several baseplate studs and track details remain softer than their targets. [Paired per-view measurements](../results/recovery/lego/dense-budget-comparison.csv) · [Targets, RGB, opacity and predicted depth](../results/recovery/lego/dense-heldout.png)

**C4 failed.** The dense mean was 0.894780 dB below the window's 30.54 dB lower bound. On 24 November 2026 the operator chose to report this miss and use 75,000 for matched within-backend comparisons. The approved budget is the original 50,000-step exponential schedule followed by 25,000 at 0.00005. Every arm starts from its own seed-42 initialization. The reference is the measured 29.645220 dB Lego result. Sparse held-out scores and gap recovery are pending.

The Stage 7 ZIP SHA-256 is `4f6c16c5cf6dbb4b9875a179eae0b986863b7d522fefd160f999cef248c31e97`. All 24 payload hashes and ZIP CRCs passed. The checkpoint SHA-256 is `471a21d7057f2890d2f899061f2f0e6e2c341464d4625ae0f6a241cf3a3b2ccb`. The saved model fingerprint agrees with the scorer; all 48 Adam counters equal 75,000, weights and moments were finite and both RNG states were present. Recomputed each mean from the 25 frame records and compared the evaluation settings, ordered targets and LPIPS weights with Stage 4. The three exported targets agree pixel for pixel with the original composited Lego images. Scoring took 523.72 seconds on one Tesla T4. The final 15,000 updates took 2,534.59 seconds in three segments.

## Stage 8 local checks

The notebook verified the Stage 0 kit's data identity, the exact eight training cameras, and the hashes of both the original prefix config and the approved 75,000-step config. Two independent CPU subprocesses ran the frozen training CLI with smaller fields and ray batches. Each performed a real optimizer update, restored its checkpoint and performed another. The test checked separate device assignments and paths for vanilla and regularised, distinct run IDs, both previews and the failure-review export. Fixtures with simulated 500- and 2,000-step counters and health records tested the pass path, both ZIP and expanded input, payload hashes, embedded download, and reuse of completed checkpoints in a fresh session. The full-size two-T4 sparse outcomes and simultaneous throughput remain to be measured.

## Sparse Lego at 2,000 updates on two T4s

Both eight-view runs passed at 500 and 2,000 updates. The table uses the fine field on 128 fixed foreground training rays across all eight training cameras. Clear-background opacity uses separate rays from the same views. The white foreground control scored 4.16265 dB.

| Step 2,000 | Vanilla | Regularised |
| --- | ---: | ---: |
| Foreground PSNR | 16.37306 dB | 14.72782 dB |
| Foreground opacity | 0.89028 | 0.89110 |
| Clear-background opacity | 0.01727 | 0.01650 |
| Mean foreground sample density | 6.91463 | 4.56718 |
| Density-head gradient norm | 0.011979 | 0.015612 |
| Colour-head gradient norm | 0.026180 | 0.030371 |
| Updates/s while both T4s were active, final segment | 5.71365 | 6.14100 |

The previews show the yellow loader against white. The regularised prediction has a smoother silhouette and less detail at this early checkpoint. Rerendering the fixed rays on CPU from each returned model reproduced both fine foreground PSNR values within 0.00002 dB. Coarse and fine density, opacity, mean RGB, MSE and gradient norms also agreed to numerical tolerance. Training fit on eight full images and quality on the test cameras remain open.

The Stage 8 ZIP SHA-256 is `4fc7094fa39a5f43b77f45e730394644633612635b4fb7458d8f5f812a3ddc14`. All 18 members passed CRC verification and all 17 payload hashes matched the receipt. The sparse IDs, seed, data fingerprint and two-phase settings match the approved comparison. All 48 Adam counters per arm equal 2,000, all weights and moments are finite and both RNG states are present. Vanilla checkpoint SHA-256 is `87b8450e7bb81ed8231f7f0432be7192359ba6afc2f2a157982767f3974eb197`. Regularised checkpoint SHA-256 is `f2e17832c90fd27dd4ec24e25b19d48905d7fc660b85e41ecb9452e6a4592342`.

## Stage 9 local checks

The notebook restored both reviewed checkpoints byte for byte and loaded the calibrated offline AlexNet v0.1 metric on CPU. Two independent workers resumed copies of the real models, each performing one four-ray CPU optimizer update from 2,000 to 2,001. All 48 Adam counters per arm advanced. The project suite passed all 28 tests.

Fixtures with simulated milestone counters exercised eight portable checkpoint exports and the 50,000-step transition. Every model tensor, Adam slot, parameter group and RNG state matched each parent after its fork. A fresh working directory restored the final training snapshot and skipped completed updates. A fixture with synthetic score records exercised model and comparison fingerprints, camera order, image sizes, metric means, embedded ZIP downloads and reuse of scored outputs from an expanded snapshot. A separate control-flow check exercised the live 50k and 75k backup links and the interrupted-evaluation export. Full-batch sparse training through 75,000, actual held-out metrics and the eight-camera fit scores still require the two T4s.

## Lego sparse pair at 75,000 and matched held-out review

The returned Stage 9 ZIP SHA-256 is `99651cd8c61f078db42ac52c96ff60ad26c87fc311282b6d4dfbb287254ec603`. All 62 members passed CRC and all 61 payload digests matched. Verified both step-75,000 checkpoint hashes, all 48 Adam counters per arm, finite model weights and moments, both RNG states, and the constant terminal learning rate. Each checkpoint's model matches its scored model fingerprint. The Stage 8 IDs and log prefixes match, and the 50,000-step parent digests recorded in the child metadata reproduce the child IDs. The 50,000-step sparse parent checkpoints are referenced by digest in this return.

All three arms share the same schedule, target hash, test camera order and LPIPS weight fingerprint. Recomputed the target hash and 25 white-image scores from the local data. The nine example targets matched the original composited RGB targets pixel for pixel. The example RGB PNGs reproduced the reported full-precision PSNR to within 0.05 dB after quantisation. Recomputed every held-out mean and both eight-camera training means. The final 73,000 updates took 13,417.75 s for sparse vanilla and 11,923.80 s for regularised concurrently on separate T4s. The 25 held-out renders and scores took 580.82 s and 505.42 s respectively.

The approved dense reference scored 29.6452 dB. Eight-view vanilla reached 25.6183 dB and eight-view regularised reached 25.3533 dB. Their signed PSNR gap recovery was −6.58% with a paired 95% interval of −15.65% to −0.04%. Both sparse arms passed the 30 dB mean training-fit check. The dense C4 failure stands. [Metrics, per-camera measurements, images and interval interpretation](../results/recovery/lego/seed-42/README.md)

## Stage 10 local checks

Using the three original 75,000-update models on CPU, sampled an 8 × 8 ray grid from orbit frames 0 and 97 for each arm. Each first-frame PNG matched the frozen video exporter's output on the same rays byte for byte. A fresh session restored a six-frame cache. Two concurrent CPU subprocesses exercised the missing-frame assignment. A separate encoding fixture produced three 400 × 400 H.264 clips and FFmpeg decoded all 120 frames of each. The archive's 366 payload hashes, embedded download and expanded-folder restore passed. The published evaluator archive also reproduced the entire 10,000-draw Lego report from its three original metric files. All 28 project tests passed. Full-resolution orbit inference, the three final movies and actual two-T4 video throughput still require Stage 10 on Kaggle.

## Stage 10 matched Lego videos on Kaggle

The returned ZIP has SHA-256 `dd6a10bbf99a736bdafdfcff05a32fd612e1f23e39f832364e81321056fa6abc`. All 367 members passed CRC and all 366 payload hashes matched. The three descriptors match the reviewed 75,000-update run IDs, models and checkpoints. All use the same trajectory fingerprint, `a439091ea45f0bf3164b894df78c2071ed01ec746fa1481dd61612b806d7914e`.

Each H.264 clip decoded to 120 ordered RGB frames at 400 × 400, 24 fps and five seconds. Each decoded frame's closest 32 × 32 thumbnail among its 120 lossless source images had the same frame index. For each model, CPU replay on an 8 × 8 grid at orbit frames 7, 41 and 89 reproduced every sampled 8-bit channel exactly. The two T4s rendered the 360 frames in 4,180.38 s (69 min 40 s). A labelled comparison clip was encoded locally from the original PNGs and decoded to 120 frames. The full frame archive is retained in local recovery storage. [Original movies, comparison and fingerprints](../results/recovery/lego/seed-42/README.md#matched-novel-view-videos)

## Stage 11 local Chair checks

The notebook verified the prepared Chair kit and the same two-phase schedule used on Lego. On all 100 dense and the eight sparse training cameras, the centre rays passed within 0.00514 scene units of the origin, with centre depth about 4.031 inside the frozen 2–6 bounds. Each diagnostic used 128 foreground rays across eight training cameras. The white control was 6.68204 dB for the dense diagnostic and 6.36737 dB for the sparse diagnostic. The full seed-42 model had finite, nonzero density and colour gradients at initialisation for all three arms.

Reduced-width CPU fixtures each completed an actual optimizer update, checkpoint restore and second update on Chair. Simulated milestone counters and health records exercised the 500- and 2,000-update exports, failure stop, mixed-step resume, embedded download, and fresh restores from ZIPs and expanded folders. A restored failed health check prevented further training. All 28 repository tests passed. Full-batch CUDA learning on Chair and its measured throughput are pending the Stage 11 review.

## Stage 11 Chair results on Kaggle

| Training arm | Last saved update | Diagnostic update | Fine foreground PSNR | White control | Foreground opacity | Fine colour-head gradient norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense vanilla | 500 | 500 | 13.203 dB | 6.682 dB | 0.803817 | 0.077998 |
| Eight-view vanilla | 500 | 500 | 14.760 dB | 6.367 dB | 0.614505 | 0.045835 |
| Eight-view regularised | 0 | 128 | 6.367 dB | 6.367 dB | 0.001911 | 0 |

The regularised foreground RGB mean was [1, 1, 1] on the 128 fixed foreground rays. Fine mean density was 0.0005853 and the fine density-head gradient norm was 2.22 × 10⁻⁹. Coarse mean density was 0.02465, coarse foreground opacity was 0.09558, and its density and colour gradients were nonzero. The fine colour gradient triggered the stop. The step-128 health record was written before the trainer raised, so the portable regularised checkpoint is still at initialisation. The eight-view vanilla probe also scored at the white control to logged precision at 128 and recovered by 500. No arm reached 2,000.

The archive SHA-256 is `2a4ecd1a99c193c38c7f782ee4c0d38c5201ac97ffac4ed48d55c03498b7d4ed`. CRCs, all 21 payload digests, checkpoint fingerprints, Adam states and both RNG states verified. CPU replay reproduced the two step-500 fine foreground PSNRs within 0.00002 dB. The step-zero regularised model reproduced its initial diagnostics. Dense and sparse centre rays passed within 0.00514 scene units of the origin with centre depth about 4.031 inside the 2–6 bounds. Replaying the eight-view CPU pixel sampler yielded 73–125 opaque foreground pixels in every one of the first 128 batches of 512. Opaque pixels accounted for 19.45% overall. The replay's CPU RNG state at 500 matched the saved sparse vanilla state. The measurements establish the symptom; they leave open the contribution of the near-camera penalty and whether the white state would persist. [Saved diagnostic values](../results/recovery/chair/seed-42/stage11-health.json)

## Stage 12 local checks

Two CPU workers started from copies of the actual Chair regularised initialisation. Each ran a real optimiser update with eight rays and the full-size networks, restored its checkpoint and ran another update. Both copies consumed identical CPU random streams. The full two-term copy agreed with a separate run of the unmodified trainer to within 10⁻⁷ per model parameter. The frequency-only copy had a zero weighted near-loss gradient. A separately saturated colour-head fixture produced a white foreground, a zero colour-head gradient and a measurable near-loss density gradient in the new probes.

Snapshot payload hashes, embedded download, fresh restore from a ZIP and expanded folders, and reuse of completed milestones all passed. The project suite passed all 28 tests. Reproducing the original step-128 failure, checking persistence at 500 and measuring the effect of the penalty on these two full-batch trajectories still require the T4s.

## Remaining experimental stages

1. Review the regularised Chair replay and its frequency-only control at 128 and 500 updates. Both are training-only diagnostic runs. Resume the healthy vanilla checkpoints from 500 when proceeding with Chair. [Stage 12 notebook](kaggle_stage12_chair_probe.ipynb)
2. Finish Chair and the accepted personal capture at the matched 75,000-update endpoint. Record training fit, held-out scores, images, matched videos and time for each scene. Apply the same gated gap calculation to each matched comparison.
3. Register the personal capture, check the camera frustums and reprojection statistics, then freeze its pixels, poses, split and depth bounds before training. [Capture instructions](CAPTURE.md)
