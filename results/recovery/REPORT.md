# Does regularisation recover the sparse-view NeRF gap?

Eight training runs reached 75,000 updates. On Lego, the two-term regulariser slightly widened the PSNR gap. On the photographed scene, its small PSNR gain was uncertain and perceptual quality measured by LPIPS worsened. The regularised Chair run stopped early. These results give no consistent evidence of gap recovery for the tested configuration and seed.

| Scene | Training views and method | PSNR (dB) ↑ | SSIM ↑ | LPIPS-Alex v0.1 ↓ |
| --- | --- | ---: | ---: | ---: |
| Lego | 100 vanilla | 29.645 | 0.9400 | 0.0409 |
| Lego | 8 vanilla | 25.618 | 0.8944 | 0.0788 |
| Lego | 8 regularised | 25.353 | 0.8831 | 0.0809 |
| Chair | 100 vanilla | 31.706 | 0.9641 | 0.0364 |
| Chair | 8 vanilla | 24.930 | 0.8959 | 0.1078 |
| Chair | 8 regularised | Stopped at update 128 | — | — |
| Personal | 64 vanilla | 20.045 | 0.6602 | 0.3684 |
| Personal | 8 vanilla | 16.433 | 0.5245 | 0.4732 |
| Personal | 8 regularised | 16.522 | 0.5119 | 0.5183 |

All endpoint scores are arithmetic means over held-out cameras. The synthetic scenes use 25 cameras from the published Blender test split at 400 × 400. COLMAP registered 76 of 77 photographs. Twelve registered views were held out from training and scored at 240 × 320. Their poses were estimated as part of the full registration, so this measures known-pose view synthesis. [Lego evidence](lego/seed-42/README.md) · [Chair evidence](chair/seed-42/README.md) · [Capture registration](personal/seed-42/registration.json) · [Personal per-view scores](personal/seed-42/held-out.csv)

The matched runs used the project's hierarchical PyTorch NeRF, seed 42, 512 rays per update and the same 50,000-update schedule followed by a 25,000-update hold. The regularised arms applied both frequency annealing and the near-camera fine-density penalty. The backend kept the camera and image pipeline available for diagnostics. Differences from the upstream experiments include the training schedule, ray budget, density integration and LPIPS backbone. Dense Lego missed the proposal's 30.54 dB minimum by 0.895 dB. The operator accepted 75,000 updates for the matched comparisons and recorded C4 as failed. [Upstream configurations and reference numbers](../../recovery/REFERENCES.md) · [Validation record](../../recovery/VALIDATION.md)

## How much of the PSNR gap was recovered?

Recovery is 100 × (regularised − sparse vanilla) / (dense vanilla − sparse vanilla). It requires a measured dense advantage of at least 3 dB and a paired interval for that advantage excluding zero. Intervals use 10,000 paired resamples of held-out views with seed 42. They describe variation across these test cameras, conditional on one trained model per arm.

| Scene | Dense advantage (dB) | Regularised gain (dB) | Signed recovery, 95% interval |
| --- | ---: | ---: | --- |
| Lego | 4.027 | −0.265 | −6.58% [−15.65%, −0.04%] |
| Chair | 6.777 | — | Unavailable after the regularised arm stopped |
| Personal | 3.612 | +0.088 | +2.45% [−6.16%, +13.14%] |

On the photographs, both sparse arms fitted their eight training images well (34.43 and 31.12 dB). Held-out PSNR fell to about 16.5 dB in both arms. The regularised gain interval was −0.200 to +0.419 dB. LPIPS increased by 0.0451, with a paired 95% interval of +0.0116 to +0.0809 for the increase. All three endpoints beat the 5.092 dB all-white control. [Personal model identities, means and intervals](personal/seed-42/stage17-results.json)

## Scope and media

The verified synthetic comparisons include RGB movies and representative predicted-depth images. Stage 17 produced private personal-scene target, RGB, depth and opacity examples. Stage 18 exports the three personal RGB movies. The original multi-seed 102-job design, RegNeRF comparator, extensive sweeps and standalone depth movies remain outside the measured results. Treat the CPU pilots and earlier white-render failure as diagnostic history. [Lego movies](lego/seed-42/videos/index.json) · [Chair movies](chair/seed-42/videos/index.json) · [Operator guide](../../recovery/OPERATIONS.md)
