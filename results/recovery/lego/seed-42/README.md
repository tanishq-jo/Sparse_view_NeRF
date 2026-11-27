# Lego held-out results

The 100-view dense and two eight-view sparse runs used seed 42, the same recovery PyTorch renderer, 75,000 updates with 512 rays each, and deterministic scoring on 25 published test cameras at 400 × 400. The regularised arm used both frequency annealing and the near-camera density penalty.

| Arm | Held-out PSNR (dB) ↑ | 7 × 7 SSIM ↑ | AlexNet LPIPS v0.1 ↓ | Foreground PSNR (dB) ↑ |
| --- | ---: | ---: | ---: | ---: |
| 100-view vanilla | 29.6452 | 0.94004 | 0.04095 | 24.8224 |
| 8-view vanilla | 25.6183 | 0.89437 | 0.07882 | 21.6159 |
| 8-view regularised | 25.3533 | 0.88310 | 0.08090 | 20.9100 |

The all-white prediction scored 9.4926 dB on the same full images. The two sparse models averaged 32.5261 and 30.2863 dB respectively across their eight full training images, passing the 30 dB mean-fit check. Eight of eight vanilla training images and four of eight regularised training images individually reached 30 dB. The dense reference missed the proposal's 30.54 dB lower bound by 0.8948 dB. The operator retained it for matched comparisons. [Held-out rows](held-out.csv) · [Training rows](training-fit.csv) · [Protocol and C4 decision](../../../../recovery/VALIDATION.md)

## Did the regulariser recover the sparse-view gap?

| Quantity | PSNR | AlexNet LPIPS |
| --- | ---: | ---: |
| Dense advantage over sparse vanilla | 4.0270 dB [3.1926, 4.8867] | 0.03788 [0.02296, 0.05415] |
| Regularised gain over sparse vanilla (positive favours regularised) | −0.2649 dB [−0.5107, −0.0019] | −0.00207 [−0.00959, 0.00683] |
| Signed gap recovery | −6.58% [−15.65%, −0.04%] | −5.47% [−39.64%, 13.15%] |

The dense advantage exceeded 3 dB, and its paired interval excluded zero. The regulariser widened the measured PSNR gap. Its test PSNR was lower on 18 of the 25 cameras, and the gain interval only narrowly excludes zero. The LPIPS gain interval spans zero. Intervals are percentile intervals from 10,000 paired resamples of test cameras, conditional on this single seed. [Full precision means, paired intervals and settings](comparison.json)

## Held-out RGB and predicted depth

The examples use the first, middle and last test cameras in the frozen evaluation order. On camera 192, the vanilla sparse render has stray structure to the upper left of the loader. The regularised render reduces that artifact and improves this camera's PSNR from 22.494 to 23.571 dB. Camera 96 goes the other way, from 30.226 to 29.147 dB. All depth maps use camera-axis depth on the same 2–6 scale. [RGB and depth comparison](held-out-comparison.png) · [Original scores, targets, RGB, opacity and 16-bit depth PNGs](held-out-examples.zip)

To reproduce the table and intervals on CPU, extract the example archive into `lego-eval` and run `python -m recovery.report --runs lego-eval --output lego-comparison.json` from the repository root. Each arm's `eval/metrics.json` is the original scorer output.

## Matched novel-view videos

Each 400 × 400 clip follows the same 120-position orbit in five seconds at 24 fps. The comparison places the three original PNG sequences side by side, aligned by camera index. Around zero-based frame 41, stray structure is visible to the right of the loader in the eight-view vanilla render and is less apparent in the regularised render. The held-out table gives the quantitative comparison across the test cameras. [Watch side by side](videos/comparison.mp4)

Original H.264 exports and their checkpoint identities are available individually. [100-view vanilla](videos/dense-vanilla.mp4) · [Eight-view vanilla](videos/sparse-vanilla.mp4) · [Eight-view regularised](videos/sparse-regularized.mp4) · [Movie and checkpoint hashes](videos/index.json)
