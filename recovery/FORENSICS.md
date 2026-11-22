# What happened to the previous Lego run

The archived dense Lego checkpoint worked. Both eight-view checkpoints learned a coarse field, then returned white from their fine field on the probed foreground pixels. The fine density head used a ReLU. Its input was negative at every probed fine sample in both sparse checkpoints, which made the foreground RGB loss gradient zero at both fine output heads. The checkpoint captures the final state. It cannot date the first bad update or establish what drove the preactivations below zero.

## Archived scores

These are seed-42 **legacy** measurements at 400 × 400, 512 rays per update and 50,000 steps on one Tesla T4. Each mean covers all 200 test views. LPIPS used AlexNet v0.1. Both sparse arms have identical per-view PSNR and LPIPS arrays.

| Lego arm | Views | PSNR (dB) | LPIPS |
| --- | ---: | ---: | ---: |
| Dense vanilla | 100 | 28.925725 | 0.048138 |
| Sparse vanilla | 8 | 9.490950 | 0.540947 |
| Sparse frequency annealing | 8 | 9.490950 | 0.540947 |

An independent CPU replay scored a solid white image against each of the 200 local targets. Its mean was **9.490949817 dB / 0.540947479 AlexNet LPIPS**. Across every view, the largest absolute difference from the archived sparse scores was 1.78 × 10⁻¹⁵ dB for PSNR and 1.20 × 10⁻⁷ for LPIPS. The checkpoint probes below check the predicted pixels and gradients directly.

Selecting every eighth archived test frame retrospectively gives 25 views (0 through 192). The dense mean on those frames was **28.946953 dB and 0.048942 LPIPS**, 1.593 dB below the proposal's 30.54 dB reference. Both sparse checkpoint means on the same frames were 9.492596 dB and 0.541166 LPIPS. This selection reuses archived measurements and predates the recovery backend and its protocol.

The archive contains all three 50,000-step checkpoints, optimizer states, configs and complete score files. Its 48-byte temporary sparse-regularized MP4 is incomplete. `OldLogs.txt` records frame 67/120 at 43,201.1 seconds, after evaluation finished. The successful dense score is a reference point for recovering useful scene geometry with the new protocol.

## CPU checkpoint probe

The probe uses identical rays for every checkpoint. Four foreground and four background pixels come from each of the eight archived training views and the first and last test views. Foreground requires alpha ≥ 250/255 and mean white-composited target RGB < 0.8. Background requires zero alpha. These 80 stratified rays diagnose the renderer and are unsuitable for estimating held-out PSNR.

| Arm | Field | Positive density samples | Mean foreground opacity | Foreground RGB MSE | Density-head gradient norm | Color-head gradient norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dense | coarse | 22.64% | 1.00000 | 0.00361 | 0.00537 | 0.01614 |
| Dense | fine | 30.87% | 0.99982 | 0.00158 | 0.00054 | 0.01515 |
| Eight-view vanilla | coarse | 10.16% | 0.99978 | 0.00573 | 0.01009 | 0.01351 |
| Eight-view vanilla | fine | 0% | 0 | 0.42514 | 0 | 0 |
| Eight-view annealed | coarse | 38.36% | 1.00000 | 0.00472 | 0.00054 | 0.01844 |
| Eight-view annealed | fine | 0% | 0 | 0.42514 | 0 | 0 |

The average foreground target RGB was (0.5131, 0.4303, 0.2624). Both sparse fine fields predicted exactly (1, 1, 1) for every sampled pixel, including that foreground. Across 15,360 fine samples per arm, the pre-ReLU ranges were [−0.2473, −0.0316] for vanilla and [−1.1980, −0.5680] for annealing. Backpropagation used the sum of coarse and fine foreground RGB MSEs, with deterministic samples and the final encoding schedule. The reported gradients are weight-gradient L2 norms for each head.

The ten center rays had cosines above 0.999999 with the vector toward the scene origin. The camera-axis depth of closest approach was 4.03112, inside the [2, 6] marching bounds. Dense fine foreground opacity was 0.99982 and background opacity was 0.00097. These checks and the dense result support the coordinate convention on the sampled cameras. The sparse checkpoints' coarse fields retained both color and opacity signal on the same rays.

## Runtime recorded on the old stack

Dense completed its first 100 updates at 6.23 steps/s and finished at 5.80 steps/s. Sparse frequency annealing completed its first 100 at 5.78 steps/s and finished at 5.87 steps/s. Dense training took 2 h 24 min, followed by 1 h 13 min to evaluate 200 images. The sparse-vanilla 120-frame video added about 44 min. New-backend time estimates require measuring its own batch size, image scale, model and GPU.

## CPU pilot provenance

The six local 80 × 80, 1,000-step pilots have complete metrics. Both the committed summary and local `scratch/*-pilot.log` give sparse-vanilla PSNR of 12.134 dB on Lego and 15.853 dB on Chair. The 12.57 and 16.83 dB figures from the project handoff appear to describe another export or run. The local files do not establish their provenance. All of these numbers belong in the pilot record.

## Reproduce

From the repository root with the legacy dependencies installed, run this PowerShell command. The script reads the ZIP and local dataset, checks the local renderer, loader, evaluator and manifest against the archive, and loads only tensor checkpoint state with `weights_only=True`. It reads named members directly and never executes the archived checkout. The optional LPIPS calculation uses the local AlexNet weights in this workspace. Omitting `--lpips` retains the full PSNR comparison and model probes.

```powershell
$env:TORCH_HOME = (Resolve-Path scratch/torch).Path
.\.venv\Scripts\python -m recovery.forensics --archive "$env:USERPROFILE\Downloads\results.zip" --lpips
```

To retain machine-readable results, add `--output scratch/recovery-evidence/probe.json` and select a new filename for each replay.

| Input | Identifier |
| --- | --- |
| Archived checkout | `71d53facf53437407fddfd47b60145b1f04b73ba` |
| results.zip SHA-256 | `e36b5dc0bad5feef6907987b4b0711f58f92ee55de84817bf27b5e7d18189c8f` |
| OldLogs.txt SHA-256 | `dd5c5c34b7d2d107751515b09cd6e5793ae4ee9841f87a79e50fb57dc9214958` |
