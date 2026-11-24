# Photograph and register the personal scene

You can collect photographs while the synthetic scenes train. Choose a matte, textured object about 20–40 cm across, on a nonwhite surface with a close backdrop. A little fixed texture on the support helps COLMAP match features. Keep the subject, background and lighting still. Walk the camera around the object. Turning the object on a turntable while the room stays still gives COLMAP incompatible motion. Keep people and private details outside the frame.

1. Capture two candidate subjects, each in its own folder. Aim for 60–80 sharp overlapping JPG or PNG stills per subject. Three arcs at low, middle and high elevations with roughly 20–25 photos per arc work for an object you can walk around. Keep visible overlap with adjacent arcs. A textured detail should be easy to recognize in several neighboring photos.
2. Use one lens, orientation, crop and resolution. Lock focus, exposure and white balance. Avoid flash, changing shadows, digital zoom and portrait mode. Inspect the full-resolution files for motion blur. Replace blurred photographs before registration. Keep an unmodified folder of the exact stills passed to processing. The freeze command counts original JPG, JPEG and PNG files directly inside that folder.
3. On a machine with Nerfstudio, COLMAP and FFmpeg installed, register all views for each candidate. Commands below use a shell on Linux or Kaggle and one candidate named A. Substitute B for the second candidate. Keep the input and processed folders separate.

```bash
ns-process-data images --data photos/A --output-dir prepared/A
colmap model_analyzer --path prepared/A/colmap/sparse/0
mkdir -p prepared/A/colmap-text
colmap model_converter --input_path prepared/A/colmap/sparse/0 \
  --output_path prepared/A/colmap-text --output_type TXT
python -m recovery.data --scene personal --source prepared/A --raw photos/A \
  --points prepared/A/colmap-text/points3D.txt --test-count 12 --split-seed 42 \
  --output recovery-work/candidates/A.json
```

Use the sparse model from which Nerfstudio generated this candidate's `transforms.json`. Standard `ns-process-data images` writes `colmap/sparse/0`. If your registration outputs several disconnected models, inspect the largest and its matching transforms. View its camera frustums and sparse points. Look for coherent, inward-facing arcs and features on the object. Two disconnected halves or a cluster of cameras floating outside the orbit warrants recapture. [Nerfstudio custom-data guide](https://docs.nerf.studio/quickstart/custom_dataset.html) · [COLMAP command-line tools](https://colmap.github.io/cli.html)

The acceptance test requires at least 95% of submitted stills registered and mean per-point reprojection error below 1 pixel. Send the original count, `model_analyzer` output, the freeze summary for each candidate, and one screenshot of the camera orbit. Choose on registration quality before any training. Prefer the larger registered fraction, then the smaller reprojection error. If neither candidate passes, photograph more intermediate positions around the poorly connected arc and rerun registration.

The manifest records an eight-camera farthest-point selection seeded at 42, then chooses 12 held-out cameras from the remaining positions using the same deterministic procedure. All non-test cameras make up the dense training set. Camera normalization uses the dense training poses. Near and far are the 5th and 95th percentiles of positive depths of registered sparse points observed by those cameras, scaled to the normalized scene. Each depth is measured along the optical axis, matching the renderer's unnormalized OpenGL rays. Registration uses all photos, so report the personal result as view synthesis with known poses. The capture has RGB targets and no ground-truth foreground matte or depth. [COLMAP point and track format](https://colmap.github.io/legacy/3.13/format.html) · [Nerfstudio coordinate conventions](https://docs.nerf.studio/quickstart/data_conventions.html)

Preserve the chosen raw stills, processed images, `transforms.json`, the COLMAP model and text export, the candidate summaries, and the chosen manifest. In a fresh session, rebind the saved manifest to the mounted scene directory with `python -m recovery.data --verify saved/A.json --source /kaggle/input/my-capture/prepared/A --output recovery-work/personal.json`. This verifies the image files and preserves the original camera matrices, bounds, split and hash exactly. Keep images in their original orientation, crop and resolution throughout.
