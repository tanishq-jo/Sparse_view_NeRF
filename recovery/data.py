"""Freeze the published Blender splits or a registered personal scene. Recheck the same bytes after relocation."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from sparse_nerf.prepare import _center_capture, _coverage_subset, _load_frames


# FreeNeRF's NeRF (Blender) split: legacy RandomState(0).choice(100, 8, replace=False).
SPARSE = [26, 86, 2, 55, 75, 93, 16, 73]
REPO = Path(__file__).resolve().parents[1]
# SHA-256 of the ordered, concatenated ASCII file digests of the validated local copies.
PIXEL_FILES = {
    "lego": "ffcb54486c5f9debe0f948b1e9aee200f3d942cb2f6373d3207d93ce4f50d160",
    "chair": "ee36d7f07d5978c0d0b6215cee7a11487a80898c8685208771cd145a8aac3d22",
}


def manifest_hash(manifest):
    portable = {k: v for k, v in manifest.items() if k not in {"source", "data_hash"}}
    canonical = json.dumps(portable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(path, source_override=None):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if m.get("data_hash") != manifest_hash(m):
        raise ValueError("Manifest fingerprint differs. Freeze it again from the original scene.")
    source = Path(source_override if source_override is not None else m["source"]).resolve()
    for frame in m["frames"]:
        image = (source / frame["file_path"]).resolve()
        if not image.is_relative_to(source) or _sha(image) != frame["image_sha256"]:
            raise ValueError(f"Image differs from the frozen scene: {frame['file_path']}")
    m["source"] = str(source)
    return m


def _blender(source, name):
    frames, seen, camera = [], set(), {}
    dense = _load_frames(source, "transforms_train.json", "blender", frames, seen, camera)
    test = _load_frames(source, "transforms_test.json", "blender", frames, seen, camera)
    reference = json.loads((REPO / "manifests" / f"{name}.json").read_text(encoding="utf-8"))
    if len(dense) != 100 or len(test) != 200 or len(frames) != len(reference["frames"]):
        raise ValueError("Expected the published Blender 100 training and 200 testing metadata entries")
    if not all(np.isclose(camera[k], v, rtol=1e-10) for k, v in reference["camera"].items()):
        raise ValueError(f"{name} calibration differs from the checked-in manifest")
    for frame, original in zip(frames, reference["frames"]):
        if frame["file_path"] != original["file_path"] or not np.allclose(
            frame["transform_matrix"], original["transform_matrix"], rtol=0, atol=1e-7
        ):
            raise ValueError(f"{name} frame order or pose differs at {frame['file_path']}")
    # Use the audited focal value after verifying it, avoiding platform-specific libm roundoff.
    return dict(format="blender", camera=reference["camera"], white_background=True, near=2., far=6.,
                normalization=dict(center=[0., 0., 0.], scale=1.), frames=frames,
                splits=dict(dense=dense, sparse=SPARSE.copy(), test=test[::8]),
                split_protocol="FreeNeRF Blender / DietNeRF fixed-8, testskip 8")


def _seeded_fps(pool, frames, count, rng):
    shuffled = [int(i) for i in rng.permutation(pool)]
    return _coverage_subset(shuffled, frames, count)


def _capture_stats(source, frames, dense, points, raw):
    photos = [p for p in raw.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    registered = len(frames)
    if not photos or not 0.95 <= registered / len(photos) <= 1:
        raise ValueError(f"Registration gate failed: {registered} registered, {len(photos)} raw JPG/PNG stills")
    meta = json.loads((source / "transforms.json").read_text(encoding="utf-8"))
    # ns-process-data stores its fixed COLMAP-world -> transforms.json mapping.
    mapping = np.asarray(meta.get("applied_transform", np.eye(4)[:3]), dtype=float)
    if mapping.shape != (3, 4) or not np.isfinite(mapping).all() or meta.get("applied_scale", 1) != 1:
        raise ValueError("Expected original ns-process-data poses with the rigid COLMAP world transform")
    ids = {Path(f["file_path"]).as_posix(): int(f["colmap_im_id"]) for f in meta["frames"]}
    if len(set(ids.values())) != registered:
        raise ValueError("Missing or repeated COLMAP image ID in transforms.json")
    dense_by_id = {ids[frames[i]["file_path"]]: frames[i]["transform_matrix"] for i in dense}
    all_ids = set(ids.values())
    errors, depths = [], []
    with points.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 12 or len(fields[8:]) % 2:
                raise ValueError("Malformed COLMAP points3D.txt track")
            position = mapping[:, :3] @ np.array(fields[1:4], dtype=float) + mapping[:, 3]
            errors.append(float(fields[7]))
            for image_id in map(int, fields[8::2]):
                if image_id not in all_ids:
                    raise ValueError("Sparse points and registered image IDs differ. Use the same COLMAP model.")
                pose = dense_by_id.get(image_id)
                if pose is not None:
                    z = np.dot(position - pose[:3, 3], -pose[:3, 2])
                    if z > 0:
                        depths.append(z)
    if not errors or not np.isfinite(errors).all() or min(errors) < 0 or np.mean(errors) >= 1:
        raise ValueError("Registration gate failed: mean per-point COLMAP reprojection error must be below 1 px")
    if not depths or not np.isfinite(depths).all():
        raise ValueError("No usable positive training-camera point depths")
    return dict(raw_stills=len(photos), registered=registered, registered_fraction=registered/len(photos),
                mean_reprojection_error_px=float(np.mean(errors)), sparse_points=len(errors),
                points3D_sha256=_sha(points), positive_training_observations=len(depths),
                raw_z_p05_p95=np.percentile(depths, [5, 95]).tolist())


def _personal(source, raw, points, test_count, split_seed):
    if raw is None or points is None:
        raise ValueError("Personal capture requires --raw and --points (COLMAP points3D.txt)")
    if not 12 <= test_count <= 15:
        raise ValueError("Choose 12–15 held-out personal views")
    frames, camera = [], {}
    pool = _load_frames(source, "transforms.json", "nerfstudio", frames, set(), camera)
    if len(pool) <= test_count + 8:
        raise ValueError("Need more registered images than the held-out views plus eight")
    rng = np.random.default_rng(split_seed)
    sparse = _seeded_fps(pool, frames, 8, rng)
    test = _seeded_fps([i for i in pool if i not in sparse], frames, test_count, rng)
    dense = [i for i in pool if i not in test]
    quality = _capture_stats(source, frames, dense, Path(points), Path(raw))
    norm = _center_capture(frames, dense)
    near, far = (d * norm["scale"] for d in quality["raw_z_p05_p95"])
    if not 0 < near < far:
        raise ValueError("Sparse point depth percentiles cannot define render bounds")
    return dict(format="nerfstudio", camera=camera, near=near, far=far, white_background=False,
                frames=frames, normalization=norm, splits=dict(dense=dense, sparse=sparse, test=test),
                split_protocol="seeded farthest-point sparse then disjoint farthest-point test",
                split_seed=split_seed, registration=quality)


def _write_manifest(output, m):
    if output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        if old.get("data_hash") != m["data_hash"]:
            raise ValueError(f"Refusing to overwrite a different manifest at {output}. Use a new output path.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False, mode="w", encoding="utf-8") as f:
        temp = Path(f.name)
        json.dump(m, f, indent=2, allow_nan=False)
        f.write("\n")
    try:
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)


def freeze(scene, source, output, *, raw=None, points=None, test_count=12, split_seed=42):
    source, output = Path(source).resolve(), Path(output)
    if scene in {"lego", "chair"}:
        m = _blender(source, scene)
    elif scene == "personal":
        m = _personal(source, raw, points, test_count, split_seed)
    else:
        raise ValueError(f"Unknown scene {scene}")
    for frame in m["frames"]:
        photo = source / frame["file_path"]
        with Image.open(photo) as image:
            if scene != "personal" and image.mode != "RGBA":
                raise ValueError(f"Expected the original RGBA image at {photo}")
            image.load()
        frame["image_sha256"] = _sha(photo)
        frame["transform_matrix"] = frame["transform_matrix"].tolist()
    dense_hashes = {m["frames"][i]["image_sha256"] for i in m["splits"]["dense"]}
    if scene in PIXEL_FILES:
        observed = "".join(f["image_sha256"] for f in m["frames"]).encode("ascii")
        if hashlib.sha256(observed).hexdigest() != PIXEL_FILES[scene]:
            raise ValueError(f"{scene} image bytes differ from the verified local copy. Reuse the original files.")
    if any(m["frames"][i]["image_sha256"] in dense_hashes for i in m["splits"]["test"]):
        raise ValueError("Test photograph duplicated in training")
    m.update(version=1, name=scene, source=str(source))
    m["data_hash"] = manifest_hash(m)
    _write_manifest(output, m)
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["lego", "chair", "personal"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="verify an existing manifest and all frozen image bytes")
    parser.add_argument("--raw", type=Path, help="original personal capture folder of JPG/PNG stills")
    parser.add_argument("--points", type=Path, help="points3D.txt exported from the same accepted COLMAP model")
    parser.add_argument("--test-count", type=int, default=12, help="personal capture only")
    parser.add_argument("--split-seed", type=int, default=42, help="personal camera selection only")
    args = parser.parse_args()
    try:
        if args.verify:
            m = verify_manifest(args.verify, args.source)
            if args.output:
                _write_manifest(args.output, m)
        elif args.scene and args.source and args.output:
            m = freeze(args.scene, args.source, args.output, raw=args.raw, points=args.points,
                       test_count=args.test_count, split_seed=args.split_seed)
        else:
            parser.error("provide --scene, --source and --output, or --verify MANIFEST")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Data check failed: {exc}\n")
    print(json.dumps(dict(scene=m["name"], data_hash=m["data_hash"], images=len(m["frames"]),
                          splits={k: len(v) for k, v in m["splits"].items()}, near=m["near"], far=m["far"],
                          registration=m.get("registration")), indent=2))


if __name__ == "__main__":
    main()
