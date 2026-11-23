"""Package a verified Blender scene and the recovery code for offline use on Kaggle."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from recovery.data import freeze


ROOT = Path(__file__).resolve().parents[1]
MARKER = "RECOVERY_STAGE0.txt"


def build_kit(output, scene="lego"):
    if scene not in ("lego", "chair"):
        raise ValueError("Expected a verified Blender scene (lego or chair)")
    marker_name = MARKER if scene == "lego" else "RECOVERY_CHAIR.txt"
    output = Path(output).resolve()
    if output.suffix.lower() != ".zip" or output.exists():
        raise ValueError("Choose a new output filename ending in .zip")
    source = ROOT / "data" / "blender" / scene
    with tempfile.TemporaryDirectory() as scratch:
        manifest = freeze(scene, source, Path(scratch) / "verified.json")

    packages = ("recovery", "sparse_nerf")
    code = [p for package in packages for p in sorted((ROOT / package).glob("*.py"))]
    paths = code + [ROOT / "manifests" / f"{scene}.json"]
    paths.extend(source / f"transforms_{split}.json" for split in ("train", "test"))
    images = {source / frame["file_path"]: frame["image_sha256"] for frame in manifest["frames"]}
    paths.extend(images)
    required = {ROOT / "recovery" / name for name in ("__init__.py", "data.py", "model.py", "training.py")}
    if not required.issubset(code) or len(images) != 300:
        raise ValueError(f"Expected the recovery training modules and 300 distinct images for {scene}")
    if any(not p.resolve().is_relative_to(ROOT) or not p.is_file() for p in paths):
        raise ValueError("Every bundle member must be a regular file inside the source tree")
    members = {p.relative_to(ROOT).as_posix() for p in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects a previously uploaded kit from accidental replacement.
    stream = output.open("xb")
    try:
        with stream, zipfile.ZipFile(stream, "w") as archive:
            for path in paths:
                payload = path.read_bytes()
                if path in images and hashlib.sha256(payload).hexdigest() != images[path]:
                    raise ValueError(f"Image changed while building the kit: {path.name}")
                compression = zipfile.ZIP_STORED if path in images else zipfile.ZIP_DEFLATED
                archive.writestr(path.relative_to(ROOT).as_posix(), payload, compress_type=compression)
            marker = dict(scene=scene, data_hash=manifest["data_hash"], images=len(images))
            if scene == "lego":
                marker["stage"] = 0
            archive.writestr(marker_name, json.dumps(marker, indent=2) + "\n")
        with zipfile.ZipFile(output) as archive:
            if set(archive.namelist()) != members | {marker_name} or archive.testzip() is not None:
                raise ValueError("The written ZIP failed its member or CRC check")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return dict(output=str(output), bytes=output.stat().st_size, members=len(members) + 1,
                images=len(images), data_hash=manifest["data_hash"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", choices=("lego", "chair"), default="lego")
    args = parser.parse_args()
    print(json.dumps(build_kit(args.output, args.scene), indent=2))


if __name__ == "__main__":
    main()
