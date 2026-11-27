"""Package a verified Blender scene and the recovery code for offline use on Kaggle."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from recovery.data import freeze, verify_manifest


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


def build_personal_kit(output, manifest):
    """Package accepted RGB images, their frozen cameras, and the offline trainer."""
    from recovery.training import implementation_hash

    data = verify_manifest(manifest)
    if data["name"] != "personal" or data["format"] != "nerfstudio":
        raise ValueError("Provide the frozen personal capture manifest.")
    source = Path(data["source"])
    output = Path(output).resolve()
    if output.suffix.lower() != ".zip":
        raise ValueError("Choose a new output filename ending in .zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    members = {p.relative_to(ROOT).as_posix(): p for package in ("recovery", "sparse_nerf")
               for p in sorted((ROOT / package).glob("*.py"))}
    members["data/personal/transforms.json"] = source / "transforms.json"
    for frame in data["frames"]:
        name = frame["file_path"]
        members["data/personal/" + name] = source / name
    marker = dict(scene="personal", data_hash=data["data_hash"], images=len(data["frames"]),
                  implementation_hash=implementation_hash())
    stream = output.open("xb")
    try:
        with stream, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr("RECOVERY_PERSONAL.txt", json.dumps(marker, indent=2))
            archive.writestr("manifests/personal.json", json.dumps(dict(data, source="data/personal"), indent=2))
            for name, path in members.items():
                archive.write(path, name)
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(output) as archive:
            if archive.testzip() or set(archive.namelist()) != set(members) | {"RECOVERY_PERSONAL.txt","manifests/personal.json"}:
                raise ValueError("Personal kit failed its member or CRC check.")
            archive.extractall(tmp)
            verify_manifest(Path(tmp)/"manifests/personal.json", Path(tmp)/"data/personal")
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return dict(marker, output=str(output), bytes=output.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", choices=("lego", "chair", "personal"), default="lego")
    parser.add_argument("--manifest", type=Path, help="Frozen personal capture manifest")
    args = parser.parse_args()
    if args.scene == "personal":
        if args.manifest is None:
            parser.error("--scene personal requires --manifest")
        result = build_personal_kit(args.output, args.manifest)
    else:
        result = build_kit(args.output, args.scene)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
