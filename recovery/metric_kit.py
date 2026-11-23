"""Build the private offline LPIPS input for Kaggle evaluation.

Fetch the original wheel with pip download --no-deps --only-binary=:all: lpips==0.1.4.
The matching ImageNet backbone is available from torchvision at
https://download.pytorch.org/models/alexnet-owt-7be5be79.pth.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


WHEEL = "wheels/lpips-0.1.4-py3-none-any.whl"
BACKBONE = "torch-cache/hub/checkpoints/alexnet-owt-7be5be79.pth"
HASHES = {
    WHEEL: "fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e",
    BACKBONE: "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
}
MARKER = "RECOVERY_LPIPS.txt"


def build(wheel, weights, output):
    if output.suffix.lower() != ".zip":
        raise ValueError("Choose a new output filename ending in .zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("xb")
    try:
        with stream, zipfile.ZipFile(stream, "w") as archive:
            for member, path in ((WHEEL, wheel), (BACKBONE, weights)):
                digest = hashlib.sha256()
                # Fixed timestamps keep the output bytes identical across rebuilds.
                entry = zipfile.ZipInfo(member)
                entry.create_system = 3
                with path.open("rb") as source, archive.open(entry, "w") as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        target.write(chunk)
                if digest.hexdigest() != HASHES[member]:
                    raise ValueError(f"SHA-256 mismatch for {member}")
            marker = dict(lpips="0.1.4", net="alex", version="0.1", sha256=HASHES)
            entry = zipfile.ZipInfo(MARKER)
            entry.create_system = 3
            archive.writestr(entry, json.dumps(marker, indent=2) + "\n")
        with zipfile.ZipFile(output) as archive:
            if set(archive.namelist()) != set(HASHES) | {MARKER} or archive.testzip() is not None:
                raise ValueError("Offline LPIPS kit failed its member or CRC check")
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.wheel, args.weights, args.output)
    print(f"Verified {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
