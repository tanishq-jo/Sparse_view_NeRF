import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from recovery.data import SPARSE, freeze, verify_manifest


class RecoveryDataTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "processed"
        images = self.source / "images"
        images.mkdir(parents=True)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.points = self.root / "points3D.txt"
        self.output = self.root / "manifest.json"
        frames = []
        for i in range(32):
            angle = 2 * math.pi * i / 32
            c, s = math.cos(angle), math.sin(angle)
            pose = [[-s, 0, c, 2*c], [c, 0, s, 2*s], [0, 1, 0, 0], [0, 0, 0, 1]]
            name = f"frame_{i:03d}.png"
            image = Image.new("RGB", (31, 32), (i, 2*i, 3*i))
            image.save(images / name)
            image.save(self.raw / name)
            frames.append(dict(file_path=f"images/{name}", transform_matrix=pose, colmap_im_id=i+1))
        meta = dict(w=31, h=32, fl_x=25, fl_y=25, cx=15.5, cy=16, frames=frames)
        (self.source / "transforms.json").write_text(json.dumps(meta))
        track = " ".join(f"{i+1} 0" for i in range(32))
        self.points.write_text(f"1 0 0 0 100 100 100 0.25 {track}\n2 .2 .1 0 100 100 100 0.5 {track}\n")

    def freeze_capture(self):
        return freeze("personal", self.source, self.output, raw=self.raw, points=self.points)

    def test_split_bounds_and_relocation(self):
        m = self.freeze_capture()
        self.assertEqual(m, self.freeze_capture())
        self.assertEqual(m["registration"]["registered_fraction"], 1)
        self.assertEqual(m["registration"]["mean_reprojection_error_px"], .375)
        self.assertLess(m["near"], 4)
        self.assertGreater(m["far"], 4)
        dense, sparse, test = (m["splits"][s] for s in ("dense", "sparse", "test"))
        self.assertEqual([len(dense), len(sparse), len(test)], [20, 8, 12])
        self.assertFalse(set(dense) & set(test))
        self.assertTrue(set(sparse).issubset(dense))
        moved = self.root / "new-mount"
        self.source.rename(moved)
        relocated = verify_manifest(self.output, moved)
        self.assertEqual(m["data_hash"], relocated["data_hash"])
        frame = moved / m["frames"][0]["file_path"]
        frame.write_bytes(frame.read_bytes() + b"modified")
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_manifest(self.output, moved)

    def test_requires_registration_quality(self):
        for i in range(4):
            (self.raw / f"unregistered_{i}.jpg").touch()
        with self.assertRaisesRegex(ValueError, "Registration gate failed"):
            self.freeze_capture()

    def test_reprojection_gate(self):
        self.points.write_text(self.points.read_text().replace("0.25", "2"))
        with self.assertRaisesRegex(ValueError, "reprojection"):
            self.freeze_capture()

    def test_published_sparse_ids(self):
        self.assertEqual(SPARSE, np.random.RandomState(0).choice(100, 8, replace=False).tolist())


if __name__ == "__main__":
    unittest.main()
