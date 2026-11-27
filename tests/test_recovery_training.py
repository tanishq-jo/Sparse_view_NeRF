import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image, ImageDraw
import torch

from recovery.data import manifest_hash
from recovery.model import Renderer, integrate, near_camera_loss
from recovery.training import load_final, settings, train
from sparse_nerf.model import ModelConfig


class RecoveryTests(unittest.TestCase):
    def test_finite_intervals_and_regularizer_scale(self):
        z = torch.tensor([[2.1, 3., 5., 5.9]])
        rgb = torch.zeros(1, 4, 3)
        sigma = torch.full_like(z, .25)
        colors, weights, residual = integrate(rgb, sigma, z, 2, 6, torch.ones(1, 1), True)
        expected = torch.exp(torch.tensor(-1.))
        torch.testing.assert_close(colors, expected.expand_as(colors))
        torch.testing.assert_close(weights.sum() + residual.sum(), torch.tensor(1.))
        sig = torch.tensor([[2., 4., 9., 1.]], requires_grad=True)
        near_camera_loss(sig, 2).backward()
        torch.testing.assert_close(sig.grad, torch.tensor([[.25, .25, 0., 0.]]))
        self.assertEqual(near_camera_loss(sig, 2).item(), 1.5)

    def test_negative_density_logits_keep_foreground_gradients(self):
        torch.manual_seed(31)
        net = Renderer(ModelConfig(width=24, depth=2, coarse_samples=8, fine_samples=8))
        for field in (net.coarse, net.fine):
            with torch.no_grad():
                field.density.bias.fill_(-2.)
                field.density.weight.zero_()
        o = torch.zeros(8, 3)
        d = torch.randn(8, 3)
        out = net(o, d, 2, 6, regularized=True, progress=.1)
        loss = out["rgb"].square().mean() + out["rgb_coarse"].square().mean() + .01*out["near_loss"]
        loss.backward()
        self.assertTrue((out["opacity"] > 0).all())
        for field in (net.coarse, net.fine):
            self.assertGreater(field.density.bias.grad.abs().item(), 0)
            self.assertGreater(field.color.bias.grad.norm().item(), 0)
        a = net(o, d, 2, 6, regularized=False)
        b = net(o, d, 2, 6, regularized=True)
        torch.testing.assert_close(a["rgb"], b["rgb"])
        self.assertFalse(torch.allclose(a["rgb"], out["rgb"]))

    def test_fresh_directory_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photos = root / "photos"
            photos.mkdir()
            frames = []
            for i in range(11):
                image = Image.new("RGBA", (32, 32))
                ImageDraw.Draw(image).rectangle((8, 8, 24, 24), fill=(50+i, 90, 120, 255))
                path = photos / f"{i}.png"
                image.save(path)
                pose = torch.eye(4)
                pose[2, 3] = 4
                frames.append(dict(file_path=path.name, image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   transform_matrix=pose.tolist()))
            camera = dict(w=32, h=32, fx=45, fy=45, cx=16, cy=16, k1=0,k2=0,k3=0,p1=0,p2=0)
            m = dict(version=1, name="lego", format="blender", source=str(photos.resolve()), camera=camera,
                     frames=frames, splits=dict(dense=list(range(9)), sparse=list(range(8)),test=[9,10]),
                     near=2., far=6., white_background=True)
            m["data_hash"] = manifest_hash(m)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(m))
            cfg = settings("main", camera)
            cfg.update(steps=3, batch_size=12, model=asdict(ModelConfig(width=12,depth=2,
                       coarse_samples=4,fine_samples=3)), near_samples=2)
            def fit(folder, p, **opts):
                train(folder, p, "sparse-regularized", 42, "cpu", cfg, **opts)
            fit(root / "whole", manifest)
            fit(root / "part", manifest, stop_at=1)
            with self.assertRaisesRegex(ValueError, "completed main"):
                load_final(root / "part", manifest, "cpu")
            # A fresh session has new absolute data paths and a writable copy of
            # the previous output. Only the path-independent fingerprint persists.
            shutil.copytree(photos, root / "remounted")
            m["source"] = str((root / "remounted").resolve())
            moved = root / "relocated.json"
            moved.write_text(json.dumps(m))
            (root / "continued").mkdir()
            for name in ("checkpoint.pt", "run.json"):
                shutil.copy2(root / "part" / name, root / "continued" / name)
            fit(root / "continued", moved, resume=True)
            a, _ = load_final(root / "whole", manifest, "cpu")
            b, _ = load_final(root / "continued", moved, "cpu")
            for x, y in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "Resume requires"):
                train(root / "continued", moved, "sparse-vanilla", 42, "cpu", cfg, resume=True)
            (root / "remounted" / "1.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Image differs"):
                load_final(root / "continued", moved, "cpu")


if __name__ == "__main__":
    unittest.main()
