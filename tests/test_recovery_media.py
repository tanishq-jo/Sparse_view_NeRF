import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from recovery.media import psnr, render_maps, ssim


class MediaTests(unittest.TestCase):
    def test_ssim_against_independent_sliding_window_equation(self):
        x = np.random.default_rng(4).uniform(size=(8, 9, 3))
        y = np.flip(x, axis=0).copy()
        scores = []
        for row in range(2):
            for col in range(3):
                for channel in range(3):
                    a = x[row:row+7, col:col+7, channel].ravel()
                    b = y[row:row+7, col:col+7, channel].ravel()
                    luminance = (2*a.mean()*b.mean() + 0.01**2) / (a.mean()**2 + b.mean()**2 + 0.01**2)
                    structure = (2*np.cov(a, b)[0, 1] + 0.03**2) / (a.var(ddof=1) + b.var(ddof=1) + 0.03**2)
                    scores.append(luminance * structure)
        self.assertAlmostEqual(ssim(torch.from_numpy(x), torch.from_numpy(y)), np.mean(scores))

    def test_render_maps_chunk_order_and_channel_scaling(self):
        dataset = SimpleNamespace(height=2, width=3, near=2., far=6., white_background=True)
        directions = torch.arange(18).view(2, 3, 3).float() / 18

        def model(origins, rays, near, far, **kwargs):
            self.assertFalse(kwargs["randomized"])
            self.assertTrue(kwargs["regularized"])
            return {"rgb": rays, "depth": near+rays[:, 0], "opacity": rays[:, 2]}

        maps = render_maps(model, dataset, (torch.zeros_like(directions), directions), True, "cpu", 4)
        for img, expected in zip(maps, (directions, 2+directions[..., 0], directions[..., 2])):
            torch.testing.assert_close(img, expected)
        self.assertAlmostEqual(psnr(torch.zeros(4, 3), torch.full((4, 3), .5)), -10*math.log10(.25))
