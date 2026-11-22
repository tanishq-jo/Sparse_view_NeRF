"""Hierarchical vanilla NeRF with finite ray support and optional FreeNeRF losses."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from sparse_nerf.model import FourierEncoding, ModelConfig, RadianceField, _importance_samples


class SmoothField(RadianceField):
    """Retain the original trunk and heads, with differentiable positive density."""

    def forward(self, positions, directions):
        hidden = positions
        for index, layer in enumerate(self.layers):
            if index == 5:
                hidden = torch.cat((hidden, positions), dim=-1)
            hidden = F.relu(layer(hidden))
        sigma = F.softplus(self.density(hidden).squeeze(-1) - 1.0)
        views = directions[:, None, :].expand(-1, positions.shape[1], -1)
        hidden_rgb = F.relu(self.view(torch.cat((self.feature(hidden), views), dim=-1)))
        return torch.sigmoid(self.color(hidden_rgb)), sigma


def integrate(rgb, sigma, z, near, far, ray_lengths, white_background):
    """Voronoi intervals span [near, far]. Residual transmittance sees the background."""
    mid = (z[:, 1:] + z[:, :-1]) * 0.5
    edges = torch.cat((torch.full_like(z[:, :1], near), mid, torch.full_like(z[:, :1], far)), -1)
    optical = sigma.float() * edges.diff(dim=-1) * ray_lengths
    cumulative = optical.cumsum(-1)
    prefix = torch.cat((torch.zeros_like(optical[:, :1]), cumulative[:, :-1]), -1)
    weights = (-prefix).exp() * -torch.expm1(-optical)
    residual = (-cumulative[:, -1:]).exp()
    color = (weights[..., None] * rgb).sum(1)
    if white_background:
        color = color + residual
    return color, weights, residual


def near_camera_loss(sigma, count):
    # The reference zeroes all except the first M density samples, then takes
    # the mean over the entire tensor, including those zeros.
    return sigma[:, :count].sum() / sigma.numel()


class Renderer(nn.Module):
    def __init__(self, config: ModelConfig, near_samples=20):
        super().__init__()
        self.config = config
        self.near_samples = near_samples
        self.position = FourierEncoding(config.pos_bands)
        self.direction = FourierEncoding(config.dir_bands)
        self.coarse = SmoothField(config, self.position.out_dim, self.direction.out_dim)
        self.fine = SmoothField(config, self.position.out_dim, self.direction.out_dim)

    def forward(self, origins, directions, near, far, *, progress=1., regularized=False,
                randomized=False, white_background=True):
        if not 0 <= progress <= 1 or not 0 <= near < far or not math.isfinite(far):
            raise ValueError("Invalid training progress or ray bounds")
        fraction = min(progress / self.config.anneal_fraction, 1.) if regularized else 1.
        directions_pe = self.direction(F.normalize(directions, dim=-1), fraction)
        lengths = directions.norm(dim=-1, keepdim=True)

        def query(field, z):
            xyz = origins[:, None] + z[..., None] * directions[:, None]
            rgb, sigma = field(self.position(xyz, fraction), directions_pe)
            return (*integrate(rgb, sigma, z, near, far, lengths, white_background), sigma)

        z = torch.linspace(near, far, self.config.coarse_samples,
                           device=origins.device, dtype=origins.dtype).expand(len(origins), -1)
        mid = (z[:, 1:] + z[:, :-1]) * 0.5
        if randomized:
            z = torch.lerp(torch.cat((z[:, :1], mid), -1),
                           torch.cat((mid, z[:, -1:]), -1), torch.rand_like(z))
            mid = (z[:, 1:] + z[:, :-1]) * 0.5
        coarse_rgb, weights, _, coarse_sigma = query(self.coarse, z)
        new = _importance_samples(mid, weights[:, 1:-1], self.config.fine_samples, randomized)
        z_fine = torch.cat((z, new), -1).sort(dim=-1).values
        rgb, fine_weights, residual, fine_sigma = query(self.fine, z_fine)
        return dict(rgb=rgb, rgb_coarse=coarse_rgb, depth=(z_fine * fine_weights).sum(-1)
                    + far * residual.squeeze(-1), opacity=fine_weights.sum(-1),
                    opacity_coarse=weights.sum(-1), density=fine_sigma, density_coarse=coarse_sigma,
                    near_loss=near_camera_loss(fine_sigma, self.near_samples))
