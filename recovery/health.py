"""Fixed training rays for checking gradients, density, opacity and the white control."""

import math

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def psnr(mse):
    return -10 * math.log10(max(float(mse), 1e-12))


def camera_check(dataset):
    poses = dataset.poses
    row, col = dataset.height // 2, dataset.width // 2
    origin, direction = dataset.rays(torch.arange(len(poses)), row, col)
    t = -(origin * direction).sum(-1) / direction.square().sum(-1)
    offset = (origin + t[:, None] * direction).norm(dim=-1)
    record = dict(center_t_min=t.min().item(), center_t_max=t.max().item(),
                  max_centerline_distance=offset.max().item(), near=dataset.near, far=dataset.far)
    if dataset.manifest["format"] == "blender" and (
            not ((t > dataset.near) & (t < dataset.far)).all() or offset.max() > 0.1):
        raise ValueError(f"Blender center rays miss the origin or bounds: {record}")
    return record


def fixed_rays(dataset, ids, device, pixels_per_class=16):
    """Use a private RNG. Read only training cameras and keep masks out of the loss."""
    gen = torch.Generator().manual_seed(297)
    origins, directions, colors, labels = [], [], [], []
    for i in ids:
        target = dataset.image(i)
        if dataset.manifest["format"] == "blender":
            with Image.open(dataset.source / dataset.frame_id(i)) as image:
                alpha = image.getchannel("A").resize((dataset.width, dataset.height),
                                                     Image.Resampling.BOX)
            mask = torch.from_numpy(np.array(alpha))
            regions = (mask >= 245, mask <= 10)
        else:
            # Opaque photographs lack an object mask. This is an explicitly labeled
            # central-region diagnostic, separate from any reported foreground metric.
            center = torch.zeros(target.shape[:2], dtype=torch.bool)
            center[dataset.height//4:3*dataset.height//4, dataset.width//4:3*dataset.width//4] = True
            regions = (center, ~center)
        for is_object, region in zip((True, False), regions):
            points = region.nonzero()
            if not len(points):
                raise ValueError(f"No diagnostic pixels in {dataset.frame_id(i)} for region {is_object}")
            picks = points[torch.randperm(len(points), generator=gen)[:pixels_per_class]]
            row, col = picks.unbind(-1)
            o, d = dataset.rays(i, row, col)
            origins.append(o)
            directions.append(d)
            colors.append(target[row, col])
            labels.append(torch.full((len(picks),), is_object))
    return tuple(torch.cat(parts).to(device) for parts in (origins, directions, colors, labels))


def diagnostics(model, dataset, rays, progress, regularized):
    o, d, target, fg = rays
    out = model(o, d, dataset.near, dataset.far, regularized=regularized, progress=progress,
                randomized=False, white_background=dataset.white_background)
    white_mse = (1 - target[fg]).square().mean().item()
    record = dict(region="alpha>=245" if dataset.manifest["format"] == "blender" else "central rectangle",
                  white_region_psnr=psnr(white_mse), foreground_rays=int(fg.sum().item()))
    for name, field, suffix in (("coarse", model.coarse, "_coarse"), ("fine", model.fine, "")):
        rgb, sigma, opacity = (out[key + suffix] for key in ("rgb", "density", "opacity"))
        loss = F.mse_loss(rgb[fg], target[fg])
        grad_density, grad_color = torch.autograd.grad(loss, (field.density.weight, field.color.weight))
        with torch.no_grad():
            values = dict(region_psnr=psnr(loss.item()), region_mse=loss.item(),
                          avg_density=sigma[fg].mean().item(), region_opacity=opacity[fg].mean().item(),
                          outside_opacity=opacity[~fg].mean().item(), region_rgb=rgb[fg].mean(0).tolist(),
                          density_grad=grad_density.norm().item(), color_grad=grad_color.norm().item())
            if not all(torch.isfinite(x).all() for x in (rgb, sigma, opacity, grad_density, grad_color)):
                raise FloatingPointError(f"Nonfinite {name} diagnostic")
            record[name] = values
    return record


def dead_field(record):
    # Catch the archived failure on either pass. Positive softplus can approach
    # zero numerically, so test the actual rendered rays and gradients.
    return any(level["region_opacity"] < 1e-5 or level["color_grad"] == 0
               or level["density_grad"] == 0 for level in (record["coarse"], record["fine"]))
