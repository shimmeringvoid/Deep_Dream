"""Gradient ascent on the image: the dream loop itself.

Three layers, smallest first:

* `gradients` / `tiled_gradients` — one normalized ascent direction for an
  image. The tiled variant keeps memory flat at any resolution by summing
  per-tile gradients, with a random roll each pass so tile seams never print.
* `dream_tensor` — the octave loop, tensor in / tensor out. This is the one
  `coherence/` and the video code should call, because chaining frames must
  not round-trip through uint8.
* `dream` — the human-facing wrapper: uint8 in, uint8 out.
"""

from __future__ import annotations

import numpy as np
import torch

from .image import deprocess, preprocess, random_roll, resize, to_batch
from .model import load_backbone
from .objective import Objective

OCTAVE_SCALE = 1.30
OCTAVES = range(-2, 3)
TILE_SIZE = 512


def gradients(objective: Objective, img: torch.Tensor) -> torch.Tensor:
    """Normalized d(loss)/d(pixels) for a whole HWC image."""
    img = img.detach().requires_grad_(True)
    loss = objective.loss(img)
    grad = torch.autograd.grad(loss, img)[0]
    return grad / (grad.std() + 1e-8)


def tiled_gradients(
    objective: Objective, img: torch.Tensor, tile_size: int = TILE_SIZE
) -> torch.Tensor:
    """Normalized gradient assembled from tiles, so memory stays flat.

    `loss.backward()` accumulates each tile's gradient into `img.grad` — the
    running-sum bookkeeping the TF tutorial carried by hand.
    """
    shift, rolled = random_roll(img, tile_size)
    rolled = rolled.detach().requires_grad_(True)

    h, w = rolled.shape[:2]
    # Skip the last (ragged) tile unless there is only one; the random roll
    # ensures every pixel is still visited across steps.
    xs = list(range(0, w, tile_size))[:-1] or [0]
    ys = list(range(0, h, tile_size))[:-1] or [0]

    for x in xs:
        for y in ys:
            tile = rolled[y : y + tile_size, x : x + tile_size]
            objective.loss(tile).backward()

    grad = torch.roll(
        rolled.grad, shifts=(-int(shift[0]), -int(shift[1])), dims=(0, 1)
    )
    return grad / (grad.std() + 1e-8)


def _as_objective(objective, backbone=None, device=None) -> Objective:
    if isinstance(objective, Objective):
        return objective
    return Objective(objective, backbone or load_backbone(device=device))


def dream_tensor(
    img: torch.Tensor,
    objective,
    *,
    steps: int = 100,
    step_size: float = 0.01,
    octaves=OCTAVES,
    octave_scale: float = OCTAVE_SCALE,
    tiled: bool | None = None,
    tile_size: int = TILE_SIZE,
    restore_size: bool = True,
    callback=None,
) -> torch.Tensor:
    """Run the octave loop on a preprocessed HWC tensor; return one back.

    `tiled=None` decides per image: tile once the image outgrows one tile.
    `restore_size` returns the result at the input's size (the last octave is
    the *largest*, so without it the caller inherits an upscaled image) — the
    behaviour chained/video callers want.

    `callback(octave, step, img)` is called every 10 steps for progress.
    """
    objective = _as_objective(objective)
    base_shape = img.shape[:2]

    for octave in octaves:
        new_size = tuple(int(round(s * octave_scale**octave)) for s in base_shape)
        img = resize(img, new_size)
        use_tiles = max(new_size) > tile_size if tiled is None else tiled

        for step in range(steps):
            grad = (
                tiled_gradients(objective, img, tile_size)
                if use_tiles
                else gradients(objective, img)
            )
            with torch.no_grad():
                img = (img + grad * step_size).clamp(-1.0, 1.0)
            if callback is not None and step % 10 == 0:
                callback(octave, step, img)

    if restore_size and img.shape[:2] != base_shape:
        img = resize(img, base_shape).clamp(-1.0, 1.0)
    return img.detach()


def dream(
    img,
    objective,
    *,
    backbone=None,
    device=None,
    steps: int = 100,
    step_size: float = 0.01,
    octaves=OCTAVES,
    octave_scale: float = OCTAVE_SCALE,
    tiled: bool | None = None,
    tile_size: int = TILE_SIZE,
    callback=None,
) -> np.ndarray:
    """Dream a uint8 image and get a uint8 image back.

    `objective` may be an `Objective`, a list of `Target`s, or a list of
    `"layer:channel:weight"` specs.
    """
    if backbone is None and not isinstance(objective, Objective):
        backbone = load_backbone(device=device)
    objective = _as_objective(objective, backbone, device)
    dev = objective.backbone.device
    out = dream_tensor(
        preprocess(img, dev),
        objective,
        steps=steps,
        step_size=step_size,
        octaves=octaves,
        octave_scale=octave_scale,
        tiled=tiled,
        tile_size=tile_size,
        callback=callback,
    )
    return deprocess(out)
