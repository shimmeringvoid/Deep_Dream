"""Image plumbing — the doors between human space and model space.

Model space: float32, HWC, values in [-1, 1], on the device.
Human space: numpy uint8, HWC, values in [0, 255].

`preprocess` and `deprocess` are the only two functions that cross the line.
Everything else in `engine/` speaks model space; anything that reads or writes
a file speaks human space.
"""

from __future__ import annotations

import pathlib

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F


def preprocess(img, device=None) -> torch.Tensor:
    """uint8/float [0, 255] HWC array -> float32 HWC tensor in [-1, 1].

    Same convention as `tf.keras.applications.inception_v3.preprocess_input`,
    which is why `transform_input=False` backbones consume it directly.
    Passing an already-preprocessed tensor through is a no-op courtesy, so
    callers can accept "an image" without interrogating it first.
    """
    if torch.is_tensor(img) and img.dtype == torch.float32 and img.min() >= -1.0001:
        return img.to(device) if device is not None else img
    t = torch.as_tensor(np.asarray(img), dtype=torch.float32, device=device)
    return t / 127.5 - 1.0


def deprocess(img: torch.Tensor) -> np.ndarray:
    """float [-1, 1] tensor (HWC or NHWC) -> uint8 numpy array on the CPU."""
    img = 255 * (img + 1.0) / 2.0
    return img.clamp(0, 255).to(torch.uint8).cpu().numpy()


def to_batch(img: torch.Tensor) -> torch.Tensor:
    """HWC -> NCHW batch of one (what the backbones eat)."""
    return img.permute(2, 0, 1).unsqueeze(0)


def from_batch(batch: torch.Tensor) -> torch.Tensor:
    """NCHW batch of one -> HWC."""
    return batch.squeeze(0).permute(1, 2, 0)


def resize(img, size, device=None) -> torch.Tensor:
    """HWC (numpy or tensor, any range) -> bilinear-resized float32 HWC tensor.

    `size` is (height, width). The device of a tensor input is preserved
    unless `device` overrides it.
    """
    if torch.is_tensor(img):
        t = img.to(dtype=torch.float32)
        if device is not None:
            t = t.to(device)
    else:
        t = torch.as_tensor(np.asarray(img), dtype=torch.float32, device=device)
    t = t.permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(
        t, size=tuple(int(s) for s in size), mode="bilinear", align_corners=False
    )
    return t.squeeze(0).permute(1, 2, 0)


def resize_batch(batch: torch.Tensor, size) -> torch.Tensor:
    """NCHW -> bilinear-resized NCHW. The batched sibling of `resize`."""
    return F.interpolate(
        batch, size=tuple(int(s) for s in size), mode="bilinear", align_corners=False
    )


def random_roll(img: torch.Tensor, maxroll: int, generator=None):
    """Randomly shift an HWC image so tile boundaries never print into it.

    Returns `(shift, rolled)`; roll back with the negated shift.
    """
    shift = torch.randint(-maxroll, maxroll, (2,), generator=generator)
    rolled = torch.roll(img, shifts=(int(shift[0]), int(shift[1])), dims=(0, 1))
    return shift, rolled


def roll_batch(batch: torch.Tensor, maxroll: int, generator=None):
    """The NCHW sibling of `random_roll` — one shared shift for the batch."""
    shift = torch.randint(-maxroll, maxroll, (2,), generator=generator)
    rolled = torch.roll(batch, shifts=(int(shift[0]), int(shift[1])), dims=(2, 3))
    return shift, rolled


def zoom(frame: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale an HWC frame about its center by `factor`, crop back to size.

    The warp `coherence/` will use for fractal zooms (milestone 5), and what
    the notebook's `dream_zoom_video` feedback loop runs on.
    """
    h, w = frame.shape[:2]
    big = resize(frame, (int(h * factor), int(w * factor)))
    y0 = (big.shape[0] - h) // 2
    x0 = (big.shape[1] - w) // 2
    return big[y0 : y0 + h, x0 : x0 + w]


def load_image(path, max_dim: int | None = None) -> np.ndarray:
    """Read an image file as uint8 HWC RGB, optionally thumbnailed."""
    img = PIL.Image.open(path).convert("RGB")
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


def save_image(img, path) -> pathlib.Path:
    """Write uint8 HWC (or a [-1, 1] tensor) to disk, creating parent dirs."""
    if torch.is_tensor(img):
        img = deprocess(img) if img.dtype.is_floating_point else img.cpu().numpy()
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(np.asarray(img).astype(np.uint8)).save(path)
    return path


def even_dims(img: torch.Tensor) -> torch.Tensor:
    """Trim to even height/width — yuv420p video encoding requires it."""
    h, w = img.shape[:2]
    return img[: h // 2 * 2, : w // 2 * 2]
