"""Feature discovery — what does channel 417 of mixed7 actually *see*?

Channel indices are meaningless until you look at them, and `mixed7` has 768 of
them (`mixed9` has 2048). This module dreams a small seed image per channel and
lays the results out as **contact sheets** you can eye-pick from: "that one's a
flower, that one's eyes, that one's a spiral". The picks become a named preset
(see `presets.py`) that the schedule can later reference and blend.

Two cost-saving moves, both from the plan in CLAUDE.md:

* **Batch.** N seeds go through the net in one pass with a per-item
  single-channel loss. The nets are convolutional and in eval mode (BatchNorm
  uses running statistics), so nothing couples the batch items — each item's
  gradient is exactly what it would have been alone.
* **Pre-filter.** One forward pass ranks all channels by mean activation on a
  real image; `--top K` then contact-sheets only the shortlist. Dreaming all
  768 is affordable on a Titan X, so the pre-filter is an option, not a gate.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import time

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import torch

from .image import deprocess, preprocess, resize_batch, roll_batch
from .model import Backbone

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def deterministic_conv(enabled: bool = True):
    """Force cuDNN's deterministic convolution algorithms inside the block.

    Not a nicety here. cuDNN's default backward-convolution kernels accumulate
    with atomics, so two identical runs differ in the last bits — and
    single-channel ascent amplifies that chaotically: measured on the Titan X,
    two runs of the same 96-step browse differ by a mean of 71/255 per pixel,
    i.e. completely different pictures of the same feature. A contact sheet
    whose tiles cannot be regenerated is a sheet you cannot pick from. The
    determinism costs ~13% wall-clock, which is a bargain for that.
    """
    prev = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = enabled
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = prev


def _one_seed(size, kind, device, generator, amount) -> torch.Tensor:
    if kind == "noise":
        noise = torch.rand((1, 3, size, size), device=device, generator=generator)
        return (noise * 2 - 1) * 0.3
    if kind != "gray":
        raise ValueError(f"Unknown seed kind {kind!r} (gray | noise | array)")
    low = max(4, size // 8)
    noise = torch.randn((1, 3, low, low), device=device, generator=generator)
    return (resize_batch(noise, (size, size)) * amount).clamp(-1.0, 1.0)


def make_seeds(
    n: int,
    size: int,
    kind="gray",
    device=None,
    item_seeds=None,
    amount: float = 0.10,
) -> torch.Tensor:
    """N NCHW seed images in [-1, 1] to dream from.

    * `gray` — flat mid-gray plus *low-frequency* noise (noise generated at
      1/8 scale and upsampled). Smooth blobs give gradient ascent something to
      grab without the high-frequency hash a white-noise seed starts with, so
      the feature reads more clearly. This is the default and the useful one.
    * `noise` — full-resolution uniform noise, the classic harsher look.
    * a uint8 HWC array — a real image, tiled across the batch. Use this to ask
      "what would this channel do to *my* footage?" rather than "what is this
      channel", which is a different and also useful question.

    `item_seeds` gives each batch item its own RNG seed, keyed off its channel
    number, so the seed a channel starts from does not depend on the company
    it was batched with. (That fixes the *seed*; see `dream_channels` for why
    it does not by itself buy reproducibility across batch sizes.)
    """
    if isinstance(kind, np.ndarray) or torch.is_tensor(kind):
        img = preprocess(kind, device)
        batch = img.permute(2, 0, 1).unsqueeze(0)
        batch = resize_batch(batch, (size, size))
        return batch.repeat(n, 1, 1, 1).clone()
    gens = [None] * n
    if item_seeds is not None:
        gens = [
            torch.Generator(device=device).manual_seed(int(s) & 0x7FFFFFFF)
            for s in item_seeds
        ]
    return torch.cat([_one_seed(size, kind, device, g, amount) for g in gens])


def _blur_kernel(sigma: float, device, dtype=torch.float32):
    """Separable Gaussian, radius 3σ, as a (1, 1, k) kernel."""
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return (k / k.sum()).view(1, 1, -1)


def blur_gradient(grad: torch.Tensor, sigma: float) -> torch.Tensor:
    """Low-pass an NCHW gradient with a separable Gaussian.

    The regularizer that makes a contact sheet *readable*. Unconstrained
    ascent on a single channel piles up high-frequency hash — every tile ends
    up looking like the same rainbow static, which is useless when the job is
    "point at the flower one". Blurring the ascent direction biases growth
    toward the low frequencies where recognizable shape lives; the objective
    still gets maximized, just in the legible part of the spectrum.
    """
    if sigma <= 0:
        return grad
    k = _blur_kernel(sigma, grad.device, grad.dtype)
    n, c = grad.shape[:2]
    pad = k.shape[-1] // 2
    g = grad.reshape(n * c, 1, *grad.shape[2:])
    g = torch.nn.functional.conv2d(
        torch.nn.functional.pad(g, (0, 0, pad, pad), mode="reflect"), k.view(1, 1, -1, 1)
    )
    g = torch.nn.functional.conv2d(
        torch.nn.functional.pad(g, (pad, pad, 0, 0), mode="reflect"), k.view(1, 1, 1, -1)
    )
    return g.reshape(grad.shape)


# ---------------------------------------------------------------------------
# Cheap pre-filter
# ---------------------------------------------------------------------------


def rank_channels(
    backbone: Backbone, layer: str, seed=None, size: int = 256
) -> list[tuple[int, float]]:
    """Rank every channel of `layer` by mean activation on one image.

    One forward pass, no gradients. Returns `[(channel, score), ...]` sorted
    high to low — a cheap shortlist of "what this image already excites",
    which is the right pre-filter when you are hunting for channels that will
    do something to a *particular* piece of footage.
    """
    dev = backbone.device
    if seed is None:
        seed = "gray"
    batch = make_seeds(1, size, seed, device=dev, item_seeds=[0])
    with torch.no_grad():
        acts = backbone.extractor([layer])(batch)[layer]
    scores = acts[0].mean(dim=(1, 2)).cpu().numpy()
    order = np.argsort(-scores)
    return [(int(c), float(scores[c])) for c in order]


# ---------------------------------------------------------------------------
# Batched per-channel dreaming
# ---------------------------------------------------------------------------


def dream_channels(
    backbone: Backbone,
    layer: str,
    channels,
    *,
    size: int = 256,
    steps: int = 96,
    step_size: float = 0.05,
    seed="gray",
    batch_size: int = 16,
    octaves=(-2, -1, 0),
    octave_scale: float = 1.3,
    jitter: int = 8,
    grad_blur: float = 1.0,
    seed_value: int | None = 0,
    deterministic: bool = True,
    progress=None,
) -> np.ndarray:
    """Dream one image per channel. Returns uint8 `(len(channels), size, size, 3)`.

    The loss for a batch is `sum_i mean(acts[i, channel_i])`: each item chases
    its own channel, and because no operation mixes batch items the gradients
    stay independent (BatchNorm is in eval mode, so it uses running statistics
    and does not couple them either). Gradients are normalized **per item**, so
    a quiet channel gets the same step length as a loud one and the sheet stays
    comparable.

    **Reproducibility is per batch size, not absolute.** A single ascent step
    is bit-identical whether an item was dreamed in a batch of 64 or alone —
    that is the independence claim, and `tests/` checks it. But cuDNN picks
    different convolution algorithms for different batch shapes, and ascent on
    a single channel is chaotic: those last-bit differences amplify, until by
    ~100 steps the batch-of-64 and batch-of-1 tiles for one channel differ
    about as much as two *different* channels do. Both are honest pictures of
    the same feature; they are not the same picture. So to reproduce a tile off
    a sheet, re-run with the `batch_size` recorded in that sheet's
    `index.json` — which is why it is recorded.
    """
    channels = [int(c) for c in channels]
    n_ch = backbone.channels(layer)
    bad = [c for c in channels if not 0 <= c < n_ch]
    if bad:
        raise ValueError(f"{backbone.name}/{layer} has {n_ch} channels; out of range: {bad[:8]}")

    dev = backbone.device
    extractor = backbone.extractor([layer])

    out = np.empty((len(channels), size, size, 3), dtype=np.uint8)
    t0 = time.time()

    with deterministic_conv(deterministic):
        for start in range(0, len(channels), batch_size):
            chunk = channels[start : start + batch_size]
            idx = torch.tensor(chunk, device=dev)
            rows = torch.arange(len(chunk), device=dev)
            batch = make_seeds(
                len(chunk),
                size,
                seed,
                device=dev,
                item_seeds=(
                    None if seed_value is None else [seed_value * 1000003 + c for c in chunk]
                ),
            )
            # The jitter sequence is reseeded per chunk rather than run from the
            # global RNG, for the same reason the seeds are per-channel: a tile
            # must not depend on how the run happened to be batched.
            jitter_gen = (
                None
                if seed_value is None
                else torch.Generator().manual_seed(seed_value & 0x7FFFFFFF)
            )

            for octave in octaves:
                oct_size = max(backbone.min_size, int(round(size * octave_scale**octave)))
                batch = resize_batch(batch, (oct_size, oct_size))
                roll = min(jitter, max(1, oct_size // 8))

                for _ in range(steps):
                    shift, rolled = roll_batch(batch, roll, generator=jitter_gen)
                    rolled = rolled.detach().requires_grad_(True)
                    acts = extractor(rolled)[layer]
                    # Per-item single-channel objective, summed: one backward pass
                    # yields every item's own gradient.
                    loss = acts[rows, idx].mean(dim=(1, 2)).sum()
                    grad = torch.autograd.grad(loss, rolled)[0]
                    grad = torch.roll(
                        grad, shifts=(-int(shift[0]), -int(shift[1])), dims=(2, 3)
                    )
                    grad = blur_gradient(grad, grad_blur)
                    grad = grad / (grad.std(dim=(1, 2, 3), keepdim=True) + 1e-8)
                    with torch.no_grad():
                        batch = (batch + grad * step_size).clamp(-1.0, 1.0)

            batch = resize_batch(batch, (size, size)).clamp(-1.0, 1.0)
            out[start : start + len(chunk)] = deprocess(batch.permute(0, 2, 3, 1))

            if progress is not None:
                done = start + len(chunk)
                progress(done, len(channels), time.time() - t0)

    return out


# ---------------------------------------------------------------------------
# Contact sheets
# ---------------------------------------------------------------------------


def _font(px: int):
    """A readable TrueType font if one is around, else PIL's bitmap default."""
    candidates = []
    try:  # matplotlib ships DejaVu and is already in this env
        import matplotlib

        candidates.append(
            pathlib.Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        )
    except Exception:
        pass
    candidates += [
        pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        pathlib.Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for path in candidates:
        try:
            if path.exists():
                return PIL.ImageFont.truetype(str(path), px)
        except Exception:
            continue
    return PIL.ImageFont.load_default()


def contact_sheet(
    tiles,
    labels=None,
    *,
    cols: int = 8,
    tile_px: int | None = None,
    pad: int = 6,
    label_px: int = 14,
    title: str | None = None,
    bg=(18, 18, 20),
    fg=(235, 235, 235),
) -> PIL.Image.Image:
    """Lay uint8 tiles out in a labelled grid.

    Every tile carries its channel number underneath — the sheet is useless as
    a picking tool if you cannot read the index straight off it.
    """
    tiles = np.asarray(tiles)
    n = len(tiles)
    if n == 0:
        raise ValueError("Nothing to lay out.")
    src = tiles.shape[1]
    tile_px = tile_px or src
    labels = list(labels) if labels is not None else [str(i) for i in range(n)]

    rows = (n + cols - 1) // cols
    band = label_px + 6 if labels else 0
    title_h = label_px + 12 if title else 0
    cell_w, cell_h = tile_px + pad, tile_px + pad + band

    sheet = PIL.Image.new(
        "RGB", (cols * cell_w + pad, title_h + rows * cell_h + pad), bg
    )
    draw = PIL.ImageDraw.Draw(sheet)
    font = _font(label_px)

    if title:
        draw.text((pad + 2, 6), title, fill=fg, font=font)

    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * cell_w
        y = title_h + pad + r * cell_h
        im = PIL.Image.fromarray(tile.astype(np.uint8))
        if tile_px != src:
            im = im.resize((tile_px, tile_px), PIL.Image.LANCZOS)
        sheet.paste(im, (x, y))
        if labels:
            draw.text((x + 1, y + tile_px + 2), str(labels[i]), fill=fg, font=font)

    return sheet


def browse_layer(
    backbone: Backbone,
    layer: str,
    *,
    channels=None,
    top: int | None = None,
    rank_seed=None,
    out_dir="out/browse",
    size: int = 256,
    tile_px: int = 128,
    cols: int = 8,
    per_page: int = 64,
    steps: int = 96,
    step_size: float = 0.05,
    seed="gray",
    batch_size: int = 16,
    octaves=(-2, -1, 0),
    jitter: int = 8,
    grad_blur: float = 1.0,
    deterministic: bool = True,
    save_tiles: bool = False,
    progress=None,
) -> dict:
    """Dream a layer's channels and write contact sheets + an index.

    Output under `<out_dir>/<backbone>-<layer>/`:
      * `sheet_000.png`, ... — the pages to actually look at
      * `index.json` — channel order, page/slot of each tile, and the settings
        used, so a pick can be traced back and a run reproduced
      * `tiles/ch####.png` — full-size individual tiles (`save_tiles=True`)
    """
    n_ch = backbone.channels(layer)
    if channels is None:
        channels = list(range(n_ch))
    channels = [int(c) for c in channels]

    ranking = None
    if top:
        ranking = rank_channels(backbone, layer, seed=rank_seed, size=size)
        keep = {c for c, _ in ranking[:top]}
        channels = [c for c in channels if c in keep]

    out_dir = pathlib.Path(out_dir) / f"{backbone.name}-{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tiles = dream_channels(
        backbone,
        layer,
        channels,
        size=size,
        steps=steps,
        step_size=step_size,
        seed=seed,
        batch_size=batch_size,
        octaves=octaves,
        jitter=jitter,
        grad_blur=grad_blur,
        deterministic=deterministic,
        progress=progress,
    )

    if save_tiles:
        tile_dir = out_dir / "tiles"
        tile_dir.mkdir(exist_ok=True)
        for ch, tile in zip(channels, tiles):
            PIL.Image.fromarray(tile).save(tile_dir / f"ch{ch:04d}.png")

    pages, entries = [], []
    for page, start in enumerate(range(0, len(channels), per_page)):
        chunk = channels[start : start + per_page]
        path = out_dir / f"sheet_{page:03d}.png"
        contact_sheet(
            tiles[start : start + len(chunk)],
            labels=chunk,
            cols=cols,
            tile_px=tile_px,
            title=(
                f"{backbone.name} / {layer} — channels {chunk[0]}..{chunk[-1]} "
                f"(page {page + 1} of {-(-len(channels) // per_page)}, "
                f"{n_ch} channels total)"
            ),
        ).save(path)
        pages.append(str(path))
        entries += [
            {"channel": ch, "page": page, "slot": i} for i, ch in enumerate(chunk)
        ]

    index = {
        "backbone": backbone.name,
        "layer": layer,
        "layer_channels": n_ch,
        "browsed": len(channels),
        "settings": {
            "size": size,
            "steps": steps,
            "step_size": step_size,
            "seed": seed if isinstance(seed, str) else "image",
            "octaves": list(octaves),
            "jitter": jitter,
            "grad_blur": grad_blur,
            "deterministic": deterministic,
            "batch_size": batch_size,
            "cols": cols,
            "per_page": per_page,
            "tile_px": tile_px,
        },
        "pages": pages,
        "channels": entries,
    }
    if ranking is not None:
        index["ranking_top"] = [
            {"channel": c, "score": s} for c, s in ranking[: top or 0]
        ]
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def parse_channel_spec(spec: str, n_channels: int) -> list[int]:
    """Parse `all`, `0-767`, `10,20,30-40`, `::4` into a channel list."""
    spec = str(spec).strip()
    if spec in ("", "all", "*"):
        return list(range(n_channels))
    if spec.startswith("::"):
        return list(range(0, n_channels, int(spec[2:])))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    seen, unique = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique
