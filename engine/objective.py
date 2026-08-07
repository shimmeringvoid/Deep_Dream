"""The objective: a weighted shopping list of feature channels.

Design commitment #2 of the project — dream toward a *weighted combination* of
`(layer, channel, weight)` targets, with negative weights suppressing a
feature. Two degenerate cases fall out for free:

* one target with `weight=1.0` is single-channel dreaming;
* `channel=None` means "the whole layer's mean activation", which is exactly
  the stock TF tutorial's objective. `mixed7 + mixed9` at weight 1.0 each
  reproduces the notebook's opening dream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .image import to_batch


@dataclass(frozen=True)
class Target:
    """One term of the objective."""

    layer: str
    channel: int | None = None  # None -> the whole-layer mean
    weight: float = 1.0

    def __str__(self) -> str:
        ch = "*" if self.channel is None else self.channel
        return f"{self.layer}:{ch}:{self.weight:g}"

    def to_dict(self) -> dict:
        return {"layer": self.layer, "channel": self.channel, "weight": self.weight}

    @classmethod
    def from_dict(cls, d: dict) -> "Target":
        ch = d.get("channel")
        return cls(
            layer=d["layer"],
            channel=None if ch in (None, "*", "") else int(ch),
            weight=float(d.get("weight", 1.0)),
        )


def parse_target(spec: str) -> Target:
    """Parse `layer[:channel[:weight]]`, e.g. `mixed7:10:0.7`, `mixed9:*:-0.5`.

    A bare `mixed7` is the whole-layer mean at weight 1.0.
    """
    parts = [p.strip() for p in str(spec).split(":")]
    if not parts or not parts[0]:
        raise ValueError(f"Empty target spec: {spec!r}")
    layer = parts[0]
    channel = None
    if len(parts) > 1 and parts[1] not in ("", "*"):
        channel = int(parts[1])
    weight = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
    if len(parts) > 3:
        raise ValueError(f"Too many fields in target spec: {spec!r}")
    return Target(layer, channel, weight)


def parse_targets(specs) -> list[Target]:
    """Parse a list of specs, or one comma-separated string of them."""
    if isinstance(specs, str):
        specs = [s for s in specs.split(",") if s.strip()]
    return [s if isinstance(s, Target) else parse_target(s) for s in specs]


class Objective:
    """A target list bound to a backbone, exposing a differentiable loss.

    The loss is *maximized* (gradient ascent) — that is the whole trick.
    """

    def __init__(self, targets, backbone):
        self.targets = parse_targets(targets)
        if not self.targets:
            raise ValueError("An objective needs at least one target.")
        self.backbone = backbone
        self.extractor = backbone.extractor(
            dict.fromkeys(t.layer for t in self.targets)
        )
        for t in self.targets:
            if t.channel is not None:
                n = backbone.channels(t.layer)
                if not 0 <= t.channel < n:
                    raise ValueError(
                        f"{backbone.name}/{t.layer} has {n} channels; "
                        f"channel {t.channel} is out of range."
                    )

    @property
    def layers(self) -> list[str]:
        return list(self.extractor.layers)

    def loss(self, img: torch.Tensor) -> torch.Tensor:
        """Weighted objective for an HWC (or NCHW) image in [-1, 1]."""
        batch = img if img.dim() == 4 else to_batch(img)
        acts = self.extractor(batch)
        return sum(
            t.weight
            * (
                acts[t.layer].mean()
                if t.channel is None
                else acts[t.layer][:, t.channel].mean()
            )
            for t in self.targets
        )

    def __repr__(self) -> str:
        return f"Objective({', '.join(str(t) for t in self.targets)})"
