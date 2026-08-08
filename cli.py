#!/usr/bin/env python
"""deepdream-video command line.

    conda activate deepdream          # or: conda run -n deepdream python cli.py ...

    python cli.py layers                                   # what can I tap?
    python cli.py browse --layer mixed7                    # contact-sheet all 768
    python cli.py browse --layer mixed9 --gpus 4           # all 2048, four cards
    python cli.py browse --layer mixed9 --top 128          # shortlist first
    python cli.py dream photo.jpg --targets mixed7:10:1,mixed9:4:-0.5
    python cli.py dream photo.jpg --preset flowers
    python cli.py presets

Milestone 2 lives in `browse`: dream one small image per channel and lay them
out as labelled sheets, so channels stop being anonymous integers. Picks go
into `presets/*.json` (see `engine/presets.py`), which milestone 4's schedule
will reference by name.
"""

from __future__ import annotations

import argparse
import sys
import time

from engine import discover, image as imageio_, presets as presets_mod, shard
from engine.dream import OCTAVES, dream
from engine.model import DEFAULT_BACKBONE, available_backbones, get_device, load_backbone
from engine.objective import Objective, parse_targets


def _progress(done, total, elapsed):
    rate = done / max(elapsed, 1e-9)
    eta = (total - done) / max(rate, 1e-9)
    print(
        f"  {done}/{total} channels  {elapsed:6.1f}s elapsed  "
        f"{rate:5.2f} ch/s  ETA {eta / 60:5.1f} min",
        flush=True,
    )


def cmd_layers(args):
    backbone = load_backbone(args.backbone, args.device)
    print(f"{backbone.name} on {backbone.device}")
    for name in backbone.layers:
        print(f"  {name:<12} {backbone.layer_map[name]:<16} {backbone.channels(name):>5} channels")


def cmd_browse(args):
    gpus = shard.parse_gpu_spec(args.gpus)
    # When the workers own the cards, the parent has no business holding a CUDA
    # context: it only needs the channel count and (for --top) one ranking pass.
    backbone = load_backbone(args.backbone, "cpu" if gpus else args.device)
    n_ch = backbone.channels(args.layer)
    channels = discover.parse_channel_spec(args.channels, n_ch)

    seed = args.seed
    if seed not in ("gray", "noise"):
        seed = imageio_.load_image(seed)  # a real image as the seed
    rank_seed = imageio_.load_image(args.rank_image) if args.rank_image else None

    print(
        f"{backbone.name}/{args.layer}: {n_ch} channels, browsing {len(channels)}"
        f"{f' (top {args.top} by activation)' if args.top else ''} "
        f"at {args.size}px x {args.steps} steps x {len(args.octaves)} octaves, "
        f"batch {args.batch}"
    )
    if gpus:
        spans = shard.plan_shards(len(channels), len(gpus), args.batch)
        if len(spans) < len(gpus):
            print(
                f"note: {len(channels)} channels at batch {args.batch} is only "
                f"{len(spans)} batch-aligned block(s), so {len(spans)} of "
                f"{len(gpus)} GPU(s) get work. Shrink --batch to spread it "
                f"wider (tiles are only reproducible at their own batch size, "
                f"so the shards cannot be cut anywhere else).",
                file=sys.stderr,
            )
        print(f"sharding across GPU(s) {', '.join(str(g) for g in gpus[: len(spans)])}:")
    t0 = time.time()
    index = discover.browse_layer(
        backbone,
        args.layer,
        channels=channels,
        top=args.top,
        rank_seed=rank_seed,
        out_dir=args.out,
        size=args.size,
        tile_px=args.tile_px,
        cols=args.cols,
        per_page=args.per_page,
        steps=args.steps,
        step_size=args.step_size,
        seed=seed,
        batch_size=args.batch,
        octaves=args.octaves,
        jitter=args.jitter,
        grad_blur=args.grad_blur,
        deterministic=not args.nondeterministic,
        save_tiles=args.save_tiles,
        gpus=gpus,
        progress=None if gpus else _progress,
    )
    print(f"\n{len(index['pages'])} sheet(s) in {time.time() - t0:.1f}s:")
    for p in index["pages"]:
        print("  ", p)


def cmd_dream(args):
    backbone = load_backbone(args.backbone, args.device)
    if args.preset:
        preset = presets_mod.load_preset(args.preset)
        if preset["backbone"] != backbone.name:
            print(
                f"warning: preset '{preset['name']}' was picked on "
                f"{preset['backbone']}; channel numbers do not carry over to "
                f"{backbone.name}.",
                file=sys.stderr,
            )
        targets = preset["targets"]
    else:
        targets = parse_targets(args.targets)

    objective = Objective(targets, backbone)
    print(f"{objective} on {backbone.name} / {backbone.device}")

    img = imageio_.load_image(args.input, max_dim=args.max_dim)
    t0 = time.time()
    out = dream(
        img,
        objective,
        steps=args.steps,
        step_size=args.step_size,
        octaves=args.octaves,
        tile_size=args.tile_size,
        callback=lambda o, s, _: print(f"  octave {o:+d} step {s}", flush=True),
    )
    path = imageio_.save_image(out, args.out)
    print(f"wrote {path} in {time.time() - t0:.1f}s")


def cmd_presets(args):
    names = presets_mod.list_presets()
    if not names:
        print("No presets yet — browse a layer, pick channels, then save one.")
        return
    for name in names:
        p = presets_mod.load_preset(name)
        print(f"{name}  [{p['backbone']}]  {p['notes']}")
        for t in p["targets"]:
            print(f"    {t}")


def _octave_range(spec: str) -> list[int]:
    lo, hi = (int(x) for x in spec.split(":"))
    return list(range(lo, hi + 1))


def build_parser():
    ap = argparse.ArgumentParser(prog="cli.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=available_backbones())
    ap.add_argument("--device", default=None, help="cuda:0 | cuda:1 | cpu (default: cuda:0)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("layers", help="list tappable layers and channel counts")
    p.set_defaults(func=cmd_layers)

    p = sub.add_parser("browse", help="contact-sheet a layer's channels")
    p.add_argument("--layer", default="mixed7")
    p.add_argument("--channels", default="all",
                   help="all | 0-767 | 10,20,30-40 | ::4  (default: all)")
    p.add_argument("--top", type=int, default=None,
                   help="pre-filter to the top-K channels by mean activation")
    p.add_argument("--rank-image", default=None,
                   help="image to rank against for --top (default: a gray seed)")
    p.add_argument("--out", default="out/browse")
    p.add_argument("--size", type=int, default=256, help="dream resolution per tile")
    p.add_argument("--tile-px", type=int, default=128, help="tile size on the sheet")
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--per-page", type=int, default=64)
    p.add_argument("--steps", type=int, default=96, help="ascent steps per octave")
    p.add_argument("--step-size", type=float, default=0.05)
    p.add_argument("--seed", default="gray", help="gray | noise | path/to/image")
    p.add_argument("--batch", type=int, default=16, help="channels per GPU pass")
    p.add_argument("--gpus", default=None,
                   help="fan out to one worker process per GPU: a count (4), an "
                        "explicit list (0,2,3), or 'all'. Omitted = one process "
                        "on --device. Shards are cut on --batch boundaries, so "
                        "the sheets come out identical to the single-GPU run")
    p.add_argument("--octaves", type=_octave_range, default=[-2, -1, 0],
                   help="lo:hi inclusive (default -2:0)")
    p.add_argument("--jitter", type=int, default=8)
    p.add_argument("--grad-blur", type=float, default=1.0,
                   help="Gaussian sigma for the gradient low-pass; 0 disables. "
                        "Higher = larger, more legible shapes")
    p.add_argument("--nondeterministic", action="store_true",
                   help="let cuDNN pick its faster atomics-based conv backward "
                        "(~13%% quicker, but tiles are then unreproducible)")
    p.add_argument("--save-tiles", action="store_true", help="also write per-channel PNGs")
    p.set_defaults(func=cmd_browse)

    p = sub.add_parser("dream", help="dream a single image")
    p.add_argument("input")
    p.add_argument("--out", default="out/dream.png")
    p.add_argument("--targets", default="mixed7,mixed9",
                   help="comma-separated layer[:channel[:weight]] (default: the "
                        "notebook's mixed7+mixed9 layer means)")
    p.add_argument("--preset", default=None, help="a name from presets/ (overrides --targets)")
    p.add_argument("--max-dim", type=int, default=None)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--step-size", type=float, default=0.01)
    p.add_argument("--octaves", type=_octave_range, default=list(OCTAVES))
    p.add_argument("--tile-size", type=int, default=512)
    p.set_defaults(func=cmd_dream)

    p = sub.add_parser("presets", help="list saved target presets")
    p.set_defaults(func=cmd_presets)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    get_device(args.device)  # fail fast on a bad device spec
    args.func(args)


if __name__ == "__main__":
    main()
