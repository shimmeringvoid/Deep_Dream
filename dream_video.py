#!/usr/bin/env python
"""dream_video.py — dream an existing video, frame by frame, with coherence.

A bridge to milestone 3, built only on what `engine/` already exposes.
Implements the coherence rule from CLAUDE.md with the identity warp:

    seed(t) = a * dream[t-1] + (1 - a) * frame[t]

which is the right default for footage whose frame-to-frame change happens
in place (a Julia parameter morph, ordinary footage).

For fractal *zooms* the previous dream is a frame behind the footage, and
blending it unwarped smears dream features radially outward. `--zoom F`
supplies the warp: the previous dream is magnified about centre by F (the
footage's per-SOURCE-frame zoom factor) before it seeds the current frame.
A clip zooming R times per second at `fps` wants `F = R ** (1/fps)`. This is
milestone 5's constant-rate case; reading a variable rate from a Fractal
Studio sidecar is still to come, as is a non-centred zoom (`image.zoom`
scales about the frame centre, so check that is where the clip zooms into).

    conda activate deepdream

    # 5-second test slice, 1280 px wide, flowers preset:
    python dream_video.py julia.mp4 --preset flowers --start 60 --duration 5 --max-dim 1280

    # a fractal zoom at 1.5x per second, 30 fps -> 1.5**(1/30):
    python dream_video.py zoom.mp4 --preset flowers --zoom 1.013607

    # full-length render on one card:
    python dream_video.py julia.mp4 --preset flowers --max-dim 1920

    # chunked across cards (run one per terminal; --warmup hides the seams):
    python dream_video.py julia.mp4 --preset flowers --device cuda:1 \
        --start 45 --duration 45 --warmup 1 --frames-dir out/video/julia_flowers/frames

Dreamed frames land as numbered PNGs (so a killed run RESUMES where it
stopped — just re-run the same command), then get assembled into an mp4 with
the repo's proven x264 settings. Frames dreamed by parallel chunk runs into a
shared --frames-dir are assembled into one video by whichever run finishes
last (or by --assemble-only).
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import time

import imageio.v2 as imageio

# Modules (not bare names) so the seams stay patchable/testable. `dream` is the
# one exception: `engine/__init__.py` re-exports the dream() *function* under
# that name, which shadows the `engine.dream` submodule attribute — so even
# `import engine.dream as m` binds the function. Take it directly, as cli.py
# and the tests already do.
from engine.dream import dream_tensor
from engine import image as engine_image
from engine import model as engine_model
from engine import objective as engine_objective
from engine import presets as engine_presets


def _octave_range(spec: str) -> list[int]:
    """`lo:hi` inclusive, or a single octave like `0`."""
    if ":" in spec:
        lo, hi = (int(x) for x in spec.split(":"))
        return list(range(lo, hi + 1))
    return [int(spec)]


def _even(x: float) -> int:
    return max(2, int(x) // 2 * 2)


def _work_size(h: int, w: int, max_dim: int | None) -> tuple[int, int]:
    """Target (height, width): longest side <= max_dim, both sides even."""
    scale = 1.0 if not max_dim else min(1.0, max_dim / max(h, w))
    return _even(round(h * scale)), _even(round(w * scale))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dream_video.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="input video (anything imageio-ffmpeg reads)")
    ap.add_argument("--backbone", default=engine_model.DEFAULT_BACKBONE,
                    choices=engine_model.available_backbones())
    ap.add_argument("--device", default=None, help="cuda:0 | cuda:1 | cpu")
    ap.add_argument("--preset", default=None,
                    help="a name from presets/ (overrides --targets)")
    ap.add_argument("--targets", default="mixed7,mixed9",
                    help="comma-separated layer[:channel[:weight]] specs")

    ap.add_argument("--out", default=None,
                    help="output mp4 (default: derived, under out/video/)")
    ap.add_argument("--frames-dir", default=None,
                    help="where dreamed PNG frames accumulate (default: next to "
                         "--out). Point chunked runs at the SAME dir.")

    ap.add_argument("--max-dim", type=int, default=1920,
                    help="downscale so the longest side is this (default 1920; "
                         "0 = native resolution)")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the source")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds of source to dream (default: to the end)")
    ap.add_argument("--every", type=int, default=1,
                    help="dream every Nth source frame (output fps drops to match)")
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="seconds of extra frames dreamed BEFORE --start and "
                         "discarded, so a chunk's first kept frame already has "
                         "coherent history (use ~1 on chunked runs)")

    ap.add_argument("--zoom", type=float, default=1.0,
                    help="per-SOURCE-frame zoom-in factor of the footage; the "
                         "previous dream is warped to match before blending.")
    ap.add_argument("--coherence", type=float, default=0.5,
                    help="a in seed = a*dream[t-1] + (1-a)*frame[t]. "
                         "0 = independent frames (flickery), higher = smoother "
                         "and dreamier trails (default 0.5)")
    ap.add_argument("--steps", type=int, default=24,
                    help="ascent steps per octave per frame (coherence carries "
                         "detail forward, so video wants far fewer than stills)")
    ap.add_argument("--step-size", type=float, default=0.01)
    ap.add_argument("--octaves", type=_octave_range, default=[-1, 0],
                    help="lo:hi inclusive (default -1:0 — small for video). A "
                         "negative low octave needs an equals sign: "
                         "--octaves=-2:0, else argparse reads it as a flag.")
    ap.add_argument("--tile-size", type=int, default=512)

    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--ffmpeg-preset", default="slow")
    ap.add_argument("--fresh", action="store_true",
                    help="re-dream frames even if their PNGs already exist")
    ap.add_argument("--no-assemble", action="store_true",
                    help="stop after writing PNG frames")
    ap.add_argument("--assemble-only", action="store_true",
                    help="skip dreaming; just assemble --frames-dir into --out")
    return ap


def resolve_paths(args) -> tuple[pathlib.Path, pathlib.Path]:
    stem = pathlib.Path(args.input).stem
    tag = args.preset or "custom"
    span = f"__{args.start:g}s" + (f"-{args.duration:g}s" if args.duration else "")
    base = f"{stem}__{tag}__{args.max_dim or 'native'}px"
    out = pathlib.Path(args.out) if args.out else pathlib.Path("out/video") / f"{base}{span}.mp4"
    frames = pathlib.Path(args.frames_dir) if args.frames_dir else out.with_suffix("") / "frames"
    return out, frames


def assemble(frames_dir: pathlib.Path, out: pathlib.Path, fps: float, args) -> None:
    paths = sorted(frames_dir.glob("*.png"))
    if not paths:
        sys.exit(f"No frames in {frames_dir} to assemble.")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-crf", str(args.crf), "-preset", args.ffmpeg_preset],
        macro_block_size=1,  # dims are already even; don't let imageio resize
    )
    for p in paths:
        writer.append_data(imageio.imread(p))
    writer.close()
    print(f"wrote {out}  ({len(paths)} frames at {fps:g} fps, silent)")
    print(f"frames kept in {frames_dir} — delete when happy, or re-run "
          f"--assemble-only after more chunks land.")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    engine_model.get_device(args.device)  # fail fast on a bad device spec
    out_path, frames_dir = resolve_paths(args)

    reader = imageio.get_reader(args.input)
    meta = reader.get_meta_data()
    fps = float(meta["fps"])
    src_dur = float(meta.get("duration") or math.nan)
    out_fps = fps / args.every

    if args.assemble_only:
        reader.close()
        assemble(frames_dir, out_path, out_fps, args)
        return

    backbone = engine_model.load_backbone(args.backbone, args.device)
    if args.preset:
        preset = engine_presets.load_preset(args.preset)
        if preset["backbone"] != backbone.name:
            print(f"warning: preset '{preset['name']}' was picked on "
                  f"{preset['backbone']}; channel numbers do not carry over to "
                  f"{backbone.name}.", file=sys.stderr)
        targets = preset["targets"]
    else:
        targets = engine_objective.parse_targets(args.targets)
    objective = engine_objective.Objective(targets, backbone)

    # Seek (fast, keyframe-approximate) to warmup seconds before the kept span.
    eff_start = max(0.0, args.start - args.warmup)
    if eff_start > 0:
        reader.close()
        reader = imageio.get_reader(args.input, input_params=["-ss", f"{eff_start:.3f}"])
    base_idx = round(eff_start * fps)        # absolute index of reader's frame 0
    save_from = round(args.start * fps)      # first absolute index we keep
    span = (args.duration + (args.start - eff_start)) if args.duration else (
        src_dur - eff_start if math.isfinite(src_dur) else None)
    n_src = round(span * fps) if span else None

    stop_at = base_idx + n_src if n_src else None
    expected = (len(range(save_from, stop_at)) + args.every - 1) // args.every if stop_at else None
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Warmup exists to give the first KEPT frame coherent history. If that
    # frame is already on disk (a resumed run), the resume-load supplies the
    # history and dreaming the warmup span would be wasted work — skip it.
    first_kept = ((save_from + args.every - 1) // args.every) * args.every
    warm_needed = base_idx < save_from and (
        args.fresh or not (frames_dir / f"{first_kept:08d}.png").exists())

    print(f"{objective} on {backbone.name} / {backbone.device}")
    print(f"source {meta['size'][0]}x{meta['size'][1]} @ {fps:g} fps"
          + (f", {src_dur:g}s" if math.isfinite(src_dur) else "")
          + f"  ->  dreaming from {args.start:g}s"
          + (f" for {args.duration:g}s" if args.duration else " to the end")
          + (f", every {args.every}th frame" if args.every > 1 else "")
          + (f", {args.warmup:g}s warmup" if eff_start < args.start else ""))

    prev = None          # last dreamed frame, model space, on-device
    done = skipped = 0
    work_hw = None
    t0 = time.time()

    for i, frame in enumerate(reader):
        abs_idx = base_idx + i
        if stop_at is not None and abs_idx >= stop_at:
            break
        if abs_idx % args.every:
            continue
        if abs_idx < save_from and not warm_needed:
            continue
        if work_hw is None:
            work_hw = _work_size(frame.shape[0], frame.shape[1], args.max_dim)
            print(f"dreaming at {work_hw[1]}x{work_hw[0]}, steps {args.steps} x "
                  f"octaves {args.octaves}, coherence {args.coherence:g}"
                  + (f", zoom warp {args.zoom:.6g}/frame" if args.zoom != 1.0
                     else ""))

        fpath = frames_dir / f"{abs_idx:08d}.png"
        if fpath.exists() and not args.fresh:
            # Resume: the saved frame becomes the coherence seed for the next.
            prev = engine_image.preprocess(engine_image.load_image(fpath),
                                           backbone.device)
            skipped += 1
            continue

        # Frame -> model space at working size. NOTE: resize() returns a float
        # tensor still in [0, 255], which preprocess() would mistake for an
        # already-preprocessed image — so do the [-1, 1] mapping explicitly.
        t = engine_image.resize(frame[..., :3], work_hw, backbone.device)
        t = (t / 127.5 - 1.0).clamp(-1.0, 1.0)
        # Milestone 5, constant-rate case: the previous dream is one source
        # frame (or `every` of them) behind the footage, so magnify it to where
        # its features have travelled before it seeds this frame. Without this
        # the dream lags the zoom and smears radially outward.
        if prev is not None and args.zoom != 1.0:
            prev = engine_image.zoom(prev, args.zoom ** args.every)
        if prev is not None and args.coherence > 0:
            t = (args.coherence * prev + (1.0 - args.coherence) * t).clamp(-1.0, 1.0)
            steps = args.steps
        else:
            steps = max(args.steps, 3 * args.steps)  # ramp the very first frame in

        dreamed = dream_tensor(
            t, objective,
            steps=steps,
            step_size=args.step_size,
            octaves=args.octaves,
            tile_size=args.tile_size,
        )
        prev = dreamed

        if abs_idx >= save_from:
            tmp = fpath.with_suffix(".tmp.png")
            engine_image.save_image(dreamed, tmp)
            os.replace(tmp, fpath)  # atomic: resume never sees a torn frame
            done += 1
            rate = (time.time() - t0) / max(done, 1)
            eta = f"  ETA {rate * (expected - done - skipped) / 60:6.1f} min" \
                if expected else ""
            print(f"  frame {abs_idx:>8}  {done + skipped}"
                  f"/{expected or '?'}  {rate:5.1f} s/frame{eta}", flush=True)

    reader.close()
    print(f"\n{done} frame(s) dreamed, {skipped} already existed.")
    if not args.no_assemble:
        assemble(frames_dir, out_path, out_fps, args)


if __name__ == "__main__":
    main()
