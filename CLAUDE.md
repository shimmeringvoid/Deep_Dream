# deepdream-video — project context for Claude

Read this first. This is a standalone project. It is deliberately independent
of Fractal Studio (which is only one possible source of input frames).

## What this is

A general **video-to-video DeepDream filter**. Input: any video (or a folder
of numbered frames). Output: a dreamed video, where a chosen set of CNN
feature detectors is amplified in each frame via iterated gradient ascent on
the image ("Inceptionism", Mordvintsev/Olah/Tyka, Google 2015).

Origin: the owner (rafa) independently discovered this effect years ago on an
early CNN — feeding a "flower detector" node's output back as input amplified
flower hallucinations over ~5 iterations. This project generalizes that into a
flexible, controllable video filter.

Design commitments (these are the point of the project, not extras):
1. **Any input video**, not just fractals. Fractal Studio frames are one case.
2. **Flexible targets:** dream toward a *weighted combination* of feature
   nodes — a list of (layer, channel, weight) — not just a single channel.
   Weights can be negative (suppress a feature). Single-channel is the
   one-element case.
3. **Time-varying parameters:** every dream parameter (target weights, overall
   strength, layer selection, octave count) can vary smoothly per frame,
   driven by a keyframed/parametrized envelope over the timeline. E.g. flowers
   morphing into spirals across a clip; strength swelling on a musical build.

## Why separate from Fractal Studio

* Different, fragile dependency stacks that shouldn't co-resolve: Fractal
  Studio needs a conda-forge PySide6 env pinned by Ubuntu-18.04 glibc 2.27;
  this needs PyTorch/CUDA pinned by GPU compute capability. One env file
  trying to satisfy both is asking for trouble.
* This filter is independently useful on arbitrary footage.
* Clean seam: the two projects meet only through a **folder of numbered PNG
  frames**. No shared code.

## Hardware (shared with Fractal Studio owner)

* Laptop: GTX 1650 (Turing, sm_75), i7-9750H, Ubuntu 18.04.6. Good dev/test
  GPU — sm_75 is supported by any recent PyTorch. Prototype here.
* Workstation (render farm, not yet set up): 4x "Titan X", ~16 cores.
  **Ambiguous GPU:** Titan X Maxwell (sm_52) vs Titan X Pascal (sm_61).
  Recent PyTorch binaries may have dropped Maxwell. FIRST STEP on that box:
  `nvidia-smi`, then check `torch.cuda.get_arch_list()` vs the card; if
  Maxwell, pin an older torch (cu11x, ~1.13/2.0-era). Frames are
  embarrassingly parallel → shard frame ranges across the 4 GPUs.

## Proposed architecture

    io/            video<->frames (ffmpeg via imageio-ffmpeg), frame folders,
                   optional sidecar metadata (e.g. per-frame zoom factor)
    engine/        the dreamer:
                     - model loader (pretrained torchvision nets) + layer hooks
                     - objective = weighted sum over a target list of
                       (layer, channel, weight); .backward() to the image
                     - octave loop (scale ~1.4, re-add detail), jitter roll,
                       normalized-gradient ascent step, clamp
    schedule/      time-varying parameter envelopes: map frame index/time ->
                   {targets+weights, strength, layer, octaves...}. Keyframes
                   with easing; or parametrized curves. Design rule: envelope
                   evolves SLOWLY relative to frame rate so it doesn't fight
                   temporal coherence.
    coherence/     seed frame t from frame t-1 to kill flicker:
                     init(t) = a * warp(dream[t-1]) + (1-a) * frame[t]
                   warp = identity by default; = known transform if provided
                   (for fractal zooms, a pure scale-about-center by the
                   per-frame factor, read from the sidecar). Fallbacks:
                   plain blend, or optical flow for arbitrary footage.
    cli.py         run the whole pipeline from a config file / args.

## Feature discovery

Batch-dream noise/gray images per channel of a layer into a contact sheet;
eye-pick "flower"-like (or whatever) channels. ImageNet-trained nets have
many flora/texture detectors in mid layers. Save chosen targets as named
presets ("flowers", "eyes", "swirls") that the schedule can reference and
blend.

## Model choice (open question for rafa)

* GoogLeNet / Inception-v1 — the classic DeepDream look; ornate, intricate.
  Mid layers like inception4c/4d.
* VGG16 — smooth, painterly.
* AlexNet — chunkier, larger-scale features; likely closest to rafa's
  original 2010s memory. Worth trying for nostalgic fidelity.

## Milestones

1. Single-image dream on the laptop GPU (one channel), verify the look.
2. Weighted multi-target objective; contact-sheet channel browser.
3. Coherent 5-second dreamed clip (frame-to-frame seeding), arbitrary input.
4. Time-varying schedule (targets/strength morphing across the clip).
5. Optional zoom-warp coherence using Fractal Studio's per-frame zoom sidecar.
6. Multi-GPU frame sharding on the workstation.

## Open design questions for rafa

* Network aesthetic (GoogLeNet vs AlexNet-era vs VGG)?
* Dream strength constant, or ramping/pulsing over the video?
* Per-video single channel, or weighted multi-node blends that shift over time?
* Dream resolution (dream at 1080p–2K, upscale/tile to 4K) vs native 4K?

## Interop with Fractal Studio

Fractal Studio (separate repo) emits numbered PNG frames via its video
dialogs' "keep PNG frames" option; for zoom videos it can also write a sidecar
(e.g. frames_meta.json) with the per-frame zoom factor. This project consumes
that folder (+ optional sidecar) and knows nothing else about fractals.
