# deepdream-video — project context for Claude

Read this first. This is a standalone project. It is deliberately independent
of Fractal Studio (which is only one possible source of input frames).

## Status (2026-08-07)

* **Milestone 1 is DONE and verified on the workstation** ("monster"):
  `deepdream_pytorch.ipynb` — a PyTorch port of the TF DeepDream tutorial
  (simple dream, octaves, rolled tiled gradients), plus two forward-looking
  extras: a **working weighted multi-target objective** (`TargetDream` /
  `run_deep_dream_targets` — design commitment 2 has a reference
  implementation) and `dream_zoom_video` (frame t seeded from dreamed frame
  t-1 — the embryo of `coherence/`).
* The notebook is the **reference implementation**. When building `engine/`,
  lift and modularize from it rather than re-deriving.
* The GPU question below is resolved (cards are **Maxwell sm_52**) and the
  environment is settled and pinned — do not revisit the torch pin.

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

## Hardware & environment (settled 2026-07)

* Primary: "monster" workstation — Ubuntu 22.04, 4x GTX TITAN X **Maxwell
  (sm_52)**, 12 GB each, driver 535. Set up, verified, dreams rendered.
* Secondary: the newer Ubuntu 22.04 laptop.
* The Ubuntu 18.04 laptop (GTX 1650) is **out of scope** for this project:
  torch >= 2.7 wheels require glibc >= 2.28, and conda-forge's baseline is
  headed there too. (Documented fallback if it's ever needed:
  `torch==2.6.0+cu118` / `torchvision==0.21.0+cu118` — same code runs.)
* Env: `conda env create -f environment.yml` → env **`deepdream`**, Jupyter
  kernel registered as `deepdream`. Run project code inside that env
  (`conda activate deepdream`, or `conda run -n deepdream python ...`).
* **Pinned hard: `torch==2.7.1+cu118` / `torchvision==0.22.1+cu118`. NEVER
  upgrade torch in this env.** 2.7.1 is the final release with cu118 wheels
  and the last whose binaries ship Maxwell (sm_50) kernels; 2.8+ removes
  both. This pin changes only if the GPUs do. Full rationale lives in the
  comments of `environment.yml`.
* Maxwell constraints: **eager fp32 only** — no `torch.compile` (Triton needs
  sm_70+), no fp16/tensor cores. fp32 *is* the fast lane on these cards.
* Multi-GPU: frames are embarrassingly parallel → one worker process per card
  via `CUDA_VISIBLE_DEVICES`; no DDP/NCCL needed. **Caveat:** frame-seeded
  coherence makes frames within a chain sequential — shard by scene/chunk
  (each coherent run stays on one GPU), not by interleaved frame index.

## Proposed architecture

    io/            video<->frames (ffmpeg via imageio-ffmpeg), frame folders,
                   optional sidecar metadata (e.g. per-frame zoom factor)
    engine/        the dreamer:
                     - model loader (pretrained torchvision nets) + layer hooks
                     - objective = weighted sum over a target list of
                       (layer, channel, weight); .backward() to the image
                     - octave loop (scale 1.3, re-add detail), jitter roll,
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

Conventions already established by the notebook — keep them:
* User-facing layer names are Keras-style `mixed0..mixed10`, mapped to
  torchvision InceptionV3 modules (`Mixed_5b..Mixed_7c`) via the
  `KERAS_TO_TV` dict; feature taps via
  `torchvision.models.feature_extraction.create_feature_extractor`, one
  extractor per objective (build it from exactly the layers the targets use).
* `base_model.transform_input = False` + preprocessing `x/127.5 - 1`: inside
  the machinery images are float32 **HWC in [-1, 1]** on-device; at the human
  edges they are plain numpy uint8. `preprocess()` / `deprocess()` are the
  doors. (The AttributeError you get from mixing these up is the convention
  announcing itself.)
* Octaves: scale 1.30 over `range(-2, 3)`. Tiled gradients with random roll,
  `tile_size=512`.
* Video writing: imageio + pip imageio-ffmpeg (static ffmpeg with libx264):
  `codec="libx264", pixelformat="yuv420p",
  output_params=["-crf","20","-preset","slow"]`; keep frame dimensions even.

## Feature discovery

Batch-dream noise/gray images per channel of a layer into a contact sheet;
eye-pick "flower"-like (or whatever) channels. ImageNet-trained nets have
many flora/texture detectors in mid layers. Save chosen targets as named
presets ("flowers", "eyes", "swirls") that the schedule can reference and
blend.

Notes for the implementation (milestone 2, next up):
* Channel counts at the current taps: `mixed7` → 768, `mixed9` → 2048.
* Dream many channels per GPU pass by **batching**: N low-res seeds
  (~256 px; gray/noise or a downscaled photo) with a per-batch-item
  single-channel loss (sum of `acts[layer][i, channel_i].mean()` over i) —
  gradients stay independent per item.
* Cheap pre-filter first: one forward pass on a seed image, rank channels by
  mean activation, contact-sheet the shortlist.

## Model choice (open question for rafa)

Milestone 1 runs torchvision **InceptionV3** (`IMAGENET1K_V1`,
`transform_input=False`) — the working default. The alternatives below stay
open for aesthetic comparison; the engine's model loader should make swapping
backbones easy.

* GoogLeNet / Inception-v1 — the classic DeepDream look; ornate, intricate.
  Mid layers like inception4c/4d.
* VGG16 — smooth, painterly.
* AlexNet — chunkier, larger-scale features; likely closest to rafa's
  original 2010s memory. Worth trying for nostalgic fidelity.

## Milestones

1. Single-image dream, verify the look — **DONE 2026-07**, verified on the
   monster (scope moved off the old laptop): `deepdream_pytorch.ipynb`.
2. Weighted multi-target objective (**done in-notebook**); contact-sheet
   channel browser — **next up**.
3. Coherent 5-second dreamed clip (frame-to-frame seeding), arbitrary input —
   embryo exists as `dream_zoom_video` in the notebook.
4. Time-varying schedule (targets/strength morphing across the clip).
5. Optional zoom-warp coherence using Fractal Studio's per-frame zoom sidecar.
6. Multi-GPU frame sharding on the workstation — see the coherence×sharding
   caveat under Hardware.

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
