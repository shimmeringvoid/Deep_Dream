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
* **Milestone 2 is DONE**: the notebook's machinery is lifted into `engine/`,
  and the contact-sheet channel browser works — all 768 `mixed7` channels
  sheeted and eye-pickable, three presets picked from them. See
  "Architecture", "Feature discovery" and "Decisions settled 2026-08-07" below.

      conda activate deepdream
      python cli.py layers                  # taps and channel counts
      python cli.py browse --layer mixed7   # contact sheets -> out/browse/
      python cli.py dream photo.jpg --preset flowers
      python -m pytest tests -q             # 30 tests, ~6 s
* The notebook is still the **reference implementation** for the *look*, and
  the place to play interactively. `engine/` is the same machinery, importable;
  `cli.py dream ... --targets mixed7,mixed9` reproduces the notebook's
  labrador. Extend `engine/` from here rather than re-deriving from the
  notebook.
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

## Architecture

Built (milestone 2):

    engine/image.py      the [-1,1] <-> uint8 doors, resize, roll, zoom,
                         load/save. Everything above this speaks model space.
    engine/model.py      Backbone = frozen eval-mode net + layer map + input
                         adapter; extractor cache; channel counts by probe.
    engine/objective.py  Target(layer, channel, weight) and Objective.loss().
    engine/dream.py      gradients / tiled_gradients / dream_tensor / dream —
                         the octave loop from the notebook.
    engine/discover.py   milestone 2: batched per-channel dreaming, the
                         gradient-blur regularizer, contact sheets, ranking.
    engine/presets.py    named target lists as JSON in presets/.
    cli.py               subcommands: layers | browse | dream | presets.
    tests/test_engine.py 30 tests; the slow ones need a GPU + weights.

Still to build:

    io/            video<->frames (ffmpeg via imageio-ffmpeg), frame folders,
                   optional sidecar metadata (e.g. per-frame zoom factor)
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
                   `image.zoom()` is already the fractal-zoom warp.
    cli.py         grows a `video` subcommand for the whole pipeline.

`dream_tensor()` is the seam `coherence/` should call: tensor in, tensor out,
at the input's size, so chained frames never round-trip through uint8.

Conventions already established by the notebook — keep them:
* User-facing layer names are Keras-style `mixed0..mixed10`, mapped to
  torchvision InceptionV3 modules (`Mixed_5b..Mixed_7c`) via the
  `KERAS_TO_TV` dict; feature taps via
  `torchvision.models.feature_extraction.create_feature_extractor`, one
  extractor per objective (build it from exactly the layers the targets use).
  Generalized in `engine/model.py`: each backbone carries its **own** layer
  map, so `mixed*` is InceptionV3's dialect, not a global vocabulary.
* `base_model.transform_input = False` + preprocessing `x/127.5 - 1`: inside
  the machinery images are float32 **HWC in [-1, 1]** on-device; at the human
  edges they are plain numpy uint8. `preprocess()` / `deprocess()` are the
  doors. (The AttributeError you get from mixing these up is the convention
  announcing itself.) Backbones whose weights want something else (VGG16,
  AlexNet want ImageNet normalization) convert in their **input adapter**, so
  the [-1, 1] convention holds everywhere above `engine/model.py`.
* Octaves: scale 1.30 over `range(-2, 3)`. Tiled gradients with random roll,
  `tile_size=512`.
* Video writing: imageio + pip imageio-ffmpeg (static ffmpeg with libx264):
  `codec="libx264", pixelformat="yuv420p",
  output_params=["-crf","20","-preset","slow"]`; keep frame dimensions even.

## Feature discovery — BUILT (`engine/discover.py`, `cli.py browse`)

Batch-dream gray/noise images per channel of a layer into a contact sheet;
eye-pick "flower"-like (or whatever) channels. Save chosen targets as named
presets that the schedule can reference and blend.

    python cli.py browse --layer mixed7                  # all 768, ~19 min
    python cli.py browse --layer mixed9 --top 128        # shortlist first
    python cli.py browse --layer mixed7 --channels 0-63 --steps 200

Output lands in `out/browse/<backbone>-<layer>/`: `sheet_NNN.png` (64 tiles a
page, 8 wide, each labelled with its channel number), `index.json` (channel →
page/slot, plus every setting used), and `tiles/chNNNN.png` with `--save-tiles`.

Confirmed channel counts: `mixed7` → 768, `mixed9` → 2048 (`cli.py layers`
prints them all).

Three things that had to be got right, all now defaults:

* **Batching works and is safe.** N seeds per GPU pass with a per-item
  single-channel loss (`sum_i acts[layer][i, channel_i].mean()`). BatchNorm is
  in eval mode so nothing couples the items — verified bit-exact against
  unbatched at one step (`tests/test_engine.py`). Batch 16 at 256 px uses
  ~6.5 GB, so it fits a Titan X with room.
* **The gradient-blur regularizer is not optional.** Unregularized ascent on a
  single channel piles up high-frequency hash and every tile ends up looking
  like the same rainbow static — useless for picking. Low-passing the *ascent
  direction* (separable Gaussian, `--grad-blur`, default σ=1.0) pushes growth
  into the low frequencies where recognizable shape lives. Compare σ=0 against
  σ=1.0 once and the reason is obvious.
* **cuDNN determinism is on by default**, and this one is a trap worth
  remembering: single-channel ascent is *chaotic*. cuDNN's default atomics-based
  conv backward differs in the last bits run to run, and by ~96 steps that
  amplifies into a completely different picture (measured: mean 71/255 per
  pixel between two identical runs — as different as two unrelated channels).
  Without determinism a channel you liked on a sheet cannot be regenerated at
  all. Costs ~13% wall-clock; `--nondeterministic` opts out.
  Corollary: reproducing a tile also needs the same `--batch`, since cuDNN
  picks different kernels per batch shape. `index.json` records it.

Browse defaults: 256 px seeds, 96 steps per octave, octaves -2..0, step 0.05,
low-frequency gray seed (noise generated at 1/8 scale and upsampled — a
white-noise seed starts the image in the hash regime the blur is fighting),
per-channel RNG seeds so a tile depends on its channel, not its batch-mates.

Cheap pre-filter, for the other question: `--top K --rank-image shot.png` does
one forward pass and keeps the K channels *that image* already excites. Use it
when hunting for channels that will do something to particular footage; for
"what is this layer made of", just browse all of them.

### Presets picked from that first sheet (2026-08-07)

All `mixed7` / InceptionV3, eye-picked off `out/browse/inception_v3-mixed7`,
verified by rendering each on the labrador (`cli.py dream ... --preset X`):

* **flowers** — 634 (dense florets, the best in the layer), 12 (buds and
  pinecones), 602 (pale open blossoms), 292 (spiky thistle, adds bite).
  The origin-story preset.
* **spirals** — 632 (ornate scrollwork), 631 (concentric shell), 8 (loose
  vortex), 603 (wavy ribbons, loosens the tighter three). Renders beautifully;
  the natural partner for a flowers→spirals morph in milestone 4.
* **scales** — 55 (roof shingles), 43 (overlapping scales), 610 (scalloped
  rows), 611 (fish scales), 10 (fan arcs). Strong on flat regions.

**Not found: eyes.** `mixed7` has eye-*ish* channels (274, 309) but nothing
convincing. Expected — `mixed7` is a texture/part layer; whole-object features
like eyes and faces live deeper. Browse `mixed9` (2048 channels, ~50 min at
defaults) or `mixed10` when an eyes preset is wanted.

Layers other than `mixed7` have not been browsed yet.

## Model choice (still open for rafa — but now a flag away)

**InceptionV3** (`IMAGENET1K_V1`, `transform_input=False`) stays the default.
All four candidates are wired into `engine/model.py` and selectable with
`--backbone`; each brings its own layer map and input adapter, so trying one
costs nothing. `cli.py --backbone X layers` lists its taps and channel counts.

* `inception_v3` — default. Taps `mixed0..mixed10`.
* `googlenet` — Inception-v1, the classic DeepDream look; ornate, intricate.
  Taps `inception3a..inception5b`; `inception4c`/`4d` are the famous ones.
* `vgg16` — smooth, painterly. Taps `relu1_2..relu5_3`.
* `alexnet` — chunkier, larger-scale features; likely closest to rafa's
  original 2010s memory. Taps `conv1..conv5`. Worth trying for nostalgic
  fidelity.

All four were smoke-tested on the labrador 2026-08-07 and behave as the notes
above predict — GoogLeNet ornate and dog-slug classic, VGG16 dense floral,
AlexNet big swirling bursts. The real aesthetic comparison is an eye-judgment
for rafa; `cli.py browse --backbone googlenet --layer inception4c` is the way
to look at each one's vocabulary.

Practical note: **step sizes don't port between backbones either.** VGG16 and
AlexNet activations are much larger than Inception's, so `--step-size 0.01`
(fine on InceptionV3) drives them to saturation and erases the source image.
Start an order of magnitude lower on those two and work up.

**Channel numbers are backbone-specific and do not port.** A preset records
which backbone it was picked on, and `cli.py dream` warns on a mismatch.

## Milestones

1. Single-image dream, verify the look — **DONE 2026-07**, verified on the
   monster (scope moved off the old laptop): `deepdream_pytorch.ipynb`.
2. Weighted multi-target objective; contact-sheet channel browser —
   **DONE 2026-08-07**. Notebook machinery lifted into `engine/`, targets and
   presets are first-class, all 768 `mixed7` channels sheeted. Tests in
   `tests/`.
3. Coherent 5-second dreamed clip (frame-to-frame seeding), arbitrary input —
   **next up**. Embryo exists as `dream_zoom_video` in the notebook; the seam
   to build on is `engine.dream.dream_tensor` + `engine.image.zoom`. Needs
   `io/` (video↔frames) and `coherence/`.
4. Time-varying schedule (targets/strength morphing across the clip). The
   preset format is ready to be referenced by name from a schedule.
5. Optional zoom-warp coherence using Fractal Studio's per-frame zoom sidecar.
6. Multi-GPU frame sharding on the workstation — see the coherence×sharding
   caveat under Hardware.

## Decisions settled 2026-08-07 (the engine/ + browser session)

These are commitments now; change them deliberately, not by drift.

* **Target representation.** `Target(layer, channel, weight)`, with
  `channel=None` meaning the whole-layer mean — so the stock tutorial
  objective is the degenerate case of design commitment 2, not a separate code
  path. Text form is `layer[:channel[:weight]]`: `mixed7:10:0.7`,
  `mixed9:*:-0.5`, or a bare `mixed7`. Comma-separated on the CLI.
* **Preset format.** JSON at `presets/<name>.json`:
  `{name, backbone, notes, targets: [{layer, channel, weight}]}`. Objects, not
  tuples, so a preset stays readable and diffable. `backbone` is required
  because channel numbers don't port. `notes` is where the provenance goes
  ("picked off sheet 003, 2026-08-07"). `engine/presets.py` reads and writes.
* **Backbone abstraction.** A `Backbone` is a frozen eval-mode net + a layer
  map + an input adapter. The adapter is what keeps [-1, 1] universal while
  letting VGG/AlexNet have their ImageNet normalization. Extractors are cached
  per (backbone, layer tuple) — `create_feature_extractor` re-traces otherwise,
  ~1 s each time.
* **Determinism in discovery is on by default.** See the Feature discovery
  section for why. Rendering (`cli.py dream`) leaves it off — a dream of a real
  photo is not chaotic in the same way, and the 13% matters more there.
* **Testing.** `conda run -n deepdream python -m pytest tests -q`. Pure-logic
  tests are instant; `-m slow` ones want the GPU. The load-bearing test is
  `test_batched_channel_dream_matches_single` — if batching ever starts
  coupling items, every contact sheet silently becomes wrong, and that test is
  the only thing that would notice.
* **Not adopted:** no `torch.compile` (Maxwell), no fp16, no pip-installable
  package (`python cli.py` from the repo root, inside the `deepdream` env).

## Open design questions for rafa

* Network aesthetic (GoogLeNet vs AlexNet-era vs VGG)?
* Dream strength constant, or ramping/pulsing over the video?
* Per-video single channel, or weighted multi-node blends that shift over time?
  (Partly answered by the sheets: single channels are *too* uniform across a
  frame — the 4-5 channel weighted blends in `presets/` read much better. The
  open part is how fast blends should shift over a clip.)
* Dream resolution (dream at 1080p–2K, upscale/tile to 4K) vs native 4K?

## Interop with Fractal Studio

Fractal Studio (separate repo) emits numbered PNG frames via its video
dialogs' "keep PNG frames" option; for zoom videos it can also write a sidecar
(e.g. frames_meta.json) with the per-frame zoom factor. This project consumes
that folder (+ optional sidecar) and knows nothing else about fractals.
