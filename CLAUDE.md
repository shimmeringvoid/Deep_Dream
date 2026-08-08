# deepdream-video — project context for Claude

Read this first. This is a standalone project. It is deliberately independent
of Fractal Studio (which is only one possible source of input frames).

## Status (2026-08-08)

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
* **The browser now shards across all four GPUs** (2026-08-08): `--gpus 4`,
  ~3.3x, tiles **bit-identical** to the one-card run because the shards are cut
  on `batch_size` boundaries. With it, `mixed9` (all 2048 channels) has been
  browsed in 15.8 min. See "Multi-GPU browsing" and "Decisions settled
  2026-08-08". Nothing has been eye-picked out of those sheets yet.
* **Milestone 3 has a working bridge** (2026-08-08): `dream_video.py` in the
  repo root dreams an existing video frame by frame with the coherence rule
  under the identity warp, resumable numbered PNGs, house x264 assembly.
  Verified on real weights (see "Video dreaming" below). It is deliberately a
  script, not a package: `io/` and `coherence/` as proper modules, and a
  `cli.py video` subcommand, are still to build.

      conda activate deepdream
      python cli.py layers                       # taps and channel counts
      python cli.py browse --layer mixed7        # contact sheets -> out/browse/
      python cli.py browse --layer mixed9 --gpus 4   # 2048 channels, four cards
      python cli.py dream photo.jpg --preset flowers
      python dream_video.py clip.mp4 --preset flowers --max-dim 1920
      python -m pytest tests -q                  # 46 tests, ~11 s
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
  `engine/shard.py` already does exactly this for the channel browser
  (`--gpus 4`, ~3.3x); milestone 6 should copy its process/env/merge shape.
* GPU 0 also hosts an unrelated long-lived Jupyter kernel (`inception` env,
  ~4.7 GB, running since late July). Harmless at browse batch sizes (~1.7 GB
  per worker) but it does make GPU 0 the slow shard — check before blaming the
  code, and it is rafa's to kill, not Claude's.

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
    engine/shard.py      one browse worker process per GPU: batch-aligned
                         channel spans, CUDA_VISIBLE_DEVICES, merge. Also the
                         worker entry point (`python -m engine.shard job.json`).
    engine/presets.py    named target lists as JSON in presets/.
    cli.py               subcommands: layers | browse | dream | presets.
    tests/test_engine.py 46 tests; the slow ones need a GPU + weights, and one
                         is marked `multigpu`.

Bridged by a script (milestone 3, 2026-08-08):

    dream_video.py   repo root, not a package. Video -> dreamed video: reads
                     frames with imageio-ffmpeg, seeds each frame from the
                     last dreamed one (identity warp), writes absolute-indexed
                     PNGs atomically so a killed run resumes, assembles with
                     the house x264 settings. Chunked multi-GPU is manual —
                     one process per card over --start/--duration into a
                     shared --frames-dir, with --warmup hiding the seams.
                     It calls the seams below and adds no new machinery, so
                     io/ and coherence/ can absorb it without a rewrite.

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
    python cli.py browse --layer mixed9 --gpus 4         # all 2048, ~16 min
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

### Multi-GPU browsing — BUILT (`engine/shard.py`, `--gpus`) 2026-08-08

    python cli.py browse --layer mixed9 --gpus 4     # 2048 channels, ~3.3x
    python cli.py browse --layer mixed7 --gpus 0,2,3 # leave a card alone
    python cli.py browse --layer mixed7              # unchanged: one process

`--gpus` takes a count (`4`), an explicit list (`0,2,3`), or `all`; omitted
means one process on `--device`, exactly as before. To pin a single-process
browse to one card, that is still `--device cuda:3` — a bare `--gpus 3` means
*three cards*, not card 3.

**The alignment rule is the whole design.** A tile is only reproducible at the
batch size it was dreamed in (cuDNN picks kernels by batch shape; see the
determinism trap above), and `dream_channels` walks its list in chunks of
`batch_size`. So shards are **contiguous spans cut on `batch_size`
boundaries**, and every worker runs the *same* `batch_size`. Then each chunk
holds the same channels, in the same order, at the same shape as it would have
unsharded — and the sheets are bit-identical to the one-card run, so
`index.json`'s "re-run at this batch size to regenerate this tile" promise
survives sharding untouched. The obvious `channels[gpu::4]` round-robin would
repack every chunk with different neighbours and silently invalidate the whole
sheet; `plan_shards()` is the function that refuses to do that.

Consequence worth knowing: with fewer batch-aligned blocks than cards, spare
cards get nothing rather than the split being nudged off a boundary — 32
channels at batch 16 is two blocks, so two GPUs idle and the CLI says so.
Shrink `--batch` to spread work wider.

Mechanics: one worker process per card under `CUDA_VISIBLE_DEVICES=<n>` (each
sees its card as `cuda:0`), tiles handed back as `.npy` through
`out/.shards/`, parent concatenates in channel order and writes one sheet set
and one `index.json` — which now also records `settings.gpus`. The parent
holds **no CUDA context**: with `--gpus` it loads the backbone on the CPU,
since all it needs is the channel count and (for `--top`) the ranking pass.
One caveat from that: with `--gpus`, `--top`'s ranking is computed on the CPU,
so near-ties can order differently than the GPU would have. It is a shortlist
heuristic and the scores go into `index.json`, so this is noted, not a problem.
If any shard dies the rest are terminated immediately rather than grinding out
a quarter-browse that gets thrown away.

Verified 2026-08-08, at full browse defaults, tiles compared byte-for-byte:

* 32 channels (`mixed7` 0-31), `--batch 8`, 4 GPUs vs 1 GPU — all 32 tiles and
  the sheet PNG **identical**. 19.1 s vs 63.0 s (**3.3x**).
* 40 channels (`mixed7` 100-139), `--batch 16`, 4 GPUs vs 1 — identical too.
  This is the ragged case: 40 is 2.5 blocks, so three GPUs get 16/16/8 and the
  partial chunk stays at the end of the list where the unsharded run puts it.
* In the suite: `test_plan_shards*` (pure logic, instant) and
  `test_sharded_browse_matches_single_gpu` (`-m multigpu`, a cheap 2-GPU
  version of the check above).

Scaling is sublinear-ish because the cards are not identical in load — GPU 0
also hosts an unrelated long-lived Jupyter kernel (~4.7 GB), which is worth
remembering before blaming the sharding for a slow shard: on the `mixed9` run
the three clean cards finished 512 channels in 14.3 min and GPU 0 took
15.7 min. Memory is not the constraint: a worker at batch 16 / 256 px sits at
~1.7 GB, so batch could go a lot higher if wall-clock ever mattered more than
matching an existing sheet's batch size.

### `mixed9` browsed — all 2048 channels (2026-08-08)

    python cli.py browse --layer mixed9 --gpus 4     # 948.6 s = 15.8 min

32 sheets in `out/browse/inception_v3-mixed9/`, at browse defaults (256 px, 96
steps, octaves -2..0, batch 16, grad-blur 1.0, deterministic). Aggregate
2.18 ch/s against ~0.6 ch/s on one card — the four-card run is the difference
between "after lunch" and "while you make coffee", which is what makes
browsing a 2048-channel layer a normal thing to do rather than an expedition.

Two things to know before picking off these sheets:

* **`mixed9` tiles read much busier than `mixed7`'s.** Deeper layer, larger
  receptive field, and at 256 px the features are cramped: lots of ornate
  high-frequency filigree, fewer of the clean single-motif tiles that made
  `mixed7` easy to eye-pick. If picking proves hard, the knobs to try first
  are a bigger `--size` and a stronger `--grad-blur` (2.0), not more steps.
* **~9% of the layer is effectively blank at these settings** — 191 of 2048
  tiles come out near-flat gray (tile std < 15 against a median of 64), e.g.
  1028, 1037, 1040, 1043, 1049, 1053, 1070. They are spread across the whole
  layer, not clustered. These are channels a gray seed simply does not excite;
  `--top K --rank-image <your footage>` is the right way to skip them rather
  than paying to dream them.

Nothing has been eye-picked into a preset from `mixed9` yet — that is rafa's
call, and the eyes hunt (below) is the obvious first errand.

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
like eyes and faces live deeper. `mixed9`'s 2048 sheets now exist (2026-08-08,
15.8 min on four cards) and are where to hunt; `mixed10` is the next stop.

Browsed so far: `mixed7` (768) and `mixed9` (2048), both InceptionV3, both at
browse defaults. No other layer or backbone has been sheeted.

## Video dreaming — BRIDGED (`dream_video.py`) 2026-08-08

    python dream_video.py clip.mp4 --preset flowers --start 60 --duration 5 \
        --max-dim 1280                          # a review slice first
    python dream_video.py clip.mp4 --preset flowers --max-dim 1920
    python dream_video.py clip.mp4 --preset flowers --device cuda:1 \
        --start 45 --duration 45 --warmup 1 --frames-dir D --no-assemble
    python dream_video.py clip.mp4 --assemble-only --frames-dir D --out v.mp4

Milestone 3's behaviour, in one script at the repo root, built only on
`engine/` seams. `seed(t) = a*dream[t-1] + (1-a)*frame[t]` with the identity
warp — right for footage that changes in place (a Julia parameter morph,
ordinary video); fractal *zooms* want `image.zoom` as the warp, which is
milestone 5 and not done here.

Things worth knowing before using it:

* **Video wants far fewer steps than stills.** Defaults are steps 24, octaves
  `-1:0`, coherence 0.5 — feedback carries detail across frames, so each frame
  only has to add a little. The first frame of a chain gets 3x steps to ramp
  in. `--coherence` is the flicker/trails knob: 0 is independent frames and
  visibly boils, higher is smoother and dreamier.
* **Frames are absolute-indexed PNGs written atomically**, so a killed run
  resumes by re-running the same command, and a resumed run reloads the last
  frame on disk as its coherence seed rather than re-dreaming warmup.
* **Chunked multi-GPU is manual and works today**: one process per card over
  `--start`/`--duration` into a shared `--frames-dir` with `--no-assemble`,
  `--warmup 1` on every chunk but the first so a chunk's first kept frame has
  coherent history, then one `--assemble-only` pass. This is the scene-chunk
  shard milestone 6 wants, done by hand — note it is *not* `engine/shard.py`'s
  aligned-span planner, because frames in a coherent chain are sequential.
* **New settings need a new `--frames-dir`.** Resume matches on frame index
  alone; it cannot tell a frame dreamed at other settings from a current one.
* `--max-dim` defaults to 1920 (`0` = native). Sides are forced even for x264.

Verified 2026-08-08 on `julia_morph_z3_p_c.mp4` (3 min, 3840x2160 @ 30 fps),
`flowers` preset, InceptionV3, one card (`cuda:1`): 2 s slice at 640 px /
8 steps → 60 PNGs in 53 s wall (0.8 s/frame), assembled mp4 decodes to 60
frames at 640x360 @ 30 fps and plays. The dream is present but light at that
step count — a smoke test, not a look test.

One real bug was found and fixed by that first run: `from engine import dream`
gets the re-exported **function**, not the `engine.dream` submodule
(`engine/__init__.py` binds the name), so the module-style call raised
`AttributeError`. Even `import engine.dream as m` binds the function. Anything
new should import `from engine.dream import dream_tensor`, the way `cli.py`
and the tests already do.

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
   **BRIDGED 2026-08-08, not yet packaged.** `dream_video.py` does it end to
   end on real footage (see "Video dreaming" below); the coherence rule,
   resume, and chunked multi-GPU rendering all work. What remains for the
   milestone proper is structural: `io/` (video↔frames) and `coherence/` as
   modules, and a `cli.py video` subcommand, so the script's behaviour stops
   living in a script. The embryo it grew from is `dream_zoom_video` in the
   notebook; the seams are `engine.dream.dream_tensor` + `engine.image.zoom`.
4. Time-varying schedule (targets/strength morphing across the clip). The
   preset format is ready to be referenced by name from a schedule.
5. Optional zoom-warp coherence using Fractal Studio's per-frame zoom sidecar.
6. Multi-GPU frame sharding on the workstation — see the coherence×sharding
   caveat under Hardware. **Half-done early**: the channel browser already
   shards across all four cards (`engine/shard.py`, 2026-08-08); frames want
   the same process/env/merge shape with a scene-chunk planner instead of an
   aligned-span one.

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

## Decisions settled 2026-08-08 (the multi-GPU browse session)

* **Shards are batch-aligned contiguous spans, never round-robin.** The full
  reasoning is under "Multi-GPU browsing"; the one-line version is that a tile
  is reproducible only at its own batch size, so a cut anywhere but a
  `batch_size` boundary would repack the chunks and quietly break every sheet.
  `plan_shards()` owns this rule and `test_plan_shards_is_a_batch_aligned_
  partition` guards it.
* **`batch_size` is uniform across shards**, deliberately, for the same
  reason. `index.json` records one `batch_size` because there *is* one.
* **Processes, not threads or DDP.** `CUDA_VISIBLE_DEVICES` per worker,
  `.npy` through a scratch dir, merge in the parent. Same shape milestone 6
  wants for frame sharding, so `engine/shard.py` is the pattern to copy (the
  planner is not, though: frames in a coherent chain are sequential, so that
  split is by scene/chunk, not by aligned index spans).
* **The parent stays off the GPUs** when sharding — CPU backbone for metadata
  only. Cheap, and it keeps a card's worth of memory out of the picture.
* **Sheet writing is separate from dreaming** (`discover.write_sheets`), so
  the sharded and single-GPU paths converge on one implementation of "sheets +
  index" rather than growing two.

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
