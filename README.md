# Deep_Dream

**A flexible DeepDream video filter in Python (PyTorch).** Amplifies chosen CNN feature detectors in any input video via iterated gradient ascent on the image, with weighted combinations of feature nodes, parameters that shift smoothly across frames, and frame-to-frame coherence to avoid flicker. Works on any footage — fractals included.

<!--
Hero image / example dreams go here. Note that .gitignore excludes *.png and
*.jpg repo-wide, so committed examples need either a docs/ exception
(e.g. `!docs/*.jpg`) in .gitignore or a `git add -f`.
-->

The project generalizes an effect the author stumbled on years ago with an early CNN: feeding a "flower detector" node's output back into its input amplified flower hallucinations within a handful of iterations. This repo turns that into a controllable filter — pick *which* features to amplify (or suppress), blend several at once, and (soon) morph the blend over the course of a clip.

The technique is ["Inceptionism"](https://research.google/blog/inceptionism-going-deeper-into-neural-networks/) (Mordvintsev, Olah & Tyka, Google Research, 2015); the included notebook is a PyTorch port of the [TensorFlow DeepDream tutorial](https://www.tensorflow.org/tutorials/generative/deepdream), extended well past it.

## Design commitments

These three ideas are the point of the project, not extras:

1. **Any input video.** Fractal-zoom footage is one interesting case, not a requirement.
2. **Flexible targets.** Dream toward a *weighted combination* of feature nodes — a list of `(layer, channel, weight)` — not just a single channel or layer. Weights can be negative to suppress a feature. Single-channel dreaming is just the one-element case.
3. **Time-varying parameters.** Every dream parameter (target weights, strength, layer selection, octave count) will be drivable by a keyframed envelope over the timeline — flowers morphing into spirals across a clip, strength swelling on a musical build.

## Status

| Milestone | State |
|---|---|
| 1. Single-image dream (notebook: simple dream, octaves, tiled gradients) | **Done** |
| 2. Weighted multi-target objective + contact-sheet channel browser (`engine/`, `cli.py`) | **Done** |
| — Multi-GPU channel browsing (`--gpus`, bit-identical to single-GPU output) | **Done** |
| 3. Coherent dreamed clips (frame-to-frame seeding) — working as `dream_video.py`; `io/` + `coherence/` modules and a `cli.py video` subcommand still to come | **Bridged** |
| 4. Time-varying schedules (`schedule/`, presets blended by name over time) | Planned |
| 5. Zoom-warp coherence from per-frame metadata sidecars | Planned |
| 6. Multi-GPU frame sharding | Planned |

Milestones 1 and 2 are fully usable today: dream single images with arbitrary weighted targets, and browse a network layer channel-by-channel to build your own presets. Video works too, via `dream_video.py` at the repo root — coherent frame-to-frame seeding, resumable frames, chunked rendering across cards — pending its promotion from a script into `io/` + `coherence/` modules behind a `cli.py video` subcommand.

## Quick start

```bash
conda env create -f environment.yml     # creates the `deepdream` env
conda activate deepdream

python cli.py layers                    # what can I tap?
python cli.py dream photo.jpg --preset flowers
python cli.py browse --layer mixed7     # contact-sheet all 768 channels
python dream_video.py clip.mp4 --preset flowers   # dream a video (see dream_video.py -h)
python -m pytest tests -q               # sanity-check the install
```

The repo is run in place from its root (`python cli.py ...`); it is deliberately **not** pip-installable. `pyproject.toml` only carries pytest configuration.

## The CLI

Global flags `--backbone` and `--device` go **before** the subcommand:

```bash
python cli.py --backbone googlenet layers
python cli.py --device cuda:1 dream photo.jpg --preset spirals
```

`--device` accepts `cuda:N` or `cpu` (default `cuda:0`). Everything works on CPU, just slowly.

### `layers` — list what you can tap

```bash
python cli.py layers
python cli.py --backbone vgg16 layers
```

Prints each tappable layer's friendly name, the torchvision module it maps to, and its channel count (e.g. `mixed7` → 768 channels, `mixed9` → 2048 on InceptionV3).

### `dream` — dream a single image

```bash
python cli.py dream photo.jpg                                  # tutorial look: mixed7+mixed9 layer means
python cli.py dream photo.jpg --preset flowers                 # a saved preset
python cli.py dream photo.jpg --targets mixed7:634:1,mixed7:12:0.8
python cli.py dream photo.jpg --targets mixed7:10:0.7,mixed9:*:-0.5
```

| Flag | Default | Meaning |
|---|---|---|
| `--targets` | `mixed7,mixed9` | comma-separated `layer[:channel[:weight]]` specs |
| `--preset` | — | a name from `presets/` (overrides `--targets`) |
| `--out` | `out/dream.png` | output path |
| `--max-dim` | — | downscale input so its longest side is this |
| `--steps` | `100` | ascent steps per octave |
| `--step-size` | `0.01` | ascent step size |
| `--octaves` | `-2:2` | inclusive `lo:hi` range of octave exponents |
| `--tile-size` | `512` | tile size for memory-flat tiled gradients |

### `browse` — contact-sheet a layer's channels

Channel indices are meaningless until you look at them. `browse` dreams one small seed image per channel and lays the results out as labelled contact sheets you can eye-pick from: *that one's a flower, that one's a spiral*. Picks become presets.

```bash
python cli.py browse --layer mixed7                    # all 768, ~19 min on one GTX Titan X
python cli.py browse --layer mixed9 --gpus 4           # all 2048 on four cards, ~16 min
python cli.py browse --layer mixed9 --top 128 --rank-image shot.png
python cli.py browse --layer mixed7 --channels 0-63 --steps 200
```

Output lands in `out/browse/<backbone>-<layer>/`:

- `sheet_NNN.png` — 64 tiles a page (8 wide), each labelled with its channel number
- `index.json` — channel → page/slot, plus every setting used (so any tile can be regenerated)
- `tiles/chNNNN.png` — per-channel PNGs, with `--save-tiles`

| Flag | Default | Meaning |
|---|---|---|
| `--layer` | `mixed7` | layer to browse |
| `--channels` | `all` | `all` \| `0-767` \| `10,20,30-40` \| `::4` |
| `--top` | — | pre-filter to the top-K channels by mean activation |
| `--rank-image` | gray seed | image to rank against for `--top` |
| `--size` | `256` | dream resolution per tile |
| `--steps` | `96` | ascent steps per octave |
| `--step-size` | `0.05` | ascent step size |
| `--octaves` | `-2:0` | octave range |
| `--seed` | `gray` | `gray` \| `noise` \| path to an image |
| `--batch` | `16` | channels dreamed per GPU pass (~6.5 GB at 256 px) |
| `--gpus` | — | fan out across GPUs: a count (`4`), a list (`0,2,3`), or `all` |
| `--grad-blur` | `1.0` | Gaussian σ for the gradient low-pass; `0` disables |
| `--nondeterministic` | off | ~13% faster, but tiles become unreproducible |
| `--jitter` / `--tile-px` / `--cols` / `--per-page` / `--out` / `--save-tiles` | | layout and output knobs |

Three defaults worth knowing about, because they were each hard-won:

- **The gradient-blur regularizer is not optional in practice.** Unregularized ascent on a single channel piles up high-frequency hash until every tile looks like the same rainbow static. Low-passing the *ascent direction* (`--grad-blur`, default σ=1.0) pushes growth into the low frequencies where recognizable shape lives.
- **cuDNN determinism is on by default.** Single-channel ascent is chaotic: cuDNN's default atomics-based convolution backward differs in the last bits from run to run, and by ~96 steps that amplifies into a completely different picture. Without determinism, a channel you liked on a sheet cannot be regenerated at all. Reproducing a tile also requires the same `--batch` (cuDNN picks kernels by batch shape) — `index.json` records everything needed.
- **The seed is low-frequency gray** (noise generated at 1/8 scale and upsampled), with per-channel RNG seeds, so a tile depends on its channel rather than its batch-mates or a hash-regime starting point.

**Multi-GPU browsing** (`--gpus`) runs one worker process per card via `CUDA_VISIBLE_DEVICES` — no DDP, no NCCL. Shards are contiguous channel spans cut **on `--batch` boundaries**, which is exactly what makes the sheets come out *bit-identical* to a single-GPU run (a round-robin split would silently change every tile — see the reproducibility note above). Note that a bare `--gpus 3` means *three cards*; to pin a single-process browse to card 3, use `--device cuda:3` instead.

Practical notes from browsing InceptionV3 so far: `mixed7` (768 channels) eye-picks easily; `mixed9` (2048) reads much busier at 256 px — try a bigger `--size` or `--grad-blur 2.0` before more steps — and ~9% of its channels come out near-blank on a gray seed, which `--top K --rank-image <your footage>` is the right way to skip.

### `presets` — list saved target presets

```bash
python cli.py presets
```

## Targets

A target is one term of the dream objective, written `layer[:channel[:weight]]`:

| Spec | Meaning |
|---|---|
| `mixed7` | whole-layer mean activation, weight 1.0 (the stock tutorial objective) |
| `mixed7:634` | channel 634 of `mixed7`, weight 1.0 |
| `mixed7:634:0.7` | channel 634, weight 0.7 |
| `mixed9:*:-0.5` | whole-layer mean of `mixed9`, weight −0.5 (**suppress** it) |

Comma-separate multiple targets on the CLI. The loss is the weighted sum of each target's mean activation; per-layer/per-channel means keep differently sized layers comparable.

## Presets

A preset is the durable output of a browsing session: named target lists as JSON in `presets/<name>.json`.

```json
{
  "name": "flowers",
  "backbone": "inception_v3",
  "notes": "picked by eye off out/browse/inception_v3-mixed7 sheets ...",
  "targets": [
    {"layer": "mixed7", "channel": 634, "weight": 1.0},
    {"layer": "mixed7", "channel": 12,  "weight": 0.8}
  ]
}
```

`"channel": null` means the whole-layer mean. The `backbone` field is required because **channel numbers are not portable between backbones** — `cli.py dream` warns if a preset is used with a different backbone than it was picked on. `notes` is for provenance.

Three presets ship with the repo, all eye-picked from the InceptionV3 `mixed7` sheets and verified by rendering:

| Preset | Channels (weight) | Character |
|---|---|---|
| `flowers` | 634 (1.0), 12 (0.8), 602 (0.7), 292 (0.5) | dense florets, buds, pale blossoms, a spiky thistle for bite |
| `spirals` | 632 (1.0), 631 (0.8), 8 (0.7), 603 (0.5) | ornate scrollwork, concentric shells, loose vortices, wavy ribbons |
| `scales` | 55 (1.0), 43 (0.8), 610 (0.7), 611 (0.6), 10 (0.5) | shingles, overlapping scales, scalloped rows — strong on flat regions |

Weighted blends of 4–5 channels read much better on real footage than any single channel, which tends to be too uniform across a frame.

## Backbones

Four pretrained torchvision backbones are wired in and selectable with `--backbone`; each carries its own layer map and input adapter, so trying one costs nothing.

| Backbone | Tappable layers | Character |
|---|---|---|
| `inception_v3` (default) | `mixed0` … `mixed10` | the TF-tutorial look; Keras-style layer names |
| `googlenet` (Inception-v1) | `inception3a` … `inception5b` | the classic 2015 DeepDream look; ornate, intricate (`inception4c`/`4d` are the famous ones) |
| `vgg16` | `relu1_2` … `relu5_3` | smooth, painterly, dense |
| `alexnet` | `conv1` … `conv5` | chunkier, larger-scale features |

Two portability caveats:

- **Channel numbers do not port between backbones.** Browse each backbone's layers yourself (`python cli.py --backbone googlenet browse --layer inception4c`).
- **Step sizes do not port either.** VGG16 and AlexNet activations are much larger than Inception's, so `--step-size 0.01` drives them to saturation and erases the source image. Start an order of magnitude lower on those two and work up.

## How the dream works

The core loop is gradient **ascent on the image**: compute the objective's gradient with respect to the pixels, normalize it, and step the image along it. Around that, two standard refinements from the Inceptionism lineage:

- **Octaves.** The dream runs at a pyramid of scales (factor 1.30 per octave, exponents −2…+2 by default), carrying detail gained at each scale up to the next — this is what gives dreams structure at multiple granularities instead of uniform noise.
- **Tiled gradients.** Above a tile size (512 px), gradients are computed per tile and summed, keeping memory flat at any resolution; a random roll each pass keeps tile seams from ever printing into the image.

Internally, everything above the image-I/O layer speaks one convention: images are float32 **HWC tensors in [−1, 1]** on the device; at the human edges they are plain numpy uint8. `engine.image.preprocess` / `deprocess` are the only doors between the two, and each backbone's input adapter converts [−1, 1] to whatever its weights expect — which is what makes swapping backbones cheap.

## Using `engine/` as a library

The notebook is the reference implementation and the place to play; `engine/` is the same machinery in importable form:

```python
from engine import load_backbone, Objective, parse_targets, dream, load_image, save_image

backbone  = load_backbone("inception_v3")          # frozen, eval-mode, cached extractors
objective = Objective(parse_targets("mixed7:634:1,mixed7:12:0.8"), backbone)

img = load_image("photo.jpg", max_dim=1024)        # uint8 HWC in, human space
out = dream(img, objective, steps=100, step_size=0.01)
save_image(out, "out/dream.png")
```

For chaining frames (the video pipeline), `engine.dream.dream_tensor` is the seam to call — tensor in, tensor out, at the input's size — so consecutive frames never round-trip through uint8.

| Module | What it owns |
|---|---|
| `engine/image.py` | the [−1, 1] ↔ uint8 doors, resize, roll, zoom, load/save |
| `engine/model.py` | `Backbone` = frozen net + layer map + input adapter; extractor cache; channel counts |
| `engine/objective.py` | `Target(layer, channel, weight)` and the weighted loss |
| `engine/dream.py` | `gradients` / `tiled_gradients` / `dream_tensor` / `dream` — the octave loop |
| `engine/discover.py` | batched per-channel dreaming, gradient-blur regularizer, contact sheets, ranking |
| `engine/shard.py` | one browse worker per GPU: batch-aligned spans, `CUDA_VISIBLE_DEVICES`, merge |
| `engine/presets.py` | named target lists as JSON in `presets/` |

## The notebook

`deepdream_pytorch.ipynb` is a self-contained PyTorch port of the TensorFlow DeepDream tutorial — simple dream, octaves, rolled tiled gradients — plus two forward-looking extras: a working weighted multi-target objective (the reference implementation of design commitment 2) and `dream_zoom_video`, the classic Inceptionism feedback loop (dream a little, zoom a little, feed the result back) that is the embryo of the coherence machinery. It remains the reference for *the look*; extend `engine/` rather than re-deriving from it.

## Project layout

```
cli.py                    subcommands: layers | browse | dream | presets
deepdream_pytorch.ipynb   reference implementation + interactive playground
engine/                   the importable machinery (see table above)
presets/                  flowers.json, spirals.json, scales.json
tests/test_engine.py      the test suite
environment.yml           the pinned `deepdream` conda env (read its comments)
pyproject.toml            pytest configuration only — the repo runs in place
```

Planned: `io/` (video ↔ frames via imageio-ffmpeg, plus optional per-frame metadata sidecars), `schedule/` (keyframed parameter envelopes), `coherence/` (seed frame *t* from dreamed frame *t−1*, with warp hooks for known transforms like fractal zooms), and a `video` subcommand tying the pipeline together.

## Environment & GPU notes

`environment.yml` creates a conda env named `deepdream` (Python 3.12, conda-forge) with **`torch==2.7.1+cu118` / `torchvision==0.22.1+cu118` pinned hard** — see the comments in that file for the full rationale. The short version:

- 2.7.1 is the final PyTorch release with CUDA 11.8 wheels, and the cu118 builds are the last compiled with **Maxwell-era GPU support** (sm_5x). Newer releases drop both, so upgrading torch in this env would remove GPU support on those cards entirely. If your GPU is newer, the pinned wheels still run fine.
- The wheels bundle their own CUDA 11.8 runtime and cuDNN; no system CUDA toolkit is needed. Any reasonably recent NVIDIA driver satisfies CUDA 11.8.
- torch ≥ 2.7 wheels require glibc ≥ 2.28 (Ubuntu 20.04+/equivalent). On older systems, the documented fallback is `torch==2.6.0+cu118` / `torchvision==0.21.0+cu118` — the code is identical under either pin.
- On pre-Volta GPUs the fast path is eager fp32: `torch.compile`/Triton needs sm_70+, and fp16/tensor-core paths don't exist there. The code assumes nothing newer.
- Video writing (when the pipeline lands) uses imageio + the pip `imageio-ffmpeg` static ffmpeg with libx264: `codec="libx264"`, `pixelformat="yuv420p"`, `-crf 20 -preset slow`, even frame dimensions.

No GPU at all? Everything runs with `--device cpu` — fine for the tests and small experiments, slow for real dreams.

## Testing

```bash
python -m pytest tests -q                       # full suite: 46 tests, ~11 s with a GPU
python -m pytest tests -q -m "not slow"         # pure-logic tests only, no weights/GPU needed
```

Tests marked `slow` need the pretrained backbone (and really want a GPU); one marked `multigpu` needs at least two cards. The load-bearing test is `test_batched_channel_dream_matches_single`: batching is only safe because eval-mode BatchNorm keeps batch items independent, and if that ever broke, every contact sheet would silently become wrong — that test is the thing that would notice.

## Background & credits

- A. Mordvintsev, C. Olah, M. Tyka, [*Inceptionism: Going Deeper into Neural Networks*](https://research.google/blog/inceptionism-going-deeper-into-neural-networks/), Google Research Blog, 2015.
- The [TensorFlow DeepDream tutorial](https://www.tensorflow.org/tutorials/generative/deepdream), of which the notebook is a faithful PyTorch port before it goes further.
- Pretrained backbones and feature extraction from [torchvision](https://pytorch.org/vision/stable/index.html) (`create_feature_extractor`).

## License

[MIT](LICENSE).
