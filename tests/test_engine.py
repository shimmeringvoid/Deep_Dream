"""Tests for engine/ — run with `conda run -n deepdream python -m pytest tests -q`.

The parsing/preset/sheet tests are pure CPU and fast. The dream tests need a
backbone (they will download InceptionV3 weights on first run) and are marked
`slow`; skip them with `-m 'not slow'`.
"""

import numpy as np
import pytest
import torch

from engine import discover, image, presets, shard
from engine.model import load_backbone
from engine.objective import Objective, Target, parse_target, parse_targets

# ---------------------------------------------------------------------------
# Target parsing
# ---------------------------------------------------------------------------


def test_parse_target_forms():
    assert parse_target("mixed7") == Target("mixed7", None, 1.0)
    assert parse_target("mixed7:10") == Target("mixed7", 10, 1.0)
    assert parse_target("mixed7:10:0.7") == Target("mixed7", 10, 0.7)
    assert parse_target("mixed9:*:-0.5") == Target("mixed9", None, -0.5)


def test_parse_targets_from_comma_string():
    got = parse_targets("mixed7:10:1,mixed9:4:-0.5")
    assert got == [Target("mixed7", 10, 1.0), Target("mixed9", 4, -0.5)]


def test_parse_target_rejects_junk():
    with pytest.raises(ValueError):
        parse_target("mixed7:10:1:extra")
    with pytest.raises(ValueError):
        parse_target("")


def test_target_str_roundtrips_through_parse():
    for t in [Target("mixed7", 10, 0.7), Target("mixed9", None, -0.5)]:
        assert parse_target(str(t)) == t


# ---------------------------------------------------------------------------
# Channel specs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("all", list(range(8))),
        ("0-3", [0, 1, 2, 3]),
        ("1,3,5", [1, 3, 5]),
        ("0-2,5,6-7", [0, 1, 2, 5, 6, 7]),
        ("::3", [0, 3, 6]),
        ("2,2,2", [2]),  # de-duped, order kept
    ],
)
def test_parse_channel_spec(spec, expected):
    assert discover.parse_channel_spec(spec, 8) == expected


# ---------------------------------------------------------------------------
# Image conventions — the [-1, 1] / uint8 door
# ---------------------------------------------------------------------------


def test_preprocess_deprocess_roundtrip():
    src = np.random.randint(0, 256, (16, 24, 3), dtype=np.uint8)
    t = image.preprocess(src)
    assert t.dtype == torch.float32 and t.shape == (16, 24, 3)
    assert -1.0 <= float(t.min()) and float(t.max()) <= 1.0
    # 255 levels through a [-1, 1] float and back: exact to within rounding.
    assert np.abs(image.deprocess(t).astype(int) - src.astype(int)).max() <= 1


def test_preprocess_passes_tensors_through():
    t = image.preprocess(np.zeros((8, 8, 3), dtype=np.uint8))
    assert image.preprocess(t) is t


def test_batch_roundtrip():
    t = torch.randn(5, 7, 3)
    assert torch.equal(image.from_batch(image.to_batch(t)), t)


def test_zoom_preserves_shape():
    t = torch.randn(32, 48, 3)
    assert image.zoom(t, 1.05).shape == t.shape


def test_random_roll_is_invertible():
    t = torch.randn(16, 16, 3)
    shift, rolled = image.random_roll(t, 8)
    back = torch.roll(rolled, shifts=(-int(shift[0]), -int(shift[1])), dims=(0, 1))
    assert torch.equal(back, t)


def test_even_dims():
    assert image.even_dims(torch.zeros(15, 9, 3)).shape == (14, 8, 3)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_preset_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "PRESET_DIR", tmp_path)
    targets = [Target("mixed7", 10, 1.0), Target("mixed9", None, -0.5)]
    presets.save_preset("spirals", targets, notes="test")
    assert presets.list_presets() == ["spirals"]
    got = presets.load_preset("spirals")
    assert got["targets"] == targets
    assert got["backbone"] == "inception_v3"


def test_load_missing_preset(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "PRESET_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        presets.load_preset("nope")


# ---------------------------------------------------------------------------
# Contact sheets
# ---------------------------------------------------------------------------


def test_contact_sheet_geometry():
    tiles = np.zeros((5, 64, 64, 3), dtype=np.uint8)
    sheet = discover.contact_sheet(tiles, cols=3, tile_px=32, pad=4, label_px=10)
    # 3 columns x 2 rows of 32 px cells, plus padding and a label band.
    assert sheet.size == (3 * 36 + 4, 2 * (32 + 4 + 16) + 4)


def test_contact_sheet_rejects_empty():
    with pytest.raises(ValueError):
        discover.contact_sheet(np.zeros((0, 8, 8, 3), dtype=np.uint8))


def test_blur_gradient_preserves_shape_and_lowers_energy():
    g = torch.randn(2, 3, 32, 32)
    blurred = discover.blur_gradient(g, 1.0)
    assert blurred.shape == g.shape
    # A low-pass must reduce the variance of white noise.
    assert blurred.std() < g.std()
    assert torch.equal(discover.blur_gradient(g, 0.0), g)


# ---------------------------------------------------------------------------
# The real thing (needs weights + ideally a GPU)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def backbone():
    return load_backbone("inception_v3")


@pytest.mark.slow
def test_layer_channel_counts(backbone):
    # The numbers CLAUDE.md quotes; if these move, something is very wrong.
    assert backbone.channels("mixed7") == 768
    assert backbone.channels("mixed9") == 2048


@pytest.mark.slow
def test_objective_rejects_out_of_range_channel(backbone):
    with pytest.raises(ValueError):
        Objective([Target("mixed7", 768)], backbone)


@pytest.mark.slow
def test_dream_changes_the_image_and_raises_the_loss(backbone):
    src = np.full((128, 128, 3), 128, dtype=np.uint8)
    objective = Objective("mixed7:10:1.0", backbone)
    img = image.preprocess(src, backbone.device)
    before = float(objective.loss(img))
    from engine.dream import dream_tensor

    out = dream_tensor(img, objective, steps=10, step_size=0.05, octaves=[0])
    assert float(objective.loss(out)) > before
    assert out.shape == img.shape


@pytest.mark.slow
def test_dream_restores_base_size_across_octaves(backbone):
    from engine.dream import dream

    src = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    out = dream(src, "mixed7", backbone=backbone, steps=2, octaves=[-1, 0, 1])
    assert out.shape == src.shape and out.dtype == np.uint8


@pytest.mark.slow
def test_batchnorm_is_in_eval_mode(backbone):
    """Batching the browser is only sound because BN uses running stats."""
    bns = [m for m in backbone.module.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and not any(m.training for m in bns)


@pytest.mark.slow
def test_batched_channel_dream_matches_single(backbone):
    """The load-bearing claim of the browser: batching does not couple items.

    Checked at *one* step, where it is exact. If BatchNorm ran in train mode,
    or anything else mixed the batch, item 0's gradient would depend on items
    1 and 2 and this would fail immediately — and every contact sheet would be
    subtly wrong.

    It is deliberately not checked at 96 steps: cuDNN chooses different kernels
    per batch shape, and single-channel ascent amplifies those last-bit
    differences chaotically, so the *tiles* legitimately differ across batch
    sizes even though the *gradients* never couple. See `dream_channels`.
    """
    chans = [10, 99, 417]
    kw = dict(size=128, steps=1, octaves=[0])
    together = discover.dream_channels(
        backbone, "mixed7", chans, batch_size=3, **kw
    )
    apart = np.concatenate(
        [
            discover.dream_channels(backbone, "mixed7", [c], batch_size=1, **kw)
            for c in chans
        ]
    )
    assert np.array_equal(together, apart)


@pytest.mark.slow
def test_channel_dream_is_reproducible_at_a_fixed_batch_size(backbone):
    """The guarantee actually on offer, and the one index.json records."""
    kw = dict(size=128, steps=8, batch_size=2, octaves=[0])
    a = discover.dream_channels(backbone, "mixed7", [10, 99], **kw)
    b = discover.dream_channels(backbone, "mixed7", [10, 99], **kw)
    assert np.array_equal(a, b)


@pytest.mark.slow
def test_dream_channels_are_distinct(backbone):
    tiles = discover.dream_channels(
        backbone, "mixed7", [10, 417], size=128, steps=20, batch_size=2, octaves=[0]
    )
    assert np.abs(tiles[0].astype(int) - tiles[1].astype(int)).mean() > 5


@pytest.mark.slow
def test_rank_channels_covers_the_layer(backbone):
    ranked = discover.rank_channels(backbone, "mixed7", size=128)
    assert len(ranked) == 768
    assert {c for c, _ in ranked} == set(range(768))
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Multi-GPU shard planning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,gpus,batch,expected",
    [
        # The even case: 2048 channels, batch 16, four cards.
        (2048, 4, 16, [(0, 512), (512, 1024), (1024, 1536), (1536, 2048)]),
        # A ragged tail stays at the very end, where the unsharded run puts it.
        (100, 3, 16, [(0, 48), (48, 80), (80, 100)]),
        # Fewer blocks than cards: the spare cards get nothing, rather than the
        # shards being cut off a batch boundary to keep everyone busy.
        (32, 4, 16, [(0, 16), (16, 32)]),
        (32, 4, 8, [(0, 8), (8, 16), (16, 24), (24, 32)]),
        (1, 4, 16, [(0, 1)]),
        (0, 4, 16, []),
    ],
)
def test_plan_shards(n, gpus, batch, expected):
    assert shard.plan_shards(n, gpus, batch) == expected


@pytest.mark.parametrize("n,gpus,batch", [(2048, 4, 16), (768, 3, 16), (100, 4, 7)])
def test_plan_shards_is_a_batch_aligned_partition(n, gpus, batch):
    """Every boundary on a batch multiple, no gaps, no overlaps, nothing lost.

    That alignment is the whole reason the sharded sheets match the one-card
    sheets: it guarantees each shard chunks the channel list exactly where the
    unsharded run would have.
    """
    spans = shard.plan_shards(n, gpus, batch)
    assert [c for a, b in spans for c in range(a, b)] == list(range(n))
    assert all(start % batch == 0 for start, _ in spans)
    assert all(stop % batch == 0 for _, stop in spans[:-1])


@pytest.mark.parametrize(
    "spec,expected",
    [("4", [0, 1, 2, 3]), ("0,2,3", [0, 2, 3]), ("1-3", [1, 2, 3]),
     (None, []), ("0", []), ("", [])],
)
def test_parse_gpu_spec(spec, expected):
    assert shard.parse_gpu_spec(spec) == expected


@pytest.mark.slow
@pytest.mark.multigpu
def test_sharded_browse_matches_single_gpu(tmp_path):
    """Two cards, split down the middle, must give bit-identical tiles.

    The full-size version of this check (32 channels over 4 GPUs at browse
    defaults) is in CLAUDE.md; this is the cheap one that can live in the
    suite.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("needs at least 2 GPUs")
    chans = list(range(8))
    kw = dict(size=128, steps=6, batch_size=4, octaves=(0,))
    sharded, used = shard.dream_channels_sharded(
        "inception_v3", "mixed7", chans, [0, 1], work_dir=tmp_path, **kw
    )
    single = discover.dream_channels(
        load_backbone("inception_v3", "cuda:0"), "mixed7", chans, **kw
    )
    assert used == [0, 1]
    assert np.array_equal(sharded, single)
