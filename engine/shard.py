"""Sharding a channel browse across the workstation's four GPUs.

Browsing a layer is embarrassingly parallel — every channel is an independent
dream — so the only interesting question is *where to cut*, and the answer is
fixed by the reproducibility guarantee `index.json` makes.

**The alignment rule.** A tile is bit-reproducible only at the batch size it
was dreamed in: cuDNN picks its convolution algorithm from the batch shape,
and single-channel ascent amplifies last-bit differences into a different
picture within ~100 steps (see `discover.dream_channels`). `dream_channels`
walks its channel list in chunks of `batch_size`, so a shard reproduces the
unsharded run exactly when its boundaries fall on multiples of `batch_size` —
then every chunk contains the same channels, in the same order, at the same
shape, as it would have unsharded. `plan_shards` cuts on those boundaries and
nowhere else, and every worker runs the *same* `batch_size`. The one ragged
chunk a run can have (when the channel count is not a multiple of the batch)
stays where it was: at the very end, in the last shard.

That is why the split is by aligned contiguous blocks rather than the obvious
round-robin `channels[gpu::4]` — round-robin would repack every chunk with
different neighbours and quietly invalidate every tile on the sheet.

Mechanically: one worker *process* per GPU, selected with
`CUDA_VISIBLE_DEVICES` (so each worker sees its card as `cuda:0`), each
dreaming its slice and dropping raw uint8 tiles in a `.npy`. The parent holds
no CUDA context of its own — it resolves the channel list on the CPU, streams
the workers' progress, then concatenates the slices in channel order and
writes the sheets. Same reason as milestone 6's frame sharding: no DDP, no
NCCL, nothing shared but the filesystem.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_shards(n_items: int, n_shards: int, batch_size: int) -> list[tuple[int, int]]:
    """Cut `n_items` into at most `n_shards` contiguous `[start, stop)` spans.

    Every boundary is a multiple of `batch_size` (the last stop excepted, since
    it is the end of the list), so each shard's chunking is identical to the
    unsharded run's — see this module's docstring for why that is the whole
    ballgame. Shards that would get no work are dropped, so the result can be
    shorter than `n_shards`: with 32 channels at batch 16 there are only two
    blocks to hand out, and two GPUs sit the round out.
    """
    if n_items <= 0:
        return []
    n_shards = max(1, int(n_shards))
    batch_size = max(1, int(batch_size))

    n_blocks = -(-n_items // batch_size)  # ceil
    per, extra = divmod(n_blocks, n_shards)

    spans, block = [], 0
    for s in range(n_shards):
        take = per + (1 if s < extra else 0)
        if take == 0:
            continue
        start = block * batch_size
        block += take
        spans.append((start, min(block * batch_size, n_items)))
    return spans


def parse_gpu_spec(spec) -> list[int]:
    """`4` -> [0,1,2,3]; `0,2,3` -> [0,2,3]; `all` -> every visible card.

    A bare count is the common case ("use four cards"); the explicit list is
    for leaving a card alone — someone else's notebook may be sitting on one.
    A count of zero (or nothing at all) means "don't shard"; to pin a single
    dream to one card, that is what `--device cuda:N` is for.
    """
    if spec is None:
        return []
    spec = str(spec).strip()
    if spec in ("", "0", "none"):
        return []
    if spec == "all":
        import torch

        return list(range(torch.cuda.device_count()))
    if "," in spec or "-" in spec:
        out = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part.lstrip("-"):
                lo, hi = part.split("-", 1)
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(part))
        return list(dict.fromkeys(out))
    n = int(spec)
    return list(range(n))


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


def _worker_main(job_path: str) -> None:
    """Dream one shard's channels on the single GPU this process can see.

    Runs as `python -m engine.shard <job.json>` under
    `CUDA_VISIBLE_DEVICES=<n>`, which is why the device is hard-coded to
    `cuda:0`: masking happens in the environment, before torch initializes.
    """
    job = json.loads(pathlib.Path(job_path).read_text())

    import torch

    from . import discover
    from .model import load_backbone

    if not torch.cuda.is_available():  # a masked-off or broken card
        raise RuntimeError(
            f"shard {job['shard']}: no CUDA device visible "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
        )

    backbone = load_backbone(job["backbone"], "cuda:0")
    seed = job["seed"]
    if isinstance(seed, dict):  # a real image, handed over as a .npy
        seed = np.load(seed["npy"])

    def progress(done, total, elapsed):
        print(f"PROGRESS {done} {total} {elapsed:.3f}", flush=True)

    tiles = discover.dream_channels(
        backbone,
        job["layer"],
        job["channels"],
        size=job["size"],
        steps=job["steps"],
        step_size=job["step_size"],
        seed=seed,
        batch_size=job["batch_size"],
        octaves=tuple(job["octaves"]),
        jitter=job["jitter"],
        grad_blur=job["grad_blur"],
        deterministic=job["deterministic"],
        progress=progress,
    )
    np.save(job["out_npy"], tiles)
    print(f"DONE {len(job['channels'])}", flush=True)


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


class _Relay(threading.Thread):
    """Read one worker's stdout, tally its progress, echo everything else."""

    def __init__(self, proc, gpu, state, lock, report):
        super().__init__(daemon=True)
        self.proc, self.gpu = proc, gpu
        self.state, self.lock, self.report = state, lock, report
        self.tail: list[str] = []

    def run(self):
        for raw in self.proc.stdout:
            line = raw.rstrip("\n")
            if line.startswith("PROGRESS "):
                _, done, _total, _elapsed = line.split()
                with self.lock:
                    self.state[self.gpu] = int(done)
                    self.report(self.gpu)
                continue
            if line.startswith("DONE "):
                continue
            self.tail.append(line)
            del self.tail[:-40]
            with self.lock:
                print(f"  [gpu {self.gpu}] {line}", flush=True)


def dream_channels_sharded(
    backbone_name: str,
    layer: str,
    channels,
    gpus,
    *,
    size: int = 256,
    steps: int = 96,
    step_size: float = 0.05,
    seed="gray",
    batch_size: int = 16,
    octaves=(-2, -1, 0),
    jitter: int = 8,
    grad_blur: float = 1.0,
    deterministic: bool = True,
    work_dir=None,
    keep_work: bool = False,
    progress=None,
) -> tuple[np.ndarray, list[int]]:
    """`discover.dream_channels`, run as one process per GPU.

    Returns `(tiles, gpus_used)` — tiles in the same order as `channels`, and
    bit-identical to the single-GPU run at this `batch_size`, which is what
    `tests/` and the 32-channel check in CLAUDE.md verify.
    """
    channels = [int(c) for c in channels]
    gpus = list(gpus)
    spans = plan_shards(len(channels), len(gpus), batch_size)
    if not spans:
        raise ValueError("Nothing to dream.")
    gpus = gpus[: len(spans)]

    work = pathlib.Path(work_dir or (REPO_ROOT / "out" / ".shards" / f"{backbone_name}-{layer}"))
    work.mkdir(parents=True, exist_ok=True)

    seed_job = seed
    if not isinstance(seed, str):  # an image seed: hand it over on disk
        seed_path = work / "seed.npy"
        np.save(seed_path, np.asarray(seed))
        seed_job = {"npy": str(seed_path)}

    total = len(channels)
    state: dict[int, int] = {g: 0 for g in gpus}
    lock = threading.Lock()
    t0 = time.time()

    def report(gpu):
        done = sum(state.values())
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-9)
        eta = (total - done) / max(rate, 1e-9)
        start, stop = spans[gpus.index(gpu)]
        print(
            f"  [gpu {gpu}] {state[gpu]:>5}/{stop - start:<5} "
            f"| all {done:>5}/{total:<5} {elapsed / 60:5.1f} min  "
            f"{rate:5.2f} ch/s  ETA {eta / 60:5.1f} min",
            flush=True,
        )
        if progress is not None:
            progress(done, total, elapsed)

    procs, relays, jobs = [], [], []
    for gpu, (start, stop) in zip(gpus, spans):
        job = {
            "shard": gpu,
            "backbone": backbone_name,
            "layer": layer,
            "channels": channels[start:stop],
            "size": size,
            "steps": steps,
            "step_size": step_size,
            "seed": seed_job,
            "batch_size": batch_size,
            "octaves": list(octaves),
            "jitter": jitter,
            "grad_blur": grad_blur,
            "deterministic": deterministic,
            "out_npy": str(work / f"tiles_gpu{gpu}.npy"),
        }
        job_path = work / f"job_gpu{gpu}.json"
        job_path.write_text(json.dumps(job, indent=2) + "\n")
        jobs.append(job)

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [sys.executable, "-m", "engine.shard", str(job_path)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append(proc)
        relay = _Relay(proc, gpu, state, lock, report)
        relay.start()
        relays.append(relay)
        print(f"  gpu {gpu}: channels[{start}:{stop}]  ({stop - start} channels)", flush=True)

    # Fail fast: one card OOMing 30 seconds in should not leave the other three
    # grinding out a quarter of a browse each that will be thrown away.
    codes: dict[int, int] = {}
    trouble = threading.Event()

    def wait_for(gpu, proc):
        codes[gpu] = proc.wait()
        if codes[gpu] != 0:
            trouble.set()

    waiters = [threading.Thread(target=wait_for, args=(g, p), daemon=True)
               for g, p in zip(gpus, procs)]
    for w in waiters:
        w.start()
    while len(codes) < len(procs) and not trouble.is_set():
        trouble.wait(0.5)
    if trouble.is_set():
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
    for w in waiters:
        w.join()
    for relay in relays:
        relay.join()

    failed = [
        (g, codes[g], "\n".join(r.tail[-20:]))
        for g, r in zip(gpus, relays)
        if codes.get(g)
    ]
    if failed:
        detail = "\n".join(
            f"  gpu {g} exited {rc}"
            + (" (terminated because a sibling shard failed)" if rc == -15 else "")
            + (f"\n{tail}" if tail else "")
            for g, rc, tail in failed
        )
        raise RuntimeError(f"{len(failed)} browse shard(s) failed:\n{detail}")

    tiles = np.concatenate([np.load(job["out_npy"]) for job in jobs])
    if len(tiles) != total:
        raise RuntimeError(f"shards returned {len(tiles)} tiles, expected {total}")

    if not keep_work:
        for job in jobs:
            pathlib.Path(job["out_npy"]).unlink(missing_ok=True)

    return tiles, gpus


if __name__ == "__main__":
    _worker_main(sys.argv[1])
