"""Named target lists ("flowers", "eyes", "swirls") as JSON on disk.

A preset is the durable output of the contact-sheet browsing session: you look
at a sheet, you write down which channels of which layer looked like flowers,
and that list becomes something `schedule/` can reference and blend by name
(milestone 4) instead of hard-coding integers in a script.

Format — one JSON object per preset, in `presets/<name>.json`:

    {
      "name": "flowers",
      "backbone": "inception_v3",
      "notes": "picked off out/browse/inception_v3-mixed7 sheets, 2026-08",
      "targets": [
        {"layer": "mixed7", "channel": 10,  "weight": 1.0},
        {"layer": "mixed7", "channel": 99,  "weight": 0.7},
        {"layer": "mixed9", "channel": 4,   "weight": -0.5}
      ]
    }

`channel: null` means the whole-layer mean. The `backbone` field matters:
channel *numbers are not portable between backbones*, so a preset that names
one is a preset that can be checked.
"""

from __future__ import annotations

import json
import pathlib

from .objective import Target

PRESET_DIR = pathlib.Path(__file__).resolve().parent.parent / "presets"


def _path(name_or_path) -> pathlib.Path:
    p = pathlib.Path(name_or_path)
    if p.suffix == ".json" or p.exists():
        return p
    return PRESET_DIR / f"{name_or_path}.json"


def list_presets() -> list[str]:
    """Names of the presets in `presets/`, alphabetically."""
    if not PRESET_DIR.is_dir():
        return []
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def load_preset(name_or_path) -> dict:
    """Load a preset. Returns `{name, backbone, notes, targets: [Target]}`."""
    path = _path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No preset {name_or_path!r} ({path}). Known: {list_presets()}"
        )
    data = json.loads(path.read_text())
    data["targets"] = [Target.from_dict(t) for t in data.get("targets", [])]
    data.setdefault("name", path.stem)
    data.setdefault("backbone", "inception_v3")
    data.setdefault("notes", "")
    return data


def save_preset(name, targets, backbone="inception_v3", notes="") -> pathlib.Path:
    """Write a preset to `presets/<name>.json` and return the path."""
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": path.stem,
        "backbone": backbone,
        "notes": notes,
        "targets": [
            (t if isinstance(t, Target) else Target.from_dict(t)).to_dict()
            for t in targets
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
