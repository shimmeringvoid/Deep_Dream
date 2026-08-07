"""engine — the dreamer, lifted out of `deepdream_pytorch.ipynb`.

The notebook stays the reference implementation and the place to play; this
package is the same machinery in importable form, with the conventions it
established kept intact:

* User-facing layer names are Keras-style (`mixed0..mixed10` on InceptionV3),
  mapped to torchvision module paths by the backbone's layer map.
* Inside the machinery an image is a float32 **HWC tensor in [-1, 1]** on the
  device; at the human edges it is a plain numpy **uint8 HWC** array.
  `image.preprocess()` / `image.deprocess()` are the doors, and every backbone
  adapts [-1, 1] to whatever its own weights expect.
* The objective is a weighted list of `(layer, channel, weight)` targets;
  `channel=None` means the whole-layer mean, which is the stock tutorial's
  objective as the degenerate case.
"""

from .image import (
    deprocess,
    load_image,
    preprocess,
    random_roll,
    resize,
    save_image,
    zoom,
)
from .model import KERAS_TO_TV, Backbone, available_backbones, load_backbone
from .objective import Objective, Target, parse_target, parse_targets
from .dream import dream, dream_tensor, gradients, tiled_gradients
from .presets import list_presets, load_preset, save_preset

__all__ = [
    "Backbone",
    "KERAS_TO_TV",
    "Objective",
    "Target",
    "available_backbones",
    "deprocess",
    "dream",
    "dream_tensor",
    "gradients",
    "list_presets",
    "load_backbone",
    "load_image",
    "load_preset",
    "parse_target",
    "parse_targets",
    "preprocess",
    "random_roll",
    "resize",
    "save_image",
    "save_preset",
    "tiled_gradients",
    "zoom",
]
