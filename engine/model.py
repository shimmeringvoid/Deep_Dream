"""Backbone loading and layer taps.

A `Backbone` bundles three things the rest of the engine needs and nothing else:

1. a pretrained torchvision net in eval mode with frozen weights,
2. a **layer map** from friendly names to torchvision module paths, and
3. an **input adapter** from the engine's [-1, 1] convention to whatever that
   particular net's weights were trained on.

(3) is what makes swapping backbones cheap. InceptionV3 and GoogLeNet accept
[-1, 1] directly once `transform_input=False`; VGG16 and AlexNet want
ImageNet-normalized input, so their adapter does the conversion and the
convention holds everywhere above this file.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field

import torch
from torchvision.models.feature_extraction import create_feature_extractor

# Keras InceptionV3 concat-layer names -> torchvision module names. The
# notebook's dict, kept verbatim so layer picks keep their familiar names.
KERAS_TO_TV = {
    "mixed0": "Mixed_5b", "mixed1": "Mixed_5c", "mixed2": "Mixed_5d",
    "mixed3": "Mixed_6a", "mixed4": "Mixed_6b", "mixed5": "Mixed_6c",
    "mixed6": "Mixed_6d", "mixed7": "Mixed_6e", "mixed8": "Mixed_7a",
    "mixed9": "Mixed_7b", "mixed10": "Mixed_7c",
}

# GoogLeNet / Inception-v1 — the classic DeepDream look. torchvision already
# uses the paper's names, so the map is the identity over the taps worth having.
GOOGLENET_LAYERS = {
    name: name
    for name in [
        "inception3a", "inception3b",
        "inception4a", "inception4b", "inception4c", "inception4d", "inception4e",
        "inception5a", "inception5b",
    ]
}

# VGG16 / AlexNet expose one flat `features` Sequential; tap the ReLUs, which
# is where feature-visualization work conventionally hangs its objectives.
VGG16_LAYERS = {
    "relu1_2": "features.3", "relu2_2": "features.8", "relu3_3": "features.15",
    "relu4_3": "features.22", "relu5_3": "features.29",
}
ALEXNET_LAYERS = {
    "conv1": "features.1", "conv2": "features.4", "conv3": "features.7",
    "conv4": "features.9", "conv5": "features.11",
}

# ImageNet statistics, for the backbones whose weights expect them.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _identity_adapter(batch: torch.Tensor) -> torch.Tensor:
    return batch


def _imagenet_adapter(batch: torch.Tensor) -> torch.Tensor:
    # [-1, 1] -> [0, 1] -> standardized. Buffers are built per call on the
    # batch's own device; they are 3 floats, so the cost is noise.
    mean = torch.tensor(_IMAGENET_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=batch.device).view(1, 3, 1, 1)
    return ((batch + 1.0) / 2.0 - mean) / std


class Extractor:
    """A feature tap: NCHW in [-1, 1] -> {friendly layer name: activations}."""

    def __init__(self, fx, adapter, layers):
        self.fx = fx
        self.adapter = adapter
        self.layers = tuple(layers)

    def __call__(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.fx(self.adapter(batch))


@dataclass
class Backbone:
    """A pretrained net plus everything needed to tap and feed it."""

    name: str
    module: torch.nn.Module
    layer_map: dict[str, str]
    adapter: object
    device: torch.device
    min_size: int = 96
    _extractors: dict = field(default_factory=dict, repr=False)
    _channels: dict = field(default_factory=dict, repr=False)

    @property
    def layers(self) -> list[str]:
        """Tappable layer names, in network order."""
        return list(self.layer_map)

    def extractor(self, layers) -> Extractor:
        """Build (and cache) an extractor covering exactly `layers`.

        `create_feature_extractor` prunes the graph after the deepest requested
        layer, so asking for only what the objective uses is also the fast
        path. Tracing costs ~a second, hence the cache.
        """
        layers = tuple(dict.fromkeys(layers))  # de-dupe, keep order
        unknown = [n for n in layers if n not in self.layer_map]
        if unknown:
            raise KeyError(
                f"{self.name} has no layer(s) {unknown}. Available: {self.layers}"
            )
        if layers not in self._extractors:
            with warnings.catch_warnings():
                # torchvision warns that eval-mode nodes are a subsequence of
                # train-mode ones. We only ever tap in eval mode; that is the
                # point. Silence it so the CLI output stays readable.
                warnings.filterwarnings("ignore", message=".*nodes obtained by tracing.*")
                fx = create_feature_extractor(
                    self.module, return_nodes={self.layer_map[n]: n for n in layers}
                )
            self._extractors[layers] = Extractor(fx.eval(), self.adapter, layers)
        return self._extractors[layers]

    def channels(self, layer: str) -> int:
        """Channel count at `layer`, discovered by one throwaway forward pass."""
        if layer not in self._channels:
            probe = torch.zeros(
                1, 3, self.min_size, self.min_size, device=self.device
            )
            with torch.no_grad():
                acts = self.extractor([layer])(probe)
            self._channels[layer] = int(acts[layer].shape[1])
        return self._channels[layer]


def _load_inception_v3(device):
    from torchvision.models import Inception_V3_Weights, inception_v3

    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.transform_input = False  # consume TF-style [-1, 1] input directly
    return Backbone(
        name="inception_v3",
        module=model,
        layer_map=dict(KERAS_TO_TV),
        adapter=_identity_adapter,
        device=device,
        # The strided stem eats ~5 px of margin per conv; below ~96 px the
        # deep taps degenerate to a couple of spatial positions.
        min_size=96,
    )


def _load_googlenet(device):
    from torchvision.models import GoogLeNet_Weights, googlenet

    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    model.transform_input = False
    return Backbone(
        name="googlenet",
        module=model,
        layer_map=dict(GOOGLENET_LAYERS),
        adapter=_identity_adapter,
        device=device,
        min_size=96,
    )


def _load_vgg16(device):
    from torchvision.models import VGG16_Weights, vgg16

    return Backbone(
        name="vgg16",
        module=vgg16(weights=VGG16_Weights.IMAGENET1K_V1),
        layer_map=dict(VGG16_LAYERS),
        adapter=_imagenet_adapter,
        device=device,
        min_size=64,
    )


def _load_alexnet(device):
    from torchvision.models import AlexNet_Weights, alexnet

    return Backbone(
        name="alexnet",
        module=alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
        layer_map=dict(ALEXNET_LAYERS),
        adapter=_imagenet_adapter,
        device=device,
        min_size=64,
    )


_LOADERS = {
    "inception_v3": _load_inception_v3,
    "googlenet": _load_googlenet,
    "vgg16": _load_vgg16,
    "alexnet": _load_alexnet,
}

DEFAULT_BACKBONE = "inception_v3"


def available_backbones() -> list[str]:
    return list(_LOADERS)


def get_device(spec=None) -> torch.device:
    """Resolve a device spec ('cuda:1', 'cpu', None) to a torch.device.

    The workstation has 4 Titan X; a single dream stays on one card, and
    multi-GPU work happens by sharding *frames* across processes (milestone 6),
    not by splitting one dream.
    """
    if isinstance(spec, torch.device):
        return spec
    if spec:
        return torch.device(spec)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@functools.lru_cache(maxsize=None)
def _load_cached(name: str, device_str: str) -> Backbone:
    if name not in _LOADERS:
        raise KeyError(f"Unknown backbone {name!r}. Available: {available_backbones()}")
    device = torch.device(device_str)
    backbone = _LOADERS[name](device)
    backbone.module.eval().requires_grad_(False).to(device)
    return backbone


def load_backbone(name: str = DEFAULT_BACKBONE, device=None) -> Backbone:
    """Load a pretrained backbone, frozen and in eval mode, on `device`.

    Cached per (name, device): loading is a ~100 MB checkpoint read, and the
    extractor cache inside the Backbone is worth keeping warm too.
    """
    return _load_cached(name, str(get_device(device)))
