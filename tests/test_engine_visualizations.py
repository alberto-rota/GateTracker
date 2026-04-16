import sys
import types
from types import SimpleNamespace

import torch


fake_geometry_module = types.ModuleType("gatetracker.geometry.pipeline")


class DummyGeometryPipeline:
    def __init__(self, *args, **kwargs):
        pass


fake_geometry_module.GeometryPipeline = DummyGeometryPipeline
sys.modules.setdefault("gatetracker.geometry.pipeline", fake_geometry_module)

import gatetracker.engine as engine_module


def test_create_architecture_images_shares_pca_and_stacks_source_target(
    monkeypatch,
):
    engine = engine_module.Engine.__new__(engine_module.Engine)
    num_layers = 3
    num_patches = 4
    engine.matcher = SimpleNamespace(
        model=SimpleNamespace(
            latest_diagnostics={
                "source": {
                    "effective_layer": torch.ones(1, num_patches),
                    "max_weight": torch.full((1, num_patches), 0.5),
                    "layer_weights": torch.full(
                        (1, num_layers, num_patches), 1.0 / num_layers
                    ),
                    "layer_indices": torch.arange(num_layers),
                },
                "target": {
                    "effective_layer": torch.ones(1, num_patches),
                    "max_weight": torch.full((1, num_patches), 0.5),
                    "layer_weights": torch.full(
                        (1, num_layers, num_patches), 1.0 / num_layers
                    ),
                    "layer_indices": torch.arange(num_layers),
                },
            },
        ),
    )
    engine.config = SimpleNamespace(
        IMAGE_HEIGHT=32,
        IMAGE_WIDTH=32,
        RESAMPLED_PATCH_SIZE=16,
    )

    framestack = torch.rand(1, 2, 3, 32, 32)
    images = engine.create_architecture_images(
        framestack=framestack,
        descriptors={
            "source_embedding_match": torch.randn(1, 8, num_patches),
            "target_embedding_match": torch.randn(1, 8, num_patches),
        },
    )
    assert isinstance(images, dict)
