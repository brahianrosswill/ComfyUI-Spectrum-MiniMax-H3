from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.minimax_h3 import (
    _sanitize_prediction,
    is_native_minimax_h3,
    locate_minimax_h3_inner,
    require_native_minimax_h3,
)
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
from comfyui_spectrum_h3.sampling import (
    BINDING_KEY,
    SpectrumH3Binding,
    model_clone_callback,
)


def _native_shaped_fake():
    cls = type("MiniMaxH3Model", (), {})
    cls.__module__ = "comfy.ldm.minimax.model"
    instance = cls()
    for name, value in {
        "blocks": [object()],
        "final_layer": object(),
        "hidden_size": 8,
        "patch_size": (1, 2, 2),
        "latents_dim": 24,
        "audio_latents_dim": 32,
        "sigma_shift_video": 12.0,
        "sigma_shift_audio": 3.0,
    }.items():
        setattr(instance, name, value)
    return instance


def test_model_detection_accepts_only_native_h3_identity_and_shape():
    inner = _native_shaped_fake()
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=inner))
    assert locate_minimax_h3_inner(patcher) == (inner, "model.diffusion_model")
    assert is_native_minimax_h3(inner)
    assert require_native_minimax_h3(patcher)[0] is inner
    with pytest.raises(TypeError, match="requires ComfyUI's native"):
        require_native_minimax_h3(SimpleNamespace(model=SimpleNamespace(diffusion_model=torch.nn.Linear(2, 2))))


def test_forecast_sanitization_clamps_and_replaces_nonfinite_values():
    source = torch.tensor([float("nan"), float("inf"), -float("inf"), 1e20, 2.0])
    sanitized, event = _sanitize_prediction(source, torch.float16)
    assert event is not None
    assert torch.isfinite(sanitized).all()
    assert sanitized.dtype == torch.float16
    all_bad, event = _sanitize_prediction(torch.tensor([float("nan"), float("inf")]), torch.float16)
    assert all_bad is None
    assert "no finite" in event["reason"]


def test_model_clone_callback_provisions_an_isolated_runtime():
    source_runtime = SpectrumH3Runtime(SpectrumH3Config(degree=1, max_history=4))
    source = SimpleNamespace(model_options={BINDING_KEY: SpectrumH3Binding(source_runtime)})
    clone = SimpleNamespace(model_options={BINDING_KEY: source.model_options[BINDING_KEY]})
    model_clone_callback(source, clone)
    clone_runtime = clone.model_options[BINDING_KEY].runtime
    assert clone_runtime is not source_runtime
    assert clone_runtime.config == source_runtime.config
