from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

RES_VARIANTS = (
    "sample_res_multistep",
    "sample_res_multistep_cfg_pp",
)

ANCESTRAL_VARIANTS = (
    "sample_euler_ancestral",
    "sample_euler_ancestral_RF",
    "sample_res_multistep_ancestral",
    "sample_res_multistep_ancestral_cfg_pp",
)


def _native_sampling_functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if not comfyui_path:
        pytest.skip("COMFYUI_PATH is required for native sampler contract tests")
    source_path = Path(comfyui_path) / "comfy" / "k_diffusion" / "sampling.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _named_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name
    ]


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }


def test_native_res_multistep_makes_one_model_call_per_solver_iteration():
    function = _native_sampling_functions()["res_multistep"]
    loops = [node for node in function.body if isinstance(node, (ast.For, ast.AsyncFor))]

    assert len(loops) == 1
    assert len(_named_calls(loops[0], "model")) == 1
    assert len(_named_calls(function, "model")) == 1


def test_native_res_multistep_second_order_update_reuses_current_and_previous_denoised():
    function = _native_sampling_functions()["res_multistep"]
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "x" for target in node.targets)
    ]

    assert any({"denoised", "old_denoised"} <= _loaded_names(node.value) for node in assignments)


def test_native_euler_makes_one_model_call_per_solver_iteration():
    function = _native_sampling_functions()["sample_euler"]
    loops = [node for node in function.body if isinstance(node, (ast.For, ast.AsyncFor))]

    assert len(loops) == 1
    assert len(_named_calls(loops[0], "model")) == 1
    assert len(_named_calls(function, "model")) == 1


@pytest.mark.parametrize("function_name", RES_VARIANTS)
def test_native_res_variant_delegates_once_to_reviewed_core(function_name):
    function = _native_sampling_functions()[function_name]

    assert len(_named_calls(function, "res_multistep")) == 1
    assert not _named_calls(function, "model")


@pytest.mark.parametrize("function_name", ANCESTRAL_VARIANTS)
def test_native_ancestral_variants_inject_or_delegate_to_noise(function_name):
    function = _native_sampling_functions()[function_name]
    loaded = _loaded_names(function)
    delegates_to_ancestral_core = bool(
        _named_calls(function, "sample_euler_ancestral_RF")
        or _named_calls(function, "res_multistep")
    )

    assert "noise_sampler" in loaded or delegates_to_ancestral_core
