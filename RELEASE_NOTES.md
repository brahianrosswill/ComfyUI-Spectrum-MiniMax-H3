# Spectrum MiniMax H3 v0.1.3

Stabilizes RES multistep audio-video forecasting and adds Comfy Registry publishing metadata.

## Changed

- Force an actual H3 refresh after every RES multistep forecast so the second-order update never combines two forecasted denoised results.
- Default `tail_actual_steps` to `1` for the standard and aggressive presets.
- Add the xmarre Comfy Registry publisher metadata and publishing workflow.

## Highlights

- Forecasts selected post-transformer H3 features while preserving the native current-step output heads, reconstruction, sigma mapping, and return structure.
- Keeps runtime and history state isolated per model clone and rolls incomplete split-branch transactions back to a complete native step.
- Supports native Euler, Euler ancestral, and RES multistep sampling, with solver-aware RES refreshes and explicit native fallback for unsupported samplers, incompatible topology, invalid forecasts, and multi-GPU parallel sampling.
- Bounds retained history on CPU and streams forecast accumulation in chunks to avoid persistent full-feature FP32 coefficient or right-hand-side tensors.
- Leaves the separate FLUX-focused ComfyUI-Spectrum-Proper repository unchanged.

## Validation

- The local suite passes against native ComfyUI commit `e377e263049f9338b4d12a3dd417b36ae62948ff`.
- Automated tests cover forecasting mathematics, scheduling, rollback, clone isolation, the actual ComfyUI loader shape, native-path equivalence, and zero transformer-block execution on forecast steps.
- RES contract tests verify that the native second-order update consumes both current and previous denoised results and that Spectrum inserts an actual refresh between forecasts.

## Current limits

The reported sharp-audio reproduction requires a full MiniMax H3 checkpoint that is unavailable in the automated environment. The recurrence path is covered structurally; decoded output quality and the effective RES speedup still require real-generation confirmation.
