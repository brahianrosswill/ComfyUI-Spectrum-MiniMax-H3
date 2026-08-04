# Spectrum MiniMax H3 v0.1.3

Stabilizes deterministic Euler and RES multistep audio-video forecasting and adds Comfy Registry publishing metadata.

## Changed

- Force an actual H3 refresh after every RES multistep forecast so the second-order update never combines two forecasted denoised results.
- Apply the same one-forecast limit to Euler to prevent late forecast streaks from accumulating temporal audio/video errors on short schedules.
- Keep ancestral Euler and RES variants on the native path because their injected noise breaks the forecaster's smooth deterministic trajectory assumption.
- Default `tail_actual_steps` to `1` for the standard and aggressive presets.
- Add the xmarre Comfy Registry publisher metadata and publishing workflow.

## Highlights

- Forecasts selected post-transformer H3 features while preserving the native current-step output heads, reconstruction, sigma mapping, and return structure.
- Keeps runtime and history state isolated per model clone and rolls incomplete split-branch transactions back to a complete native step.
- Supports deterministic Euler and RES multistep sampling with a native refresh after every forecast, plus explicit native fallback for stochastic samplers, unsupported samplers, incompatible topology, invalid forecasts, and multi-GPU parallel sampling.
- Bounds retained history on CPU and streams forecast accumulation in chunks to avoid persistent full-feature FP32 coefficient or right-hand-side tensors.
- Leaves the separate FLUX-focused ComfyUI-Spectrum-Proper repository unchanged.

## Validation

- The local suite passes against native ComfyUI commit `e377e263049f9338b4d12a3dd417b36ae62948ff`.
- Automated tests cover forecasting mathematics, scheduling, rollback, clone isolation, the actual ComfyUI loader shape, native-path equivalence, and zero transformer-block execution on forecast steps.
- Sampler contract tests verify one model call per deterministic solver step, RES's current/previous-denoised recurrence, ancestral noise injection, and an actual refresh between every pair of forecasts.

## Current limits

The reported audio and high-motion video artifacts require a full MiniMax H3 checkpoint that is unavailable in the automated environment. Consecutive forecast accumulation is removed structurally; decoded output quality and effective speedup still require real-generation confirmation.
