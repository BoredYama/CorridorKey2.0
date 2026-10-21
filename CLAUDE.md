# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CorridorKey 2.0: an inference-only release of a two-stage transformer video matting (green-screen keying) model. It produces alpha and foreground RGB at native resolution. Requires an NVIDIA GPU with CUDA PyTorch (tested torch 2.9.1 / torchvision 0.24.1) and `ffmpeg` on PATH. There is no training code, test suite, linter, or build step.

## Commands

```bash
# setup (weights ~5 GB go into weights/ and vfms/, both gitignored)
python -m venv venv && source venv/bin/activate
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/download_weights.py

# sanity check (checkpoint format only; add --load_model to build the model on CUDA)
python scripts/smoke_test_runtime.py [--load_model]

# inference
python infer.py --input clip.mp4 --output_dir out [--hint_first_frame matte.png | --hint_video hint.mp4]
python infer.py --input clip.mp4 --output_dir out_test --num_frames 20   # quick partial-clip test

# guided interactive session (what the drag-and-drop .sh/.bat launchers call)
python run_interactive.py /path/to/clip.mp4

# browser GUI (Gradio; wraps infer.py as a subprocess, never imports the model)
pip install -r gui/requirements.txt && ./gui/launch.sh [--listen --auth user:pass]
python gui/test_command.py   # tests for the GUI's settings -> infer.py argv builder
```

The README documents every `infer.py` flag (outputs, EXR/HDR, Hann windows, low-VRAM knobs, color handling) in detail.

## Architecture

**Entry points.** `infer.py` parses the CLI into `InferenceConfig` and `WriterConfig`, then calls `load_model` and `run_inference` from `corridorkey/runtime.py`. `run_interactive.py` is a prompt-driven wrapper that builds an `infer.py` command line and runs it as a subprocess. It also auto-detects companion hints (`<clip>FrameHint.png`, `<clip>_AlphaHint.mp4`) and Windows→Linux path mappings (`path_mappings.json` / `CORRIDORKEY_PATH_MAPPINGS`).

**Model (`corridorkey/model.py`, `CorridorKeyV3`; `CorridorKeyV2` in `__init__.py` is just a public alias).** Pipeline:
1. `foundations.py` `FrozenFoundations`: three frozen VFMs each produce /64-grid features, normalized with stats from `weights/foundation_stats.safetensors`. They are C-RADIOv4-SO400M (input H/4), MoGe-2 ViT-B (input 7H/32, from vendored `third_party/MoGe`), and RVM (`rvm.py`, a PyTorch port loading DeepMind's JAX `.npz`; recurrent, carries `rvm_state` across frames).
2. `stage1.py`: temporal transformer (full spatiotemporal attention at /64, Swin-3D at /16). It takes asinh-encoded RGB, the hint, and hint quality, and outputs a /16 "baton" plus a /2 auxiliary alpha/fg prediction.
3. `stage2.py`: per-frame Swin-2D refinement of baton + RGB to full-resolution alpha + fg.
`blocks.py` holds the shared attention/MLP blocks and `rope.py` the 2D/3D RoPE. H and W must be multiples of 64. The runtime reflect-pads to `shape_quantum` (128) and crops back.

**Runtime (`corridorkey/runtime.py`).** The core of inference:
- `RuntimePaths` resolves asset locations relative to the package root. It prefers `.safetensors` and falls back to legacy `.pt`. Checkpoint format string: `corridorkey_v2_runtime_checkpoint`, with `model_config` stored in the safetensors header.
- `prepare_runtime_environment` forces `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` and puts `third_party/MoGe` on `sys.path`. Everything loads from local folders and nothing is downloaded at runtime.
- `hann_stage1_stream`: stage 1 runs on overlapping temporal windows (`hann_chunk`/`hann_stride`). Window hidden states are blended with Hann taper weights (`video_io.taper_weights`), and stage 2 streams final frames. With carry-hint on, the stage-1 auxiliary alpha from earlier windows is fed back as the hint for overlapping frames.
- Auto-VRAM (`estimate_peak_vram_gb` / `select_vram_profile` / `apply_auto_vram`) picks the window size and low-VRAM mode from free GPU memory, unless `--hann_chunk` or `--low_vram` is passed.
- Low-VRAM mode precomputes RVM and foundation features to disk memmaps (`precompute_*_to_memmap`, in `--temp_dir`). `configure_low_vram_inference` sets chunking attributes on blocks by class name (`SwiGLU`, `SwinAttention3D`, etc.), so renaming those classes silently breaks it.
- Rendering to video (`ThreadedVideoWriter`, ffmpeg 16-bit pipe) or EXR (`ExrSequenceWriter`) happens in `_render_predictions_torch` / `_render_predictions_exr_torch`.

**I/O and color (`video_io.py`, `color_utils.py`).** Video/frame-dir probing, sequential window readers, hint readers, writers. The color contract: display-coded input goes through `srgb_to_linear` to linear and then through signed asinh (`linear_to_asinh`) into the model. The VFMs get SDR proxies instead. `--frame_dir_linear` skips the sRGB decode. The network runs in bfloat16.

## Conventions
- Fail loudly: missing weights, assets, ffmpeg or unsupported inputs raise errors. Do not add silent fallbacks or mode switches.
- `third_party/MoGe` is vendored upstream code (MIT). Avoid editing it.
- Training-only paths (MAE masks, recurrent stage-1 sites, gradient checkpointing) are still in the model code but are disabled by `load_model`. The checkpoint must not contain recurrence tensors (the smoke test checks this).
- License: custom CC BY-NC-SA–based terms (see LICENSE). Derivatives must stay free and keep the CorridorKey name.
