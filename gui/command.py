"""Translate GUI settings into an infer.py command line.

Pure logic, no Gradio import, so it can be tested on its own. Only flags whose
value differs from the infer.py default are emitted, which keeps the copyable
command short and lets infer.py's own defaults (notably automatic VRAM
selection) stay in charge.
"""

from __future__ import annotations

import os
import shlex
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INFER_PY = REPO_ROOT / "infer.py"

MODE_NONE = "No hint"
MODE_FIRST_FRAME = "First-frame hint"
MODE_HINT_VIDEO = "Full hint video"
MODES = (MODE_NONE, MODE_FIRST_FRAME, MODE_HINT_VIDEO)

OUTPUT_CHOICES = ("alpha", "fg", "checker", "cutout")
START_MODES = ("begin", "middle", "random_middle")

# infer.py defaults; flags equal to these are left out of the command.
DEFAULT_OUTPUTS = ("alpha", "fg")
DEFAULT_HINT_QUALITY = 0.95
DEFAULT_CARRY_HINT_QUALITY = 0.95
DEFAULT_DESPILL = 0.5
DEFAULT_FRAME_DIR_FPS = 24.0

DEFAULTS = {
    "input_path": "",
    "output_dir": "",
    "frame_dir_fps": DEFAULT_FRAME_DIR_FPS,
    "frame_dir_linear": False,
    "mode": MODE_NONE,
    "hint_path": "",
    "hint_quality": DEFAULT_HINT_QUALITY,
    "no_carry_hint": False,
    "carry_hint_quality": DEFAULT_CARRY_HINT_QUALITY,
    "outputs": list(DEFAULT_OUTPUTS),
    "despill_strength": DEFAULT_DESPILL,
    "cutout_linear": False,
    "output_format": "video",
    "exr_dtype": "half",
    "memory_mode": "Auto",
    "low_vram": False,
    "set_window": False,
    "hann_chunk": 24,
    "hann_stride": 12,
    "temp_dir": "",
    "quick_test": False,
    "num_frames": 100,
    "start_mode": "begin",
}


def clean_path(value) -> str:
    """Normalize a typed/pasted path: strip quotes, expand ~, make absolute."""
    if value is None:
        return ""
    s = str(value).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    if not s:
        return ""
    if os.name != "nt":
        s = s.replace("\\ ", " ")
    return os.path.abspath(os.path.expanduser(s))


def default_output_dir(input_path: str, uploaded: bool) -> str:
    """Mirror run_interactive.py: <input parent>/out_<stem>.

    Uploaded files live in Gradio's temp cache, so their outputs go under
    <repo>/outputs/ instead (matched by the repo's `out*/` gitignore rule).
    """
    if not input_path:
        return ""
    p = Path(input_path)
    stem = p.name if p.is_dir() else p.stem
    if uploaded:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return str(REPO_ROOT / "outputs" / f"{stem}_{stamp}")
    return str(p.parent / f"out_{stem}")


def _fmt(x: float) -> str:
    return f"{float(x):g}"


def _differs(a, b) -> bool:
    return abs(float(a) - float(b)) > 1e-9


def build_command(settings: dict, python: str, infer_path: str | os.PathLike = INFER_PY,
                  check_paths: bool = True) -> tuple[list[str], list[str]]:
    """Return (argv, errors). argv is still built when errors exist, for display."""
    s = {**DEFAULTS, **(settings or {})}
    errors: list[str] = []
    argv = [str(python), str(infer_path)]

    input_path = clean_path(s["input_path"])
    if not input_path:
        errors.append("No input selected.")
    elif check_paths and not os.path.exists(input_path):
        errors.append(f"Input does not exist: {input_path}")
    argv += ["--input", input_path or "<input>"]

    output_dir = clean_path(s["output_dir"])
    if not output_dir:
        errors.append("No output directory set.")
    elif check_paths and os.path.isfile(output_dir):
        errors.append(f"Output directory is an existing file: {output_dir}")
    argv += ["--output_dir", output_dir or "<output_dir>"]

    if input_path and os.path.isdir(input_path):
        try:
            fps = float(s["frame_dir_fps"])
        except (TypeError, ValueError):
            fps = 0.0
        if fps <= 0:
            errors.append("Frame-sequence FPS must be greater than 0.")
        elif _differs(fps, DEFAULT_FRAME_DIR_FPS):
            argv += ["--frame_dir_fps", _fmt(fps)]
        if s["frame_dir_linear"]:
            argv.append("--frame_dir_linear")

    mode = s["mode"]
    if mode not in MODES:
        errors.append(f"Unknown inference mode: {mode}")
    elif mode != MODE_NONE:
        hint = clean_path(s["hint_path"])
        flag = "--hint_first_frame" if mode == MODE_FIRST_FRAME else "--hint_video"
        if not hint:
            errors.append(f"{mode} mode needs a hint file.")
        elif check_paths and not os.path.exists(hint):
            errors.append(f"Hint does not exist: {hint}")
        argv += [flag, hint or "<hint>"]
        q = float(s["hint_quality"])
        if not 0.0 <= q <= 1.0:
            errors.append("Hint quality must be between 0 and 1.")
        if _differs(q, DEFAULT_HINT_QUALITY):
            argv += ["--hint_quality", _fmt(q)]
        if s["no_carry_hint"]:
            argv.append("--no_carry_hint")
        elif _differs(s["carry_hint_quality"], DEFAULT_CARRY_HINT_QUALITY):
            argv += ["--carry_hint_quality", _fmt(s["carry_hint_quality"])]

    outputs = [o for o in OUTPUT_CHOICES if o in (s["outputs"] or [])]
    if not outputs:
        errors.append("Select at least one output.")
    elif sorted(outputs) != sorted(DEFAULT_OUTPUTS):
        argv += ["--outputs", *outputs]
    if ("checker" in outputs or "cutout" in outputs) and _differs(
            s["despill_strength"], DEFAULT_DESPILL):
        argv += ["--despill_strength", _fmt(s["despill_strength"])]

    exr = s["output_format"] == "exr"
    # EXR output is always scene-linear, so --cutout_linear only matters for video.
    if s["cutout_linear"] and "cutout" in outputs and not exr:
        argv.append("--cutout_linear")
    if exr:
        argv += ["--output_format", "exr"]
        if s["exr_dtype"] == "float":
            argv += ["--exr_dtype", "float"]

    # Passing none of --low_vram/--hann_chunk/--hann_stride keeps infer.py's
    # automatic per-clip VRAM selection on.
    if s["memory_mode"] == "Manual":
        if s["low_vram"]:
            argv.append("--low_vram")
            temp_dir = clean_path(s["temp_dir"])
            if temp_dir:
                argv += ["--temp_dir", temp_dir]
        if s["set_window"]:
            try:
                chunk, stride = int(s["hann_chunk"]), int(s["hann_stride"])
            except (TypeError, ValueError):
                chunk, stride = 0, 0
            if chunk < 1 or not 1 <= stride <= chunk:
                errors.append("Hann window needs chunk >= 1 and 1 <= stride <= chunk.")
            argv += ["--hann_chunk", str(chunk), "--hann_stride", str(stride)]
        if not s["low_vram"] and not s["set_window"]:
            errors.append("Manual memory mode: enable Low VRAM and/or set a Hann window "
                          "(or switch back to Auto).")

    if s["quick_test"]:
        try:
            n = int(s["num_frames"])
        except (TypeError, ValueError):
            n = 0
        if n < 1:
            errors.append("Quick test needs at least 1 frame.")
        else:
            argv += ["--num_frames", str(n)]
            if s["start_mode"] != "begin":
                argv += ["--start_mode", s["start_mode"]]

    return argv, errors


def display_command(argv: list[str]) -> str:
    return shlex.join(argv)
