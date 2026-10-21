#!/usr/bin/env python3
"""CorridorKey 2.0 browser GUI.

Wraps infer.py: the GUI builds an infer.py command line, runs it as a
subprocess, streams its log, and transcodes the results into browser-playable
previews. Nothing here imports the model code, so upstream changes to the
runtime only matter if infer.py's flags change.
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from command import (  # noqa: E402
    DEFAULTS, INFER_PY, MODE_FIRST_FRAME, MODE_NONE, MODES,
    OUTPUT_CHOICES, REPO_ROOT, START_MODES, build_command, clean_path,
    default_output_dir, display_command,
)

PREVIEW_ROOT = Path(tempfile.gettempdir()) / "corridorkey_gui" / "previews"
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mxf")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff")
HANN_PRESETS = {
    "80 / 40 (max continuity)": (80, 40),
    "48 / 24": (48, 24),
    "24 / 12": (24, 12),
    "8 / 4 (low memory)": (8, 4),
    "1 / 1 (single frame, lowest memory)": (1, 1),
}
MAX_LOG_LINES = 2000
SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)  # Windows has no SIGKILL
# Preview slot per output, matched against infer.py's output file names.
PREVIEW_SLOTS = (
    ("alpha", "_alpha.", "Alpha"),
    ("fg", "_fg.", "Foreground (raw fg)"),
    ("checker", "_checker", "Checker composite"),
    ("cutout", "_cutout.", "Cutout (over grey; alpha flattened for preview)"),
)

APP = argparse.Namespace(python=sys.executable, infer=str(INFER_PY), ffmpeg="ffmpeg")


class Job:
    """The single running inference process (one GPU, one job)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.cancelled = False

    def start(self, argv, env) -> subprocess.Popen | None:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return None
            self.cancelled = False
            # Own process group, so Cancel also reaches infer.py's ffmpeg writers.
            group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                     if os.name == "nt" else {"start_new_session": True})
            self.proc = subprocess.Popen(
                argv, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                bufsize=1, errors="replace", **group)
            return self.proc

    def cancel(self) -> bool:
        with self.lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                return False
            self.cancelled = True
        # infer.py spawns ffmpeg writers; signal the whole process group.
        _signal_group(proc, signal.SIGTERM)
        threading.Thread(target=_kill_after, args=(proc, 10.0), daemon=True).start()
        return True

    def finish(self):
        with self.lock:
            self.proc = None


def _signal_group(proc, sig):
    if os.name == "nt":
        # No process groups to signal on Windows; kill the process tree instead.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
        return
    # start_new_session makes the child a group leader, so its pid is the pgid;
    # this still reaches leftover children after the leader itself has exited.
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _kill_after(proc, timeout):
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _signal_group(proc, SIGKILL)


JOB = Job()
# Output files written by runs in this process; the only ones offered for download.
PRODUCED_FILES: set[str] = set()

# ---------------------------------------------------------------------------
# Settings plumbing
# ---------------------------------------------------------------------------

SETTING_KEYS = [
    "input_source", "input_upload", "input_server_path",
    "frame_dir_fps", "frame_dir_linear",
    "mode", "hint_source", "hint_upload", "hint_server_path",
    "hint_quality", "no_carry_hint", "carry_hint_quality",
    "outputs", "despill_strength", "cutout_linear", "output_format", "exr_dtype",
    "output_dir",
    "memory_mode", "low_vram", "set_window", "hann_chunk", "hann_stride", "temp_dir",
    "quick_test", "num_frames", "start_mode",
]


def _file_path(value) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    return getattr(value, "name", None) or getattr(value, "path", "") or ""


def _settings(values) -> dict:
    s = dict(zip(SETTING_KEYS, values))
    upload = s.pop("input_source") == "Upload"
    s["input_path"] = _file_path(s["input_upload"]) if upload else s["input_server_path"]
    hint_upload = s.pop("hint_source") == "Upload"
    s["hint_path"] = _file_path(s["hint_upload"]) if hint_upload else s["hint_server_path"]
    s["output_format"] = "exr" if s["output_format"] == "EXR image sequence" else "video"
    s["exr_dtype"] = "float" if s["exr_dtype"].startswith("float") else "half"
    for key in ("input_upload", "input_server_path", "hint_upload", "hint_server_path"):
        s.pop(key)
    return s


def preview_command(*values) -> str:
    argv, errors = build_command(_settings(values), APP.python, APP.infer, check_paths=True)
    text = display_command(argv)
    if errors:
        text += "\n\n# Not ready:\n" + "\n".join(f"#  - {e}" for e in errors)
    return text


def on_input_changed(source, upload, server_path):
    uploaded = source == "Upload"
    path = clean_path(_file_path(upload) if uploaded else server_path)
    is_dir = bool(path) and os.path.isdir(path)
    out = default_output_dir(path, uploaded=uploaded) if path and os.path.exists(path) else ""
    return (gr.update(value=out), gr.update(visible=is_dir), gr.update(visible=is_dir))


def on_source_toggle(source):
    upload = source == "Upload"
    return gr.update(visible=upload), gr.update(visible=not upload)


def on_mode_changed(mode, hint_source):
    hinted = mode != MODE_NONE
    upload = hint_source == "Upload"
    types = list(IMAGE_EXTS) if mode == MODE_FIRST_FRAME else list(VIDEO_EXTS)
    label = ("First-frame matte (white = subject; image with alpha also works)"
             if mode == MODE_FIRST_FRAME else "Hint video (matte / roto for every frame)")
    path_info = ("Image file on the GPU machine" if mode == MODE_FIRST_FRAME
                 else "Video file or frame-sequence folder on the GPU machine")
    return (
        gr.update(visible=hinted),
        gr.update(visible=hinted and upload, file_types=types, label=label, value=None),
        gr.update(visible=hinted and not upload, info=path_info),
    )


def on_hint_source_toggle(mode, hint_source):
    hinted = mode != MODE_NONE
    upload = hint_source == "Upload"
    return gr.update(visible=hinted and upload), gr.update(visible=hinted and not upload)


def on_outputs_changed(outputs, output_format):
    outputs = outputs or []
    despill = "checker" in outputs or "cutout" in outputs
    linear = "cutout" in outputs and output_format != "EXR image sequence"
    return gr.update(interactive=despill), gr.update(interactive=linear)


def on_format_changed(outputs, output_format):
    exr = output_format == "EXR image sequence"
    despill, linear = on_outputs_changed(outputs, output_format)
    return gr.update(visible=exr), linear


def on_preset(name):
    chunk, stride = HANN_PRESETS.get(name, (None, None))
    if chunk is None:
        return gr.update(), gr.update()
    return gr.update(value=chunk), gr.update(value=stride)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _find_outputs(output_dir: str, since: float) -> list[Path]:
    root = Path(output_dir)
    if not root.is_dir():
        return []
    found = []
    for p in sorted(root.iterdir()):
        try:
            fresh = p.stat().st_mtime >= since - 1
        except OSError:
            continue
        if not fresh:
            continue
        if p.is_file() and p.suffix.lower() in (".mp4", ".mov"):
            found.append(p)
        elif p.is_dir() and p.name.endswith("_exr"):
            found.append(p)
    return found


def _make_preview(src: Path, dst: Path, has_alpha: bool) -> str | None:
    scale = "scale='min(1280,iw)':-2,format=yuv420p"
    if has_alpha:
        # Flatten straight-alpha ProRes over mid grey so transparency is visible.
        vf = ["-filter_complex",
              "[0:v]split[a][b];[a]drawbox=c=0x808080@1:t=fill,format=yuv420p[bg];"
              f"[bg][b]overlay=format=auto,{scale}[v]", "-map", "[v]"]
    else:
        vf = ["-vf", scale]
    cmd = [APP.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), *vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an",
           "-movflags", "+faststart", str(dst)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"preview failed for {src.name}: {e}"
    if r.returncode != 0:
        return f"preview failed for {src.name}: {r.stderr.strip()[-500:]}"
    return None


def _no_previews():
    return [gr.update(value=None, visible=False) for _ in PREVIEW_SLOTS]


def _result(log, status, previews=None, files_state=None, running=False):
    previews = previews if previews is not None else [gr.update() for _ in PREVIEW_SLOTS]
    if files_state is None:
        files_update, dl_update = gr.update(), gr.update()
    else:
        files_update, dl_update = files_state, gr.update(visible=bool(files_state))
    return (
        log, status, *previews, files_update,
        gr.update(interactive=not running),
        gr.update(interactive=running),
        dl_update,
    )


def run_job(*values):
    s = _settings(values)
    argv, errors = build_command(s, APP.python, APP.infer)
    if errors:
        yield _result("", "**Cannot start:** " + " ".join(errors), _no_previews(), [])
        return
    output_dir = argv[argv.index("--output_dir") + 1]
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        yield _result("", f"**Cannot create output directory:** {e}", _no_previews(), [])
        return

    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.time()
    proc = JOB.start(argv, env)
    if proc is None:
        yield _result(gr.update(), "**Another job is already running.** Cancel it or wait.")
        return

    lines = [f"$ {display_command(argv)}", ""]
    q: queue.Queue = queue.Queue()

    def pump():
        for line in proc.stdout:
            q.put(line.rstrip("\n"))
        q.put(None)

    threading.Thread(target=pump, daemon=True).start()
    yield _result("\n".join(lines), "Running…", _no_previews(), [], running=True)

    done = False
    exited_at = None
    while not done:
        try:
            item = q.get(timeout=0.25)
            while True:
                if item is None:
                    done = True
                    break
                lines.append(item)
                item = q.get_nowait()
        except queue.Empty:
            pass
        # A leftover child (e.g. an ffmpeg writer after a crash) can hold the
        # log pipe open after infer.py exits; stop waiting and clean it up.
        if not done and proc.poll() is not None:
            exited_at = exited_at or time.time()
            if time.time() - exited_at > 3:
                _signal_group(proc, SIGKILL)
                lines.append("[infer.py exited; stopped leftover child processes]")
                done = True
        if len(lines) > MAX_LOG_LINES:
            lines = lines[:2] + ["[… earlier log lines trimmed …]"] + lines[-MAX_LOG_LINES:]
        elapsed = time.time() - started
        yield _result("\n".join(lines), f"Running… {elapsed:.0f} s", running=True)

    code = proc.wait()
    JOB.finish()
    elapsed = time.time() - started

    if JOB.cancelled:
        lines.append("\n[Cancelled. Partial outputs, if any, were left in the output directory.]")
        yield _result("\n".join(lines), f"**Cancelled** after {elapsed:.0f} s.", _no_previews(), [])
        return
    if code != 0:
        yield _result("\n".join(lines),
                      f"**Failed** with exit code {code} after {elapsed:.0f} s. See the log.",
                      _no_previews(), [])
        return

    outputs = _find_outputs(output_dir, started)
    lines.append(f"\n[Finished in {elapsed:.0f} s. Outputs in {output_dir}]")
    yield _result("\n".join(lines), "Building previews…", running=True)

    preview_dir = PREVIEW_ROOT / time.strftime("%Y%m%d-%H%M%S")
    preview_dir.mkdir(parents=True, exist_ok=True)
    previews = _no_previews()
    for i, (_, marker, label) in enumerate(PREVIEW_SLOTS):
        src = next((p for p in outputs if p.is_file() and marker in p.name), None)
        if src is None:
            continue
        dst = preview_dir / f"{src.stem}_preview.mp4"
        err = _make_preview(src, dst, has_alpha=src.suffix.lower() == ".mov")
        if err:
            lines.append(f"[{err}]")
        else:
            previews[i] = gr.update(value=str(dst), visible=True, label=label)

    listing = "\n".join(f"- `{p}`" + ("  (EXR sequence)" if p.is_dir() else "") for p in outputs)
    status = f"**Done** in {elapsed:.0f} s. Output directory: `{output_dir}`"
    if listing:
        status += "\n\n" + listing
    if any(p.is_dir() for p in outputs):
        status += "\n\nEXR sequences have no browser preview; open them in your compositor."
    files = [str(p) for p in outputs if p.is_file()]
    PRODUCED_FILES.update(files)
    yield _result("\n".join(lines), status, previews, files)


def cancel_job():
    if JOB.cancel():
        return "Cancelling… (stopping infer.py and its ffmpeg writers)"
    return "Nothing is running."


def prepare_downloads(files):
    """Expose the full-quality output files for download.

    Gradio only serves files from allowed paths, so each output file is added
    to them. This copies the files into Gradio's cache, which can take a while
    for long 4K ProRes cutouts, so it is opt-in. Only files produced by a run in
    this GUI qualify, so the endpoint can't be used to read arbitrary files.
    """
    files = [f for f in (files or []) if f in PRODUCED_FILES]
    if not files:
        return gr.update(value=None, visible=False)
    for f in files:
        if f not in DEMO.allowed_paths:
            DEMO.allowed_paths.append(f)
    return gr.update(value=files, visible=True)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="CorridorKey 2.0") as demo:
        gr.Markdown("# CorridorKey 2.0\nAI green/blue screen keyer. This page builds and runs "
                    "an `infer.py` command; see the README for what each option does.")
        c: dict[str, gr.components.Component] = {}

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group():
                    gr.Markdown("### Input")
                    c["input_source"] = gr.Radio(["Upload", "Path on GPU machine"],
                                                 value="Upload", show_label=False)
                    c["input_upload"] = gr.File(label="Video clip", file_types=list(VIDEO_EXTS),
                                                type="filepath")
                    c["input_server_path"] = gr.Textbox(
                        label="Video file or frame-sequence folder", visible=False,
                        placeholder="/path/to/shot.mp4 or /path/to/frames",
                        info="Folders: EXR/PNG/JPG frames, read in sorted filename order.")
                    with gr.Row():
                        c["frame_dir_fps"] = gr.Number(DEFAULTS["frame_dir_fps"],
                                                       label="Frame-sequence FPS",
                                                       minimum=1, visible=False)
                        c["frame_dir_linear"] = gr.Checkbox(
                            False, label="Frames are scene-linear (e.g. linear EXR plates)",
                            visible=False)

                with gr.Group():
                    gr.Markdown("### Inference mode")
                    c["mode"] = gr.Radio(list(MODES), value=MODE_NONE, show_label=False)
                    with gr.Column(visible=False) as hint_box:
                        c["hint_source"] = gr.Radio(["Upload", "Path on GPU machine"],
                                                    value="Upload", show_label=False)
                        c["hint_upload"] = gr.File(type="filepath", visible=False)
                        c["hint_server_path"] = gr.Textbox(label="Hint path", visible=False)
                        c["hint_quality"] = gr.Slider(
                            0, 1, DEFAULTS["hint_quality"], step=0.05, label="Hint quality",
                            info="0.95 clean artist matte · 0.75 rough automated hint · "
                                 "≤0.5 loose/unreliable")
                        c["no_carry_hint"] = gr.Checkbox(
                            False, label="No carry hint",
                            info="Don't feed the model's own predictions back as hints on "
                                 "overlapping frames; use the explicit hint literally.")
                        c["carry_hint_quality"] = gr.Slider(
                            0, 1, DEFAULTS["carry_hint_quality"], step=0.05,
                            label="Carry hint quality")

                with gr.Group():
                    gr.Markdown("### Outputs")
                    c["outputs"] = gr.CheckboxGroup(list(OUTPUT_CHOICES),
                                                    value=list(DEFAULTS["outputs"]),
                                                    show_label=False)
                    with gr.Row():
                        c["despill_strength"] = gr.Slider(
                            0, 1, DEFAULTS["despill_strength"], step=0.05,
                            label="Despill strength", info="Applies to checker and cutout.",
                            interactive=False, scale=2)
                        c["cutout_linear"] = gr.Checkbox(
                            False, label="Cutout as scene-linear RGB", interactive=False)
                    with gr.Row():
                        c["output_format"] = gr.Radio(["Video", "EXR image sequence"],
                                                      value="Video", label="Format")
                        c["exr_dtype"] = gr.Radio(["half (float16)", "float (float32)"],
                                                  value="half (float16)",
                                                  label="EXR precision", visible=False)
                    c["output_dir"] = gr.Textbox(
                        label="Output directory (on the GPU machine)",
                        info="Filled in from the input; created if missing.")

                with gr.Accordion("Memory", open=False):
                    c["memory_mode"] = gr.Radio(
                        ["Auto", "Manual"], value="Auto", show_label=False,
                        info="Auto picks the Hann window and low-VRAM mode per clip from "
                             "free GPU memory.")
                    with gr.Column(visible=False) as manual_box:
                        c["low_vram"] = gr.Checkbox(
                            False, label="Low VRAM mode",
                            info="Caches foundation features to disk; lower peak VRAM, slower.")
                        c["temp_dir"] = gr.Textbox(
                            label="Temp dir for feature cache (low VRAM)",
                            placeholder=str(Path(tempfile.gettempdir()) / "corridorkey_v2"))
                        c["set_window"] = gr.Checkbox(False, label="Set Hann window")
                        preset = gr.Dropdown(list(HANN_PRESETS), value="24 / 12",
                                             label="Preset (chunk / stride)")
                        with gr.Row():
                            c["hann_chunk"] = gr.Number(DEFAULTS["hann_chunk"], precision=0,
                                                        minimum=1, label="hann_chunk")
                            c["hann_stride"] = gr.Number(DEFAULTS["hann_stride"], precision=0,
                                                         minimum=1, label="hann_stride")

                with gr.Accordion("Quick test", open=False):
                    c["quick_test"] = gr.Checkbox(False, label="Only process part of the clip")
                    with gr.Row():
                        c["num_frames"] = gr.Number(DEFAULTS["num_frames"], precision=0,
                                                    minimum=1, label="Frames")
                        c["start_mode"] = gr.Dropdown(list(START_MODES), value="begin",
                                                      label="Start")

            with gr.Column(scale=4):
                cmd_box = gr.Code(label="Command", language="shell", lines=3,
                                  interactive=False, wrap_lines=True, buttons=["copy"])
                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary")
                    cancel_btn = gr.Button("Cancel", variant="stop", interactive=False)
                status = gr.Markdown("Idle.")
                log = gr.Textbox(label="Log", lines=18, max_lines=18, autoscroll=True,
                                 interactive=False, buttons=["copy"])
                with gr.Row():
                    videos = [gr.Video(label=label, visible=False, interactive=False,
                                       loop=True)
                              for _, _, label in PREVIEW_SLOTS]
                files_state = gr.State([])
                dl_btn = gr.Button("Download full-quality outputs", visible=False)
                downloads = gr.File(label="Full-quality outputs", file_count="multiple",
                                    visible=False, interactive=False)

        inputs = [c[k] for k in SETTING_KEYS]

        # Visibility / interactivity wiring.
        c["input_source"].change(on_source_toggle, c["input_source"],
                                 [c["input_upload"], c["input_server_path"]])
        gr.on([c["input_source"].change, c["input_upload"].change,
               c["input_server_path"].blur, c["input_server_path"].submit],
              on_input_changed,
              [c["input_source"], c["input_upload"], c["input_server_path"]],
              [c["output_dir"], c["frame_dir_fps"], c["frame_dir_linear"]])
        c["mode"].change(on_mode_changed, [c["mode"], c["hint_source"]],
                         [hint_box, c["hint_upload"], c["hint_server_path"]])
        c["hint_source"].change(on_hint_source_toggle, [c["mode"], c["hint_source"]],
                                [c["hint_upload"], c["hint_server_path"]])
        c["no_carry_hint"].change(lambda v: gr.update(visible=not v), c["no_carry_hint"],
                                  c["carry_hint_quality"])
        c["outputs"].change(on_outputs_changed, [c["outputs"], c["output_format"]],
                            [c["despill_strength"], c["cutout_linear"]])
        c["output_format"].change(on_format_changed, [c["outputs"], c["output_format"]],
                                  [c["exr_dtype"], c["cutout_linear"]])
        c["memory_mode"].change(lambda m: gr.update(visible=m == "Manual"),
                                c["memory_mode"], manual_box)
        preset.change(on_preset, preset, [c["hann_chunk"], c["hann_stride"]])

        # Live command preview.
        gr.on([comp.change for comp in inputs] + [demo.load], preview_command, inputs,
              cmd_box, show_progress="hidden", queue=False)

        run_outputs = [log, status, *videos, files_state, run_btn, cancel_btn, dl_btn]
        run_btn.click(lambda: gr.update(value=None, visible=False), None, downloads,
                      queue=False).then(
            run_job, inputs, run_outputs, concurrency_limit=None, show_progress="hidden")
        cancel_btn.click(cancel_job, None, status, queue=False)
        dl_btn.click(prepare_downloads, files_state, downloads)
    return demo


DEMO: gr.Blocks | None = None


def main():
    global DEMO
    ap = argparse.ArgumentParser(description="CorridorKey 2.0 Gradio GUI")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Interface to bind. Default: localhost only.")
    ap.add_argument("--listen", action="store_true",
                    help="Bind 0.0.0.0 so other machines on the network can connect.")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--auth", default=None, metavar="USER:PASS",
                    help="Require a login. Strongly recommended with --listen.")
    ap.add_argument("--share", action="store_true",
                    help="Create a public gradio.live tunnel (off by default).")
    ap.add_argument("--python", default=sys.executable,
                    help="Python used to run infer.py. Default: this interpreter.")
    ap.add_argument("--infer", default=str(INFER_PY), help=argparse.SUPPRESS)
    ap.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg used for previews.")
    args = ap.parse_args()

    APP.python, APP.infer, APP.ffmpeg = args.python, args.infer, args.ffmpeg
    if shutil.which(APP.ffmpeg) is None:
        print(f"[WARNING] {APP.ffmpeg!r} not found on PATH; inference and previews need it.")
    auth = None
    if args.auth:
        user, sep, password = args.auth.partition(":")
        if not sep or not user:
            ap.error("--auth must be USER:PASS")
        auth = (user, password)
    host = "0.0.0.0" if args.listen else args.host
    if host != "127.0.0.1" and auth is None:
        print("[WARNING] Listening on the network without --auth: anyone who can reach "
              f"port {args.port} can run jobs and read files the GUI can access.")

    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    DEMO = build_ui()
    DEMO.queue(default_concurrency_limit=None)
    DEMO.launch(server_name=host, server_port=args.port, auth=auth, share=args.share,
                allowed_paths=[str(PREVIEW_ROOT)], max_file_size=None,
                theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
