# CorridorKey 2.0 GUI

A browser front end for `infer.py`. The GUI builds the same command you would
type yourself (shown, copyable, above the Run button), runs it as a
subprocess, streams its log, and shows playable previews of the results. It
does not import or change the model code.

## Install

Finish the main setup in the repository README first (venv, PyTorch,
`pip install -r requirements.txt`, `python scripts/download_weights.py`). Then,
in the same venv:

```
source venv/bin/activate
pip install -r gui/requirements.txt
```

## Launch

```
./gui/launch.sh            # Linux (uses ./venv if present)
gui\launch.bat             # Windows
```

Then open http://127.0.0.1:7860.

Options (`python gui/app.py --help`):

- `--port 7860`
- `--listen`: bind `0.0.0.0` so other machines can connect
- `--auth user:pass`: require a login (use it with `--listen`)
- `--share`: public gradio.live tunnel (off by default)
- `--python /path/to/python`: interpreter for `infer.py` (default: the one running the GUI)

## Running on a remote GPU machine

Safest: keep the default localhost binding and tunnel the port over SSH from
your PC, then open http://127.0.0.1:7860 locally:

```
ssh -L 7860:localhost:7860 user@gpu-box
```

Or run `./gui/launch.sh --listen --auth me:secret` on the GPU machine and open
`http://gpu-box:7860`. Anyone who can log in can run jobs and write to any
folder the GUI's user can write to.

Paths typed into "Path on GPU machine" and "Output directory" are on the
machine running the GUI. Uploaded files are copied to it. Previews stream back
to the browser, and "Download full-quality outputs" sends the original files.

## Notes

- **Memory: Auto** passes no memory flags, so `infer.py` picks the Hann
  window and low-VRAM mode per clip from free GPU memory. **Manual** lets you
  set Low VRAM and/or a Hann window yourself.
- Frame-sequence folders can only be given as a path (browsers can't upload a
  folder as a clip). The FPS and scene-linear options appear once a folder is
  selected.
- Previews are 8-bit H.264 copies (max 1280 px wide) made with ffmpeg, because
  browsers can't play the 10-bit 4:4:4 x264 or ProRes 4444 originals. The
  cutout preview is flattened over grey. EXR output has no preview.
- Only one job runs at a time. Cancel stops `infer.py` and its ffmpeg
  writers. Partial outputs are left in place.
- Uploaded clips write their outputs to `outputs/<clip>_<timestamp>/` in the
  repository (git-ignored) unless you set another output directory.

## Tests

```
python gui/test_command.py
```
