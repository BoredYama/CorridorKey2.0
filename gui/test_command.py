#!/usr/bin/env python3
"""Tests for gui/command.py. Run with: python gui/test_command.py"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from command import (  # noqa: E402
    MODE_FIRST_FRAME, MODE_HINT_VIDEO, build_command, clean_path,
    default_output_dir, display_command,
)

TMP = Path(tempfile.mkdtemp(prefix="ck_gui_test_"))
CLIP = TMP / "shot.mp4"
CLIP.write_bytes(b"")
FRAMES = TMP / "frames_dir"
FRAMES.mkdir()
HINT = TMP / "matte.png"
HINT.write_bytes(b"")
OUT = str(TMP / "out")


def build(**kw):
    s = {"input_path": str(CLIP), "output_dir": OUT, **kw}
    return build_command(s, "python", "infer.py")


def flags(argv):
    return [a for a in argv if a.startswith("--")]


def test_minimal():
    argv, errors = build()
    assert errors == [], errors
    assert argv == ["python", "infer.py", "--input", str(CLIP), "--output_dir", OUT]


def test_auto_memory_emits_nothing():
    argv, _ = build(low_vram=True, set_window=True, temp_dir="/x")
    assert not {"--low_vram", "--hann_chunk", "--hann_stride", "--temp_dir"} & set(argv)


def test_manual_memory():
    argv, errors = build(memory_mode="Manual", set_window=True, hann_chunk=8, hann_stride=4)
    assert errors == []
    assert argv[-4:] == ["--hann_chunk", "8", "--hann_stride", "4"]
    assert "--low_vram" not in argv
    argv, errors = build(memory_mode="Manual", low_vram=True, temp_dir="/scratch/ck")
    assert errors == []
    assert "--low_vram" in argv and argv[argv.index("--temp_dir") + 1] == "/scratch/ck"
    assert "--hann_chunk" not in argv


def test_manual_memory_errors():
    _, errors = build(memory_mode="Manual")
    assert errors
    _, errors = build(memory_mode="Manual", set_window=True, hann_chunk=4, hann_stride=8)
    assert errors


def test_first_frame_hint():
    argv, errors = build(mode=MODE_FIRST_FRAME, hint_path=str(HINT), hint_quality=0.75)
    assert errors == []
    assert argv[argv.index("--hint_first_frame") + 1] == str(HINT)
    assert argv[argv.index("--hint_quality") + 1] == "0.75"
    assert "--carry_hint_quality" not in argv


def test_hint_video_no_carry_suppresses_carry_quality():
    argv, errors = build(mode=MODE_HINT_VIDEO, hint_path=str(HINT), no_carry_hint=True,
                         carry_hint_quality=0.5)
    assert errors == []
    assert "--hint_video" in argv and "--no_carry_hint" in argv
    assert "--carry_hint_quality" not in argv
    assert "--hint_quality" not in argv  # 0.95 is the default
    argv, _ = build(mode=MODE_HINT_VIDEO, hint_path=str(HINT), carry_hint_quality=0.5)
    assert argv[argv.index("--carry_hint_quality") + 1] == "0.5"


def test_hint_mode_requires_hint():
    _, errors = build(mode=MODE_FIRST_FRAME)
    assert errors
    _, errors = build(mode=MODE_HINT_VIDEO, hint_path=str(TMP / "missing.mp4"))
    assert errors


def test_no_hint_ignores_hint_fields():
    argv, _ = build(hint_path=str(HINT), hint_quality=0.5, no_carry_hint=True)
    assert flags(argv) == ["--input", "--output_dir"]


def test_frame_dir_flags_only_for_folders():
    argv, _ = build(frame_dir_fps=25, frame_dir_linear=True)
    assert "--frame_dir_fps" not in argv and "--frame_dir_linear" not in argv
    argv, errors = build(input_path=str(FRAMES), frame_dir_fps=25, frame_dir_linear=True)
    assert errors == []
    assert argv[argv.index("--frame_dir_fps") + 1] == "25"
    assert "--frame_dir_linear" in argv


def test_outputs_and_despill():
    argv, _ = build(outputs=["fg", "alpha"])
    assert "--outputs" not in argv
    argv, _ = build(outputs=["cutout", "alpha", "checker"], despill_strength=0.75)
    i = argv.index("--outputs")
    assert argv[i + 1:i + 4] == ["alpha", "checker", "cutout"]
    assert argv[argv.index("--despill_strength") + 1] == "0.75"
    argv, _ = build(outputs=["alpha"], despill_strength=0.75)
    assert "--despill_strength" not in argv
    _, errors = build(outputs=[])
    assert errors


def test_cutout_linear_and_exr():
    argv, _ = build(outputs=["cutout"], cutout_linear=True)
    assert "--cutout_linear" in argv
    argv, _ = build(outputs=["alpha"], cutout_linear=True)
    assert "--cutout_linear" not in argv
    argv, _ = build(outputs=["cutout"], cutout_linear=True, output_format="exr",
                    exr_dtype="float")
    assert "--cutout_linear" not in argv
    assert argv[-4:] == ["--output_format", "exr", "--exr_dtype", "float"]
    argv, _ = build(output_format="exr")
    assert argv[-2:] == ["--output_format", "exr"]


def test_quick_test():
    argv, _ = build(num_frames=20)
    assert "--num_frames" not in argv
    argv, _ = build(quick_test=True, num_frames=20)
    assert argv[-2:] == ["--num_frames", "20"]
    argv, _ = build(quick_test=True, num_frames=20, start_mode="middle")
    assert argv[-4:] == ["--num_frames", "20", "--start_mode", "middle"]
    _, errors = build(quick_test=True, num_frames=0)
    assert errors


def test_missing_input():
    argv, errors = build(input_path="")
    assert errors and "<input>" in argv
    _, errors = build(input_path=str(TMP / "nope.mp4"))
    assert errors


def test_paths_and_display():
    assert clean_path('  "~/a b/c.mp4" ') == os.path.expanduser("~/a b/c.mp4")
    assert clean_path("") == ""
    assert default_output_dir(str(CLIP), uploaded=False) == str(TMP / "out_shot")
    assert default_output_dir(str(FRAMES), uploaded=False) == str(TMP / "out_frames_dir")
    assert "/outputs/shot_" in default_output_dir(str(CLIP), uploaded=True)
    assert display_command(["python", "infer.py", "--input", "/a b.mp4"]) == \
        "python infer.py --input '/a b.mp4'"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")
