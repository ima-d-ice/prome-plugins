"""Generate deterministic synthetic video fixtures for the video benchmark."""

from __future__ import annotations

import subprocess
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

BITEXACT = ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact", "-metadata", "creation_time=2000-01-01T00:00:00Z"]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        *BITEXACT,
        str(FIXTURES / "testsrc.mp4"),
    ])

    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=red:size=320x240:rate=10:duration=1.5",
        "-f", "lavfi", "-i", "color=blue:size=320x240:rate=10:duration=1.5",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
        *BITEXACT,
        str(FIXTURES / "scene_cut.mp4"),
    ])

    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:a", "aac", "-vn",
        *BITEXACT,
        str(FIXTURES / "audio_only.mp4"),
    ])

    (FIXTURES / "corrupt.mp4").write_bytes(b"\x00\x00\x00\x18ftypNOTAVIDEO" + b"\xff" * 128)
    (FIXTURES / "empty.mp4").write_bytes(b"")
    print(f"Generated video fixtures in {FIXTURES}")


if __name__ == "__main__":
    main()
