"""Generate deterministic synthetic audio fixtures for the audio benchmark.

Uses macOS `say` for speech synthesis and ffmpeg for format conversion and
silence generation. Re-running re-creates the same fixtures (speech sample
count is voice/text dependent but stable on a given macOS version).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

EN_TEXT = "The quick brown fox jumps over the lazy dog. Prome rag plugins parser benchmark."
FR_TEXT = "Le renard brun rapide saute par dessus le chien paresseux."
LONG_TEXT = (
    "This is a longer fixture for the audio transcription benchmark. "
    "It contains several sentences so that segment boundaries and timestamps can be checked. "
    "The fox jumped over the fence and landed safely on the other side of the field. "
    "Parsers must preserve temporal relationships between segments."
)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def say(text: str, voice: str, out_aiff: Path) -> None:
    run(["say", "-v", voice, "-o", str(out_aiff), text])


def to_wav(src: Path, dst: Path) -> None:
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), str(dst)])


def to_mp3(src: Path, dst: Path) -> None:
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-codec:a", "libmp3lame", str(dst)])


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    tmp = FIXTURES / "_tmp"
    tmp.mkdir(exist_ok=True)

    say(EN_TEXT, "Samantha", tmp / "en.aiff")
    to_wav(tmp / "en.aiff", FIXTURES / "speech_en.wav")
    to_mp3(tmp / "en.aiff", FIXTURES / "speech_en.mp3")

    say(FR_TEXT, "Thomas", tmp / "fr.aiff")
    to_wav(tmp / "fr.aiff", FIXTURES / "speech_fr.wav")

    say(LONG_TEXT, "Samantha", tmp / "long.aiff")
    to_wav(tmp / "long.aiff", FIXTURES / "long_speech.wav")

    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "2", str(FIXTURES / "silence.wav")])

    (FIXTURES / "corrupt.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\xff" * 64)
    (FIXTURES / "empty.wav").write_bytes(b"")

    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    print(f"Generated audio fixtures in {FIXTURES}")


if __name__ == "__main__":
    sys.exit(main())
