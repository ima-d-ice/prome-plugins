"""PyAV/ffmpeg video parser: structured metadata + frame extraction.

Extracts container/stream metadata and selectively decodes frames (never all
frames by default) into FRAME blocks with timestamps and frame references.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedBlock,
    ParsedDocument,
    ParserDependencyError,
)
from parsers.contracts.validation import validate_source

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".mp4", ".m4v", ".mkv", ".webm", ".mov", ".avi", ".mpg", ".mpeg", ".ts",
}

FRAME_STRATEGIES = ("interval", "scene")


def _load_av():
    try:
        import av
        import numpy as np

        return av, np
    except ImportError as exc:
        raise ParserDependencyError(
            "PyAV is not installed. Install with:\n"
            "  pip install 'prome-rag-plugins[video]'\n"
            "or:\n"
            "  pip install av numpy\n"
            "PyAV wheels bundle ffmpeg; a system ffmpeg binary is not required."
        ) from exc


def _stream_duration(stream) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    return None


def _rational_to_float(rate) -> float | None:
    try:
        if rate is None:
            return None
        num, den = rate.numerator, rate.denominator
        if num == 0 or den == 0:
            return None
        return round(num / den, 6)
    except Exception:
        return None


def _frame_luma_mean_abs_diff(prev_luma, curr_luma) -> float:
    import numpy as np

    diff = np.abs(curr_luma.astype(np.int16) - prev_luma.astype(np.int16))
    return float(diff.mean()) / 255.0


def _luma_plane(frame):
    plane = frame.to_ndarray(format="gray")
    return plane


class FFmpegVideoParser:
    """Parse video files into metadata + selected FRAME blocks via PyAV."""

    name = "ffmpeg_video"

    def __init__(
        self,
        frame_strategy: str = "interval",
        interval_seconds: float = 1.0,
        max_frames: int = 10,
        scene_threshold: float = 0.08,
        frame_dir: str | Path | None = None,
    ) -> None:
        if frame_strategy not in FRAME_STRATEGIES:
            raise ValueError(f"frame_strategy must be one of {FRAME_STRATEGIES}, got {frame_strategy!r}")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        if not 0.0 < scene_threshold < 1.0:
            raise ValueError("scene_threshold must be in (0, 1)")
        self.frame_strategy = frame_strategy
        self.interval_seconds = float(interval_seconds)
        self.max_frames = int(max_frames)
        self.scene_threshold = float(scene_threshold)
        self.frame_dir = Path(frame_dir) if frame_dir else None
        if self.frame_dir:
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, source: str | Path) -> ParsedDocument:
        av, _ = _load_av()
        path = validate_source(source)

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise CorruptDocumentError(
                f"Unsupported video format '{ext}' for file '{path}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        try:
            container = av.open(str(path))
        except Exception as exc:
            raise CorruptDocumentError(f"Cannot open video file '{path}': {exc}") from exc

        try:
            video_streams = [s for s in container.streams if s.type == "video"]
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not video_streams:
                raise CorruptDocumentError(f"No video stream found in '{path}'.")
            vstream = video_streams[0]
            vcc = vstream.codec_context

            duration = None
            if container.duration is not None:
                duration = float(container.duration) / 1_000_000
            elif (d := _stream_duration(vstream)) is not None:
                duration = d

            fps = _rational_to_float(getattr(vstream, "average_rate", None)) or _rational_to_float(
                getattr(vstream, "base_rate", None)
            )

            audio_info = []
            for astream in audio_streams:
                acc = astream.codec_context
                audio_info.append(
                    {
                        "codec": acc.name if acc else None,
                        "sample_rate": getattr(acc, "sample_rate", None),
                        "channels": getattr(acc, "channels", None),
                        "duration": _stream_duration(astream),
                        "bit_rate": getattr(astream, "bit_rate", None),
                    }
                )

            metadata: dict[str, Any] = {
                "format": container.format.name if container.format else None,
                "format_long_name": getattr(container.format, "long_name", None),
                "duration_seconds": round(duration, 4) if duration is not None else None,
                "bit_rate": getattr(container, "bit_rate", None),
                "video_codec": vcc.name if vcc else None,
                "width": getattr(vcc, "width", None),
                "height": getattr(vcc, "height", None),
                "frame_rate": fps,
                "frame_count_estimate": getattr(vstream, "frames", None) or None,
                "audio_streams": audio_info,
                "audio_stream_count": len(audio_streams),
                "frame_strategy": self.frame_strategy,
                "interval_seconds": self.interval_seconds if self.frame_strategy == "interval" else None,
                "scene_threshold": self.scene_threshold if self.frame_strategy == "scene" else None,
                "max_frames": self.max_frames,
            }

            blocks = self._extract_frames(av, container, vstream, path)
        finally:
            container.close()

        if not blocks:
            raise EmptyDocumentError(f"Video file '{path}' contains no decodable frames.")

        metadata["frames_extracted"] = len(blocks)

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version=getattr(av, "__version__", "unknown"),
            metadata=metadata,
            pages=[],
            blocks=blocks,
            warnings=[],
        )

    def _extract_frames(self, av, container, vstream, path: Path) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        next_target = 0.0
        prev_luma = None
        frame_index = 0

        try:
            for frame in container.decode(vstream):
                ts = float(frame.pts * frame.time_base) if frame.pts is not None and frame.time_base else None
                if ts is None:
                    ts = frame_index / (self.interval_seconds or 1.0)

                selected = False
                if self.frame_strategy == "interval":
                    if ts >= next_target - 1e-6:
                        selected = True
                        next_target = ts + self.interval_seconds
                else:  # scene
                    curr_luma = _luma_plane(frame)
                    if prev_luma is None:
                        selected = True
                    else:
                        score = _frame_luma_mean_abs_diff(prev_luma, curr_luma)
                        if score >= self.scene_threshold:
                            selected = True
                    prev_luma = curr_luma

                if selected:
                    blocks.append(self._make_frame_block(av, frame, frame_index, ts, path))
                    if len(blocks) >= self.max_frames:
                        break

                frame_index += 1
        except Exception as exc:
            raise CorruptDocumentError(f"Failed to decode frames from '{path}': {exc}") from exc

        return blocks

    def _make_frame_block(self, av, frame, frame_index: int, ts: float, source_path: Path) -> ParsedBlock:
        saved_path = None
        if self.frame_dir is not None:
            saved_path = self.frame_dir / f"{source_path.stem}_frame_{frame_index}_{ts:.3f}s.jpg"
            try:
                frame.to_image().save(str(saved_path), quality=85)
            except Exception as exc:
                raise CorruptDocumentError(f"Could not save frame to '{saved_path}': {exc}") from exc

        metadata = {
            "frame_number": frame_index,
            "timestamp": round(ts, 4),
            "width": frame.width,
            "height": frame.height,
            "strategy": self.frame_strategy,
            "frame_ref": saved_path.name if saved_path else None,
            "frame_path": str(saved_path) if saved_path else None,
        }

        return ParsedBlock(
            block_type=BlockType.FRAME,
            content=f"frame {frame_index} at {ts:.3f}s",
            page_number=None,
            bbox=None,
            metadata=metadata,
        )

    def extract_audio_to_wav(
        self,
        source: str | Path,
        target: str | Path,
        sample_rate: int = 16000,
    ) -> Path:
        """Extract the first audio stream to a 16-bit PCM WAV at ``sample_rate``.

        Optional loose-coupling integration path: the resulting file can be fed
        directly to ``parsers.audio.faster_whisper.FasterWhisperParser``. This
        method does not transcribe anything by itself.
        """
        av, _ = _load_av()
        path = validate_source(source)
        target_path = Path(target).expanduser()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with av.open(str(path)) as in_container:
                audio_streams = [s for s in in_container.streams if s.type == "audio"]
                if not audio_streams:
                    raise CorruptDocumentError(f"No audio stream found in '{path}'.")
                astream = audio_streams[0]

                with av.open(str(target_path), mode="w") as out_container:
                    out_stream = out_container.add_stream("pcm_s16le", rate=sample_rate)
                    out_stream.layout = "mono"
                    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
                    for frame in in_container.decode(astream):
                        for resampled in resampler.resample(frame):
                            resampled.pts = None
                            for packet in out_stream.encode(resampled):
                                out_container.mux(packet)
                    for resampled in resampler.resample(None):
                        resampled.pts = None
                        for packet in out_stream.encode(resampled):
                            out_container.mux(packet)
                    for packet in out_stream.encode():
                        out_container.mux(packet)
        except CorruptDocumentError:
            raise
        except Exception as exc:
            raise CorruptDocumentError(f"Failed to extract audio from '{path}': {exc}") from exc

        return target_path
