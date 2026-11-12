"""faster-whisper audio transcription parser.

Transcribes audio files into SEGMENT blocks with timestamps using
faster-whisper (CTranslate2). Optional model download happens on first
``parse()`` call, never at import time.
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
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus",
    ".wma", ".aiff", ".aif", ".caf", ".webm",
}


def _load_faster_whisper():
    try:
        import faster_whisper
        import av
        import numpy as np

        return faster_whisper, av, np
    except ImportError as exc:
        raise ParserDependencyError(
            "faster-whisper (and its PyAV audio backend) is not installed. "
            "Install with:\n"
            "  pip install 'prome-rag-plugins[audio]'\n"
            "or:\n"
            "  pip install faster-whisper av numpy"
        ) from exc


def _decode_audio_16k(path: Path, av, np):
    """Decode an audio file to mono float32 PCM at 16 kHz.

    faster-whisper's own decode helper passes ``metadata_errors`` to
    ``av.open``, which PyAV >= 19 removed; decoding here via the documented
    numpy-array input keeps this parser working with any PyAV version.
    """
    try:
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise CorruptDocumentError(f"No audio stream found in '{path}'.")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            chunks = []
            for frame in container.decode(stream):
                chunks.extend(resampler.resample(frame))
            chunks.extend(resampler.resample(None))
    except CorruptDocumentError:
        raise
    except Exception as exc:
        raise CorruptDocumentError(f"Cannot decode audio file '{path}': {exc}") from exc

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    arr = np.concatenate([f.to_ndarray() for f in chunks], axis=1)[0]
    return arr.astype(np.float32) / 32768.0


def _probe_container(path: Path, av) -> dict[str, Any]:
    """Collect container/stream metadata with PyAV. Raises CorruptDocumentError."""
    try:
        with av.open(str(path)) as container:
            fmt = container.format
            duration = None
            if container.duration is not None:
                duration = container.duration / 1_000_000  # PyAV container.duration is microseconds
            audio_streams = [s for s in container.streams if s.type == "audio"]
            if not audio_streams:
                raise CorruptDocumentError(f"No audio stream found in '{path}'.")
            stream = audio_streams[0]
            return {
                "format": fmt.name if fmt else None,
                "format_long_name": getattr(fmt, "long_name", None),
                "duration_seconds": duration,
                "bit_rate": getattr(container, "bit_rate", None),
                "stream_codec": stream.codec_context.name if stream.codec_context else None,
                "sample_rate": getattr(stream.codec_context, "sample_rate", None),
                "channels": getattr(stream.codec_context, "channels", None),
                "audio_stream_count": len(audio_streams),
            }
    except CorruptDocumentError:
        raise
    except Exception as exc:
        raise CorruptDocumentError(f"Cannot decode audio file '{path}': {exc}") from exc


class FasterWhisperParser:
    """Transcribe audio files with faster-whisper (CTranslate2 Whisper)."""

    name = "faster_whisper"

    def __init__(
        self,
        model_size: str = "tiny",
        language: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        vad_filter: bool = False,
        word_timestamps: bool = False,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self.word_timestamps = word_timestamps
        self._model = None

    def _get_model(self):
        if self._model is None:
            faster_whisper, _, _ = _load_faster_whisper()
            try:
                self._model = faster_whisper.WhisperModel(
                    self.model_size, device=self.device, compute_type=self.compute_type
                )
            except Exception as exc:
                raise ParserDependencyError(
                    f"Could not load faster-whisper model '{self.model_size}': {exc}.\n"
                    f"Model weights are downloaded on first use to the Hugging Face "
                    f"cache (~75MB for 'tiny', larger for other sizes). Check network "
                    f"access or pre-download the model."
                ) from exc
        return self._model

    def parse(self, source: str | Path) -> ParsedDocument:
        faster_whisper, av, np = _load_faster_whisper()
        path = validate_source(source)

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise CorruptDocumentError(
                f"Unsupported audio format '{ext}' for file '{path}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        container_meta = _probe_container(path, av)
        waveform = _decode_audio_16k(path, av, np)
        if waveform.size == 0:
            raise EmptyDocumentError(f"Audio file '{path}' decodes to zero samples.")
        model = self._get_model()

        try:
            segments_iter, info = model.transcribe(
                waveform,
                language=self.language,
                vad_filter=self.vad_filter,
                word_timestamps=self.word_timestamps,
            )
            segments = list(segments_iter)
        except Exception as exc:
            raise CorruptDocumentError(f"Transcription failed for audio file '{path}': {exc}") from exc

        blocks: list[ParsedBlock] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            blocks.append(
                ParsedBlock(
                    block_type=BlockType.SEGMENT,
                    content=text,
                    page_number=None,
                    bbox=None,
                    metadata={
                        "segment_id": seg.id,
                        "start": round(float(seg.start), 3),
                        "end": round(float(seg.end), 3),
                        "duration": round(float(seg.end) - float(seg.start), 3),
                        "avg_logprob": round(float(seg.avg_logprob), 4),
                        "no_speech_prob": round(float(seg.no_speech_prob), 4),
                        "compression_ratio": round(float(seg.compression_ratio), 4),
                        "temperature": float(seg.temperature),
                        "language": info.language,
                    },
                )
            )

        warnings: list[str] = []
        if not blocks:
            warnings.append("No speech detected in audio (transcription produced no segments).")

        duration = float(info.duration) if info.duration else container_meta.get("duration_seconds")
        metadata = {
            **container_meta,
            "audio_duration": round(duration, 3) if duration else None,
            "language": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "all_language_probs": getattr(info, "all_language_probs", None),
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "vad_filter": self.vad_filter,
            "segment_count": len(blocks),
            "diarization": False,
            "note": (
                "Speaker diarization is not supported by faster-whisper; "
                "speaker labels are never fabricated."
            ),
        }

        return ParsedDocument(
            source=str(path),
            parser_name=self.name,
            parser_version=getattr(faster_whisper, "__version__", "unknown"),
            metadata=metadata,
            pages=[],
            blocks=blocks,
            warnings=warnings,
        )
