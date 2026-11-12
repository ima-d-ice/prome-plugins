# faster-whisper Audio Parser

Audio transcription behind the common `ParsedDocument` contract, using
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 Whisper).

## Purpose

Transcribe spoken audio into structured `SEGMENT` blocks with word/temporal
provenance, instead of one flat text blob.

## Supported formats

Any container/codec supported by PyAV's ffmpeg build: `.mp3`, `.wav`, `.m4a`,
`.aac`, `.flac`, `.ogg`, `.opus`, `.wma`, `.aiff`/`.aif`, `.caf`, `.webm`.
Other extensions raise `CorruptDocumentError`.

## Installation

```bash
pip install -e '.[audio]'
# or: pip install faster-whisper av numpy
```

This installs the Python packages only. **Model weights are downloaded on first
`parse()`**, not at install/import time. The default model is `tiny`
(~75MB, Hugging Face cache, works on CPU with int8). Model sizes and rough
footprints: `tiny` ~75MB, `base` ~145MB, `small` ~466MB, `medium` ~1.5GB,
`large-v3` ~3GB. Larger models = better accuracy, slower inference. GPU/MPS
(`device="cuda"` / `compute_type="int8_float16"`) are supported by
faster-whisper but are not required; `cpu` + `int8` works everywhere.

Set `HF_HUB_CACHE`/`HF_TOKEN` as usual if you need to pre-seed models or
increase rate limits.

## External dependencies

- No system binaries are required when using the PyAV wheels (they bundle ffmpeg).
- Network access is required once per model size (first `parse()`), unless the
  model is already in the Hugging Face cache.

## Usage

```python
from parsers.audio.faster_whisper import FasterWhisperParser

parser = FasterWhisperParser(model_size="tiny", language="en")  # language=None → auto-detect
doc = parser.parse("speech.wav")

print(doc.metadata["language"], doc.metadata["audio_duration"])
for block in doc.blocks:  # BlockType.SEGMENT
    print(block.metadata["start"], block.metadata["end"], block.content)
```

## Extracted fields

Per document (`metadata`): `format`, `format_long_name`, `duration_seconds`,
`bit_rate`, `stream_codec`, `sample_rate`, `channels`, `audio_stream_count`,
`audio_duration`, `language`, `language_probability`, `model_size`, `device`,
`compute_type`, `vad_filter`, `segment_count`, `diarization` (always `false`).

Per segment (`ParsedBlock` with `block_type=segment`): `segment_id`, `start`,
`end`, `duration`, `avg_logprob`, `no_speech_prob`, `compression_ratio`,
`temperature`, `language`.

## Notes on correctness

- Empty audio (0 bytes) raises `EmptyDocumentError`.
- Audio with no detectable speech returns a document with zero segment blocks and
  a warning (`No speech detected…`) — it does **not** fabricate text.
- Undecodable/unsupported input raises `CorruptDocumentError`.
- Audio is decoded to 16 kHz mono float32 internally before transcription
  (faster-whisper's requirement). PyAV >= 19 compatibility is handled inside this
  parser; no monkey-patching of PyAV or faster-whisper is applied.

## Unsupported capabilities

- **Speaker diarization / speaker labels** — not supported by faster-whisper and
  never fabricated (`diarization: false`). Possible future extension.
- Translation task (`task="translate"`) is not exposed; use faster-whisper
  directly for that.
- Word-level timestamps are available via `word_timestamps=True` on the parser,
  but words are not emitted as blocks; segments only.

## Known limitations

- Transcription quality is bounded by the chosen model size.
- `silence.wav`-style no-speech input yields `language="nn"` with low probability
  from tiny models — treat `language_probability` as a signal, not ground truth.
- Test fixtures use synthesized speech as a deterministic regression signal, not a
  real-world accuracy measurement.
