# PyAV/ffmpeg Video Parser

Structured video extraction behind the common `ParsedDocument` contract, using
[PyAV](https://github.com/PyAV-Org/PyAV) (Python bindings for ffmpeg libraries).

## Purpose

Turn video files into a structured document: container/stream metadata plus a
small set of selected frames as `FRAME` blocks with millisecond-accurate
timestamps and frame references — never "every frame by default".

## Supported formats

`.mp4`, `.m4v`, `.mkv`, `.webm`, `.mov`, `.avi`, `.mpg`, `.mpeg`, `.ts`
(anything the bundled ffmpeg can demux). Other extensions raise
`CorruptDocumentError`.

## Installation

```bash
pip install -e '.[video]'
# or: pip install av numpy
```

PyAV wheels bundle ffmpeg libraries; **no system ffmpeg binary is required**.
System ffmpeg is only needed to *regenerate the benchmark fixtures*
(`benchmarks/parsers/video/generate_fixtures.py`).

## Usage

```python
from parsers.video.ffmpeg import FFmpegVideoParser

parser = FFmpegVideoParser(
    frame_strategy="interval",  # "interval" | "scene"
    interval_seconds=1.0,       # interval strategy: one frame per N seconds
    max_frames=10,              # hard cap on emitted frames
    scene_threshold=0.08,       # scene strategy: luma mean-abs-diff threshold
    frame_dir="/tmp/frames",    # optional: also save selected frames as JPEGs
)
doc = parser.parse("clip.mp4")
print(doc.metadata["duration_seconds"], doc.metadata["frame_rate"], doc.metadata["video_codec"])
for block in doc.blocks:  # BlockType.FRAME
    print(block.metadata["frame_number"], block.metadata["timestamp"], block.content)

# Optional loose coupling to the audio parser:
wav_path = parser.extract_audio_to_wav("clip.mp4", "/tmp/clip.wav")  # 16 kHz mono PCM
# from parsers.audio.faster_whisper import FasterWhisperParser
# audio_doc = FasterWhisperParser().parse(wav_path)
```

## Extracted fields

Document metadata: `format`, `format_long_name`, `duration_seconds`, `bit_rate`,
`video_codec`, `width`, `height`, `frame_rate`, `frame_count_estimate`,
`audio_streams[]` (`codec`, `sample_rate`, `channels`, `duration`, `bit_rate`),
`audio_stream_count`, frame strategy settings, `frames_extracted`.

Per frame (`ParsedBlock` with `block_type=frame`): `frame_number` (0-based decode
order), `timestamp` (seconds, from PTS), `width`, `height`, `strategy`,
`frame_ref` / `frame_path` (filename / absolute path when `frame_dir` is set).

## Frame extraction strategies

- **interval** (default): emit the first frame at or after each multiple of
  `interval_seconds`. Predictable coverage, O(duration/interval) frames.
- **scene**: emit the first frame, then any frame whose 8-bit luma mean-absolute
  difference from the previous frame is ≥ `scene_threshold`. Detects hard cuts
  (e.g. red→blue at 1.5s in the `scene_cut.mp4` fixture).
- Both are capped by `max_frames`. Timestamps are preserved in temporal order.

Decoding is sequential: frames are decoded (but not emitted) as needed to reach
selection points. No video seeking is used; very long files will take time.

## Unsupported capabilities

- **No visual understanding.** Frame extraction does not perform object
  detection, captioning, scene semantics or OCR. Use the image parsers for OCR
  on selected frames.
- **No automatic transcription.** Audio is never transcribed implicitly; call
  `extract_audio_to_wav()` and feed the result to an audio parser yourself.
- **No diarization, no LLM summarization, no keyframe semantics.**
- Rotation metadata (display matrix) is not surfaced; frame dimensions are the
  coded dimensions.

## Known limitations

- Scene detection is a crude global luma heuristic — robust for hard cuts, not
  for gradual or flash-frame changes.
- `frame_count_estimate` comes from the container and may be missing/inexact.
- Corrupt containers raise `CorruptDocumentError`; containers with no video
  stream (e.g. audio-only MP4) also raise `CorruptDocumentError`. Zero-byte
  files raise `EmptyDocumentError`.
- Fixtures are tiny synthetic clips; reported benchmark numbers are regression
  signals, not production baselines.
