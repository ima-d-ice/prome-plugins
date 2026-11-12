# Video Benchmark Suite

Benchmarking suite for the PyAV/ffmpeg video parser in Prome RAG Plugins.

## Covered Parsers

- `ffmpeg_video`: structured container/stream metadata + selective frame extraction
  via PyAV (wheels bundle ffmpeg; no system binary required).

## Fixtures

Small synthetic fixtures generated with ffmpeg `lavfi` sources (deterministic
content, small file size):

- `testsrc.mp4`: 3s H.264/AAC, 320x240 @ 10fps — `testsrc` pattern + 440Hz sine audio
- `scene_cut.mp4`: 3s H.264 (no audio), hard cut red→blue at 1.5s (scene-detection fixture)
- `audio_only.mp4`: MP4 container without a video stream (negative fixture)
- `corrupt.mp4` / `empty.mp4`: negative fixtures (tests only, not benchmarked)

Regenerate with:

```bash
python benchmarks/parsers/video/generate_fixtures.py
```

## Metrics

Each fixture is parsed twice where relevant: once with `frame_strategy="interval"`
(1s interval, max 10 frames) and — for `scene_cut.mp4` — once with
`frame_strategy="scene"` (threshold 0.08) to verify cut detection.

- `metadata_accuracy`: fraction of expected container/stream fields matched
  (codec, resolution, fps ±0.5, duration ±0.25s, audio stream count).
- `frames_extracted`: number of FRAME blocks returned.
- `frames_ok`: 1.0 if the interval strategy produced exactly 3 frames, or the
  scene strategy found a frame within 150ms of the 1.5s cut.
- `quality_score`: 0.7·metadata_accuracy + 0.3·frames_ok.

## Running the Benchmark

```bash
python benchmarks/parsers/video/benchmark.py
```

Reports go to `benchmarks/parsers/video/reports/` (`benchmark_report.json`,
`benchmark_results.csv`, `benchmark_summary.csv`); previous reports are archived
to `reports/archive/` before being overwritten.

If PyAV is not installed, the parser is reported as `skipped`:

```bash
pip install -e '.[video]'
```

## Known Limitations

- Frame extraction decodes (not all frames are *emitted*, but decoding is
  sequential); fixtures are tiny by design.
- This parser reports container/stream facts and selected frames only — it does
  **not** understand scenes, objects or actions (no vision model is involved).
- Audio is not transcribed here; use `FFmpegVideoParser.extract_audio_to_wav()`
  to bridge into `parsers.audio.faster_whisper`.
