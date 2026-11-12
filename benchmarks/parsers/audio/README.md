# Audio Benchmark Suite

Benchmarking suite for the faster-whisper audio transcription parser in Prome RAG Plugins.

## Covered Parsers

- `faster_whisper`: faster-whisper (CTranslate2) Whisper transcription. Default model is
  `tiny` (~75MB), downloaded to the Hugging Face cache on first use — never at import
  time. Pass `--model-size base|small|medium|large-v3` to evaluate larger models
  (larger downloads, higher accuracy). A GPU or Apple Silicon MPS device can be
  selected via the parser constructor (`device="cuda"`/`"mps"`, `compute_type="int8_float16"` etc.).

## Fixtures

Deterministic synthetic fixtures generated with macOS `say` (speech synthesis) and
ffmpeg (format conversion, silence):

- `speech_en.wav` / `speech_en.mp3`: English speech ("The quick brown fox…")
- `speech_fr.wav`: French speech (language detection fixture)
- `long_speech.wav`: ~16s multi-sentence audio (segment/timestamp fixture)
- `silence.wav`: 2s of digital silence (no-speech handling)
- `corrupt.wav` / `empty.wav`: negative fixtures (tests only, not benchmarked)

Regenerate with:

```bash
python benchmarks/parsers/audio/generate_fixtures.py
```

## Metrics

- `text_coverage`: fraction of `expected_text` substrings found in the transcript.
- `segment_coverage`: 1.0 when at least `min_blocks` segments are returned.
- `language_accuracy`: detected language vs `expected_language` (N/A = full credit when no expectation).
- `wer`: word error rate vs `expected_transcript` (only where one is defined; otherwise reported blank/N/A).
- `quality_score`: 0.6·text_coverage + 0.2·segment_coverage + 0.2·language_accuracy.

`quality_score` is deliberately coarse; `wer` and per-fixture `benchmark_results.csv`
carry the detailed signal. Silence is scored on producing no segments without crashing
(coverage defaults to full credit); it does not assert hallucinated text content.

## Running the Benchmark

```bash
python benchmarks/parsers/audio/benchmark.py                 # default model (tiny)
python benchmarks/parsers/audio/benchmark.py --model-size base
```

Reports go to `benchmarks/parsers/audio/reports/` (`benchmark_report.json`,
`benchmark_results.csv`, `benchmark_summary.csv`); previous reports are archived
to `reports/archive/` before being overwritten.

If faster-whisper or its audio backend (PyAV) is not installed, the parser is
reported as `skipped`, not failed:

```bash
pip install -e '.[audio]'
```

## Known Limitations

- faster-whisper does **not** perform speaker diarization; speaker labels are never
  emitted (the `diarization` metadata flag is always `false`). Diarization is a
  possible future extension.
- Model weights are not bundled; first parse downloads them. `tiny` is the default
  to keep this cheap; accuracy on accented/noisy speech improves with larger models.
- Transcription quality on synthetic (synthesized) voices is used here purely as a
  regression signal, not as a real-world accuracy claim.
