from benchmarks.parsers.audio.benchmark import run_benchmark, word_error_rate


def test_word_error_rate():
    assert word_error_rate("the quick brown fox", "the quick brown fox") == 0.0
    assert word_error_rate("the quick brown fox", "the quick red fox") == 0.25
    assert word_error_rate("", "anything") == 1.0


def test_audio_benchmark_runs(tmp_path):
    report = run_benchmark(
        parsers=["faster_whisper"],
        output_dir=tmp_path,
        archive=False,
    )
    assert "summaries" in report
    assert len(report["summaries"]) == 1
    summary = report["summaries"][0]
    assert summary["status"] in ("available", "skipped")
    if summary["status"] == "available":
        assert summary["success_rate"] == 1.0
    assert (tmp_path / "benchmark_report.json").exists()
