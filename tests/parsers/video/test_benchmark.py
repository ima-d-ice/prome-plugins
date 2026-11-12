from benchmarks.parsers.video.benchmark import run_benchmark


def test_video_benchmark_runs(tmp_path):
    report = run_benchmark(
        parsers=["ffmpeg_video"],
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
