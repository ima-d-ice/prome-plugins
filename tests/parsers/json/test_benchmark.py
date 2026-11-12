from benchmarks.parsers.json.benchmark import run_benchmark


def test_json_benchmark_runs(tmp_path):
    report = run_benchmark(
        parsers=["json_stdlib"],
        output_dir=tmp_path,
        archive=False,
    )
    assert "summaries" in report
    assert len(report["summaries"]) == 1
    assert (tmp_path / "benchmark_report.json").exists()
