from benchmarks.parsers.email.benchmark import run_benchmark


def test_email_benchmark_runs(tmp_path):
    report = run_benchmark(
        parsers=["eml_stdlib", "mbox_stdlib"],
        output_dir=tmp_path,
        archive=False,
    )
    assert "summaries" in report
    assert len(report["summaries"]) == 2
    assert (tmp_path / "benchmark_report.json").exists()
