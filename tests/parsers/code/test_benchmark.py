from benchmarks.parsers.code.benchmark import run_benchmark


def test_code_benchmark_runs(tmp_path):
    report = run_benchmark(
        parsers=["python_ast", "tree_sitter"],
        output_dir=tmp_path,
        archive=False,
    )
    assert "summaries" in report
    assert len(report["summaries"]) == 2
    assert (tmp_path / "benchmark_report.json").exists()
