from pathlib import Path
from benchmarks.parsers.office.benchmark import run_benchmark


def test_office_benchmark_runs(tmp_path):
    report = run_benchmark(
        parsers=["python_docx", "python_pptx"],
        repeats=1,
        output_dir=tmp_path,
        archive=False,
    )
    assert "summaries" in report
    assert len(report["summaries"]) == 2
    assert (tmp_path / "benchmark_report.json").exists()
    assert (tmp_path / "benchmark_results.csv").exists()
    assert (tmp_path / "benchmark_summary.csv").exists()
