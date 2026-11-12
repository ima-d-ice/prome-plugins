from benchmarks.parsers.xml.benchmark import run_benchmark


def test_xml_benchmark_runs(tmp_path):
    report = run_benchmark(
        parsers=["elementtree_xml", "lxml_xml"],
        output_dir=tmp_path,
        archive=False,
    )
    assert "summaries" in report
    assert len(report["summaries"]) == 2
    assert (tmp_path / "benchmark_report.json").exists()
