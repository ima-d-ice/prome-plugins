import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = ROOT / "benchmarks" / "parsers" / "spreadsheets" / "fixtures"
EXPECTATIONS_PATH = ROOT / "benchmarks" / "parsers" / "spreadsheets" / "expectations.json"


@pytest.fixture(scope="session")
def spreadsheet_fixtures_dir() -> Path:
    if not FIXTURES_DIR.exists():
        pytest.skip("Spreadsheet fixtures missing")
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def spreadsheet_expectations() -> dict:
    if not EXPECTATIONS_PATH.exists():
        pytest.skip("Spreadsheet expectations missing")
    return json.loads(EXPECTATIONS_PATH.read_text())
