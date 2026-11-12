import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = ROOT / "benchmarks" / "parsers" / "office" / "fixtures"
EXPECTATIONS_PATH = ROOT / "benchmarks" / "parsers" / "office" / "expectations.json"


@pytest.fixture(scope="session")
def office_fixtures_dir() -> Path:
    if not FIXTURES_DIR.exists():
        pytest.skip("Office fixtures missing")
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def office_expectations() -> dict:
    if not EXPECTATIONS_PATH.exists():
        pytest.skip("Office expectations.json missing")
    return json.loads(EXPECTATIONS_PATH.read_text())
