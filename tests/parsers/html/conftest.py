import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = ROOT / "benchmarks" / "parsers" / "html" / "fixtures"
EXPECTATIONS_PATH = ROOT / "benchmarks" / "parsers" / "html" / "expectations.json"

PARSER_NAMES = ["beautifulsoup", "trafilatura"]


def parser_importable(name: str) -> bool:
    module = {"beautifulsoup": "bs4", "trafilatura": "trafilatura"}[name]
    try:
        __import__(module)
    except ImportError:
        return False
    return True


def make_parser(name: str, **kwargs):
    if name == "beautifulsoup":
        from parsers.html.beautifulsoup import BeautifulSoupParser

        return BeautifulSoupParser(**kwargs)
    if name == "trafilatura":
        from parsers.html.trafilatura import TrafilaturaParser

        return TrafilaturaParser(**kwargs)
    raise ValueError(name)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    if not FIXTURES_DIR.exists() or not any(FIXTURES_DIR.glob("*.html")):
        pytest.skip("HTML fixtures are missing")
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def expectations() -> dict:
    if not EXPECTATIONS_PATH.exists():
        pytest.skip("expectations.json is missing")
    return json.loads(EXPECTATIONS_PATH.read_text())


def parser_params():
    return [
        pytest.param(name, marks=pytest.mark.skipif(not parser_importable(name), reason=f"{name} not installed"))
        for name in PARSER_NAMES
    ]


@pytest.fixture(params=parser_params(), scope="module")
def parser_name(request):
    return request.param


@pytest.fixture()
def parser(parser_name):
    return make_parser(parser_name)
