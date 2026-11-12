from pathlib import Path
import pytest
from parsers.contracts import ParserDependencyError
from parsers.images.paddleocr import PaddleOCRParser

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "parsers" / "images" / "fixtures"


def test_paddleocr_raises_dependency_error_when_missing():
    parser = PaddleOCRParser()
    # When paddleocr is not installed, it must raise ParserDependencyError
    with pytest.raises(ParserDependencyError) as exc_info:
        parser.parse(FIXTURES_DIR / "clean_text.png")
    assert "PaddleOCR" in str(exc_info.value)
    assert "pip install" in str(exc_info.value)
