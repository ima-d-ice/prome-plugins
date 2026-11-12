import pytest
from parsers.contracts import Parser
from parsers.office.docx.python_docx import PythonDocxParser
from parsers.office.docx.docling import DoclingDocxParser
from parsers.office.pptx.python_pptx import PythonPptxParser
from parsers.office.pptx.docling import DoclingPptxParser


@pytest.mark.parametrize(
    "parser_cls",
    [PythonDocxParser, DoclingDocxParser, PythonPptxParser, DoclingPptxParser],
)
def test_office_parsers_implement_protocol(parser_cls):
    parser = parser_cls()
    assert isinstance(parser, Parser)
    assert hasattr(parser, "name")
    assert callable(parser.parse)
