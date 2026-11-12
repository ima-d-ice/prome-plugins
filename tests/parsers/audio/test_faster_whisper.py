from pathlib import Path

import pytest

from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "parsers" / "audio" / "fixtures"

faster_whisper = pytest.importorskip("faster_whisper", reason="faster-whisper not installed")


@pytest.fixture(scope="module")
def parser():
    from parsers.audio.faster_whisper import FasterWhisperParser

    try:
        p = FasterWhisperParser(model_size="tiny")
        p._get_model()
    except Exception as exc:
        pytest.skip(f"faster-whisper tiny model unavailable: {exc}")
    return p


def test_parse_english_speech(parser):
    doc = parser.parse(FIXTURES_DIR / "speech_en.wav")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "faster_whisper"
    assert doc.page_count == 0
    assert doc.pages == []
    assert "quick brown fox" in doc.text.lower()
    assert doc.metadata["language"] == "en"
    assert doc.metadata["language_probability"] > 0.5
    assert doc.metadata["audio_duration"] == pytest.approx(5.0, abs=0.5)
    assert doc.metadata["diarization"] is False
    assert doc.metadata["model_size"] == "tiny"
    assert len(doc.blocks) >= 1
    for block in doc.blocks:
        assert block.block_type == BlockType.SEGMENT
        start, end = block.metadata["start"], block.metadata["end"]
        assert 0 <= start < end
        assert block.metadata["language"] == "en"


def test_parse_mp3(parser):
    doc = parser.parse(FIXTURES_DIR / "speech_en.mp3")
    assert "quick brown fox" in doc.text.lower()
    assert doc.metadata["format"] in ("mp3", "mp3_mp3", "mp3file") or "mp3" in (doc.metadata["format"] or "")


def test_parse_french_speech_language_detection(parser):
    doc = parser.parse(FIXTURES_DIR / "speech_fr.wav")
    assert doc.metadata["language"] == "fr"


def test_parse_long_speech_segments_are_time_ordered(parser):
    doc = parser.parse(FIXTURES_DIR / "long_speech.wav")
    assert doc.metadata["audio_duration"] > 10
    assert len(doc.blocks) >= 2
    starts = [b.metadata["start"] for b in doc.blocks]
    assert starts == sorted(starts)
    for block in doc.blocks:
        assert block.metadata["start"] < block.metadata["end"]


def test_silence_produces_warning_not_text(parser):
    doc = parser.parse(FIXTURES_DIR / "silence.wav")
    assert doc.blocks == []
    assert any("No speech detected" in w for w in doc.warnings)
    assert doc.metadata["segment_count"] == 0


def test_corrupt_audio_raises(parser):
    with pytest.raises(CorruptDocumentError):
        parser.parse(FIXTURES_DIR / "corrupt.wav")


def test_empty_audio_raises(parser):
    with pytest.raises(EmptyDocumentError):
        parser.parse(FIXTURES_DIR / "empty.wav")


def test_missing_file_raises(parser):
    with pytest.raises(FileNotFoundError):
        parser.parse(FIXTURES_DIR / "does_not_exist.wav")


def test_unsupported_extension_raises(parser, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not audio")
    with pytest.raises(CorruptDocumentError):
        parser.parse(bad)


def test_segment_timestamps_do_not_overlap(parser):
    doc = parser.parse(FIXTURES_DIR / "long_speech.wav")
    for prev, curr in zip(doc.blocks, doc.blocks[1:]):
        assert curr.metadata["start"] >= prev.metadata["start"]
