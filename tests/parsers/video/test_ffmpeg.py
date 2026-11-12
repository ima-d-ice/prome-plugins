from pathlib import Path

import pytest

from parsers.contracts import (
    BlockType,
    CorruptDocumentError,
    EmptyDocumentError,
    ParsedDocument,
)

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "parsers" / "video" / "fixtures"

av = pytest.importorskip("av", reason="PyAV not installed")


def test_parse_testsrc_metadata():
    from parsers.video.ffmpeg import FFmpegVideoParser

    parser = FFmpegVideoParser(frame_strategy="interval", interval_seconds=1.0, max_frames=10)
    doc = parser.parse(FIXTURES_DIR / "testsrc.mp4")
    assert isinstance(doc, ParsedDocument)
    assert doc.parser_name == "ffmpeg_video"
    meta = doc.metadata
    assert meta["duration_seconds"] == pytest.approx(3.0, abs=0.2)
    assert meta["video_codec"] == "h264"
    assert meta["width"] == 320
    assert meta["height"] == 240
    assert meta["frame_rate"] == pytest.approx(10.0, abs=0.5)
    assert meta["audio_stream_count"] == 1
    assert meta["audio_streams"][0]["codec"] == "aac"
    assert meta["audio_streams"][0]["sample_rate"] == 44100


def test_interval_frame_strategy():
    from parsers.video.ffmpeg import FFmpegVideoParser

    parser = FFmpegVideoParser(frame_strategy="interval", interval_seconds=1.0, max_frames=10)
    doc = parser.parse(FIXTURES_DIR / "testsrc.mp4")
    timestamps = [b.metadata["timestamp"] for b in doc.blocks]
    assert len(doc.blocks) == 3
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == pytest.approx(0.0, abs=0.05)
    for block in doc.blocks:
        assert block.block_type == BlockType.FRAME
        assert block.content.startswith("frame ")


def test_interval_strategy_respects_max_frames():
    from parsers.video.ffmpeg import FFmpegVideoParser

    parser = FFmpegVideoParser(frame_strategy="interval", interval_seconds=0.25, max_frames=2)
    doc = parser.parse(FIXTURES_DIR / "testsrc.mp4")
    assert len(doc.blocks) == 2


def test_scene_strategy_detects_cut():
    from parsers.video.ffmpeg import FFmpegVideoParser

    parser = FFmpegVideoParser(frame_strategy="scene", scene_threshold=0.08, max_frames=10)
    doc = parser.parse(FIXTURES_DIR / "scene_cut.mp4")
    timestamps = [b.metadata["timestamp"] for b in doc.blocks]
    # hard red->blue cut at 1.5s must be detected
    assert any(abs(t - 1.5) < 0.15 for t in timestamps)
    assert doc.metadata["audio_stream_count"] == 0


def test_frame_dir_saves_frames(tmp_path):
    from parsers.video.ffmpeg import FFmpegVideoParser

    parser = FFmpegVideoParser(frame_dir=tmp_path, max_frames=2)
    doc = parser.parse(FIXTURES_DIR / "testsrc.mp4")
    assert len(list(tmp_path.glob("*.jpg"))) == 2
    for block in doc.blocks:
        assert block.metadata["frame_path"]
        assert Path(block.metadata["frame_path"]).exists()
        assert block.metadata["frame_ref"]


def test_temporal_order_preserved():
    from parsers.video.ffmpeg import FFmpegVideoParser

    parser = FFmpegVideoParser(frame_strategy="interval", interval_seconds=0.5, max_frames=10)
    doc = parser.parse(FIXTURES_DIR / "testsrc.mp4")
    timestamps = [b.metadata["timestamp"] for b in doc.blocks]
    assert timestamps == sorted(timestamps)


def test_corrupt_video_raises():
    from parsers.video.ffmpeg import FFmpegVideoParser

    parser = FFmpegVideoParser()
    with pytest.raises(CorruptDocumentError):
        parser.parse(FIXTURES_DIR / "corrupt.mp4")


def test_empty_video_raises():
    from parsers.video.ffmpeg import FFmpegVideoParser

    parser = FFmpegVideoParser()
    with pytest.raises(EmptyDocumentError):
        parser.parse(FIXTURES_DIR / "empty.mp4")


def test_audio_only_container_raises():
    from parsers.video.ffmpeg import FFmpegVideoParser

    parser = FFmpegVideoParser()
    with pytest.raises(CorruptDocumentError):
        parser.parse(FIXTURES_DIR / "audio_only.mp4")


def test_missing_file_raises():
    from parsers.video.ffmpeg import FFmpegVideoParser

    parser = FFmpegVideoParser()
    with pytest.raises(FileNotFoundError):
        parser.parse(FIXTURES_DIR / "does_not_exist.mp4")


def test_unsupported_extension_raises(tmp_path):
    from parsers.video.ffmpeg import FFmpegVideoParser

    bad = tmp_path / "notes.txt"
    bad.write_text("not a video")
    with pytest.raises(CorruptDocumentError):
        FFmpegVideoParser().parse(bad)


def test_invalid_constructor_args():
    from parsers.video.ffmpeg import FFmpegVideoParser

    with pytest.raises(ValueError):
        FFmpegVideoParser(frame_strategy="bogus")
    with pytest.raises(ValueError):
        FFmpegVideoParser(interval_seconds=0)
    with pytest.raises(ValueError):
        FFmpegVideoParser(max_frames=0)
    with pytest.raises(ValueError):
        FFmpegVideoParser(scene_threshold=1.5)


def test_extract_audio_to_wav_roundtrip(tmp_path):
    from parsers.video.ffmpeg import FFmpegVideoParser

    parser = FFmpegVideoParser()
    out = parser.extract_audio_to_wav(FIXTURES_DIR / "testsrc.mp4", tmp_path / "audio.wav")
    assert out.exists()
    assert out.stat().st_size > 1000
    import av as _av

    with _av.open(str(out)) as container:
        streams = [s for s in container.streams if s.type == "audio"]
        assert len(streams) == 1
        assert streams[0].codec_context.sample_rate == 16000
        assert streams[0].codec_context.channels == 1
