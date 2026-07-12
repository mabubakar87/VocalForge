from pathlib import Path

from vocalforge.storage import default_paths, next_transcript_path, save_transcript


def test_default_paths_create_directories(tmp_path: Path):
    paths = default_paths(tmp_path)
    assert paths.models.is_dir()
    assert paths.audio.is_dir()
    assert paths.transcripts.is_dir()
    assert paths.log_file.name == "vocalforge.log"


def test_save_transcript_writes_file(tmp_path: Path):
    paths = default_paths(tmp_path)
    path = save_transcript(paths, "Hello world")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "Hello world"
    assert path.parent == paths.transcripts


def test_next_transcript_path_avoids_collision(tmp_path: Path):
    paths = default_paths(tmp_path)
    first = next_transcript_path(paths)
    first.write_text("a", encoding="utf-8")
    second = next_transcript_path(paths)
    assert first != second


def test_list_and_read_transcripts(tmp_path: Path):
    from datetime import datetime, timedelta

    from vocalforge.storage import list_transcripts, read_transcript, transcript_label

    paths = default_paths(tmp_path)
    base = datetime(2026, 7, 12, 12, 0, 0)
    older = save_transcript(paths, "first", when=base)
    newer = save_transcript(paths, "second", when=base + timedelta(seconds=5))
    listed = list_transcripts(paths)
    assert listed[0] == newer
    assert older in listed
    assert read_transcript(newer) == "second"
    assert transcript_label(newer)
