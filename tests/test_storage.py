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
