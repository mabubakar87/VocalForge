from pathlib import Path

from vocalforge.media import is_supported_upload


def test_supported_upload_extensions():
    assert is_supported_upload("a.wav")
    assert is_supported_upload("a.MP3")
    assert is_supported_upload(Path("clip.m4a"))
    assert is_supported_upload("talk.flac")
    assert not is_supported_upload("notes.txt")
    assert not is_supported_upload("image.png")
