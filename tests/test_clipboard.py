from vocalforge.clipboard import ClipboardSettings, deliver_text


def test_deliver_text_copy_only(monkeypatch):
    copied = {}

    def fake_copy(text):
        copied["text"] = text

    monkeypatch.setattr("vocalforge.clipboard.copy_text", fake_copy)

    def boom():
        raise AssertionError("paste should not run")

    monkeypatch.setattr("vocalforge.clipboard.paste_hotkey", boom)
    deliver_text("hello", ClipboardSettings(auto_paste=False))
    assert copied["text"] == "hello"


def test_deliver_text_swallows_clipboard_errors(monkeypatch):
    def boom(_text):
        raise RuntimeError("clipboard unavailable")

    monkeypatch.setattr("vocalforge.clipboard.copy_text", boom)
    deliver_text("hello", ClipboardSettings(auto_paste=False))
