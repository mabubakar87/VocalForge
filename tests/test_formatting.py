from vocalforge.formatting import (
    contains_arabic_script,
    format_text,
    prepare_ui_text,
)


def test_format_text_empty():
    assert format_text("") == ""
    assert format_text("   ") == ""


def test_format_text_collapses_whitespace_and_capitalizes():
    assert format_text("  hello   world  ") == "Hello world"


def test_format_text_punctuation_spacing():
    assert format_text("hello ,world") == "Hello, world"
    assert format_text("Hi.there") == "Hi. There"


def test_format_text_sentence_capitals():
    assert format_text("one. two! three?") == "One. Two! Three?"


def test_format_text_unicode():
    assert format_text("café is nice") == "Café is nice"


def test_format_text_preserves_urdu():
    urdu = "دیکھتے ہیں اردو میں ایک ایسا کام کرتا ہے"
    assert format_text(urdu) == urdu
    assert contains_arabic_script(urdu) is True


def test_prepare_ui_text_shapes_urdu():
    urdu = "دیکھتے ہیں اردو میں ایک ایسا کام کرتا ہے"
    shaped = prepare_ui_text(urdu)
    assert shaped != urdu
    assert "د" in shaped or "ﺩ" in shaped or "ﺪ" in shaped
    assert prepare_ui_text("Hello world") == "Hello world"

