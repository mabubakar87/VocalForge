from vocalforge.formatting import format_text


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
