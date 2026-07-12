from vocalforge.merge import (
    UNKNOWN_SPEAKER,
    assign_speakers,
    dominant_speaker,
    format_speaker_transcript,
    group_words_by_speaker,
)


def test_dominant_speaker_picks_max_overlap():
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    assert dominant_speaker(0.0, 4.0, turns) == "SPEAKER_00"
    assert dominant_speaker(6.0, 9.0, turns) == "SPEAKER_01"
    assert dominant_speaker(4.5, 6.5, turns) in {"SPEAKER_00", "SPEAKER_01"}


def test_assign_speakers_unknown_when_no_overlap():
    turns = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    labeled = assign_speakers(
        [{"start": 5.0, "end": 6.0, "text": "later"}],
        turns,
    )
    assert labeled[0]["speaker"] == UNKNOWN_SPEAKER
    assert labeled[0]["text"] == "later"


def test_assign_speakers_and_format():
    turns = [
        {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 9.0, "speaker": "SPEAKER_01"},
    ]
    segments = [
        {"start": 0.0, "end": 4.0, "text": "Hello, thanks for joining."},
        {"start": 4.0, "end": 9.0, "text": "Happy to be here."},
    ]
    labeled = assign_speakers(segments, turns)
    assert labeled[0]["speaker"] == "SPEAKER_00"
    assert labeled[1]["speaker"] == "SPEAKER_01"
    text = format_speaker_transcript(labeled)
    assert "[SPEAKER_00 0:00–0:04]" in text
    assert "[SPEAKER_01 0:04–0:09]" in text


def test_group_words_by_speaker_splits_on_change():
    turns = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"},
    ]
    words = [
        {"start": 0.0, "end": 0.5, "word": "Hello"},
        {"start": 0.5, "end": 1.0, "word": "there"},
        {"start": 2.1, "end": 2.5, "word": "Hi"},
        {"start": 2.5, "end": 3.0, "word": "back"},
    ]
    groups = group_words_by_speaker(words, turns)
    assert len(groups) == 2
    assert groups[0]["speaker"] == "SPEAKER_00"
    assert groups[0]["text"] == "Hello there"
    assert groups[1]["speaker"] == "SPEAKER_01"
    assert groups[1]["text"] == "Hi back"
