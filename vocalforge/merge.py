"""Merge diarization speaker turns onto timed Whisper segments."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence

UNKNOWN_SPEAKER = "SPEAKER_UNKNOWN"


def _as_mapping(segment: Any) -> Mapping[str, Any]:
    if isinstance(segment, Mapping):
        return segment
    return {
        "start": getattr(segment, "start", 0.0),
        "end": getattr(segment, "end", 0.0),
        "text": getattr(segment, "text", ""),
        "speaker": getattr(segment, "speaker", None),
    }


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def dominant_speaker(
    start: float,
    end: float,
    diarization_turns: Sequence[Mapping[str, Any]],
) -> str:
    """Return the speaker with the largest temporal overlap, or SPEAKER_UNKNOWN."""
    best_speaker = UNKNOWN_SPEAKER
    best_overlap = 0.0
    for turn in diarization_turns:
        try:
            t_start = float(turn["start"])
            t_end = float(turn["end"])
            speaker = str(turn.get("speaker") or UNKNOWN_SPEAKER)
        except (KeyError, TypeError, ValueError):
            continue
        amount = _overlap(start, end, t_start, t_end)
        if amount > best_overlap:
            best_overlap = amount
            best_speaker = speaker
    return best_speaker if best_overlap > 0 else UNKNOWN_SPEAKER


def assign_speakers(
    whisper_segments: Sequence[Any],
    diarization_turns: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach a dominant speaker id to each Whisper segment by time overlap.

    Each output item is a dict with at least: start, end, text, speaker.
    """
    labeled: list[dict[str, Any]] = []
    for raw in whisper_segments:
        seg = _as_mapping(raw)
        try:
            start = float(seg.get("start", 0.0) or 0.0)
            end = float(seg.get("end", start) or start)
        except (TypeError, ValueError):
            start, end = 0.0, 0.0
        text = str(seg.get("text", "") or "")
        speaker = dominant_speaker(start, end, diarization_turns)
        item: dict[str, Any] = {
            "start": start,
            "end": end,
            "text": text,
            "speaker": speaker,
        }
        # Preserve any extra keys from dict inputs.
        if isinstance(raw, MutableMapping):
            for key, value in raw.items():
                if key not in item:
                    item[key] = value
        labeled.append(item)
    return labeled


def group_words_by_speaker(
    words: Sequence[Mapping[str, Any]],
    diarization_turns: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Assign each timed word to a speaker, then merge consecutive same-speaker runs.

    Prefer this over segment-level assign when Whisper returns one long segment that
    spans multiple diarization turns (common with short conversational clips).
    """
    if not words:
        return []

    labeled_words: list[dict[str, Any]] = []
    for raw in words:
        word = str(raw.get("word") or raw.get("text") or "").strip()
        if not word:
            continue
        try:
            start = float(raw.get("start", 0.0) or 0.0)
            end = float(raw.get("end", start) or start)
        except (TypeError, ValueError):
            continue
        mid = (start + end) / 2.0
        speaker = dominant_speaker(mid, mid + 1e-3, diarization_turns)
        if speaker == UNKNOWN_SPEAKER:
            speaker = dominant_speaker(start, end, diarization_turns)
        labeled_words.append(
            {"start": start, "end": end, "text": word, "speaker": speaker}
        )

    if not labeled_words:
        return []

    groups: list[dict[str, Any]] = []
    current = {
        "start": labeled_words[0]["start"],
        "end": labeled_words[0]["end"],
        "speaker": labeled_words[0]["speaker"],
        "text": labeled_words[0]["text"],
    }
    for item in labeled_words[1:]:
        if item["speaker"] == current["speaker"]:
            current["end"] = item["end"]
            current["text"] = f"{current['text']} {item['text']}".strip()
        else:
            groups.append(current)
            current = {
                "start": item["start"],
                "end": item["end"],
                "speaker": item["speaker"],
                "text": item["text"],
            }
    groups.append(current)
    return groups


def format_speaker_transcript(segments: Sequence[Mapping[str, Any]]) -> str:
    """Render `[SPEAKER_00 0:00–0:04] text` lines for display / save."""
    from vocalforge.transcription import format_timestamp

    lines: list[str] = []
    for seg in segments:
        text = str(seg.get("text", "") or "").strip()
        if not text:
            continue
        speaker = str(seg.get("speaker") or UNKNOWN_SPEAKER)
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        lines.append(
            f"[{speaker} {format_timestamp(start)}–{format_timestamp(end)}] {text}"
        )
    return "\n".join(lines)
