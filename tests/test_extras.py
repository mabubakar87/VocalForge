from vocalforge.extras import (
    EXTRA_ORDER,
    EXTRAS,
    ExtraStatus,
    list_extras,
    probe_all,
    status_label,
)


def test_extras_catalog_complete():
    assert set(EXTRA_ORDER) == set(EXTRAS)
    assert EXTRA_ORDER == ("alignment", "diarization", "enhancement", "separation")
    for extra in list_extras():
        assert extra.id in EXTRAS
        assert extra.label
        assert extra.proposal_doc.startswith("docs/proposals/")
        assert extra.candidate_engines
        assert extra.import_names


def test_probe_all_returns_statuses_without_crash():
    results = probe_all()
    assert len(results) == 4
    for extra, status in results:
        assert isinstance(status, ExtraStatus)
        assert status_label(status)


def test_default_environment_reports_not_installed():
    """Extras stay not_installed unless their READY probes succeed locally."""
    from vocalforge.diarization import is_diarization_ready
    from vocalforge.enhancement import is_enhancement_available

    for extra, status in probe_all():
        if extra.id == "enhancement" and is_enhancement_available():
            assert status is ExtraStatus.READY
            continue
        if extra.id == "diarization" and is_diarization_ready():
            assert status is ExtraStatus.READY
            continue
        assert status is ExtraStatus.NOT_INSTALLED
