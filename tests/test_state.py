from vocalforge.state import AppState, InvalidTransition, StateMachine
import pytest


def test_happy_path_transitions():
    sm = StateMachine()
    assert sm.state is AppState.STARTING
    sm.transition(AppState.NO_MODEL)
    sm.transition(AppState.LOADING_MODEL)
    sm.transition(AppState.READY)
    sm.transition(AppState.RECORDING)
    sm.transition(AppState.TRANSCRIBING)
    sm.transition(AppState.READY)


def test_invalid_transition_raises():
    sm = StateMachine(AppState.READY)
    with pytest.raises(InvalidTransition):
        sm.transition(AppState.NO_MODEL)


def test_try_transition_rejects_without_raise():
    sm = StateMachine(AppState.RECORDING)
    assert sm.try_transition(AppState.LOADING_MODEL) is False
    assert sm.state is AppState.RECORDING


def test_control_helpers():
    sm = StateMachine(AppState.READY)
    assert sm.can_record()
    assert sm.can_upload()
    assert sm.can_load_model()
    sm.transition(AppState.RECORDING)
    assert sm.is_busy()
    assert not sm.can_record()
