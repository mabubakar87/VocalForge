"""Application state machine for VocalForge."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Iterable

logger = logging.getLogger(__name__)


class AppState(str, Enum):
    STARTING = "STARTING"
    NO_MODEL = "NO_MODEL"
    LOADING_MODEL = "LOADING_MODEL"
    READY = "READY"
    RECORDING = "RECORDING"
    TRANSCRIBING = "TRANSCRIBING"
    ERROR = "ERROR"
    SHUTTING_DOWN = "SHUTTING_DOWN"


ALLOWED_TRANSITIONS: dict[AppState, frozenset[AppState]] = {
    AppState.STARTING: frozenset({AppState.NO_MODEL, AppState.LOADING_MODEL, AppState.SHUTTING_DOWN}),
    AppState.NO_MODEL: frozenset({AppState.LOADING_MODEL, AppState.SHUTTING_DOWN}),
    AppState.LOADING_MODEL: frozenset(
        {AppState.READY, AppState.NO_MODEL, AppState.ERROR, AppState.SHUTTING_DOWN}
    ),
    AppState.READY: frozenset(
        {
            AppState.LOADING_MODEL,
            AppState.RECORDING,
            AppState.TRANSCRIBING,
            AppState.SHUTTING_DOWN,
        }
    ),
    AppState.RECORDING: frozenset({AppState.TRANSCRIBING, AppState.READY, AppState.ERROR, AppState.SHUTTING_DOWN}),
    AppState.TRANSCRIBING: frozenset({AppState.READY, AppState.ERROR, AppState.SHUTTING_DOWN}),
    AppState.ERROR: frozenset({AppState.READY, AppState.NO_MODEL, AppState.SHUTTING_DOWN}),
    AppState.SHUTTING_DOWN: frozenset(),
}


class InvalidTransition(ValueError):
    """Raised when a state transition is not allowed."""


class StateMachine:
    def __init__(self, initial: AppState = AppState.STARTING) -> None:
        self._state = initial

    @property
    def state(self) -> AppState:
        return self._state

    def can_transition(self, target: AppState) -> bool:
        return target in ALLOWED_TRANSITIONS.get(self._state, frozenset())

    def transition(self, target: AppState) -> AppState:
        if not self.can_transition(target):
            message = f"Invalid transition {self._state.value} -> {target.value}"
            logger.error(message)
            raise InvalidTransition(message)
        previous = self._state
        self._state = target
        logger.info("State %s -> %s", previous.value, target.value)
        return self._state

    def try_transition(self, target: AppState) -> bool:
        if not self.can_transition(target):
            logger.warning("Rejected transition %s -> %s", self._state.value, target.value)
            return False
        self.transition(target)
        return True

    def in_any(self, states: Iterable[AppState]) -> bool:
        return self._state in states

    def can_record(self) -> bool:
        return self._state is AppState.READY

    def can_upload(self) -> bool:
        return self._state is AppState.READY

    def can_load_model(self) -> bool:
        return self._state in {AppState.NO_MODEL, AppState.READY, AppState.STARTING, AppState.ERROR}

    def is_busy(self) -> bool:
        return self._state in {
            AppState.LOADING_MODEL,
            AppState.RECORDING,
            AppState.TRANSCRIBING,
            AppState.SHUTTING_DOWN,
        }
