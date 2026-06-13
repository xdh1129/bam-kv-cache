import pytest

from bam_kv_cache.metadata.record import State
from bam_kv_cache.metadata.state_machine import (
    ALLOWED_TRANSITIONS,
    IllegalTransition,
    validate_transition,
)


def test_happy_store_path():
    validate_transition(State.MISSING, State.WRITING)
    validate_transition(State.WRITING, State.READY)


def test_happy_load_path():
    validate_transition(State.READY, State.LOADING)
    validate_transition(State.LOADING, State.READY)


def test_illegal_transition_raises():
    with pytest.raises(IllegalTransition):
        validate_transition(State.READY, State.WRITING)


def test_terminal_states_have_no_exits():
    for terminal in (State.ABORTED, State.FAILED, State.EVICTED):
        assert ALLOWED_TRANSITIONS[terminal] == set()


def test_writing_can_fail_or_abort():
    validate_transition(State.WRITING, State.FAILED)
    validate_transition(State.WRITING, State.ABORTED)


def test_loading_can_fail():
    validate_transition(State.LOADING, State.FAILED)
