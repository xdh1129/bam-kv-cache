from .record import State

# Legal forward transitions. Terminal states map to the empty set.
ALLOWED_TRANSITIONS: dict[State, set[State]] = {
    State.MISSING: {State.WRITING},
    State.WRITING: {State.READY, State.ABORTED, State.FAILED},
    State.READY: {State.LOADING, State.FAILED, State.EVICTED},
    State.LOADING: {State.READY, State.FAILED},
    State.ABORTED: set(),
    State.FAILED: set(),
    State.EVICTED: set(),
}


class IllegalTransition(Exception):
    pass


def validate_transition(src: State, dst: State) -> None:
    if dst not in ALLOWED_TRANSITIONS[src]:
        raise IllegalTransition(f"{src.value} -> {dst.value} not allowed")
