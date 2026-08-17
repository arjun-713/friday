"""Small in-memory diagnostic session store for the text interaction loop."""

import re

from .models import DiagnosticSessionState, TroubleshootingRequest

_ACKNOWLEDGEMENT_ONLY = re.compile(
    r"^\s*(?:yes|yeah|yep|okay|ok|got it|understood|done|thanks|thank you)"
    r"(?:[\s,!.?]*(?:i(?:'ve| have)? (?:done|checked|got) it|what(?:'s| is) next|next|please|now))*[\s!.?]*$",
    re.IGNORECASE,
)


def is_acknowledgement_without_result(value: str) -> bool:
    """Keep an acknowledgement from silently completing a diagnostic check."""

    return bool(_ACKNOWLEDGEMENT_ONLY.fullmatch(value))


class DiagnosticSessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, DiagnosticSessionState] = {}

    def get(self, session_id: str) -> DiagnosticSessionState:
        state = self._sessions.get(session_id)
        if state is None:
            state = DiagnosticSessionState(session_id=session_id)
            self._sessions[session_id] = state
        return state

    def record_turn(self, request: TroubleshootingRequest) -> DiagnosticSessionState:
        state = self.get(request.session_id)
        observation = request.observation or request.selected_option
        has_explicit_option = request.selected_option is not None
        state.last_turn_was_acknowledgement = bool(
            observation and state.current_step_id and not has_explicit_option and is_acknowledgement_without_result(observation)
        )
        if observation and state.current_step_id and not state.last_turn_was_acknowledgement:
            state.observations[state.current_step_id] = observation
            if state.current_step_id not in state.completed_steps:
                state.completed_steps.append(state.current_step_id)
        return state
