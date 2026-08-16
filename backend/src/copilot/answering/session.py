"""Small in-memory diagnostic session store for the text interaction loop."""

from .models import DiagnosticSessionState, TroubleshootingRequest


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
        observation = request.selected_option or request.observation
        if observation and state.current_step_id:
            state.observations[state.current_step_id] = observation
            if state.current_step_id not in state.completed_steps:
                state.completed_steps.append(state.current_step_id)
        return state
