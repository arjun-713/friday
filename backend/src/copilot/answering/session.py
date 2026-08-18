"""Small in-memory diagnostic session store for the text interaction loop."""

import re
import sqlite3
from pathlib import Path
from threading import RLock

from .models import DiagnosticSessionState, DiagnosticTurn, TroubleshootingRequest

_ACKNOWLEDGEMENT_ONLY = re.compile(r"[^a-z0-9]+")
_ACKNOWLEDGEMENT_PHRASES = (
    "got it",
    "what next",
    "what is next",
    "whats next",
    "i understand",
    "i have done it",
    "i checked it",
)
_SHORT_ACKNOWLEDGEMENTS = {"yes", "yeah", "yep", "okay", "ok", "done", "thanks", "thank you", "understood"}


def is_acknowledgement_without_result(value: str) -> bool:
    """Keep an acknowledgement from silently completing a diagnostic check."""

    normalized = " ".join(part for part in _ACKNOWLEDGEMENT_ONLY.split(value.lower()) if part)
    if normalized in _SHORT_ACKNOWLEDGEMENTS:
        return True
    return len(normalized.split()) <= 12 and any(phrase in normalized for phrase in _ACKNOWLEDGEMENT_PHRASES)


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
        # The planner, rather than a pre-written decision tree, decides how to
        # interpret a result.  The store only retains the raw report and the
        # current question it belongs to so it cannot be forgotten.
        observation = _submitted_observation(state, request)
        has_explicit_option = request.selected_option is not None
        state.last_turn_was_acknowledgement = bool(
            observation
            and state.current_request is not None
            and not has_explicit_option
            and is_acknowledgement_without_result(observation)
        )
        state.pending_observation = None if state.last_turn_was_acknowledgement else observation or None
        state.pending_option_id = request.selected_option
        if observation and state.current_request is not None and not state.last_turn_was_acknowledgement:
            if state.current_request.request_id not in state.completed_steps:
                state.completed_steps.append(state.current_request.request_id)
            # Preserve a human-readable audit trail while the LLM turns the
            # report into its semantic fact updates.
            state.observations[state.current_request.request_id] = observation
        # Sessions saved before the planner contract used ``current_step``.
        # Keep them readable and allow their next result to complete normally.
        if observation and state.current_request is None and state.current_step_id and not state.last_turn_was_acknowledgement:
            state.observations[state.current_step_id] = observation
            if state.current_step_id not in state.completed_steps:
                state.completed_steps.append(state.current_step_id)
        return state

    def apply_turn(self, state: DiagnosticSessionState, turn: DiagnosticTurn) -> None:
        """Persist planner-approved facts and the next requested observation."""

        for fact in turn.facts_learned:
            state.facts[fact.key] = fact
            state.observations[fact.key] = f"{fact.label}: {fact.value}"
        if state.current_request and any(
            fact.key == state.current_request.fact_key for fact in turn.facts_learned
        ):
            request_id = state.current_request.request_id
            if request_id not in state.completed_steps:
                state.completed_steps.append(request_id)
        for cause in turn.ruled_out_causes:
            if cause not in state.ruled_out_causes:
                state.ruled_out_causes.append(cause)
        state.current_turn = turn
        state.current_request = turn.observation_request
        state.current_step_id = turn.observation_request.request_id if turn.observation_request else None
        state.current_step = None
        state.pending_observation = None
        state.pending_option_id = None
        state.last_turn_was_acknowledgement = False

    def save(self, state: DiagnosticSessionState) -> None:
        """Persist an updated state. In-memory storage already holds the object."""

        self._sessions[state.session_id] = state

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def _submitted_observation(state: DiagnosticSessionState, request: TroubleshootingRequest) -> str:
    """Resolve a clicked option to its visible/canonical value without rules by device."""

    if request.selected_option and state.current_request:
        for option in state.current_request.options:
            if option.id == request.selected_option:
                return option.value or option.label
    return request.observation or request.selected_option or request.query


class SqliteDiagnosticSessionStore(DiagnosticSessionStore):
    """Durable local session storage without collecting microphone audio."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnostic_sessions (
                    session_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def get(self, session_id: str) -> DiagnosticSessionState:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM diagnostic_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            state = DiagnosticSessionState(session_id=session_id)
            self.save(state)
            return state
        return DiagnosticSessionState.model_validate_json(str(row[0]))

    def save(self, state: DiagnosticSessionState) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO diagnostic_sessions (session_id, state_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (state.session_id, state.model_dump_json()),
            )

    def record_turn(self, request: TroubleshootingRequest) -> DiagnosticSessionState:
        state = super().record_turn(request)
        self.save(state)
        return state

    def apply_turn(self, state: DiagnosticSessionState, turn: DiagnosticTurn) -> None:
        super().apply_turn(state, turn)
        self.save(state)

    def delete(self, session_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM diagnostic_sessions WHERE session_id = ?", (session_id,))
