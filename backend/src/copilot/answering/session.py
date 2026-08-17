"""Small in-memory diagnostic session store for the text interaction loop."""

import re
import sqlite3
from pathlib import Path
from threading import RLock

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

    def save(self, state: DiagnosticSessionState) -> None:
        """Persist an updated state. In-memory storage already holds the object."""

        self._sessions[state.session_id] = state


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
