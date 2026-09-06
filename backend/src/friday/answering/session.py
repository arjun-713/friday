"""Small in-memory diagnostic session store for the text interaction loop."""

import re
import sqlite3
from pathlib import Path
from threading import RLock

from .models import DiagnosticSessionState, DiagnosticTurn, FactObservation, TroubleshootingRequest

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
        prior_action = state.current_turn.next_action if state.current_turn else None
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
            if prior_action is not None:
                action_key = _normalize_action(prior_action.instruction)
                if action_key and action_key not in state.completed_actions:
                    state.completed_actions.append(action_key)
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
        if (
            observation
            and not state.last_turn_was_acknowledgement
            and not state.current_request
            and not state.current_step_id
            and observation not in state.user_reports
        ):
            state.user_reports.append(observation)
        for action_key in _completed_actions_from_report(observation or ""):
            if action_key not in state.completed_actions:
                state.completed_actions.append(action_key)
        return state

    def apply_turn(self, state: DiagnosticSessionState, turn: DiagnosticTurn) -> None:
        """Persist planner-approved facts and the next requested observation."""

        prior_action_id = _action_id(state.current_turn)
        for fact in turn.facts_learned:
            previous = state.facts.get(fact.key)
            observed_after_action_id = (
                prior_action_id
                if state.current_request
                and state.current_request.recheck_after_action
                and state.current_request.fact_key == fact.key
                else None
            )
            event = FactObservation(
                **fact.model_dump(),
                turn_id=turn.turn_id,
                previous_value=previous.value if previous and previous.value != fact.value else None,
                observed_after_action_id=observed_after_action_id,
            )
            state.fact_history.setdefault(fact.key, []).append(event)
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
        if turn.next_action is not None:
            state.current_next_branch = turn.next_action.instruction
        else:
            state.current_next_branch = None
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


def _action_id(turn: DiagnosticTurn | None) -> str | None:
    """Give a persisted action a stable ID without trusting model-generated IDs."""

    if turn is None or turn.next_action is None:
        return None
    return f"{turn.turn_id}:action"


def _normalize_action(instruction: str) -> str:
    """Create a compact, stable action label for repeat suppression and logs."""

    normalized = " ".join(instruction.lower().split())
    patterns = (
        ("POWER_CYCLE_MODEM_ROUTER", ("power off", "modem", "router")),
        ("POWER_CYCLE_MODEM", ("power off", "modem")),
        ("POWER_CYCLE_ROUTER", ("power off", "router")),
        ("CHECK_WAN_CABLE", ("ethernet", "wan port")),
        ("CHECK_WAN_STATUS", ("wan", "status")),
        ("CHECK_WAN_TYPE", ("connection type",)),
        ("RENEW_DHCP", ("renew", "connection")),
        ("CHECK_MAC_CLONE", ("mac", "clone")),
        ("TEST_MODEM_DIRECT", ("direct", "modem")),
        ("CONTACT_ISP", ("contact", "internet provider")),
    )
    for action_id, terms in patterns:
        if all(term in normalized for term in terms):
            return action_id
    return normalized[:160]


def _completed_actions_from_report(report: str) -> set[str]:
    """Recognize explicit completed operations without choosing next steps."""

    normalized = " ".join(report.lower().split())
    actions: set[str] = set()
    if ("power-cycl" in normalized or "restarted" in normalized or "restart" in normalized) and "modem" in normalized:
        actions.add("POWER_CYCLE_MODEM")
    if (
        ("power-cycl" in normalized or "restarted" in normalized or "restart" in normalized or "powered back on" in normalized)
        and "router" in normalized
    ):
        actions.add("POWER_CYCLE_ROUTER")
    if "ethernet" in normalized and "wan" in normalized and ("connected" in normalized or "firmly" in normalized):
        actions.add("CHECK_WAN_CABLE")
    if "direct" in normalized and "modem" in normalized and ("works" in normalized or "working" in normalized):
        actions.add("TEST_MODEM_DIRECT")
    if "renew" in normalized and ("connection" in normalized or "dhcp" in normalized):
        actions.add("RENEW_DHCP")
    return actions


def repeated_actions_in_response(response: str, completed_actions: list[str]) -> list[str]:
    """Find completed operations that a new assistant reply recommends again."""

    normalized = " ".join(response.lower().split())
    phrases = {
        "POWER_CYCLE_MODEM": ("power off", "modem"),
        "POWER_CYCLE_ROUTER": ("power off", "router"),
        "CHECK_WAN_CABLE": ("ethernet", "wan"),
        "TEST_MODEM_DIRECT": ("connect", "direct", "modem"),
        "RENEW_DHCP": ("renew", "connection"),
    }
    return [
        action_id
        for action_id in completed_actions
        if action_id in phrases and all(term in normalized for term in phrases[action_id])
    ]


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
