from pathlib import Path

from copilot.answering.models import (
    DecisionBasis,
    DiagnosticAction,
    DiagnosticFact,
    DiagnosticSessionState,
    DiagnosticStep,
    DiagnosticTurn,
    ObservationRequest,
    TroubleshootingRequest,
)
from copilot.answering.session import DiagnosticSessionStore, SqliteDiagnosticSessionStore


def test_sqlite_session_store_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite3"
    first_store = SqliteDiagnosticSessionStore(path)
    state = first_store.record_turn(TroubleshootingRequest(query="Printer is offline", session_id="printer-1"))
    state.current_step_id = "check-status"
    state.current_step = DiagnosticStep(
        step_id="check-status",
        title="Check printer status",
        instruction="Open the printer status window.",
        question="What status is displayed?",
        source_ids=["manual-chunk-1"],
    )
    first_store.save(state)

    reopened_store = SqliteDiagnosticSessionStore(path)
    restored = reopened_store.record_turn(
        TroubleshootingRequest(query="It says offline", observation="offline", session_id="printer-1")
    )

    assert restored.current_step_id == "check-status"
    assert restored.current_step is not None
    assert restored.observations == {"check-status": "offline"}
    assert restored.completed_steps == ["check-status"]


def test_sqlite_session_store_deletes_saved_state(tmp_path: Path) -> None:
    path = tmp_path / "sessions.sqlite3"
    store = SqliteDiagnosticSessionStore(path)
    state = store.record_turn(TroubleshootingRequest(query="Printer is offline", session_id="printer-delete"))
    state.current_step_id = "check-status"
    store.save(state)

    store.delete("printer-delete")

    restored = SqliteDiagnosticSessionStore(path).get("printer-delete")
    assert restored.current_step_id is None


def test_session_store_preserves_fact_transition_after_recheck_action() -> None:
    store = DiagnosticSessionStore()
    state = DiagnosticSessionState(session_id="battery-transition")
    first = DiagnosticTurn(
        turn_id="turn-before-reconnect",
        mode="advance",
        response="Blinking amber is a battery state we need to recheck after the adapter connection changes.",
        next_action=DiagnosticAction(
            instruction="Reconnect the AC adapter firmly at both ends.",
            why="This distinguishes an adapter connection issue from a battery condition.",
        ),
        observation_request=ObservationRequest(
            request_id="recheck-battery-light",
            fact_key="battery_light",
            question="After reconnecting it, what does the battery light show?",
            recheck_after_action=True,
        ),
        decision_basis=DecisionBasis(
            why_not_solved="The earlier light state does not establish whether the adapter connection is being detected.",
            discriminates_between=["adapter connection issue", "battery condition"],
            expected_discrimination="A changed light state after reconnecting indicates that AC detection changed.",
        ),
        facts_learned=[
            DiagnosticFact(
                key="battery_light",
                value="blinking amber",
                label="Battery light",
                raw="It was blinking amber before reconnecting.",
            )
        ],
        source_ids=["manual-1"],
    )
    store.apply_turn(state, first)

    second = DiagnosticTurn(
        turn_id="turn-after-reconnect",
        mode="solve",
        response="The battery light changed after the reconnect.",
        facts_learned=[
            DiagnosticFact(
                key="battery_light",
                value="white",
                label="Battery light",
                raw="After reconnecting it is white.",
            )
        ],
        source_ids=["manual-1"],
    )
    store.apply_turn(state, second)

    history = state.fact_history["battery_light"]
    assert state.facts["battery_light"].value == "white"
    assert [event.value for event in history] == ["blinking amber", "white"]
    assert history[-1].previous_value == "blinking amber"
    assert history[-1].observed_after_action_id == "turn-before-reconnect:action"
