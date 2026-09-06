from pathlib import Path

from friday.answering.models import (
    DecisionBasis,
    DiagnosticAction,
    DiagnosticFact,
    DiagnosticSessionState,
    DiagnosticStep,
    DiagnosticTurn,
    ObservationRequest,
    TroubleshootingRequest,
)
from friday.answering.session import (
    DiagnosticSessionStore,
    SqliteDiagnosticSessionStore,
    repeated_actions_in_response,
)


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


def test_session_store_records_completed_action_when_observation_arrives() -> None:
    store = DiagnosticSessionStore()
    state = DiagnosticSessionState(session_id="router-actions")
    store.apply_turn(
        state,
        DiagnosticTurn(
            turn_id="turn-power-cycle",
            mode="advance",
            response="Restart the modem and router in order.",
            next_action=DiagnosticAction(
                instruction="Power off the modem and router, then power on the modem first.",
                why="This checks whether the upstream connection is restored before the router starts.",
            ),
            observation_request=ObservationRequest(
                request_id="wan-light-after-restart",
                fact_key="wan_light",
                question="What does the WAN light show after the restart?",
            ),
            decision_basis=DecisionBasis(
                why_not_solved="The router has power, but upstream connectivity is still unknown.",
                discriminates_between=["upstream outage", "router WAN issue"],
                expected_discrimination="A restored WAN light indicates the upstream connection returned.",
            ),
            source_ids=["manual-1"],
        ),
    )
    store.save(state)

    state = store.record_turn(
        TroubleshootingRequest(
            query="The router is back on and the WAN light is still off.",
            observation="WAN light still off",
            session_id="router-actions",
        )
    )

    assert state.completed_actions == ["POWER_CYCLE_MODEM_ROUTER"]
    assert state.completed_steps == ["wan-light-after-restart"]


def test_session_store_retains_initial_user_report_for_conversational_turns() -> None:
    store = DiagnosticSessionStore()
    state = store.record_turn(
        TroubleshootingRequest(
            query="My router has Wi-Fi but no internet.",
            session_id="router-report",
        )
    )

    assert state.user_reports == ["My router has Wi-Fi but no internet."]

    next_state = store.record_turn(
        TroubleshootingRequest(
            query="The WAN light is off.",
            session_id="router-report",
        )
    )
    assert next_state.user_reports == [
        "My router has Wi-Fi but no internet.",
        "The WAN light is off.",
    ]


def test_session_store_normalizes_explicit_completed_operations_from_reports() -> None:
    store = DiagnosticSessionStore()
    state = store.record_turn(
        TroubleshootingRequest(
            query="The Ethernet cable is firmly connected between the modem and the router WAN port.",
            session_id="router-normalization",
        )
    )
    state = store.record_turn(
        TroubleshootingRequest(
            query="I power-cycled the modem and the router is powered back on.",
            session_id="router-normalization",
        )
    )

    assert set(state.completed_actions) == {
        "CHECK_WAN_CABLE",
        "POWER_CYCLE_MODEM",
        "POWER_CYCLE_ROUTER",
    }


def test_repeated_actions_are_reported_for_conversational_quality_metrics() -> None:
    repeated = repeated_actions_in_response(
        "Power off the modem and router again, then power on the modem first.",
        ["POWER_CYCLE_MODEM", "POWER_CYCLE_ROUTER", "CHECK_WAN_CABLE"],
    )

    assert repeated == ["POWER_CYCLE_MODEM", "POWER_CYCLE_ROUTER"]
