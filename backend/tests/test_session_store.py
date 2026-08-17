from pathlib import Path

from copilot.answering.models import DiagnosticStep, TroubleshootingRequest
from copilot.answering.session import SqliteDiagnosticSessionStore


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
