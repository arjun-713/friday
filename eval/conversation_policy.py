"""Deterministic quality checks for complete diagnostic conversations.

These checks validate the agent's structured decisions, not prose style.  They
make soft-questionnaire behavior visible before adding an LLM-as-judge layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from copilot.answering.models import DiagnosticTurn


@dataclass(frozen=True)
class ConversationCase:
    """A manually authored expected diagnostic trace."""

    case_id: str
    turns: list[DiagnosticTurn]
    initial_evidence_sufficient: bool
    tool_calls: list[str]
    expected_terminal_mode: str
    max_turns_to_first_useful_intervention: int

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> ConversationCase:
        turns = [DiagnosticTurn.model_validate(item) for item in value["turns"]]
        if not turns:
            raise ValueError(f"{value['case_id']}: conversation needs at least one turn")
        expected_terminal_mode = str(value["expected_terminal_mode"])
        if expected_terminal_mode not in {"solve", "advance", "clarify", "abstain"}:
            raise ValueError(f"{value['case_id']}: unsupported terminal mode")
        return cls(
            case_id=str(value["case_id"]),
            turns=turns,
            initial_evidence_sufficient=bool(value.get("initial_evidence_sufficient", False)),
            tool_calls=[str(name) for name in value.get("tool_calls", [])],
            expected_terminal_mode=expected_terminal_mode,
            max_turns_to_first_useful_intervention=int(
                value.get("max_turns_to_first_useful_intervention", 1)
            ),
        )


def evaluate_case(case: ConversationCase) -> dict[str, Any]:
    """Return explicit policy violations and operational conversation metrics."""

    violations: list[str] = []
    known_facts: dict[str, str] = {}
    prior_recheck_actions: dict[str, str] = {}
    clarification_streak = 0
    max_clarification_streak = 0
    transitions = 0
    acknowledged_transitions = 0
    first_useful_intervention: int | None = None

    for turn_number, turn in enumerate(case.turns, start=1):
        request = turn.observation_request
        if turn.mode == "advance":
            if turn.next_action is None:
                violations.append(f"turn {turn_number}: advance has no action")
            if turn.next_action is None or not turn.next_action.why:
                violations.append(f"turn {turn_number}: advance has no diagnosis-specific reason")
            if turn.decision_basis is None or len(turn.decision_basis.discriminates_between) < 2:
                violations.append(f"turn {turn_number}: advance does not distinguish plausible causes")
        if turn.mode == "clarify":
            clarification_streak += 1
            max_clarification_streak = max(max_clarification_streak, clarification_streak)
            if request is None:
                violations.append(f"turn {turn_number}: clarify has no essential observation request")
            if turn.next_action is not None:
                violations.append(f"turn {turn_number}: clarify contains a consequential action")
            if turn.decision_basis is None:
                violations.append(f"turn {turn_number}: clarify does not say why the fact is essential")
        else:
            clarification_streak = 0
        if turn.mode == "solve" and (turn.next_action is not None or request is not None):
            violations.append(f"turn {turn_number}: solve continues the diagnostic chain")
        if turn.mode == "abstain" and (turn.next_action is not None or request is not None):
            violations.append(f"turn {turn_number}: abstain still asks for an action or observation")

        if turn.mode in {"advance", "solve"} and first_useful_intervention is None:
            first_useful_intervention = turn_number

        if request is not None and request.recheck_after_action:
            if turn.next_action is None:
                violations.append(
                    f"turn {turn_number}: recheck has no action capable of changing {request.fact_key}"
                )
            else:
                prior_recheck_actions[request.fact_key] = turn.turn_id
        if request is not None and request.fact_key in known_facts and not request.recheck_after_action:
            violations.append(f"turn {turn_number}: repeated durable fact {request.fact_key}")

        response_text = " ".join(filter(None, (turn.response, turn.interpretation))).casefold()
        for fact in turn.facts_learned:
            old_value = known_facts.get(fact.key)
            if old_value is not None and old_value != fact.value:
                transitions += 1
                action_turn_id = prior_recheck_actions.get(fact.key)
                if action_turn_id is None:
                    violations.append(f"turn {turn_number}: fact {fact.key} changed without a recheck action")
                if old_value.casefold() in response_text and fact.value.casefold() in response_text:
                    acknowledged_transitions += 1
                else:
                    violations.append(f"turn {turn_number}: fact transition for {fact.key} was not acknowledged")
            known_facts[fact.key] = fact.value

    if case.initial_evidence_sufficient and case.tool_calls:
        violations.append("agent searched manuals despite sufficient supplied evidence")
    if case.turns[-1].mode != case.expected_terminal_mode:
        violations.append(
            f"terminal mode was {case.turns[-1].mode}, expected {case.expected_terminal_mode}"
        )
    if first_useful_intervention is None and case.expected_terminal_mode != "abstain":
        violations.append("conversation never provided a useful intervention")
    elif (
        first_useful_intervention is not None
        and first_useful_intervention > case.max_turns_to_first_useful_intervention
    ):
        violations.append(
            "first useful intervention took "
            f"{first_useful_intervention} turns, expected at most "
            f"{case.max_turns_to_first_useful_intervention}"
        )

    return {
        "case_id": case.case_id,
        "passed": not violations,
        "violations": violations,
        "turn_count": len(case.turns),
        "turns_to_first_useful_intervention": first_useful_intervention,
        "max_clarification_streak": max_clarification_streak,
        "state_transitions": transitions,
        "acknowledged_state_transitions": acknowledged_transitions,
        "tool_call_count": len(case.tool_calls),
    }


def report(cases: list[ConversationCase]) -> dict[str, Any]:
    """Evaluate a manually verified suite and aggregate its quality metrics."""

    rows = [evaluate_case(case) for case in cases]
    passed = sum(bool(row["passed"]) for row in rows)
    return {
        "metrics": {
            "case_count": len(rows),
            "pass_rate": passed / len(rows) if rows else 0.0,
            "turns_to_first_useful_intervention": [
                row["turns_to_first_useful_intervention"] for row in rows
            ],
            "max_clarification_streak": max(
                (int(row["max_clarification_streak"]) for row in rows), default=0
            ),
            "acknowledged_state_transition_rate": (
                sum(int(row["acknowledged_state_transitions"]) for row in rows)
                / sum(int(row["state_transitions"]) for row in rows)
                if sum(int(row["state_transitions"]) for row in rows)
                else 1.0
            ),
        },
        "failures": [row for row in rows if not row["passed"]],
        "rows": rows,
    }
