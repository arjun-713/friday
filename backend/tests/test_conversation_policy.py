from eval.conversation_policy import ConversationCase, evaluate_case, report


def test_conversation_policy_cases_pass() -> None:
    case = ConversationCase.from_json(
        {
            "case_id": "resolved-after-action",
            "initial_evidence_sufficient": True,
            "tool_calls": [],
            "expected_terminal_mode": "solve",
            "max_turns_to_first_useful_intervention": 1,
            "turns": [
                {
                    "turn_id": "turn-1",
                    "mode": "advance",
                    "response": "We need to distinguish an adapter connection issue from battery charging progress.",
                    "interpretation": None,
                    "next_action": {
                        "instruction": "Reconnect the adapter.",
                        "why": "This tests adapter detection before treating the battery as the fault.",
                    },
                    "observation_request": {
                        "request_id": "check-light",
                        "fact_key": "battery_light",
                        "question": "What does the light show?",
                        "options": [],
                        "recheck_after_action": False,
                    },
                    "decision_basis": {
                        "why_not_solved": "The initial report does not show whether the adapter is detected.",
                        "discriminates_between": ["adapter connection issue", "battery condition"],
                        "expected_discrimination": "The light state identifies whether the system reports charging.",
                    },
                    "facts_learned": [],
                    "candidate_causes": ["adapter connection issue", "battery condition"],
                    "ruled_out_causes": [],
                    "source_ids": ["manual-1"],
                },
                {
                    "turn_id": "turn-2",
                    "mode": "solve",
                    "response": "The white light confirms the adapter is now detected.",
                    "interpretation": None,
                    "next_action": None,
                    "observation_request": None,
                    "decision_basis": None,
                    "facts_learned": [
                        {
                            "key": "battery_light",
                            "value": "white",
                            "label": "Battery light",
                            "raw": "It is white.",
                        }
                    ],
                    "candidate_causes": [],
                    "ruled_out_causes": ["adapter connection issue"],
                    "source_ids": ["manual-1"],
                },
            ],
        }
    )

    result = report([case])

    assert result["metrics"]["pass_rate"] == 1.0
    assert result["rows"][0]["turns_to_first_useful_intervention"] == 1


def test_conversation_policy_flags_soft_questionnaire_advance() -> None:
    case = ConversationCase.from_json(
        {
            "case_id": "weak-advance",
            "initial_evidence_sufficient": True,
            "tool_calls": [],
            "expected_terminal_mode": "advance",
            "max_turns_to_first_useful_intervention": 1,
            "turns": [
                {
                    "turn_id": "turn-1",
                    "mode": "advance",
                    "response": "Check the adapter.",
                    "interpretation": None,
                    "next_action": {"instruction": "Check the adapter.", "why": None},
                    "observation_request": None,
                    "decision_basis": None,
                    "facts_learned": [],
                    "candidate_causes": [],
                    "ruled_out_causes": [],
                    "source_ids": ["manual-1"],
                }
            ],
        }
    )

    result = evaluate_case(case)

    assert not result["passed"]
    assert "advance has no diagnosis-specific reason" in " ".join(result["violations"])
    assert "advance does not distinguish plausible causes" in " ".join(result["violations"])
