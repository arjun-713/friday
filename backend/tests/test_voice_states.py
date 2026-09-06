from friday.voice.bridge import VoiceSessionState, classify_interruption


def test_voice_session_states_cover_full_duplex_model() -> None:
    assert {state.value for state in VoiceSessionState} == {
        "idle",
        "listening",
        "thinking",
        "speaking",
        "interrupting",
        "recovering",
        "error",
    }


def test_interruption_classification_covers_required_kinds() -> None:
    assert classify_interruption("Actually it is a USB cable") == "correction"
    assert classify_interruption("The light is now blinking amber") == "new_information"
    assert classify_interruption("What does that light mean?") == "clarification"
    assert classify_interruption("Stop") == "stop"
    assert classify_interruption("Repeat that please") == "repeat"
    assert classify_interruption("Got it, thanks") == "acknowledgement"
