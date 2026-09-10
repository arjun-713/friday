# Friday's bounded agentic RAG contract

Friday uses a bounded agent loop rather than an unrestricted autonomous agent.
Initial hybrid retrieval remains the fast path. Luna receives that evidence and
may ask for a small amount of additional manufacturer evidence when the initial
context is not enough.

```text
request → diagnostic state → hybrid retrieval → Luna
                                      ├─ direct answer → stream
                                      └─ up to 2 read-only tools
                                            → one final streamed answer
```

The available tools are deliberately narrow:

| Tool | Purpose | Mutates state? |
| --- | --- | --- |
| `search_manual` | Search the selected manual for a symptom or component | No |
| `find_error_code` | Search for an exact reported code | No |
| `open_manual_page` | Inspect a page already present in retrieved evidence | No |
| `get_diagnostic_state` | Read known observations and completed actions | No |

The executor is the authority for tool behavior. The model supplies arguments,
but it cannot access the filesystem, network, shell, device, or database
directly. Evidence returned by a tool is merged into the final citation set and
the UI receives a `tool` SSE event so it can show what Friday is doing.

## Prompt ownership

All editable model instructions live in
[`backend/src/friday/prompts.py`](../backend/src/friday/prompts.py), including
the structured planner prompt, conversational prompt, tool follow-up message,
voice terminology hint, and tool schemas.

## Safety and latency limits

- Maximum two tool calls per turn.
- One final model call after tool execution.
- No recursive tool calls.
- Direct responses do not incur a second completion call.
- State changes are applied only by the session store after validation.
- Unsupported evidence produces an abstention instead of a guess.

The main local retrieval bottleneck is query embedding, not Qdrant search. Keep
the embedding provider warm and use the existing session retrieval cache before
adding more agent calls. Benchmark request-to-first-token, provider-first-token,
retrieval, embedding, tool, TTS, and completion phases independently.
