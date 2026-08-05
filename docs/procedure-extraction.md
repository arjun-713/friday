# Procedure extraction

Procedure extraction is the next chunking subphase after section detection. `copilot.ingestion.chunking.extract_procedures` recognizes numeric ordered blocks and emits candidates only when at least two steps are present.

Each candidate preserves the original step order, one-based pages, section path, parser, and raw-source evidence. Recognized prerequisite lines and warning-marked lines are attached as evidence. Numbering resets split adjacent procedures, and isolated numbered lines are ignored.

This pass does not invent a procedure title, merge unrelated prose, call an LLM, or write an index. Candidates must be reviewed before becoming final `procedure` chunks.
