# Exact-match extraction

Exact-match extraction creates occurrence-level candidates for technical identifiers that benefit from lexical retrieval. Supported kinds are error codes, blink patterns, part numbers, and model numbers.

An identifier is emitted only when its line contains a matching context signal such as `error code`, `blinking pattern`, `spare part number`, or `model number`. Generic page numbers and uncontextualized numeric values are ignored. Each candidate retains the original value, normalized lookup value, section, page, parser, context, and raw-source evidence.

Candidates are not globally deduplicated. Repeated values may have different evidence and must remain citeable; a later indexer can deduplicate normalized values while retaining all evidence references.
