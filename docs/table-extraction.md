# Table extraction

Table extraction is a deterministic chunking subphase. `copilot.ingestion.chunking.extract_table_rows` recognizes only pipe-delimited Markdown tables with an explicit header and separator row.

Each data row retains its table identifier, row index, headers, cells, normalized content, section, parser, page, and raw-source evidence. Unstructured pipe text and tables on excluded pages are ignored.

This pass does not infer table structure from prose or convert arbitrary layouts. Exact error-code and blink-pattern extraction remains a separate subphase because those formats are not uniform across manufacturers.
