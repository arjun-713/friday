# Structure detection

The first chunking subphase is deterministic structure detection. `copilot.ingestion.chunking.structure_document` reads cleaned page records and produces section-aware lines without joining or rewriting content.

It recognizes explicit Markdown headings (`#` through `######`), maintains a heading stack across page boundaries, and assigns each line a section path such as `Troubleshooting > Wi-Fi > Signal`.

Pages marked `excluded_from_chunking` remain represented with their page number and exclusion reason but produce no structure lines. No heading is inferred from prose, font size, or an LLM in this subphase. That keeps section provenance reviewable before procedure and table chunking are introduced.
