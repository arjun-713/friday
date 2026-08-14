# CPU embedding acceleration

Granite uses four intra-op CPU threads and one inter-op thread by default. The local i5-8350U benchmark selected the AVX2 INT8 ONNX artifact; generate it once after creating the backend environment:

```bash
make export-embedding
```

Run retrieval with the optimized artifact:

```bash
export EMBEDDING_BACKEND=onnx
export EMBEDDING_MODEL=data/models/granite-small-r2-onnx
export EMBEDDING_MODEL_FILE=onnx/model_int8-avx2.onnx
export EMBEDDING_CPU_THREADS=4
export EMBEDDING_INTEROP_THREADS=1
make benchmark-retrieval
```

The exported model is machine-generated and intentionally excluded from Git. Do not silently use the INT8 artifact on a CPU without AVX2 support. PyTorch remains the portable fallback when these variables are absent.

Use `make benchmark-embedding` for a repeatable PyTorch baseline. The benchmark reports cold start and P50/P70/P99/P100/max warm-query latency. Retrieval benchmarking additionally reports embedding, lexical, dense search, fusion, parent fetch, total, cache-hit, and confirmed-device scoped latency.

The proxy Recall@5/MRR labels generated from chunk prefixes are smoke checks, not a substitute for the manually verified retrieval benchmark required before production.
