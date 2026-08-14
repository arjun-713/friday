"""Compare an accelerated Granite artifact with the PyTorch reference."""

import argparse
import asyncio
import json
import math

from .embedding_benchmark import QUERIES
from .granite import GraniteEmbeddingProvider, GraniteEmbeddingSettings


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    return numerator / math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))


async def compare(model: str, model_file: str) -> dict[str, object]:
    reference = GraniteEmbeddingProvider(GraniteEmbeddingSettings(cpu_threads=4))
    candidate = GraniteEmbeddingProvider(
        GraniteEmbeddingSettings(model_name=model, backend="onnx", model_file=model_file, cpu_threads=4)
    )
    reference_vectors = await reference.embed_documents(QUERIES)
    candidate_vectors = await candidate.embed_documents(QUERIES)
    similarities = [cosine(left, right) for left, right in zip(reference_vectors, candidate_vectors, strict=True)]
    return {
        "query_count": len(QUERIES),
        "minimum_cosine_similarity": min(similarities),
        "mean_cosine_similarity": sum(similarities) / len(similarities),
        "cosine_similarities": similarities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-file", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(compare(args.model, args.model_file)), indent=2))


if __name__ == "__main__":
    main()
