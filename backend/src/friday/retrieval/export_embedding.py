"""Export optimized Granite embedding artifacts for this CPU."""

import argparse
from pathlib import Path


def export_onnx(model_name: str, output: Path) -> None:
    from sentence_transformers import (
        SentenceTransformer,
        export_dynamic_quantized_onnx_model,
        export_optimized_onnx_model,
    )

    output.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, backend="onnx")
    model.save_pretrained(output)
    export_optimized_onnx_model(model, "O3", str(output), file_suffix="o3")
    export_dynamic_quantized_onnx_model(model, "avx2", str(output), file_suffix="int8-avx2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ibm-granite/granite-embedding-small-english-r2")
    parser.add_argument("--output", type=Path, default=Path("data/models/granite-small-r2-onnx"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_onnx(args.model, args.output)
    print(f"exported optimized ONNX models to {args.output}")


if __name__ == "__main__":
    main()
