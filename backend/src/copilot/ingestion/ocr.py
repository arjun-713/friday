"""PP-OCRv6 CPU adapter for pages routed to OCR."""

import json
from pathlib import Path
from typing import Any

from .models import OcrPage, OcrTextItem


class PaddleOcrUnavailable(RuntimeError):
    """Raised when PaddleOCR is not installed."""


def _paddle_ocr() -> Any:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise PaddleOcrUnavailable(
            "Install the OCR extra with: pip install -e '.[ocr]'"
        ) from exc
    return PaddleOCR


class PpOcrAdapter:
    """Lazy, CPU-only PP-OCRv6 adapter with small/medium model tiers."""

    _models = {
        "small": ("PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
        "medium": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
    }

    def __init__(self, tier: str = "small", device: str = "cpu", engine: str = "onnxruntime") -> None:
        if tier not in self._models:
            raise ValueError(f"Unsupported PP-OCRv6 tier: {tier}")
        self.tier = tier
        self.device = device
        self.engine = engine
        self._pipeline: Any | None = None

    @property
    def model_name(self) -> str:
        return f"PP-OCRv6-{self.tier}"

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            PaddleOCR = _paddle_ocr()
            det_model, rec_model = self._models[self.tier]
            self._pipeline = PaddleOCR(
                text_detection_model_name=det_model,
                text_recognition_model_name=rec_model,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=self.device,
                engine=self.engine,
                enable_mkldnn=False,
                cpu_threads=2,
            )
        return self._pipeline

    @staticmethod
    def _result_data(result: Any) -> dict[str, Any]:
        data = getattr(result, "json", None)
        if callable(data):
            data = data()
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict) and "res" in data and isinstance(data["res"], dict):
            return data["res"]
        if isinstance(data, dict):
            return data
        if hasattr(result, "to_json"):
            data = result.to_json()
            if isinstance(data, str):
                data = json.loads(data)
            return data.get("res", data)
        raise TypeError("Unsupported PaddleOCR result object")

    def recognize(self, image_path: str | Path, page: int) -> OcrPage:
        result = next(iter(self._get_pipeline().predict(str(image_path))))
        data = self._result_data(result)
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        polygons = data.get("rec_polys", data.get("dt_polys", []))
        items = [
            OcrTextItem(
                text=str(text),
                confidence=float(score),
                page=page,
                polygon=[(float(point[0]), float(point[1])) for point in polygon],
            )
            for text, score, polygon in zip(texts, scores, polygons)
            if str(text).strip()
        ]
        return OcrPage(
            page=page,
            model=self.model_name,
            image_path=str(image_path),
            text="\n".join(item.text for item in items),
            confidence=(sum(item.confidence for item in items) / len(items)) if items else 0.0,
            items=items,
        )
