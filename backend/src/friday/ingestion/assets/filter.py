"""Deterministic usefulness scoring for extracted manual images."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


def classify_image(data: bytes) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            gray = np.asarray(rgb.convert("L"), dtype=np.float32)
            alpha = np.asarray(image.getchannel("A"), dtype=np.float32) if "A" in image.getbands() else None
    except (OSError, UnidentifiedImageError):
        return {
            "width": 0,
            "height": 0,
            "entropy": 0.0,
            "edge_density": 0.0,
            "border_edge_density": 0.0,
            "center_edge_density": 0.0,
            "near_uniform_fraction": 1.0,
            "component_count": 0,
            "alpha_coverage": 0.0,
            "perceptual_hash": None,
            "quality_score": -10,
            "classification": "reject",
            "reasons": ["invalid_image_payload"],
        }
    features = _features(gray, width, height, alpha)
    score = 0
    reasons: list[str] = []
    if width < 80 or height < 80:
        score -= 2
        reasons.append("too_small")
    if width < 100 or height < 100:
        score -= 5
        reasons.append("below_displayable_size")
    if width * height < 20_000:
        score -= 4
        reasons.append("low_pixel_area")
    if features["entropy"] < 2.0:
        score -= 2
        reasons.append("low_entropy")
    if features["edge_density"] < 0.01:
        score -= 2
        reasons.append("low_edge_density")
    if features["blank_center"] and features["strong_border"]:
        score -= 4
        reasons.append("hollow_rectangle")
    if features["center_edge_density"] < 0.005 and features["border_edge_density"] > max(
        features["center_edge_density"] * 5, 0.04
    ):
        score -= 3
        reasons.append("border_dominant_edges")
    if features["near_uniform_fraction"] >= 0.98:
        score -= 2
        reasons.append("near_uniform_pixels")
    if features["component_count"] <= 2:
        score -= 1
        reasons.append("few_components")
    if features["alpha_coverage"] < 0.08:
        score -= 1
        reasons.append("mostly_transparent")
    if features["edge_density"] >= 0.04:
        score += 1
        reasons.append("structured_edges")
    if features["entropy"] >= 3.0:
        score += 1
        reasons.append("pixel_variation")
    if width / max(height, 1) > 8 or height / max(width, 1) > 8:
        score -= 1
        reasons.append("extreme_aspect_ratio")
    if width / max(height, 1) > 6 or height / max(width, 1) > 6:
        score -= 4
        reasons.append("icon_like_aspect_ratio")
    if features["near_uniform_fraction"] >= 0.9 and features["component_count"] <= 4:
        score -= 4
        reasons.append("sparse_symbolic_asset")
    if score >= 2:
        classification = "valid"
    elif score >= 0:
        classification = "uncertain"
    else:
        classification = "reject"
    return {
        "width": width,
        "height": height,
        "entropy": round(features["entropy"], 4),
        "edge_density": round(features["edge_density"], 6),
        "border_edge_density": round(features["border_edge_density"], 6),
        "center_edge_density": round(features["center_edge_density"], 6),
        "near_uniform_fraction": round(features["near_uniform_fraction"], 6),
        "component_count": features["component_count"],
        "alpha_coverage": round(features["alpha_coverage"], 6),
        "perceptual_hash": features["perceptual_hash"],
        "quality_score": score,
        "classification": classification,
        "reasons": reasons,
    }


def mark_perceptual_duplicates(assets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark near-duplicates deterministically, retaining the first hash-sorted asset."""

    seen: dict[str, str] = {}
    rejected: list[dict[str, Any]] = []
    for asset_id in sorted(assets):
        asset = assets[asset_id]
        perceptual_hash = asset["features"]["perceptual_hash"]
        duplicate_of = seen.get(perceptual_hash)
        if duplicate_of is not None:
            asset["features"]["classification"] = "reject"
            asset["features"]["quality_score"] -= 2
            asset["features"]["reasons"].append("duplicate")
            asset["duplicate_of"] = duplicate_of
            rejected.append(asset)
        else:
            seen[perceptual_hash] = asset_id
    return rejected


def _features(gray: np.ndarray, width: int, height: int, alpha: np.ndarray | None) -> dict[str, Any]:
    histogram = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / max(histogram.sum(), 1)
    entropy = float(-(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])).sum())
    edge_density = _edge_density(gray)
    margin = max(1, int(min(width, height) * 0.1))
    center = gray
    if height > margin * 2 and width > margin * 2:
        center = gray[margin : height - margin, margin : width - margin]
    border = np.concatenate(
        [gray[:margin, :].ravel(), gray[-margin:, :].ravel(), gray[:, :margin].ravel(), gray[:, -margin:].ravel()]
    )
    center_gradient = _gradient_mean(center)
    border_gradient = _gradient_mean(border)
    blank_center = float(center.std()) < 8 and center_gradient < 3
    strong_border = border_gradient > max(center_gradient * 2.5, 8)
    border_edge_density = max(
        _edge_density(gray[:margin, :]),
        _edge_density(gray[-margin:, :]),
        _edge_density(gray[:, :margin]),
        _edge_density(gray[:, -margin:]),
    )
    center_edge_density = _edge_density(center)
    near_uniform_fraction = float((np.abs(gray - np.median(gray)) <= 3).mean())
    return {
        "entropy": entropy,
        "edge_density": edge_density,
        "border_edge_density": border_edge_density,
        "center_edge_density": center_edge_density,
        "blank_center": blank_center,
        "strong_border": strong_border,
        "near_uniform_fraction": near_uniform_fraction,
        "component_count": _component_count(gray),
        "alpha_coverage": float((alpha > 5).mean()) if alpha is not None else 1.0,
        "perceptual_hash": _dhash(gray),
    }


def _edge_density(gray: np.ndarray) -> float:
    if gray.size == 0:
        return 0.0
    densities: list[float] = []
    if gray.shape[1] >= 2:
        densities.append(float((np.abs(np.diff(gray, axis=1)) > 20).mean()))
    if gray.shape[0] >= 2:
        densities.append(float((np.abs(np.diff(gray, axis=0)) > 20).mean()))
    return sum(densities) / len(densities) if densities else 0.0


def _gradient_mean(values: np.ndarray) -> float:
    """Return a finite gradient score for normal and degenerate image regions."""

    if values.size < 2:
        return 0.0
    if values.ndim == 1:
        return float(np.abs(np.diff(values)).mean())
    gradients: list[np.ndarray] = []
    if values.shape[0] >= 2:
        gradients.append(np.diff(values, axis=0))
    if values.shape[1] >= 2:
        gradients.append(np.diff(values, axis=1))
    return float(np.concatenate([gradient.ravel() for gradient in gradients]).__abs__().mean())


def _component_count(gray: np.ndarray) -> int:
    """Count coarse connected edge regions without requiring OpenCV."""

    small = Image.fromarray(gray.astype(np.uint8)).resize((64, 64))
    pixels = np.asarray(small, dtype=np.float32)
    edges = np.zeros_like(pixels, dtype=bool)
    edges[:, 1:] |= np.abs(np.diff(pixels, axis=1)) > 20
    edges[1:, :] |= np.abs(np.diff(pixels, axis=0)) > 20
    visited = np.zeros_like(edges, dtype=bool)
    components = 0
    for row, column in zip(*np.where(edges), strict=True):
        if visited[row, column]:
            continue
        components += 1
        stack = [(int(row), int(column))]
        visited[row, column] = True
        while stack:
            current_row, current_column = stack.pop()
            for next_row in range(max(0, current_row - 1), min(64, current_row + 2)):
                for next_column in range(max(0, current_column - 1), min(64, current_column + 2)):
                    if edges[next_row, next_column] and not visited[next_row, next_column]:
                        visited[next_row, next_column] = True
                        stack.append((next_row, next_column))
    return components


def _dhash(gray: np.ndarray) -> str:
    image = Image.fromarray(gray.astype(np.uint8)).resize((9, 8))
    pixels = np.asarray(image, dtype=np.uint8)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"
