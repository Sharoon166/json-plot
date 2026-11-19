"""Read the raw-data JSON files and turn every labeled box into something to draw on a PDF page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from json_plot.relationships import (
    aligned_rows,
    bbox_to_pdf,
    detect_kind,
    load_json,
    page_entries,
    page_number,
)


# (legend title, category key). Colours for categories live in json_plot.theme.
LABEL_LEGEND = (
    ("Tags", "tag"),
    ("Doors", "door"),
    ("Callouts", "callout"),
    ("Areas", "area"),
    ("Verified", "verified"),
    ("Missing", "missing"),
    ("Bad callout", "bad"),
    ("OCR", "ocr"),
)


def review_file(output_dir: Path, page: int) -> Path:
    return output_dir / "hitl_reviews" / f"page_{page:03d}" / "human_review.json"


def pdf_page_count(path: Path) -> int:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)


def pdf_page_size(path: Path, page: int) -> tuple[float, float]:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        sheet = pdf.pages[page - 1]
        return float(sheet.width), float(sheet.height)


def collect_page_items(
    json_paths: list[Path],
    page: int,
    page_width: float,
    page_height: float,
    include_shapes: bool = True,
) -> list[dict[str, Any]]:
    del include_shapes
    items: list[dict[str, Any]] = []
    for path in json_paths:
        if not path.is_file():
            continue
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if detect_kind(data) == "ocr":
            items.extend(_ocr_items(data, path, page, page_width, page_height))
        else:
            items.extend(_page_boxes(data, path, page, page_width, page_height))
    return items


def load_review(path: Path, page: int, items: list[dict[str, Any]], page_size: list[float]) -> dict[str, Any]:
    review: dict[str, Any] = {"schema_version": 1, "page_number": page, "annotations": {}, "missed_annotations": {}}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            review.update({key: loaded.get(key, review[key]) for key in ("annotations", "missed_annotations")})
    annotations = review["annotations"] if isinstance(review.get("annotations"), dict) else {}
    for item in items:
        entry = annotations.get(item["id"])
        if not isinstance(entry, dict):
            entry = {}
        entry.setdefault("decision", "pending")
        entry.setdefault("class_name", item["class_name"])
        entry.setdefault("corrected_label", None)
        entry.setdefault("bbox_pdf", item["bbox_pdf"])
        entry.setdefault("corrected_bbox_pdf", None)
        entry.setdefault("recognized_text", item.get("ocr_text") or "")
        entry.setdefault("comment", "")
        annotations[item["id"]] = entry
    review["annotations"] = annotations
    review["page_number"] = page
    review["pdf_page_size_points"] = page_size
    if not isinstance(review.get("missed_annotations"), dict):
        review["missed_annotations"] = {}
    return review


def classify_record(record_name: str) -> tuple[str, str]:
    name = record_name.strip()
    low = name.lower()
    if low == "ocr":
        return "ocr", "OCR"
    if low.startswith("tag"):
        short = name.split(" - ", 1)[0]
        return "tag", short
    if "door" in low:
        return "door", "Door"
    if "callout" in low and "incorrect" not in low:
        return "callout", "Callout"
    if any(word in low for word in ("plan area", "schedule area", "legend")):
        return "area", name
    if "missing" in low:
        return "missing", "Missing"
    if low.startswith("incorrect"):
        return "bad", "Bad callout"
    if "verified" in low:
        return "verified", "Verified"
    return "other", name.replace("_", " ")


def _page_boxes(data: dict[str, Any], path: Path, page: int, width: float, height: float) -> list[dict[str, Any]]:
    items = []
    for key, sheet in page_entries(data):
        if page_number(key) != page or not isinstance(sheet, dict):
            continue
        for record_name, record in sheet.items():
            if not isinstance(record, dict):
                continue
            family, title = classify_record(str(record_name))
            for index, (text, box, _target) in enumerate(aligned_rows(record)):
                if box is None:
                    continue
                items.append(_item(
                    path, str(record_name), index, text, box, width, height, family, title, page_key=str(key),
                ))
    return [item for item in items if item is not None]


def _ocr_items(data: dict[str, Any], path: Path, page: int, width: float, height: float) -> list[dict[str, Any]]:
    items = []
    family, title = classify_record("ocr")
    for key, words in page_entries(data):
        if page_number(key) != page or not isinstance(words, list):
            continue
        for index, word in enumerate(words):
            if not isinstance(word, dict):
                continue
            box = word.get("bbox")
            if not isinstance(box, list):
                continue
            items.append(_item(
                path, "ocr", index, str(word.get("text") or ""), box, width, height, family, title, page_key=str(key),
            ))
    return [item for item in items if item is not None]


def _item(
    path: Path,
    kind: str,
    index: int,
    text: str,
    box: list[float],
    width: float,
    height: float,
    family: str,
    title: str,
    page_key: str = "",
) -> dict[str, Any] | None:
    try:
        left, top, right, bottom = bbox_to_pdf(box, width, height)
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return {
        "id": f"{path.name}:{page_key}:{kind}:{index}",
        "class_name": family,
        "family": family,
        "family_label": title,
        "record": kind,
        "ocr_text": text,
        "ocr_confidence": None,
        "confidence": 1.0,
        "from_json": True,
        "bbox_pdf": [round(left, 2), round(top, 2), round(right, 2), round(bottom, 2)],
        "source_bbox": [float(value) for value in box],
        "source": path.name,
        "source_path": str(path),
        "page_key": page_key,
        "row_index": index,
    }
