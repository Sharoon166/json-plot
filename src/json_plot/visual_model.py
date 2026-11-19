import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pdfplumber
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from json_plot.config import (
    ANNOTATION_PDF_FILE,
    CLASS_NAMES,
    DATASET_DIR,
    DEFAULT_WEIGHTS,
    LEGACY_REVIEW_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REFERENCE_DATA_FILE,
    REVIEW_DIR,
    SCHEDULE_DATA_FILE,
    SCHEDULE_OUTPUT_FILE,
    TRAINING_DIR,
    TRAINING_PDF_FILE,
)


CANVAS_WIDTH = 2592.0
CANVAS_HEIGHT = 1728.0
TRAINING_PAGE_WIDTH = 5400.0
TRAINING_PAGE_HEIGHT = 3600.0
RENDER_DPI = 36
TILE_SIZE = 1024
TILE_OVERLAP = 192
VALID_DECISIONS = {"pending", "accepted", "rejected"}
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
ANNOTATION_LABELS = {
    "Door Reference": "door_reference",
    "Callout Area": "callout_area",
}
SCHEDULE_OUTPUT_LABELS = {
    "verified_sched_entry": "schedule_entry",
    "missing_sched_entry": "schedule_entry",
    "verified_tag": "tag",
    "missing_from_schedule": "tag",
}
REFERENCE_SOURCE_LABELS = {
    "verified",
    "verified_from_index",
    "verified_sheets",
    "incorrect_reference_detail",
    "missing_from_index",
    "reference_missing",
}
VALIDATION_PAGES = {"2", "8", "11", "14"}
DECISION_REWARDS = {"accepted": 1, "rejected": -1}
BOX_COLORS = {
    "pending": (238, 166, 25),
    "accepted": (28, 155, 94),
    "rejected": (215, 65, 65),
}
MISSED_COLOR = (205, 75, 190)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def normalize_boxes(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        return []
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        candidates = [value]
    else:
        candidates = value

    boxes = []
    for candidate in candidates:
        if not isinstance(candidate, (list, tuple)) or len(candidate) != 4:
            continue
        try:
            x0, y0, x1, y1 = map(float, candidate)
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in (x0, y0, x1, y1)) and x1 > x0 and y1 > y0:
            boxes.append([x0, y0, x1, y1])
    return boxes


def review_box_to_canvas(bbox: list[float], review: dict[str, Any]) -> list[float]:
    page_size = review.get(
        "pdf_page_size_points",
        [TRAINING_PAGE_WIDTH, TRAINING_PAGE_HEIGHT],
    )
    if not isinstance(page_size, list) or len(page_size) != 2:
        raise ValueError("Review pdf_page_size_points must contain page width and height.")
    page_width, page_height = map(float, page_size)
    if page_width <= 0 or page_height <= 0:
        raise ValueError("Review page dimensions must be positive.")
    return [
        bbox[0] * CANVAS_WIDTH / page_width,
        bbox[1] * CANVAS_HEIGHT / page_height,
        bbox[2] * CANVAS_WIDTH / page_width,
        bbox[3] * CANVAS_HEIGHT / page_height,
    ]


def add_box(
    boxes: dict[str, list[dict[str, Any]]],
    seen: set[tuple[str, tuple[float, ...]]],
    page_id: str,
    label: str,
    bbox: list[float],
    source: str,
) -> None:
    if label not in CLASS_TO_ID:
        return
    key = (page_id, tuple(round(value, 3) for value in bbox))
    if key in seen:
        return
    seen.add(key)
    boxes[page_id].append({"class_name": label, "bbox_json": bbox, "source": source})


def read_source_boxes() -> dict[str, list[dict[str, Any]]]:
    boxes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, tuple[float, ...]]] = set()

    schedule_data = load_json(SCHEDULE_DATA_FILE)
    for page_id, page in schedule_data.items():
        if not isinstance(page, dict):
            continue
        for source_label, record in page.items():
            if not isinstance(record, dict):
                continue
            if source_label.startswith("Tag "):
                target_label = "tag"
            else:
                target_label = ANNOTATION_LABELS.get(source_label)
            if target_label is None:
                continue
            for bbox in normalize_boxes(record.get("bboxes")):
                add_box(boxes, seen, page_id, target_label, bbox, "02_schedule_data")

    schedule_outputs = load_json(SCHEDULE_OUTPUT_FILE)
    for page_id, page in schedule_outputs.items():
        if not isinstance(page, dict):
            continue
        for source_label, target_label in SCHEDULE_OUTPUT_LABELS.items():
            record = page.get(source_label)
            if not isinstance(record, dict):
                continue
            for bbox in normalize_boxes(record.get("bbox")):
                add_box(boxes, seen, page_id, target_label, bbox, "03_schedule_outputs")

    reference_data = load_json(REFERENCE_DATA_FILE)
    for page_id, page in reference_data.items():
        if not isinstance(page, dict):
            continue
        for source_label, record in page.items():
            if source_label not in REFERENCE_SOURCE_LABELS or not isinstance(record, dict):
                continue
            for bbox in normalize_boxes(record.get("bbox")):
                add_box(boxes, seen, page_id, "reference", bbox, "04_data_out")

    return boxes


def feedback_files(root: Path | None = None) -> list[Path]:
    roots = (REVIEW_DIR, LEGACY_REVIEW_DIR) if root is None else (root,)
    return sorted({path for item in roots for path in item.glob("page_*/human_review.json")})


def reviewed_training_pages(review_root: Path | None = None) -> set[str]:
    reviewed = set()
    for path in feedback_files(review_root):
        review = load_json(path)
        annotations = review.get("annotations", {})
        if any(
            isinstance(entry, dict) and entry.get("decision") in DECISION_REWARDS
            for entry in annotations.values()
        ) or review.get("missed_annotations"):
            reviewed.add(str(review.get("page_number", 0) - 1))
    return reviewed


def read_feedback_boxes(
    review_root: Path | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    additions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rewards: Counter[str] = Counter()

    for review_path in feedback_files(review_root):
        review = load_json(review_path)
        page_number = review.get("page_number")
        if not isinstance(page_number, int):
            continue
        page_id = str(page_number - 1)
        annotations = review.get("annotations", {})
        if not isinstance(annotations, dict):
            continue

        for detection_id, entry in annotations.items():
            if not isinstance(entry, dict):
                continue
            decision = entry.get("decision", "pending")
            class_name = entry.get("corrected_label") or entry.get("class_name")
            if decision in DECISION_REWARDS and class_name in CLASS_TO_ID:
                rewards[class_name] += DECISION_REWARDS[decision]
                rewards[f"{class_name}_{decision}"] += 1
            if decision != "accepted" or class_name not in CLASS_TO_ID:
                continue
            bbox = normalize_boxes(entry.get("corrected_bbox_pdf"))
            if not bbox:
                bbox = normalize_boxes(entry.get("bbox_pdf"))
            if not bbox:
                continue
            additions[page_id].append({
                "class_name": class_name,
                "bbox_json": review_box_to_canvas(bbox[0], review),
                "source": f"human_review:{detection_id}",
            })

        missed_annotations = review.get("missed_annotations", {})
        if isinstance(missed_annotations, dict):
            for missed_id, entry in missed_annotations.items():
                if not isinstance(entry, dict):
                    continue
                class_name = entry.get("class_name")
                bbox = normalize_boxes(entry.get("bbox_pdf"))
                if class_name not in CLASS_TO_ID or not bbox:
                    continue
                rewards[class_name] -= 1
                rewards[f"{class_name}_missed"] += 1
                additions[page_id].append({
                    "class_name": class_name,
                    "bbox_json": review_box_to_canvas(bbox[0], review),
                    "source": f"human_missed:{missed_id}",
                })

    return additions, rewards


def iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def apply_negative_feedback(
    page_boxes: list[dict[str, Any]],
    page_id: str,
    review_root: Path | None = None,
) -> list[dict[str, Any]]:
    rejected: list[tuple[str, list[float]]] = []
    for review_path in feedback_files(review_root):
        review = load_json(review_path)
        if str(review.get("page_number", 0) - 1) != page_id:
            continue
        for entry in review.get("annotations", {}).values():
            if not isinstance(entry, dict):
                continue
            decision = entry.get("decision")
            label = entry.get("class_name")
            bbox = normalize_boxes(entry.get("bbox_pdf"))
            corrected_bbox = normalize_boxes(entry.get("corrected_bbox_pdf"))
            corrected_label = entry.get("corrected_label")
            needs_replacement = (
                decision == "accepted"
                and (corrected_label not in (None, label) or bool(corrected_bbox))
            )
            if label in CLASS_TO_ID and bbox and (decision == "rejected" or needs_replacement):
                rejected.append((label, review_box_to_canvas(bbox[0], review)))

    kept = []
    for item in page_boxes:
        rejected_match = any(
            item["class_name"] == label and iou(item["bbox_json"], box) >= 0.5
            for label, box in rejected
        )
        if not rejected_match:
            kept.append(item)
    return kept


def tile_starts(length: int) -> list[int]:
    if length <= TILE_SIZE:
        return [0]
    stride = TILE_SIZE - TILE_OVERLAP
    starts = list(range(0, length - TILE_SIZE + 1, stride))
    final_start = length - TILE_SIZE
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def export_dataset() -> dict[str, Any]:
    if not TRAINING_PDF_FILE.exists():
        raise FileNotFoundError(f"Annotated training PDF not found: {TRAINING_PDF_FILE}")

    page_boxes = read_source_boxes()
    feedback_additions, reward_counts = read_feedback_boxes()
    for page_id, additions in feedback_additions.items():
        page_boxes[page_id].extend(additions)

    reviewed_page_ids = reviewed_training_pages()

    counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    image_root = DATASET_DIR / "images"
    label_root = DATASET_DIR / "labels"

    for root in (image_root, label_root):
        for split in ("train", "val"):
            split_dir = root / split
            if split_dir.exists():
                for old_file in split_dir.glob("page_*"):
                    if old_file.is_file():
                        old_file.unlink()

    with pdfplumber.open(TRAINING_PDF_FILE) as pdf:
        if len(pdf.pages) != 15:
            raise ValueError(f"Expected 15 pages in the annotated PDF, found {len(pdf.pages)}.")

        for page_index, page in enumerate(pdf.pages):
            page_id = str(page_index)
            page_split = (
                "val"
                if page_id in VALIDATION_PAGES and page_id not in reviewed_page_ids
                else "train"
            )
            image = page.to_image(resolution=RENDER_DPI).original.convert("RGB")
            scale_x = image.width / CANVAS_WIDTH
            scale_y = image.height / CANVAS_HEIGHT
            source_for_page = apply_negative_feedback(
                page_boxes.get(page_id, []),
                page_id,
            )
            pixel_boxes = []
            for item in source_for_page:
                x0, y0, x1, y1 = item["bbox_json"]
                box = [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]
                box[0] = min(max(box[0], 0), image.width)
                box[1] = min(max(box[1], 0), image.height)
                box[2] = min(max(box[2], 0), image.width)
                box[3] = min(max(box[3], 0), image.height)
                if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                    pixel_boxes.append({**item, "bbox_pixels": box})
                    counts[item["class_name"]] += 1

            xs = tile_starts(image.width)
            ys = tile_starts(image.height)
            for top in ys:
                for left in xs:
                    right = min(left + TILE_SIZE, image.width)
                    bottom = min(top + TILE_SIZE, image.height)
                    tile_name = f"page_{page_index:03d}_x{left:04d}_y{top:04d}"
                    image_path = image_root / page_split / f"{tile_name}.jpg"
                    label_path = label_root / page_split / f"{tile_name}.txt"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    image.crop((left, top, right, bottom)).save(
                        image_path,
                        format="JPEG",
                        quality=92,
                    )

                    labels = []
                    for item in pixel_boxes:
                        x0, y0, x1, y1 = item["bbox_pixels"]
                        center_x = (x0 + x1) / 2
                        center_y = (y0 + y1) / 2
                        if not (left <= center_x < right and top <= center_y < bottom):
                            continue
                        tile_x0 = max(x0, left) - left
                        tile_y0 = max(y0, top) - top
                        tile_x1 = min(x1, right) - left
                        tile_y1 = min(y1, bottom) - top
                        width = tile_x1 - tile_x0
                        height = tile_y1 - tile_y0
                        if width <= 0 or height <= 0:
                            continue
                        center_x = (tile_x0 + tile_x1) / 2 / (right - left)
                        center_y = (tile_y0 + tile_y1) / 2 / (bottom - top)
                        labels.append(
                            f"{CLASS_TO_ID[item['class_name']]} "
                            f"{center_x:.6f} {center_y:.6f} "
                            f"{width / (right - left):.6f} "
                            f"{height / (bottom - top):.6f}"
                        )
                        label_counts[item["class_name"]] += 1

                    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                    image_counts[page_split] += 1

    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_path.write_text(
        "path: " + json.dumps(str(DATASET_DIR)) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {index}: {json.dumps(name)}\n" for index, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_pdf": str(TRAINING_PDF_FILE),
        "coordinate_transform": {
            "json_canvas_width": CANVAS_WIDTH,
            "json_canvas_height": CANVAS_HEIGHT,
            "pdf_page_width_points": TRAINING_PAGE_WIDTH,
            "pdf_page_height_points": TRAINING_PAGE_HEIGHT,
        },
        "render_dpi": RENDER_DPI,
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "validation_page_ids_zero_based": sorted(VALIDATION_PAGES),
        "classes": CLASS_NAMES,
        "source_box_counts": dict(counts),
        "tile_label_counts": dict(label_counts),
        "tile_image_counts": dict(image_counts),
        "reward_feedback_counts": dict(reward_counts),
        "human_reward": "accepted=+1; rejected=-1; accepted corrections are added as training labels; rejected boxes remove overlapping source labels.",
    }
    save_json(DATASET_DIR / "manifest.json", manifest)
    return manifest


def train_detector(args: argparse.Namespace) -> None:
    manifest = export_dataset()
    model = YOLO(args.weights)
    results = model.train(
        data=str(DATASET_DIR / "dataset.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=0,
        amp=False,
        cache=False,
        plots=False,
        patience=0,
        project=str(TRAINING_DIR),
        name="detector",
        exist_ok=True,
        verbose=True,
    )
    best_weights = TRAINING_DIR / "detector" / "weights" / "best.pt"
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "best_weights": str(best_weights),
        "epochs_requested": args.epochs,
        "image_size": args.imgsz,
        "device": "cpu",
        "training_examples": manifest["tile_image_counts"],
        "human_reward_feedback": manifest["reward_feedback_counts"],
        "reward_calibration": collect_reward_statistics(),
        "metrics": getattr(model.trainer, "metrics", {}),
    }
    save_json(TRAINING_DIR / "training_summary.json", summary)
    print(f"Training finished. Best weights: {best_weights}")


def detection_id(page_number: int, class_name: str, bbox_pdf: list[float]) -> str:
    quantized = ",".join(f"{coordinate:.1f}" for coordinate in bbox_pdf)
    raw = f"{page_number}|{class_name}|{quantized}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"p{page_number:03d}-{digest}"


def run_nms(detections: list[dict[str, Any]], threshold: float = 0.5) -> list[dict[str, Any]]:
    kept = []
    for class_name in CLASS_NAMES:
        candidates = sorted(
            (item for item in detections if item["class_name"] == class_name),
            key=lambda item: item["confidence"],
            reverse=True,
        )
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [
                item for item in candidates
                if iou(item["bbox_pixels"], best["bbox_pixels"]) < threshold
            ]
    return kept


def collect_model_detections(
    image: Image.Image,
    page_number: int,
    scale_x: float,
    scale_y: float,
    model: YOLO,
    confidence_threshold: float,
    image_size: int,
) -> list[dict[str, Any]]:
    detections = []
    for top in tile_starts(image.height):
        for left in tile_starts(image.width):
            right = min(left + TILE_SIZE, image.width)
            bottom = min(top + TILE_SIZE, image.height)
            tile = image.crop((left, top, right, bottom))
            result = model.predict(
                source=tile,
                imgsz=image_size,
                conf=confidence_threshold,
                device="cpu",
                verbose=False,
            )[0]
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = model.names[class_id]
                x0, y0, x1, y1 = map(float, box.xyxy[0].tolist())
                bbox_pixels = [x0 + left, y0 + top, x1 + left, y1 + top]
                bbox_pdf = [
                    round(bbox_pixels[0] / scale_x, 2),
                    round(bbox_pixels[1] / scale_y, 2),
                    round(bbox_pixels[2] / scale_x, 2),
                    round(bbox_pixels[3] / scale_y, 2),
                ]
                detections.append({
                    "id": detection_id(page_number, class_name, bbox_pdf),
                    "class_name": class_name,
                    "confidence": float(box.conf.item()),
                    "bbox_pdf": bbox_pdf,
                    "bbox_pixels": [round(value, 2) for value in bbox_pixels],
                })
    return run_nms(detections)


def recognize_page_text(
    image: Image.Image,
    tesseract_executable: str,
    scale_x: float,
    scale_y: float,
) -> list[dict[str, Any]]:
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    result = subprocess.run(
        [tesseract_executable, "stdin", "stdout", "--psm", "11", "tsv"],
        input=image_buffer.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    words = []
    rows = csv.DictReader(
        io.StringIO(result.stdout.decode("utf-8", errors="replace")),
        delimiter="\t",
    )
    for row in rows:
        if row.get("level") != "5":
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row["conf"])
            left = int(row["left"])
            top = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if confidence < 0 or width <= 0 or height <= 0:
            continue
        words.append({
            "text": text,
            "confidence": round(confidence / 100, 6),
            "bbox_pixels": [left, top, left + width, top + height],
            "bbox_pdf": [
                round(left / scale_x, 2),
                round(top / scale_y, 2),
                round((left + width) / scale_x, 2),
                round((top + height) / scale_y, 2),
            ],
        })
    return words


def attach_ocr_to_detections(
    detections: list[dict[str, Any]],
    words: list[dict[str, Any]],
) -> None:
    for detection in detections:
        x0, y0, x1, y1 = detection["bbox_pixels"]
        matched_words = []
        if detection["class_name"] != "callout_area":
            pad_x = max(8, (x1 - x0) * 0.35)
            pad_y = max(8, (y1 - y0) * 0.5)
            for word in words:
                word_x0, word_y0, word_x1, word_y1 = word["bbox_pixels"]
                center_x = (word_x0 + word_x1) / 2
                center_y = (word_y0 + word_y1) / 2
                if x0 - pad_x <= center_x <= x1 + pad_x and y0 - pad_y <= center_y <= y1 + pad_y:
                    matched_words.append(word)
            matched_words.sort(key=lambda word: (word["bbox_pixels"][1], word["bbox_pixels"][0]))

        detection["ocr_words"] = matched_words
        detection["ocr_text"] = " ".join(word["text"] for word in matched_words)
        detection["ocr_confidence"] = (
            round(sum(word["confidence"] for word in matched_words) / len(matched_words), 6)
            if matched_words
            else None
        )


def collect_reward_statistics(
    review_root: Path | None = None,
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {
        name: {"accepted": 0, "rejected": 0, "reward": 0, "acceptance_rate": 0.5, "factor": 1.0}
        for name in CLASS_NAMES
    }
    for path in feedback_files(review_root):
        review = load_json(path)
        for entry in review.get("annotations", {}).values():
            if not isinstance(entry, dict):
                continue
            decision = entry.get("decision")
            name = entry.get("corrected_label") or entry.get("class_name")
            if decision not in DECISION_REWARDS or name not in stats:
                continue
            stats[name][decision] += 1
            stats[name]["reward"] += DECISION_REWARDS[decision]

        missed_annotations = review.get("missed_annotations", {})
        if isinstance(missed_annotations, dict):
            for entry in missed_annotations.values():
                if not isinstance(entry, dict):
                    continue
                name = entry.get("class_name")
                if name not in stats:
                    continue
                if entry.get("status", "open") == "recovered":
                    stats[name]["accepted"] += 1
                    stats[name]["reward"] += 1
                    stats[name]["recovered_misses"] = stats[name].get("recovered_misses", 0) + 1
                else:
                    stats[name]["rejected"] += 1
                    stats[name]["reward"] -= 1
                    stats[name]["missed"] = stats[name].get("missed", 0) + 1

    for values in stats.values():
        decisions = values["accepted"] + values["rejected"]
        rate = (values["accepted"] + 1) / (decisions + 2)
        values["acceptance_rate"] = round(rate, 4)
        values["factor"] = round(min(1.5, max(0.5, 2 * rate)), 4)
    return stats


def load_or_create_review(
    review_path: Path,
    page_number: int,
    detections: list[dict[str, Any]],
    page_size_points: list[float],
) -> dict[str, Any]:
    if review_path.exists():
        review = load_json(review_path)
        if not isinstance(review.get("annotations"), dict):
            raise ValueError(f"Invalid review file: {review_path}")
    else:
        review = {"schema_version": 1, "page_number": page_number, "annotations": {}}

    annotations = review["annotations"]
    for detection in detections:
        entry = annotations.get(detection["id"])
        if entry is None:
            entry = {
                "decision": "pending",
                "class_name": detection["class_name"],
                "corrected_label": None,
                "bbox_pdf": detection["bbox_pdf"],
                "corrected_bbox_pdf": None,
                "recognized_text": detection["ocr_text"],
                "ocr_confidence": detection["ocr_confidence"],
                "comment": "",
                "notes": "",
            }
        if not isinstance(entry, dict) or entry.get("decision", "pending") not in VALID_DECISIONS:
            raise ValueError(f"Invalid feedback entry for detection {detection['id']}")
        entry["recognized_text"] = detection["ocr_text"]
        entry["ocr_confidence"] = detection["ocr_confidence"]
        annotations[detection["id"]] = entry

    review.update({
        "schema_version": 1,
        "page_number": page_number,
        "pdf_page_size_points": page_size_points,
        "missed_annotations": review.get("missed_annotations", {}),
        "instructions": (
            "For each ID, set decision to accepted or rejected. For a corrected positive, "
            "set decision=accepted and optionally corrected_label or corrected_bbox_pdf. "
            "recognized_text and ocr_confidence are OCR output, separate from the visual class. "
            "Add objects the model missed under missed_annotations using a unique key, class_name, "
            "bbox_pdf, and optional corrected_text/comment/notes. Open misses count as -1 and "
            "become training labels; a later matching detection marks them recovered. "
            "Add comments or notes without changing the ID. Rerun training to incorporate feedback."
        ),
    })
    return review


def update_missed_recoveries(
    review: dict[str, Any],
    detections: list[dict[str, Any]],
) -> None:
    missed_annotations = review.get("missed_annotations", {})
    if not isinstance(missed_annotations, dict):
        raise ValueError("missed_annotations must be a JSON object keyed by a unique ID.")

    for missed_id, missed in missed_annotations.items():
        if not isinstance(missed, dict):
            raise ValueError(f"Missed annotation {missed_id} must be a JSON object.")
        class_name = missed.get("class_name")
        boxes = normalize_boxes(missed.get("bbox_pdf"))
        if class_name not in CLASS_TO_ID or not boxes:
            raise ValueError(
                f"Missed annotation {missed_id} needs a supported class_name and bbox_pdf."
            )

        matching_detection = next((
            detection
            for detection in detections
            if detection["class_name"] == class_name
            and detection.get("review", {}).get("decision") != "rejected"
            and iou(detection["bbox_pdf"], boxes[0]) >= 0.5
        ), None)
        missed["status"] = "recovered" if matching_detection else "open"
        if matching_detection:
            missed["recovered_by"] = matching_detection["id"]
            matching_detection.setdefault("recovered_missed_ids", []).append(missed_id)
        else:
            missed.pop("recovered_by", None)


def draw_page_detections(
    image: Image.Image,
    detections: list[dict[str, Any]],
    missed_annotations: dict[str, Any],
    page_width: float,
    page_height: float,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    for detection in detections:
        pdf_bbox = detection["effective"]["bbox_pdf"]
        x0 = int(pdf_bbox[0] * image.width / page_width)
        y0 = int(pdf_bbox[1] * image.height / page_height)
        x1 = int(pdf_bbox[2] * image.width / page_width)
        y1 = int(pdf_bbox[3] * image.height / page_height)
        decision = detection["review"]["decision"]
        color = BOX_COLORS[decision]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = detection["effective"]["class_name"]
        if detection["ocr_text"]:
            label += f": {detection['ocr_text']}"
        label += f"  D {detection['confidence']:.0%}"
        if detection["ocr_confidence"] is not None:
            label += f" O {detection['ocr_confidence']:.0%}"
        max_width = max(80, image.width - 12)
        while len(label) > 1 and draw.textbbox((0, 0), label, font=font)[2] > max_width:
            label = label[:-2].rstrip() + "..."
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        label_x = min(max(0, x0), max(0, image.width - label_width - 8))
        label_y = max(0, y0 - label_height - 8)
        draw.rectangle(
            (label_x, label_y, label_x + label_width + 8, label_y + label_height + 6),
            fill=(20, 24, 28, 220),
        )
        draw.text((label_x + 4, label_y + 2), label, fill=color, font=font)

    for missed_id, missed in missed_annotations.items():
        if not isinstance(missed, dict):
            continue
        boxes = normalize_boxes(missed.get("bbox_pdf"))
        if not boxes:
            continue
        x0 = int(boxes[0][0] * image.width / page_width)
        y0 = int(boxes[0][1] * image.height / page_height)
        x1 = int(boxes[0][2] * image.width / page_width)
        y1 = int(boxes[0][3] * image.height / page_height)
        status = missed.get("status", "open")
        color = BOX_COLORS["accepted"] if status == "recovered" else MISSED_COLOR
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        text = missed.get("corrected_text")
        label = f"{missed.get('class_name', 'object')} missed"
        if text:
            label += f": {text}"
        label_box = draw.textbbox((0, 0), label, font=font)
        width = label_box[2] - label_box[0]
        height = label_box[3] - label_box[1]
        label_x = min(max(0, x0), max(0, image.width - width - 8))
        label_y = max(0, y0 - height - 8)
        draw.rectangle(
            (label_x, label_y, label_x + width + 8, label_y + height + 6),
            fill=(20, 24, 28, 220),
        )
        draw.text((label_x + 4, label_y + 2), label, fill=color, font=font)


def annotate_page(args: argparse.Namespace) -> None:
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    if not weights.exists():
        raise FileNotFoundError(
            f"Detector weights not found: {weights}. Run `uv run json-plot-model train` first."
        )
    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    model = YOLO(str(weights))
    page_number = args.page
    page_dir = REVIEW_DIR / f"page_{page_number:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(args.pdf) as pdf:
        if page_number < 1 or page_number > len(pdf.pages):
            raise ValueError(f"Page must be in range 1-{len(pdf.pages)}.")
        page = pdf.pages[page_number - 1]
        inference_dpi = max(
            1,
            round(RENDER_DPI * TRAINING_PAGE_WIDTH / page.width),
        )
        image = page.to_image(resolution=inference_dpi).original.convert("RGB")
        page_width = page.width
        page_height = page.height
        scale_x = image.width / page.width
        scale_y = image.height / page.height
        detections = collect_model_detections(
            image,
            page_number,
            scale_x,
            scale_y,
            model,
            args.confidence,
            args.imgsz,
        )
        ocr_words = recognize_page_text(image, args.tesseract, scale_x, scale_y)
        attach_ocr_to_detections(detections, ocr_words)

    review_path = page_dir / "human_review.json"
    review = load_or_create_review(
        review_path,
        page_number,
        detections,
        [page_width, page_height],
    )
    for detection in detections:
        entry = review["annotations"][detection["id"]]
        detection["review"] = dict(entry)
        detection["effective"] = {
            "included": entry["decision"] != "rejected",
            "class_name": entry.get("corrected_label") or detection["class_name"],
            "bbox_pdf": entry.get("corrected_bbox_pdf") or detection["bbox_pdf"],
            "decision": entry["decision"],
        }

    update_missed_recoveries(review, detections)
    rewards = collect_reward_statistics()
    for detection in detections:
        entry = detection["review"]
        detection["reward"] = DECISION_REWARDS.get(entry["decision"])
        detection["class_acceptance_rate"] = rewards[detection["class_name"]]["acceptance_rate"]
        detection["reward_adjusted_confidence"] = round(
            min(1.0, detection["confidence"] * rewards[detection["class_name"]]["factor"]),
            6,
        )

    save_json(review_path, review)
    annotated = image.copy()
    draw_page_detections(
        annotated,
        detections,
        review["missed_annotations"],
        page_width,
        page_height,
    )
    annotated.save(page_dir / "annotated_page.png")
    save_json(page_dir / "detections.json", {
        "schema_version": 1,
        "page_number": page_number,
        "source_pdf": str(args.pdf),
        "pdf_page_size_points": [page_width, page_height],
        "render_dpi": inference_dpi,
        "model_weights": str(weights),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reward_calibration": rewards,
        "ocr_engine": "tesseract",
        "ocr_words": ocr_words,
        "detections": detections,
        "missed_annotations": review["missed_annotations"],
    })
    print(f"Page {page_number}: {len(detections)} visual detections")
    print(f"Review files: {page_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run the visual annotation model.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("prepare", help="Export JSON annotations as tiled detector data")

    train_parser = commands.add_parser("train", help="Train the visual detector on CPU")
    train_parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=2)

    annotate_parser = commands.add_parser("annotate", help="Detect objects on one PDF page")
    annotate_parser.add_argument("--page", type=int, required=True, help="1-based page number")
    annotate_parser.add_argument(
        "--pdf",
        type=Path,
        default=ANNOTATION_PDF_FILE,
        help="PDF to annotate; training remains tied to the labeled (6) PDF",
    )
    annotate_parser.add_argument(
        "--weights",
        default=str(TRAINING_DIR / "detector" / "weights" / "best.pt"),
    )
    annotate_parser.add_argument("--confidence", type=float, default=0.1)
    annotate_parser.add_argument("--imgsz", type=int, default=640)
    annotate_parser.add_argument("--tesseract", default="tesseract")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        manifest = export_dataset()
        print(f"Exported dataset to {DATASET_DIR}")
        print(f"Tile counts: {manifest['tile_image_counts']}")
        print(f"Class labels: {manifest['tile_label_counts']}")
        print(f"Human feedback rewards: {manifest['reward_feedback_counts']}")
    elif args.command == "train":
        train_detector(args)
    elif args.command == "annotate":
        annotate_page(args)


if __name__ == "__main__":
    main()