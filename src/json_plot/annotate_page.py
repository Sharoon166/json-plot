import argparse
import csv
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pdfplumber
from PIL import Image, ImageDraw

from json_plot.data_sources import PDF_FILE, PROJECT_ROOT
from json_plot.train_annotation_model import predict_text


MODEL_FILE = PROJECT_ROOT / "outputs" / "models" / "annotation_text_model.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "annotations"
DEFAULT_DPI = 120
MIN_OCR_CONFIDENCE = 35.0
VALID_DECISIONS = {"pending", "accepted", "rejected"}
COLORS = {
    "pending": (238, 166, 25),
    "accepted": (28, 155, 94),
    "rejected": (215, 65, 65),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def stable_detection_id(page_number: int, text: str, bbox_pixels: list[int]) -> str:
    identity = f"{page_number}|{text.strip().upper()}|{','.join(map(str, bbox_pixels))}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    return f"p{page_number:03d}-{digest}"


def run_tesseract(
    image: Image.Image,
    page_number: int,
    scale_x: float,
    scale_y: float,
    model: dict[str, Any],
    executable: str,
) -> list[dict[str, Any]]:
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    result = subprocess.run(
        [executable, "stdin", "stdout", "--psm", "11", "tsv"],
        input=image_buffer.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    rows = csv.DictReader(
        io.StringIO(result.stdout.decode("utf-8", errors="replace")),
        delimiter="\t",
    )
    detections = []

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

        if confidence < MIN_OCR_CONFIDENCE or width <= 0 or height <= 0:
            continue

        bbox_pixels = [left, top, left + width, top + height]
        try:
            predicted_label, model_confidence = predict_text(text, model)
        except ValueError:
            continue

        detections.append({
            "id": stable_detection_id(page_number, text, bbox_pixels),
            "text": text,
            "predicted_label": predicted_label,
            "model_confidence": round(model_confidence, 6),
            "ocr_confidence": round(confidence / 100, 6),
            "bbox_pdf": [
                round(left / scale_x, 3),
                round(top / scale_y, 3),
                round((left + width) / scale_x, 3),
                round((top + height) / scale_y, 3),
            ],
            "bbox_pixels": bbox_pixels,
        })

    return detections


def empty_review_entry() -> dict[str, Any]:
    return {
        "decision": "pending",
        "corrected_text": None,
        "corrected_label": None,
        "comment": "",
        "notes": "",
    }


def load_or_create_review(
    review_path: Path,
    page_number: int,
    detections: list[dict[str, Any]],
    valid_labels: set[str],
) -> dict[str, Any]:
    if review_path.exists():
        review = load_json(review_path)
        if not isinstance(review.get("annotations"), dict):
            raise ValueError(f"Invalid review file: {review_path}")
    else:
        review = {
            "schema_version": 1,
            "page_number": page_number,
            "annotations": {},
        }

    annotations = review["annotations"]
    for detection in detections:
        entry = annotations.get(detection["id"])
        if entry is None:
            annotations[detection["id"]] = empty_review_entry()
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Review entry {detection['id']} must be a JSON object.")

        merged = empty_review_entry()
        merged.update(entry)
        if merged["decision"] not in VALID_DECISIONS:
            raise ValueError(
                f"Invalid decision for {detection['id']}: {merged['decision']}"
            )
        if merged["corrected_label"] is not None and merged["corrected_label"] not in valid_labels:
            raise ValueError(
                f"Invalid corrected_label for {detection['id']}: "
                f"{merged['corrected_label']}"
            )
        annotations[detection["id"]] = merged

    review["schema_version"] = 1
    review["page_number"] = page_number
    review.setdefault(
        "instructions",
        "Set decision to accepted or rejected; optionally edit corrected_text, "
        "corrected_label, comment, or notes. Keep annotation IDs unchanged.",
    )
    return review


def apply_review(detections: list[dict[str, Any]], review: dict[str, Any]) -> None:
    for detection in detections:
        feedback = review["annotations"][detection["id"]]
        decision = feedback["decision"]
        detection["review"] = dict(feedback)
        detection["effective"] = {
            "included": decision != "rejected",
            "text": feedback["corrected_text"] or detection["text"],
            "label": feedback["corrected_label"] or detection["predicted_label"],
            "decision": decision,
        }


def draw_detections(image: Image.Image, detections: list[dict[str, Any]]) -> None:
    draw = ImageDraw.Draw(image, "RGBA")

    for detection in detections:
        x0, y0, x1, y1 = detection["bbox_pixels"]
        decision = detection["effective"]["decision"]
        color = COLORS[decision]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)


def annotate_page(
    page_number: int,
    pdf_path: Path,
    output_dir: Path,
    dpi: int,
    tesseract_executable: str,
) -> Path:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}. "
            "Run train_annotation_model.py first."
        )
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    model = load_json(MODEL_FILE)
    valid_labels = set(model["class_counts"]) | {"other"}
    page_dir = output_dir / f"page_{page_number:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    detections_path = page_dir / "detections.json"
    review_path = page_dir / "human_review.json"
    image_path = page_dir / "annotated_page.png"

    with pdfplumber.open(pdf_path) as pdf:
        if page_number < 1 or page_number > len(pdf.pages):
            raise ValueError(
                f"Page {page_number} is outside the PDF range 1-{len(pdf.pages)}."
            )

        page = pdf.pages[page_number - 1]
        rendered = page.to_image(resolution=dpi).original.convert("RGBA")
        scale_x = rendered.width / page.width
        scale_y = rendered.height / page.height
        detections = run_tesseract(
            rendered,
            page_number,
            scale_x,
            scale_y,
            model,
            tesseract_executable,
        )

    review = load_or_create_review(
        review_path,
        page_number,
        detections,
        valid_labels,
    )
    apply_review(detections, review)
    review_path.write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    detections_document = {
        "schema_version": 1,
        "page_number": page_number,
        "source_pdf": str(pdf_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coordinate_system": {
            "bbox_pdf": "[x0, y0, x1, y1] in PDF points",
            "bbox_pixels": "[x0, y0, x1, y1] in annotated_page.png pixels",
        },
        "model": {
            "file": str(MODEL_FILE.relative_to(PROJECT_ROOT)),
            "algorithm": model["algorithm"],
            "classes": sorted(model["class_counts"]),
        },
        "minimum_ocr_confidence": MIN_OCR_CONFIDENCE,
        "detections": detections,
    }
    detections_path.write_text(
        json.dumps(detections_document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    annotated_image = rendered.copy()
    draw_detections(annotated_image, detections)
    annotated_image.convert("RGB").save(image_path)
    return page_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR and annotate one PDF page for human review."
    )
    parser.add_argument("--page", type=int, required=True, help="1-based PDF page number")
    parser.add_argument("--pdf", type=Path, default=PDF_FILE, help="PDF to process")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--tesseract", default="tesseract", help="Tesseract executable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    page_dir = annotate_page(
        args.page,
        args.pdf,
        args.output_dir,
        args.dpi,
        args.tesseract,
    )
    detections = load_json(page_dir / "detections.json")["detections"]
    print(f"Page {args.page}: {len(detections)} proposals")
    print(f"Review folder: {page_dir}")


if __name__ == "__main__":
    main()