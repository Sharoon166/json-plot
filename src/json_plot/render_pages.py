
# scripts/render_pages.py

import pdfplumber

from PIL import Image, ImageDraw, ImageFont

from json_plot.data_sources import OUTPUTS_DIR, PDF_FILE, load_datasets


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OUTPUT_DIR = OUTPUTS_DIR / "rendered_pages"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 180

# JSON boxes use the selected PDF's native page coordinates.
JSON_TO_PDF_SCALE = 1.0

# Assumed coordinate format: [x0, y0, x1, y1]
CONFIDENCE_THRESHOLD = 0.70


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

datasets = load_datasets()
ocr_data = datasets.ocr_data
schedule_data = datasets.schedule_data
schedule_outputs = datasets.schedule_outputs
data_out = datasets.data_out


# --------------------------------------------------
# FONT
# --------------------------------------------------

def get_font(size=12, bold=False):

    candidates = (
        [
            "DejaVuSans-Bold.ttf",
            "arialbd.ttf",
        ]
        if bold
        else [
            "DejaVuSans.ttf",
            "arial.ttf",
        ]
    )

    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue

    return ImageFont.load_default()


FONT = get_font(12)
SMALL_FONT = get_font(10)
OCR_TEXT_FONT = get_font(24, bold=True)
LABEL_FONT = get_font(13, bold=True)


# --------------------------------------------------
# COORDINATE CONVERSION
# --------------------------------------------------

def convert_bbox(bbox, scale_x, scale_y, image_width, image_height):
    """
    Convert PDF/image coordinates into rendered image coordinates.

    Expected format:
        [x0, y0, x1, y1]

    Returns None for invalid bounding boxes.
    """

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x0, y0, x1, y1 = map(float, bbox)
    except (ValueError, TypeError):
        return None

    x0 *= JSON_TO_PDF_SCALE * scale_x
    x1 *= JSON_TO_PDF_SCALE * scale_x
    y0 *= JSON_TO_PDF_SCALE * scale_y
    y1 *= JSON_TO_PDF_SCALE * scale_y

    # Normalize reversed coordinates.
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))

    # Clip to image dimensions.
    x0 = max(0, min(x0, image_width - 1))
    x1 = max(0, min(x1, image_width - 1))
    y0 = max(0, min(y0, image_height - 1))
    y1 = max(0, min(y1, image_height - 1))

    if x1 <= x0 or y1 <= y0:
        return None

    return (
        int(x0),
        int(y0),
        int(x1),
        int(y1),
    )


# --------------------------------------------------
# DRAWING HELPERS
# --------------------------------------------------

def draw_bbox(
    draw,
    bbox,
    color,
    label=None,
    width=2,
    fill=None,
):
    if bbox is None:
        return

    draw.rectangle(
        bbox,
        outline=color,
        fill=fill,
        width=width,
    )

    if label:
        x0, y0, _, _ = bbox

        text_bbox = draw.textbbox(
            (x0, y0),
            label,
            font=SMALL_FONT,
        )

        label_height = text_bbox[3] - text_bbox[1]

        label_y = max(0, y0 - label_height - 4)

        draw.text(
            (x0, label_y),
            label,
            fill=color,
            font=SMALL_FONT,
        )


def get_record_boxes(record):
    """
    Handles:
        bbox: [x0, y0, x1, y1]
        bbox: [[x0, y0, x1, y1], ...]
    """

    if not isinstance(record, dict):
        return []

    boxes = record.get("bbox", [])

    if not isinstance(boxes, list) or not boxes:
        return []

    if len(boxes) == 4 and all(
        isinstance(value, (int, float))
        for value in boxes
    ):
        return [boxes]

    return [
        box for box in boxes
        if isinstance(box, (list, tuple))
    ]


# --------------------------------------------------
# OCR RENDERING
# --------------------------------------------------

def render_ocr(image, page_id, scale_x, scale_y):

    draw = ImageDraw.Draw(image, "RGBA")

    words = ocr_data.get(page_id, [])

    for word in words:

        if not isinstance(word, dict):
            continue

        bbox = convert_bbox(
            word.get("bbox"),
            scale_x,
            scale_y,
            image.width,
            image.height,
        )

        if bbox is None:
            continue

        confidence = float(word.get("confidence", 0))

        if confidence >= 0.85:
            color = (0, 180, 80)

        elif confidence >= CONFIDENCE_THRESHOLD:
            color = (255, 165, 0)

        else:
            color = (230, 40, 40)

        draw_bbox(
            draw,
            bbox,
            color=color,
            width=2,
        )

        text = str(word.get("text", ""))

        if text.strip():

            x0, y0, _, _ = bbox
            text_bbox = draw.textbbox(
                (x0, 0),
                text,
                font=OCR_TEXT_FONT,
            )
            text_height = text_bbox[3] - text_bbox[1]
            confidence_text = f"conf {confidence:.2f}"
            confidence_bbox = draw.textbbox(
                (x0, 0),
                confidence_text,
                font=SMALL_FONT,
            )
            confidence_height = confidence_bbox[3] - confidence_bbox[1]
            label_top = max(0, y0 - text_height - confidence_height - 6)

            draw.text(
                (x0, label_top),
                confidence_text,
                fill=color,
                font=SMALL_FONT,
            )
            draw.text(
                (x0, label_top + confidence_height + 2),
                text,
                fill=color,
                font=OCR_TEXT_FONT,
            )

    return image


# --------------------------------------------------
# DRAWING ANNOTATIONS
# --------------------------------------------------

def render_schedule_annotations(
    image,
    page_id,
    scale_x,
    scale_y,
):

    draw = ImageDraw.Draw(image, "RGBA")

    page_annotations = schedule_data.get(page_id, {})

    for label, group in page_annotations.items():

        if not isinstance(group, dict):
            continue

        boxes = group.get("bboxes", [])
        texts = group.get("text", [])

        for index, raw_bbox in enumerate(boxes):

            bbox = convert_bbox(
                raw_bbox,
                scale_x,
                scale_y,
                image.width,
                image.height,
            )

            if bbox is None:
                continue

            text = (
                texts[index]
                if index < len(texts)
                else ""
            )

            annotation_label = str(label)

            if text:
                annotation_label += f": {text}"

            draw_bbox(
                draw,
                bbox,
                color=(0, 120, 255),
                label=annotation_label,
                width=3,
            )


# --------------------------------------------------
# SCHEDULE VERIFICATION OVERLAYS
# --------------------------------------------------

def render_schedule_verification(
    image,
    page_id,
    scale_x,
    scale_y,
):

    draw = ImageDraw.Draw(image, "RGBA")

    page_results = schedule_outputs.get(page_id, {})

    colors = {
        "verified_sched_entry": (0, 190, 80),
        "missing_sched_entry": (255, 50, 50),
        "verified_tag": (0, 190, 80),
        "missing_from_schedule": (255, 140, 0),
    }

    for category, record in page_results.items():

        if not isinstance(record, dict):
            continue

        color = colors.get(
            category,
            (180, 0, 180),
        )

        boxes = get_record_boxes(record)
        texts = record.get("texts", [])

        for index, raw_bbox in enumerate(boxes):

            bbox = convert_bbox(
                raw_bbox,
                scale_x,
                scale_y,
                image.width,
                image.height,
            )

            if bbox is None:
                continue

            text = (
                str(texts[index])
                if index < len(texts)
                else ""
            )

            label = category

            if text:
                label += f": {text}"

            draw_bbox(
                draw,
                bbox,
                color=color,
                label=label,
                width=4,
            )


# --------------------------------------------------
# REFERENCE VERIFICATION OVERLAYS
# --------------------------------------------------

def render_reference_verification(
    image,
    page_id,
    scale_x,
    scale_y,
):

    draw = ImageDraw.Draw(image, "RGBA")

    page_results = data_out.get(page_id, {})

    for category, record in page_results.items():

        if not isinstance(record, dict):
            continue

        if category.startswith("incorrect_reference_detail"):
            color = (255, 0, 0)

        elif category.startswith("verified_from_index"):
            color = (0, 200, 255)

        elif category.startswith("verified"):
            color = (0, 180, 80)

        else:
            continue

        boxes = get_record_boxes(record)

        texts = record.get(
            "text",
            record.get("callout_texts", []),
        )

        for index, raw_bbox in enumerate(boxes):

            bbox = convert_bbox(
                raw_bbox,
                scale_x,
                scale_y,
                image.width,
                image.height,
            )

            if bbox is None:
                continue

            text = (
                str(texts[index])
                if isinstance(texts, list)
                and index < len(texts)
                else ""
            )

            label = category

            if text:
                label += f": {text}"

            draw_bbox(
                draw,
                bbox,
                color=color,
                label=label,
                width=3,
            )


# --------------------------------------------------
# LEGEND
# --------------------------------------------------

def draw_legend(image):

    legend_items = [
        ("OCR high confidence", (0, 180, 80)),
        ("OCR medium confidence", (255, 165, 0)),
        ("OCR low confidence", (230, 40, 40)),
        ("Drawing annotation", (0, 120, 255)),
        ("Verified schedule/tag", (0, 190, 80)),
        ("Missing schedule/tag", (255, 140, 0)),
        ("Incorrect reference", (255, 0, 0)),
        ("Verified reference", (0, 180, 80)),
    ]

    legend_height = 25
    padding = 12

    overlay_height = (
        len(legend_items) * legend_height
        + padding * 2
    )

    overlay = Image.new(
        "RGBA",
        (image.width, overlay_height),
        (255, 255, 255, 225),
    )

    draw = ImageDraw.Draw(overlay)

    for index, (label, color) in enumerate(legend_items):

        y = padding + index * legend_height

        draw.rectangle(
            (padding, y, padding + 18, y + 12),
            outline=color,
            width=3,
        )

        draw.text(
            (padding + 28, y - 2),
            label,
            fill=(20, 20, 20),
            font=FONT,
        )

    image.alpha_composite(
        overlay,
        (0, image.height - overlay_height),
    )

    return image


# --------------------------------------------------
# MAIN PDF PROCESSING
# --------------------------------------------------

def main():

    if not PDF_FILE.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_FILE}"
        )

    with pdfplumber.open(PDF_FILE) as pdf:

        print(f"Total PDF pages: {len(pdf.pages)}")

        for page_id, page in enumerate(pdf.pages):

            page_key = str(page_id)

            print(f"Processing page {page_id}...")

            rendered = page.to_image(
                resolution=DPI
            ).original.convert("RGBA")

            scale_x = rendered.width / page.width
            scale_y = rendered.height / page.height

            # --------------------------------------
            # IMAGE 1: OCR VERSION
            # --------------------------------------

            ocr_image = rendered.copy()

            render_ocr(
                ocr_image,
                page_key,
                scale_x,
                scale_y,
            )

            ocr_path = (
                OUTPUT_DIR / f"{page_id}_ocr.png"
            )

            ocr_image.convert("RGB").save(
                ocr_path,
                quality=95,
            )

            # --------------------------------------
            # IMAGE 2: ANNOTATED VERSION
            # --------------------------------------

            annotated_image = rendered.copy()

            render_schedule_annotations(
                annotated_image,
                page_key,
                scale_x,
                scale_y,
            )

            render_schedule_verification(
                annotated_image,
                page_key,
                scale_x,
                scale_y,
            )

            render_reference_verification(
                annotated_image,
                page_key,
                scale_x,
                scale_y,
            )

            annotated_image = draw_legend(
                annotated_image
            )

            annotated_path = (
                OUTPUT_DIR / f"{page_id}_annotated.png"
            )

            annotated_image.convert("RGB").save(
                annotated_path,
                quality=95,
            )

            print(f"  Saved: {ocr_path.name}")
            print(f"  Saved: {annotated_path.name}")

    print("\nAll pages processed successfully!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
