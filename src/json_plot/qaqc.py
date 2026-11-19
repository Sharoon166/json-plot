"""Turn a QAQC review sheet into page tasks, and edit one JSON row at a time.

The review counts misses and incorrect hits per sheet. It does not contain the
corrected boxes. The viewer jumps to the sheet, and Keep / Remove / Draw write
the source JSON the review was scored against.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SHEET_RE = re.compile(r"^([A-Z]{1,4}\d{1,4})\b", re.IGNORECASE)

# Longer phrases first so "door tags" is not claimed by a shorter token.
CATEGORIES = (
    ("electrical", "electricalpanels"),
    ("mechanical", "mechanicaltags"),
    ("plumbing", "plmbgtags"),
    ("door", "doortags"),
    ("keynote", "keynotes"),
    ("schedule", "schedule"),
    ("callout", "callout"),
    ("index", "index"),
)

CATEGORY_LABELS = {
    "electrical": "electrical panels",
    "mechanical": "mechanical tags",
    "plumbing": "plumbing tags",
    "door": "door tags",
    "keynote": "keynotes",
    "schedule": "schedule rows",
    "callout": "callouts",
    "index": "index entries",
}

EDITABLE_PREFIXES = (
    "data_out",
    "schedule_data",
    "schedule_outputs",
    "mechanical_data_out",
    "plumbing_data_out",
)

# Where a newly drawn box is appended. Existing records are extended in place.
ADD_DESTINATION = {
    "callout": ("data_out", "verified"),
    "index": ("data_out", "verified_from_index"),
    "schedule": ("schedule_outputs", "verified_sheets"),
    "door": ("schedule_data", "Door Reference"),
    "keynote": ("schedule_data", "Keynote Area"),
    "electrical": ("schedule_data", "Electrical Panel"),
    "plumbing": ("plumbing_data_out", "verified_plumbing"),
    "mechanical": ("mechanical_data_out", "verified_mechanical_equipment"),
}

NEW_RECORD = {
    "callout": {"status": True, "callout_texts": [], "bbox": [], "target_pages": []},
    "index": {"status": True, "value": [], "bbox": []},
    "schedule": {"texts": [], "bbox": [], "target_pages": []},
    "door": {"text": [], "bboxes": []},
    "keynote": {"text": [], "bboxes": []},
    "electrical": {"text": [], "bboxes": []},
    "plumbing": {"status": True, "value": [], "bbox": [], "target_pages": []},
    "mechanical": {"status": True, "value": [], "bbox": [], "target_pages": []},
}


@dataclass(frozen=True)
class ReviewTask:
    sheet: str
    json_key: str
    pdf_page: int
    title: str
    verb: str
    category: str
    count: int

    @property
    def key(self) -> str:
        return f"{self.sheet}|{self.category}|{self.verb}"

    def label(self) -> str:
        action = {"add": "Add", "fix": "Fix", "keep": "Leave"}[self.verb]
        title = f"  {self.title}" if self.title else ""
        return f"{self.sheet}{title}   page {self.pdf_page}   {action} {self.count} {CATEGORY_LABELS[self.category]}"


def _header_action(header: str) -> tuple[str, str] | None:
    compact = re.sub(r"[^a-z0-9]+", "", header.lower())
    if compact.startswith("missed") or compact.startswith("missing"):
        verb = "add"
    elif compact.startswith("incorrect"):
        verb = "fix"
    elif compact.startswith("actualissue"):
        verb = "keep"
    else:
        return None
    for category, token in CATEGORIES:
        if token in compact:
            return verb, category
    return None


def parse_review_csv(path: Path) -> list[dict[str, Any]]:
    """Rows of sheet code plus the non-zero review counts."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return []
    headers = rows[1]
    parsed = []
    for row in rows[2:]:
        page_cell = row[0].strip() if row else ""
        match = SHEET_RE.match(page_cell.replace(" ", ""))
        if match is None:
            spaced = SHEET_RE.match(page_cell)
            match = spaced
        if match is None:
            continue
        counts: list[tuple[str, str, int]] = []
        for index, header in enumerate(headers):
            if index >= len(row) or not str(row[index]).strip():
                continue
            action = _header_action(header)
            if action is None:
                continue
            try:
                count = int(float(row[index]))
            except ValueError:
                continue
            if count:
                counts.append((action[0], action[1], count))
        if counts:
            parsed.append({"sheet": match.group(1).upper(), "counts": counts})
    return parsed


def load_page_map(folder: Path) -> dict[str, str]:
    """Map a sheet code to the JSON page key. Keys in these files match sheet_info."""
    path = next(folder.glob("page_to_sheet_*.json"), None)
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = {}
    for key, sheet in data.items():
        if isinstance(sheet, str) and sheet.strip():
            mapping[sheet.strip().upper()] = str(key)
    return mapping


def load_sheet_titles(folder: Path, page_map: dict[str, str]) -> dict[str, str]:
    path = next(folder.glob("sheet_info_*.json"), None)
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    key_to_sheet = {key: sheet for sheet, key in page_map.items()}
    titles = {}
    for key, sheet in data.items():
        code = key_to_sheet.get(str(key))
        if code is None or not isinstance(sheet, dict):
            continue
        block = sheet.get("title_block_classification")
        title = ""
        if isinstance(block, dict):
            title = str(block.get("sheet_title") or "")
        titles[code] = title.strip()
    return titles


def build_tasks(rows: list[dict[str, Any]], page_map: dict[str, str], titles: dict[str, str]) -> tuple[list[ReviewTask], list[str]]:
    tasks: list[ReviewTask] = []
    missing: list[str] = []
    for row in rows:
        json_key = page_map.get(row["sheet"])
        if json_key is None:
            missing.append(row["sheet"])
            continue
        for verb, category, count in row["counts"]:
            tasks.append(ReviewTask(
                sheet=row["sheet"],
                json_key=json_key,
                pdf_page=int(json_key) + 1,
                title=titles.get(row["sheet"], ""),
                verb=verb,
                category=category,
                count=count,
            ))
    work = {}
    for task in tasks:
        if task.verb != "keep":
            work[task.sheet] = work.get(task.sheet, 0) + task.count

    def sort_key(task: ReviewTask) -> tuple:
        verb_order = {"fix": 0, "add": 1, "keep": 2}
        if task.sheet in work and task.verb != "keep":
            return (0, -work[task.sheet], task.pdf_page, verb_order[task.verb])
        return (1, task.pdf_page, verb_order[task.verb])

    return sorted(tasks, key=sort_key), missing


def packet_files(folder: Path) -> tuple[Path | None, list[Path]]:
    """Prefer the clean pre PDF. Skip OCR and raw detections; they flood the page."""
    pdfs = sorted(folder.glob("*.pdf"))
    pre = next((path for path in pdfs if path.name.lower().endswith("pre.pdf")), None)
    jsons = []
    for path in sorted(folder.glob("*.json")):
        name = path.name.lower()
        if name.startswith(EDITABLE_PREFIXES) and path.stat().st_size > 2:
            jsons.append(path)
    return pre, jsons


def find_named(folder: Path, prefix: str) -> Path | None:
    matches = sorted(folder.glob(f"{prefix}_*.json"))
    return matches[0] if matches else None


def item_in_category(item: dict[str, Any], category: str) -> bool:
    source = str(item.get("source") or "").lower()
    record = str(item.get("record") or "").lower()
    if category == "door":
        return "door" in record
    if category == "keynote":
        return "keynote" in record
    if category == "mechanical":
        return source.startswith("mechanical_data_out") or "mechanical" in record
    if category == "plumbing":
        return source.startswith("plumbing_data_out") or "plumb" in record
    if category == "electrical":
        return "panel" in record or "electrical" in record
    if category == "index":
        return "index" in record
    if category == "callout":
        if source.startswith("data_out") and record in {"verified", "incorrect_reference_detail", "reference_missing", "verified_sheets"}:
            return True
        return "callout" in record
    if category == "schedule":
        if source.startswith("schedule_outputs"):
            return True
        return "schedule" in record
    return False


def related_keeps(tasks: list[ReviewTask], task: ReviewTask) -> list[ReviewTask]:
    return [other for other in tasks if other.sheet == task.sheet and other.verb == "keep" and other is not task]


def _text_key(record: dict[str, Any]) -> str | None:
    for name in ("texts", "text", "callout_texts"):
        if isinstance(record.get(name), (list, str)):
            return name
    if isinstance(record.get("value"), (list, str)):
        return "value"
    return None


def _box_key(record: dict[str, Any]) -> str | None:
    for name in ("bboxes", "bbox"):
        if isinstance(record.get(name), list):
            return name
    return None


def _as_texts(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return list(value)
    return []


def _as_boxes(value: Any) -> list[Any]:
    if not isinstance(value, list) or not value:
        return []
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        return [list(value)]
    return list(value)


def _visible_indexes(texts: list[Any], boxes: list[Any], targets: list[Any]) -> list[int] | None:
    """Indexes aligned_rows would emit. None means one shared text over many boxes."""
    if len(texts) <= 1 and len(boxes) > 1 and len(targets) <= 1:
        return None
    count = max(len(texts), len(boxes), len(targets))
    visible = []
    for index in range(count):
        text = texts[index] if index < len(texts) else ""
        box = boxes[index] if index < len(boxes) else None
        target = targets[index] if index < len(targets) else None
        if text or box is not None or target is not None:
            visible.append(index)
    return visible


def _parallel_lists(record: dict[str, Any]) -> list[str]:
    return [name for name in ("target_pages", "target_shapes", "shapes") if isinstance(record.get(name), list)]


def remove_row(record: dict[str, Any], index: int) -> None:
    text_key = _text_key(record)
    box_key = _box_key(record)
    texts = _as_texts(record.get(text_key)) if text_key else []
    boxes = _as_boxes(record.get(box_key)) if box_key else []
    targets = list(record.get("target_pages")) if isinstance(record.get("target_pages"), list) else []
    shared = _visible_indexes(texts, boxes, targets)
    if shared is None:
        if not 0 <= index < len(boxes):
            raise IndexError(f"Row {index} is not in this record.")
        boxes.pop(index)
    else:
        if not 0 <= index < len(shared):
            raise IndexError(f"Row {index} is not in this record.")
        real = shared[index]
        lists = [texts, boxes, targets]
        names = _parallel_lists(record)
        lists.extend(list(record[name]) for name in names)
        for values in lists:
            if real < len(values):
                values.pop(real)
        for name, values in zip(names, lists[3:]):
            record[name] = values
    if text_key:
        original = record.get(text_key)
        record[text_key] = texts[0] if isinstance(original, str) and len(texts) == 1 else texts
    if box_key:
        record[box_key] = boxes
    if isinstance(record.get("target_pages"), list):
        record["target_pages"] = targets


def _append_parallel(values: list[Any], index: int, item: Any) -> None:
    if index < len(values):
        values.insert(index, item)
    else:
        while len(values) < index:
            values.append(None)
        values.append(item)


def append_row(record: dict[str, Any], text: str, bbox: list[float]) -> None:
    text_key = _text_key(record) or ("value" if "value" in record else "text")
    box_key = _box_key(record) or "bbox"
    if text_key not in record:
        record[text_key] = []
    if box_key not in record:
        record[box_key] = []
    texts = _as_texts(record.get(text_key))
    boxes = _as_boxes(record.get(box_key))
    box = [float(value) for value in bbox]
    if len(texts) <= 1 and len(boxes) > 1 and not isinstance(record.get("target_pages"), list):
        boxes.append(box)
        if not texts:
            texts.append(text)
    else:
        while len(texts) < len(boxes):
            texts.append("")
        slot = len(boxes)
        boxes.append(box)
        if slot < len(texts):
            texts.insert(slot, text)
        else:
            texts.append(text)
        for name in _parallel_lists(record):
            _append_parallel(record[name], slot, None)
    original = record.get(text_key)
    if isinstance(original, str) and len(texts) == 1:
        record[text_key] = texts[0]
    else:
        record[text_key] = texts
    record[box_key] = boxes


def _page(data: dict[str, Any], page_key: str) -> dict[str, Any]:
    page = data.get(page_key)
    if page is None and page_key.isdigit():
        page = data.get(int(page_key))
        if isinstance(page, dict):
            return page
    if not isinstance(page, dict):
        page = {}
        data[page_key] = page
    return page


def _write_json(path: Path, previous: str, data: Any) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(previous, encoding="utf-8")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def remove_item(item: dict[str, Any]) -> str:
    path = Path(str(item.get("source_path") or ""))
    if not path.is_file():
        raise FileNotFoundError(f"JSON file is not loaded for this box: {item.get('source')}")
    if path.name.lower().startswith(("ocr_word_dict", "raw_detections")):
        raise ValueError("OCR words and raw detections stay on disk. Remove a box from the output JSON.")
    previous = path.read_text(encoding="utf-8")
    data = json.loads(previous)
    page = _page(data, str(item.get("page_key")))
    record = page.get(item.get("record"))
    if not isinstance(record, dict):
        raise KeyError(f"Record {item.get('record')} is not on page {item.get('page_key')}.")
    remove_row(record, int(item["row_index"]))
    _write_json(path, previous, data)
    return previous


def add_item(folder: Path, task: ReviewTask, text: str, bbox: list[float]) -> tuple[str, Path, str]:
    prefix, record_name = ADD_DESTINATION[task.category]
    path = find_named(folder, prefix)
    if path is None:
        path = folder / f"{prefix}.json"
        previous = ""
        data: dict[str, Any] = {}
    else:
        previous = path.read_text(encoding="utf-8")
        data = json.loads(previous or "{}")
        if not isinstance(data, dict):
            raise ValueError(f"{path.name} is not a page map.")
    page = _page(data, task.json_key)
    record = page.get(record_name)
    if not isinstance(record, dict):
        record = json.loads(json.dumps(NEW_RECORD[task.category]))
        page[record_name] = record
    append_row(record, text, bbox)
    _write_json(path, previous if previous else "{}\n", data)
    return previous if previous else "{}\n", path, record_name


def destination_label(folder: Path, task: ReviewTask) -> str:
    prefix, record_name = ADD_DESTINATION[task.category]
    path = find_named(folder, prefix)
    name = path.name if path else f"{prefix}.json"
    return f"{name}  →  {record_name}"
