"""Build a relationship graph from annotated drawing JSON and PDF notes.

The graph is the plain shape used by editor graph views: nodes with a type and
label, edges with a source, target, and relationship. Every edge keeps the page
and box it was read from, so a click can open that spot on the PDF.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MAX_EVIDENCE = 8

CALLOUT_RE = re.compile(
    r"^(?P<detail>.+?)\s+(?P<sheet>[A-Z]{1,4}(?:\d+(?:[.\-][A-Z0-9]+)*|[.\-]\d+))$"
)
CODE_RE = re.compile(r"[A-Z]{1,4}(?:\d+(?:[.\-][A-Z0-9]+)*|[.\-]\d+)")
MARK_RE = re.compile(r"^[A-Z]{0,3}\d{2,}[A-Z]?$")

SCHEDULE_OUTPUT_KEYS = {
    "verified_sched_entry",
    "missing_sched_entry",
    "verified_tag",
    "missing_from_schedule",
}
REFERENCE_KEYS = {
    "incorrect_reference_detail",
    "verified",
    "verified_from_index",
    "verified_sheets",
    "missing_from_index",
    "reference_missing",
}
REGION_KEYS = {
    "Door Reference",
    "Callout Area",
    "Plan Area",
    "Schedule Area",
    "Legend/Key Area",
}
REFERENCE_EDGE_TYPES = {
    "verified": "calls",
    "verified_sheets": "calls",
    "incorrect_reference_detail": "bad_call",
    "verified_from_index": "indexes",
    "missing_from_index": "missing_index",
    "reference_missing": "unresolved",
}
NODE_TYPE_ORDER = ("page", "mark", "sheet", "index", "region", "class", "item")
EDGE_TYPE_ORDER = (
    "on_schedule",
    "on_plan",
    "missing",
    "calls",
    "bad_call",
    "opens",
    "indexes",
    "missing_index",
    "unresolved",
    "contains",
    "seen_on",
    "detected",
    "related_to",
)
HIDDEN_BY_DEFAULT = {"region", "seen_on"}

TYPE_LABELS = {
    "page": "Pages",
    "mark": "Marks",
    "sheet": "Sheets",
    "index": "Index",
    "region": "Regions",
    "class": "Classes",
    "item": "Items",
}
EDGE_LABELS = {
    "on_schedule": "on schedule",
    "on_plan": "on plan",
    "missing": "missing",
    "calls": "calls",
    "bad_call": "bad callout",
    "opens": "opens page",
    "indexes": "in index",
    "missing_index": "not in index",
    "unresolved": "unresolved",
    "contains": "contains",
    "seen_on": "seen on",
    "detected": "detected",
    "related_to": "related",
}
DASHED_EDGES = {"missing", "bad_call", "missing_index", "unresolved"}


@dataclass
class Evidence:
    file: str
    kind: str
    text: str = ""
    display_page: int | None = None
    pdf_page_index: int | None = None
    bbox: list[float] | None = None
    pdf: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in (None, "", [])}


@dataclass
class Node:
    id: str
    type: str
    label: str
    mentions: int = 0
    evidences: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "mentions": self.mentions,
            "evidence": [item.to_dict() for item in self.evidences],
        }


@dataclass
class Edge:
    id: str
    src: str
    dst: str
    type: str
    count: int = 0
    evidences: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "src": self.src,
            "dst": self.dst,
            "type": self.type,
            "count": self.count,
            "evidence": [item.to_dict() for item in self.evidences],
        }


@dataclass
class RelationshipGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    pdf_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    def add_node(self, node_id: str, node_type: str, label: str, evidence: Evidence | None = None) -> Node:
        node = self.nodes.get(node_id)
        if node is None:
            node = Node(id=node_id, type=node_type, label=label)
            self.nodes[node_id] = node
        node.mentions += 1
        if evidence is not None and len(node.evidences) < MAX_EVIDENCE:
            node.evidences.append(evidence)
        return node

    def add_edge(
        self,
        src: str,
        dst: str,
        edge_type: str,
        evidence: Evidence | None = None,
    ) -> Edge | None:
        if src == dst or src not in self.nodes or dst not in self.nodes:
            return None
        edge_id = f"{src}|{edge_type}|{dst}"
        edge = self.edges.get(edge_id)
        if edge is None:
            edge = Edge(id=edge_id, src=src, dst=dst, type=edge_type)
            self.edges[edge_id] = edge
        edge.count += 1
        if evidence is not None and len(edge.evidences) < MAX_EVIDENCE:
            edge.evidences.append(evidence)
        return edge

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [self.nodes[key].to_dict() for key in sorted(self.nodes)],
            "edges": [self.edges[key].to_dict() for key in sorted(self.edges)],
            "pdfs": self.pdf_paths,
            "warnings": self.warnings,
        }

    def counts_by_type(self) -> tuple[dict[str, int], dict[str, int]]:
        nodes: dict[str, int] = defaultdict(int)
        edges: dict[str, int] = defaultdict(int)
        for node in self.nodes.values():
            nodes[node.type] += 1
        for edge in self.edges.values():
            edges[edge.type] += 1
        return dict(nodes), dict(edges)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def boxes_of(record: dict[str, Any]) -> list[list[float]]:
    raw = record.get("bboxes", record.get("bbox"))
    if not isinstance(raw, list) or not raw:
        return []
    candidates = [raw] if len(raw) == 4 and all(isinstance(item, (int, float)) for item in raw) else raw
    boxes = []
    for candidate in candidates:
        if not isinstance(candidate, (list, tuple)) or len(candidate) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(item) for item in candidate)
        except (TypeError, ValueError):
            continue
        if x1 > x0 and y1 > y0:
            boxes.append([x0, y0, x1, y1])
    return boxes


def page_entries(data: Any) -> list[tuple[str, Any]]:
    if not isinstance(data, dict) or not data:
        return []
    numeric = [(str(key), value) for key, value in data.items() if str(key).isdigit()]
    if len(numeric) < max(1, int(len(data) * 0.6)):
        return []
    return numeric


def display_page(number: int) -> int:
    """Page-map JSON uses a zero-based PDF index. Reviews use a separate 1-based field."""
    return number + 1


def classify_code(code: str) -> str:
    if re.search(r"[.\-]", code):
        return "sheet"
    if re.fullmatch(r"[A-Z]{1,4}\d{1,4}", code):
        return "index"
    return "sheet"


def callout_targets(text: str) -> list[tuple[str, str, str]]:
    raw = str(text).strip()
    if not raw:
        return []
    match = CALLOUT_RE.match(raw)
    if match:
        code = match.group("sheet")
        return [(classify_code(code), code, raw)]
    codes = CODE_RE.findall(raw)
    if not codes:
        return [("sheet", raw, raw)]
    return [(classify_code(code), code, raw) for code in dict.fromkeys(codes)]


def evidence_for(
    path: Path,
    kind: str,
    text: str,
    display: int | None,
    bbox: list[float] | None,
    pdf: str | None = None,
) -> Evidence:
    pdf_index = None if display is None else display - 1
    return Evidence(
        file=path.name,
        kind=kind,
        text=text,
        display_page=display,
        pdf_page_index=pdf_index,
        bbox=bbox,
        pdf=pdf,
    )


def detect_kind(data: Any) -> str:
    if isinstance(data, dict) and isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
        return "graph"
    if isinstance(data, dict) and isinstance(data.get("detections"), list) and "page_number" in data:
        return "detections"
    if isinstance(data, dict) and isinstance(data.get("annotations"), dict) and "page_number" in data:
        return "review"
    pages = page_entries(data)
    if not pages:
        return "generic"
    sample = pages[0][1]
    if isinstance(sample, list):
        if sample and isinstance(sample[0], dict) and {"text", "bbox"} <= set(sample[0]):
            return "ocr"
        return "generic"
    keys: set[str] = set()
    for _, page in pages:
        if isinstance(page, dict):
            keys.update(str(key) for key in page)
    if keys & SCHEDULE_OUTPUT_KEYS:
        return "schedule_outputs"
    if keys & REFERENCE_KEYS:
        return "references"
    if any(key.startswith("Tag ") or key in REGION_KEYS for key in keys):
        return "schedule_annotations"
    return "generic"


def texts_of(record: dict[str, Any]) -> list[str]:
    raw = record.get("texts", record.get("text", record.get("callout_texts")))
    if raw is None and record.get("value") is not None:
        raw = record["value"]
    if isinstance(raw, str):
        return [raw.strip()]
    if isinstance(raw, list):
        return ["" if item is None else str(item).strip() for item in raw]
    return []


def targets_of(record: dict[str, Any]) -> list[Any]:
    raw = record.get("target_pages") or []
    return raw if isinstance(raw, list) else []


def aligned_rows(record: dict[str, Any]) -> list[tuple[str, list[float] | None, Any]]:
    texts = texts_of(record)
    boxes = boxes_of(record)
    targets = targets_of(record)
    if len(texts) <= 1 and len(boxes) > 1 and len(targets) <= 1:
        text = texts[0] if texts else ""
        target = targets[0] if targets else None
        return [(text, box, target) for box in boxes]
    count = max(len(texts), len(boxes), len(targets))
    rows = []
    for index in range(count):
        text = texts[index] if index < len(texts) else ""
        box = boxes[index] if index < len(boxes) else None
        target = targets[index] if index < len(targets) else None
        if text or box is not None or target is not None:
            rows.append((text, box, target))
    return rows


def page_number(value: Any) -> int | None:
    try:
        return display_page(int(value))
    except (TypeError, ValueError):
        return None


def link_mark(
    graph: RelationshipGraph,
    text: str,
    schedule_page: int | None,
    plan_page: int | None,
    path: Path,
    kind: str,
    bbox: list[float] | None,
    source_page: int | None,
    missing: bool = False,
) -> None:
    label = text.strip()
    if not label:
        return
    proof = evidence_for(path, kind, label, source_page, bbox)
    graph.add_node(f"mark:{label}", "mark", label, proof)
    if schedule_page is not None:
        schedule_id = f"page:{schedule_page}"
        graph.add_node(schedule_id, "page", f"Page {schedule_page}", proof)
        edge_type = "missing" if missing and plan_page is None else "on_schedule"
        graph.add_edge(f"mark:{label}", schedule_id, edge_type, proof)
    if plan_page is not None:
        plan_id = f"page:{plan_page}"
        graph.add_node(plan_id, "page", f"Page {plan_page}", proof)
        edge_type = "missing" if missing else "on_plan"
        graph.add_edge(f"mark:{label}", plan_id, edge_type, proof)


def ingest_schedule_outputs(data: dict[str, Any], path: Path, graph: RelationshipGraph) -> None:
    for key, page in page_entries(data):
        if not isinstance(page, dict):
            continue
        current = page_number(key)
        for record_name, record in page.items():
            if record_name not in SCHEDULE_OUTPUT_KEYS or not isinstance(record, dict):
                continue
            missing = record_name.startswith("missing")
            for text, box, target in aligned_rows(record):
                target_page = page_number(target)
                if record_name in {"verified_sched_entry", "missing_sched_entry"}:
                    link_mark(
                        graph, text, current, None if missing else target_page,
                        path, record_name, box, current, missing=missing,
                    )
                else:
                    link_mark(
                        graph, text, None if missing else target_page, current,
                        path, record_name, box, current, missing=missing,
                    )


def ingest_schedule_annotations(data: dict[str, Any], path: Path, graph: RelationshipGraph) -> None:
    for key, page in page_entries(data):
        if not isinstance(page, dict):
            continue
        current = page_number(key)
        if current is None:
            continue
        page_id = f"page:{current}"
        for record_name, record in page.items():
            if not isinstance(record, dict):
                continue
            if not (record_name.startswith("Tag ") or record_name in REGION_KEYS):
                continue
            boxes = boxes_of(record)
            if not boxes:
                continue
            proof = evidence_for(path, record_name, f"{len(boxes)} boxes", current, boxes[0])
            graph.add_node(page_id, "page", f"Page {current}", proof)
            region_id = f"region:{record_name}"
            graph.add_node(region_id, "region", record_name, proof)
            graph.add_edge(page_id, region_id, "contains", proof)


def ingest_references(data: dict[str, Any], path: Path, graph: RelationshipGraph) -> None:
    for key, page in page_entries(data):
        if not isinstance(page, dict):
            continue
        current = page_number(key)
        if current is None:
            continue
        page_id = f"page:{current}"
        for record_name, record in page.items():
            if not isinstance(record, dict):
                continue
            edge_type = REFERENCE_EDGE_TYPES.get(record_name)
            if edge_type is None and "callout_texts" not in record and "value" not in record:
                continue
            edge_type = edge_type or "calls"
            for text, box, target in aligned_rows(record):
                proof = evidence_for(path, record_name, text, current, box)
                graph.add_node(page_id, "page", f"Page {current}", proof)
                for node_type, code, raw in callout_targets(text):
                    target_id = f"{node_type}:{code}"
                    graph.add_node(target_id, node_type, code, proof)
                    graph.add_edge(page_id, target_id, edge_type, evidence_for(path, record_name, raw, current, box))
                target_page = page_number(target)
                if target_page is not None and target_page != current and text:
                    other_id = f"page:{target_page}"
                    graph.add_node(other_id, "page", f"Page {target_page}")
                    graph.add_edge(
                        page_id,
                        other_id,
                        "opens",
                        evidence_for(path, record_name, text, current, box),
                    )


def ingest_ocr(data: dict[str, Any], path: Path, graph: RelationshipGraph) -> None:
    known = {node.label: node.id for node in graph.nodes.values() if node.type == "mark"}
    if not known:
        graph.warnings.append(f"{path.name}: add a schedule or tag JSON before OCR can be tied to marks.")
        return
    for key, words in page_entries(data):
        if not isinstance(words, list):
            continue
        current = page_number(key)
        if current is None:
            continue
        for word in words:
            if not isinstance(word, dict):
                continue
            text = str(word.get("text", "")).strip()
            mark_id = known.get(text)
            if mark_id is None:
                continue
            confidence = word.get("confidence")
            if isinstance(confidence, (int, float)) and confidence < 0.45:
                continue
            box = word.get("bbox") if isinstance(word.get("bbox"), list) else None
            proof = evidence_for(path, "ocr", text, current, box if isinstance(box, list) else None)
            page_id = f"page:{current}"
            graph.add_node(page_id, "page", f"Page {current}", proof)
            plan_edge = f"{mark_id}|on_plan|{page_id}"
            if plan_edge in graph.edges:
                graph.add_edge(mark_id, page_id, "on_plan", proof)
            else:
                graph.add_edge(mark_id, page_id, "seen_on", proof)


def ingest_detected_items(
    items: list[Any],
    path: Path,
    graph: RelationshipGraph,
    display: int,
    pdf: str | None,
) -> None:
    page_id = f"page:{display}"
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("decision") == "rejected" or item.get("review", {}).get("decision") == "rejected":
            continue
        class_name = str(item.get("corrected_label") or item.get("class_name") or "object")
        text = str(item.get("ocr_text") or item.get("recognized_text") or item.get("corrected_text") or "").strip()
        raw_box = item.get("corrected_bbox_pdf") or item.get("bbox_pdf")
        box = raw_box if isinstance(raw_box, list) and len(raw_box) == 4 else None
        proof = evidence_for(path, class_name, text or class_name, display, box, pdf)
        graph.add_node(page_id, "page", f"Page {display}", proof)
        if text and class_name in {"reference", "callout_area"}:
            for node_type, code, raw in callout_targets(text):
                target_id = f"{node_type}:{code}"
                graph.add_node(target_id, node_type, code, proof)
                graph.add_edge(page_id, target_id, "calls", evidence_for(path, class_name, raw, display, box, pdf))
            continue
        if text and class_name in {"tag", "door_reference", "schedule_entry"} and MARK_RE.match(text.replace(" ", "")):
            edge_type = "on_schedule" if class_name == "schedule_entry" else "on_plan"
            link_mark(
                graph,
                text.replace(" ", ""),
                display if edge_type == "on_schedule" else None,
                display if edge_type == "on_plan" else None,
                path,
                class_name,
                box,
                display,
            )
            continue
        class_id = f"class:{class_name}"
        graph.add_node(class_id, "class", class_name.replace("_", " "), proof)
        graph.add_edge(page_id, class_id, "detected", proof)


def ingest_detections(data: dict[str, Any], path: Path, graph: RelationshipGraph) -> None:
    try:
        display = int(data["page_number"])
    except (KeyError, TypeError, ValueError):
        graph.warnings.append(f"{path.name}: detection file has no page number.")
        return
    pdf = str(data["source_pdf"]) if data.get("source_pdf") else None
    items = list(data.get("detections") or [])
    missed = data.get("missed_annotations") or {}
    if isinstance(missed, dict):
        items.extend(missed.values())
    ingest_detected_items(items, path, graph, display, pdf)


def ingest_review(data: dict[str, Any], path: Path, graph: RelationshipGraph) -> None:
    try:
        display = int(data["page_number"])
    except (KeyError, TypeError, ValueError):
        graph.warnings.append(f"{path.name}: review file has no page number.")
        return
    items = []
    annotations = data.get("annotations") or {}
    if isinstance(annotations, dict):
        items.extend(annotations.values())
    missed = data.get("missed_annotations") or {}
    if isinstance(missed, dict):
        items.extend(missed.values())
    ingest_detected_items(items, path, graph, display, None)


def ingest_graph_document(data: dict[str, Any], path: Path, graph: RelationshipGraph) -> None:
    for raw in data.get("nodes") or []:
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        label = str(raw.get("label") or raw["id"])
        graph.add_node(str(raw["id"]), str(raw.get("type") or "item"), label, evidence_for(path, "graph", label, None, None))
    for raw in data.get("edges") or []:
        if not isinstance(raw, dict):
            continue
        src = raw.get("src", raw.get("source"))
        dst = raw.get("dst", raw.get("target"))
        if not src or not dst:
            continue
        edge_type = str(raw.get("type") or raw.get("relation") or "related_to")
        graph.add_node(str(src), "item", str(src), evidence_for(path, "graph", str(src), None, None))
        graph.add_node(str(dst), "item", str(dst), evidence_for(path, "graph", str(dst), None, None))
        graph.add_edge(str(src), str(dst), edge_type, evidence_for(path, edge_type, "", None, None))


def ingest_generic(data: Any, path: Path, graph: RelationshipGraph) -> None:
    created = 0

    def walk(value: Any, depth: int) -> None:
        nonlocal created
        if created >= 400 or depth > 5:
            return
        if isinstance(value, dict):
            src = value.get("src", value.get("source", value.get("from")))
            dst = value.get("dst", value.get("target", value.get("to")))
            if isinstance(src, str) and isinstance(dst, str):
                edge_type = str(value.get("type") or value.get("relation") or value.get("relationship") or "related_to")
                proof = evidence_for(path, edge_type, "", None, None)
                graph.add_node(f"item:{src}", "item", src, proof)
                graph.add_node(f"item:{dst}", "item", dst, proof)
                graph.add_edge(f"item:{src}", f"item:{dst}", edge_type, proof)
                created += 1
            label = value.get("label", value.get("name", value.get("id")))
            if isinstance(label, str) and label.strip() and any(key in value for key in ("bbox", "bboxes", "id")):
                found = boxes_of(value)
                proof = evidence_for(path, "item", label, None, found[0] if found else None)
                graph.add_node(f"item:{label}", "item", label.strip(), proof)
                created += 1
            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child, depth + 1)
        elif isinstance(value, list):
            for child in value[:300]:
                if isinstance(child, (dict, list)):
                    walk(child, depth + 1)

    before = len(graph.nodes)
    walk(data, 0)
    if len(graph.nodes) == before:
        graph.warnings.append(f"{path.name}: no nodes or relationships were found in this JSON.")


def build_relationship_graph(json_paths: list[Path], pdf_paths: list[Path] | None = None) -> RelationshipGraph:
    graph = RelationshipGraph(pdf_paths=[str(path) for path in (pdf_paths or []) if path.is_file()])
    loaded: list[tuple[str, Path, Any]] = []
    order = {
        "graph": 0,
        "schedule_outputs": 1,
        "schedule_annotations": 2,
        "references": 3,
        "detections": 4,
        "review": 5,
        "ocr": 6,
        "generic": 7,
    }
    for path in json_paths:
        if not path.is_file():
            graph.warnings.append(f"{path.name}: file was not found.")
            continue
        try:
            data = load_json(path)
        except json.JSONDecodeError as exc:
            graph.warnings.append(f"{path.name}: not valid JSON (line {exc.lineno}).")
            continue
        loaded.append((detect_kind(data), path, data))
        graph.files.append(path.name)

    for kind, path, data in sorted(loaded, key=lambda item: order.get(item[0], 9)):
        if kind == "graph":
            ingest_graph_document(data, path, graph)
        elif kind == "schedule_outputs":
            ingest_schedule_outputs(data, path, graph)
        elif kind == "schedule_annotations":
            ingest_schedule_annotations(data, path, graph)
        elif kind == "references":
            ingest_references(data, path, graph)
        elif kind == "detections":
            ingest_detections(data, path, graph)
        elif kind == "review":
            ingest_review(data, path, graph)
        elif kind == "ocr":
            ingest_ocr(data, path, graph)
        else:
            ingest_generic(data, path, graph)
    return graph


def bbox_to_pdf(bbox: list[float], page_width: float, page_height: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if right <= page_width * 1.08 and bottom <= page_height * 1.08:
        return left, top, right, bottom
    for source_width, source_height in ((2592.0, 1728.0), (5400.0, 3600.0)):
        scale_x = page_width / source_width
        scale_y = page_height / source_height
        if right * scale_x <= page_width * 1.08 and bottom * scale_y <= page_height * 1.08:
            return left * scale_x, top * scale_y, right * scale_x, bottom * scale_y
    scale_x = page_width / max(right, 1.0)
    scale_y = page_height / max(bottom, 1.0)
    return left * scale_x, top * scale_y, min(page_width, right * scale_x), min(page_height, bottom * scale_y)
