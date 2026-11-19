"""Relationship map for annotated JSON and PDFs.

The layout follows an editor graph: typed nodes, a force pass to separate
clusters, and a side pane that opens the PDF region an edge came from.
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path
from typing import Any

import pdfplumber
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from json_plot import theme
from json_plot.relationships import (
    DASHED_EDGES,
    EDGE_LABELS,
    EDGE_TYPE_ORDER,
    HIDDEN_BY_DEFAULT,
    NODE_TYPE_ORDER,
    TYPE_LABELS,
    Evidence,
    RelationshipGraph,
    bbox_to_pdf,
    build_relationship_graph,
)


CROP_CACHE: dict[tuple[Any, ...], QImage] = {}

# Colour roles for graph nodes and edges. Names resolve to palette tokens in json_plot.theme.
TYPE_COLORS = {
    "page": "text",
    "mark": "teal",
    "sheet": "amber",
    "index": "violet",
    "region": "slate",
    "class": "blue",
    "item": "text_muted",
}
EDGE_COLORS = {
    "on_schedule": "teal",
    "on_plan": "blue",
    "missing": "red",
    "calls": "amber",
    "bad_call": "red",
    "opens": "amber",
    "indexes": "violet",
    "missing_index": "red",
    "unresolved": "red",
    "contains": "slate",
    "seen_on": "slate",
    "detected": "blue",
    "related_to": "text_muted",
}


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def chip(text: str, hue: str, checked: bool) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("chip")
    theme.set_property(button, "hue", hue)
    button.setCheckable(True)
    button.setChecked(checked)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def render_crop(pdf_path: Path, page_index: int, bbox: list[float] | None) -> QImage:
    key = (
        str(pdf_path),
        page_index,
        None if bbox is None else tuple(round(float(value), 1) for value in bbox),
    )
    cached = CROP_CACHE.get(key)
    if cached is not None:
        return cached
    with pdfplumber.open(pdf_path) as pdf:
        if page_index < 0 or page_index >= len(pdf.pages):
            raise IndexError(f"Page {page_index + 1} is outside {pdf_path.name}.")
        page = pdf.pages[page_index]
        resolution = 120 if bbox else 40
        image = page.to_image(resolution=resolution).original.convert("RGB")
        if bbox:
            left, top, right, bottom = bbox_to_pdf(bbox, float(page.width), float(page.height))
            pad = 32
            left = max(0.0, left - pad)
            top = max(0.0, top - pad)
            right = min(float(page.width), right + pad)
            bottom = min(float(page.height), bottom + pad)
            scale = resolution / 72.0
            image = image.crop((
                int(left * scale),
                int(top * scale),
                max(int(left * scale) + 2, int(right * scale)),
                max(int(top * scale) + 2, int(bottom * scale)),
            ))
    image = image.convert("RGBA")
    qimage = QImage(
        image.tobytes("raw", "RGBA"),
        image.width,
        image.height,
        image.width * 4,
        QImage.Format.Format_RGBA8888,
    ).copy()
    CROP_CACHE[key] = qimage
    return qimage


class GraphCanvas(QWidget):
    nodeActivated = Signal(str)
    edgeActivated = Signal(str)
    focusRequested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.graph = RelationshipGraph()
        self.display: dict[str, dict[str, Any]] = {}
        self.node_types: set[str] = set()
        self.edge_types: set[str] = set()
        self.query = ""
        self.focus_id: str | None = None
        self.depth = 1
        self.selected_id: str | None = None
        self.selected_edge: str | None = None
        self.hovered_id: str | None = None
        self.pan_x = 40.0
        self.pan_y = 40.0
        self.zoom = 1.0
        self._needs_fit = False
        self._drag_node: dict[str, Any] | None = None
        self._panning = False
        self._last = QPointF()

    def set_graph(self, graph: RelationshipGraph) -> None:
        self.graph = graph
        self.display = {}
        self.selected_id = None
        self.selected_edge = None
        self.hovered_id = None
        self.focus_id = None
        degrees = {node_id: 0 for node_id in graph.nodes}
        for edge in graph.edges.values():
            degrees[edge.src] = degrees.get(edge.src, 0) + 1
            degrees[edge.dst] = degrees.get(edge.dst, 0) + 1
        for node in graph.nodes.values():
            degree = degrees.get(node.id, 0)
            radius = 12 + min(16, math.sqrt(degree) * 3.2)
            if node.type == "page":
                radius += 6
            self.display[node.id] = {
                "id": node.id,
                "type": node.type,
                "label": node.label,
                "x": 0.0,
                "y": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "r": radius,
            }
        self._seed()
        self._relax()
        self._needs_fit = True
        self.update()

    def set_view_state(
        self,
        node_types: set[str],
        edge_types: set[str],
        query: str,
        focus_id: str | None,
        depth: int,
    ) -> None:
        self.node_types = node_types
        self.edge_types = edge_types
        self.query = query.strip().casefold()
        self.focus_id = focus_id
        self.depth = max(1, depth)
        self.update()

    def visible_nodes(self) -> list[dict[str, Any]]:
        allowed = [node for node in self.display.values() if node["type"] in self.node_types]
        by_id = {node["id"]: node for node in allowed}
        edges = self._edges_among(set(by_id))
        if self.query:
            seeds = {node["id"] for node in allowed if self.query in node["label"].casefold()}
            expanded = set(seeds)
            for edge in edges:
                if edge.src in seeds:
                    expanded.add(edge.dst)
                if edge.dst in seeds:
                    expanded.add(edge.src)
            by_id = {node_id: node for node_id, node in by_id.items() if node_id in expanded}
            edges = self._edges_among(set(by_id))
        if self.focus_id and self.focus_id in self.display:
            expanded = {self.focus_id}
            frontier = {self.focus_id}
            for _ in range(self.depth):
                nxt = set()
                for edge in self._edges_among(set(self.display) & self._ids_of_types()):
                    if edge.src in frontier and edge.dst in by_id:
                        nxt.add(edge.dst)
                    if edge.dst in frontier and edge.src in by_id:
                        nxt.add(edge.src)
                frontier = nxt - expanded
                expanded |= frontier
            by_id = {node_id: node for node_id, node in by_id.items() if node_id in expanded}
        return list(by_id.values())

    def visible_edges(self) -> list[Any]:
        ids = {node["id"] for node in self.visible_nodes()}
        return self._edges_among(ids)

    def fit(self) -> None:
        nodes = self.visible_nodes()
        if not nodes or self.width() < 40 or self.height() < 40:
            self._needs_fit = True
            return
        pad = 70
        min_x = min(node["x"] - node["r"] for node in nodes) - pad
        max_x = max(node["x"] + node["r"] for node in nodes) + pad
        min_y = min(node["y"] - node["r"] for node in nodes) - pad
        max_y = max(node["y"] + node["r"] + 18 for node in nodes) + pad
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        self.zoom = min(1.35, (self.width() - 24) / span_x, (self.height() - 24) / span_y)
        self.zoom = max(0.12, self.zoom)
        self.pan_x = self.width() / 2 - (min_x + max_x) / 2 * self.zoom
        self.pan_y = self.height() / 2 - (min_y + max_y) / 2 * self.zoom
        self._needs_fit = False
        self.update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._needs_fit:
            self.fit()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), theme.color("viewport"))
        if not self.display:
            painter.setPen(theme.color("text_muted"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Add annotated JSON to build the map")
            return

        nodes = self.visible_nodes()
        edges = self.visible_edges()
        if not nodes:
            painter.setPen(theme.color("text_muted"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Nothing matches these filters")
            return
        by_id = {node["id"]: node for node in nodes}
        active = self.hovered_id or self.selected_id
        neighbors = self._neighborhood(active) if active else set()
        if self.selected_edge and self.selected_edge in self.graph.edges:
            edge = self.graph.edges[self.selected_edge]
            neighbors |= {edge.src, edge.dst}

        painter.setPen(Qt.PenStyle.NoPen)
        for node in nodes:
            if node["type"] != "page":
                continue
            sx, sy = self._screen(node["x"], node["y"])
            painter.setBrush(theme.color("text", 12))
            painter.drawEllipse(QPointF(sx, sy), 76 * self.zoom, 76 * self.zoom)

        for edge in edges:
            src = by_id.get(edge.src)
            dst = by_id.get(edge.dst)
            if src is None or dst is None:
                continue
            lit = not neighbors or edge.src in neighbors or edge.dst in neighbors or edge.id == self.selected_edge
            color = theme.color(EDGE_COLORS.get(edge.type, "text_muted"))
            strength = 210 if edge.id == self.selected_edge else 78 if edge.type == "on_schedule" and lit else 150 if lit else 24
            color.setAlpha(strength)
            pen = QPen(color, 2.2 if edge.id == self.selected_edge else 1.15)
            if edge.type in DASHED_EDGES:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            x1, y1 = self._screen(src["x"], src["y"])
            x2, y2 = self._screen(dst["x"], dst["y"])
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            if lit or edge.id == self.selected_edge:
                self._arrow(painter, x1, y1, x2, y2, dst["r"] * self.zoom + 3, color)

        font = theme.ui_font(8)
        painter.setFont(font)
        for node in nodes:
            lit = not neighbors or node["id"] in neighbors
            sx, sy = self._screen(node["x"], node["y"])
            radius = node["r"] * self.zoom
            color = theme.color(TYPE_COLORS.get(node["type"], "text_muted"))
            if not lit:
                color.setAlpha(50)
            painter.setPen(QPen(theme.color("app") if lit else theme.color("app", 40), 1.4))
            painter.setBrush(color)
            self._shape(painter, node["type"], sx, sy, radius)
            if node["type"] == "page" and lit and radius >= 11:
                painter.setPen(theme.color("on_hue"))
                painter.setFont(theme.ui_font(8, bold=True))
                painter.drawText(
                    QRectF(sx - radius, sy - 8, radius * 2, 16),
                    Qt.AlignmentFlag.AlignCenter,
                    node["label"].removeprefix("Page "),
                )
                painter.setFont(font)
            if node["id"] in {self.selected_id, self.hovered_id}:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(theme.color("orange"), 2))
                painter.drawEllipse(QPointF(sx, sy), radius + 4, radius + 4)
            if self._show_label(node, lit):
                if node["type"] == "page":
                    angle = math.atan2(node["y"], node["x"])
                    self._label(
                        painter,
                        node["label"],
                        sx - math.cos(angle) * (radius + 18),
                        sy - math.sin(angle) * (radius + 12),
                        lit,
                    )
                else:
                    self._label(painter, node["label"], sx, sy + radius + 2, lit)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._last = event.position()
        node = self._node_at(event.position())
        if node is not None:
            self.selected_id = node["id"]
            self.selected_edge = None
            self._drag_node = node
            self.nodeActivated.emit(node["id"])
            self.update()
            return
        edge = self._edge_at(event.position())
        if edge is not None:
            self.selected_edge = edge.id
            self.selected_id = None
            self.edgeActivated.emit(edge.id)
            self.update()
            return
        self._panning = True

    def mouseMoveEvent(self, event) -> None:
        if self._drag_node is not None:
            world = self._world(event.position())
            previous = self._world(self._last)
            self._drag_node["x"] += world[0] - previous[0]
            self._drag_node["y"] += world[1] - previous[1]
            self._last = event.position()
            self.update()
            return
        if self._panning:
            self.pan_x += event.position().x() - self._last.x()
            self.pan_y += event.position().y() - self._last.y()
            self._last = event.position()
            self.update()
            return
        node = self._node_at(event.position())
        hovered = None if node is None else node["id"]
        if hovered != self.hovered_id:
            self.hovered_id = hovered
            self.setCursor(Qt.CursorShape.PointingHandCursor if hovered else Qt.CursorShape.ArrowCursor)
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_node = None
        self._panning = False

    def mouseDoubleClickEvent(self, event) -> None:
        node = self._node_at(event.position())
        if node is not None:
            self.focusRequested.emit(node["id"])

    def wheelEvent(self, event) -> None:
        old = self.zoom
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        world_x = (event.position().x() - self.pan_x) / old
        world_y = (event.position().y() - self.pan_y) / old
        self.zoom = min(3.2, max(0.12, old * factor))
        self.pan_x = event.position().x() - world_x * self.zoom
        self.pan_y = event.position().y() - world_y * self.zoom
        self.update()

    def leaveEvent(self, event) -> None:
        self.hovered_id = None
        self.update()

    def _seed(self) -> None:
        orbit = 980
        pages = [node for node in self.display.values() if node["type"] == "page"]
        pages.sort(key=lambda node: self._page_sort(node["label"]))
        for index, node in enumerate(pages):
            angle = -math.pi / 2 + index * 2 * math.pi / max(1, len(pages))
            node["x"] = math.cos(angle) * orbit
            node["y"] = math.sin(angle) * orbit

        marks_by_page: dict[str, list[dict[str, Any]]] = {}
        outer_by_page: dict[str, list[dict[str, Any]]] = {}
        for node in self.display.values():
            if node["type"] == "page":
                continue
            plan_pages, other_pages = self._linked_pages(node["id"])
            if node["type"] == "mark":
                home = plan_pages[0] if plan_pages else (other_pages[0] if other_pages else None)
                if home is None:
                    angle = (zlib.adler32(node["id"].encode()) % 360) * math.pi / 180
                    node["x"] = math.cos(angle) * 180
                    node["y"] = math.sin(angle) * 180
                else:
                    marks_by_page.setdefault(home["id"], []).append(node)
                continue
            home = other_pages[0] if other_pages else (plan_pages[0] if plan_pages else None)
            if home is None:
                angle = (zlib.adler32(node["id"].encode()) % 360) * math.pi / 180
                node["x"] = math.cos(angle) * (orbit + 260)
                node["y"] = math.sin(angle) * (orbit + 260)
            else:
                outer_by_page.setdefault(home["id"], []).append(node)

        for page_id, marks in marks_by_page.items():
            page = self.display[page_id]
            radius = 92 + len(marks) * 1.35
            for index, mark in enumerate(marks):
                angle = -math.pi / 2 + index * 2 * math.pi / len(marks)
                mark["x"] = page["x"] + math.cos(angle) * radius
                mark["y"] = page["y"] + math.sin(angle) * radius

        for page_id, group in outer_by_page.items():
            page = self.display[page_id]
            base = math.atan2(page["y"], page["x"])
            for index, node in enumerate(group):
                angle = base + (index - (len(group) - 1) / 2) * 0.2
                node["x"] = math.cos(angle) * (orbit + 230)
                node["y"] = math.sin(angle) * (orbit + 230)

    def _linked_pages(self, node_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        plan, other = [], []
        for edge in self.graph.edges.values():
            if node_id == edge.src:
                other_id = edge.dst
            elif node_id == edge.dst:
                other_id = edge.src
            else:
                continue
            page = self.display.get(other_id)
            if page is None or page["type"] != "page":
                continue
            if edge.type in {"on_plan", "missing", "seen_on", "detected"}:
                plan.append(page)
            else:
                other.append(page)
        return plan, other

    def _relax(self) -> None:
        nodes = list(self.display.values())
        mobile = {"mark", "class", "item", "region"}
        for _ in range(24):
            for index, left in enumerate(nodes):
                for right in nodes[index + 1:]:
                    dx = right["x"] - left["x"]
                    dy = right["y"] - left["y"]
                    distance = math.hypot(dx, dy) or 0.01
                    gap = left["r"] + right["r"] + 8
                    if distance >= gap:
                        continue
                    push = (gap - distance) / distance
                    if left["type"] in mobile and right["type"] in mobile:
                        left["x"] -= dx * push * 0.5
                        left["y"] -= dy * push * 0.5
                        right["x"] += dx * push * 0.5
                        right["y"] += dy * push * 0.5
                    elif left["type"] in mobile:
                        left["x"] -= dx * push
                        left["y"] -= dy * push
                    elif right["type"] in mobile:
                        right["x"] += dx * push
                        right["y"] += dy * push

    def _edges_among(self, ids: set[str]) -> list[Any]:
        return [
            edge for edge in self.graph.edges.values()
            if edge.type in self.edge_types and edge.src in ids and edge.dst in ids
        ]

    def _ids_of_types(self) -> set[str]:
        return {node["id"] for node in self.display.values() if node["type"] in self.node_types}

    def _neighborhood(self, node_id: str | None) -> set[str]:
        if not node_id:
            return set()
        found = {node_id}
        visible = {node["id"] for node in self.visible_nodes()}
        for edge in self.visible_edges():
            if edge.src == node_id and edge.dst in visible:
                found.add(edge.dst)
            elif edge.dst == node_id and edge.src in visible:
                found.add(edge.src)
        return found

    def _show_label(self, node: dict[str, Any], lit: bool) -> bool:
        if node["id"] in {self.selected_id, self.hovered_id}:
            return True
        if not lit:
            return False
        if node["type"] == "mark":
            return self.zoom >= 0.9
        return True

    def _shape(self, painter: QPainter, node_type: str, sx: float, sy: float, radius: float) -> None:
        if node_type == "page":
            rect = QRectF(sx - radius * 1.2, sy - radius * 0.72, radius * 2.4, radius * 1.44)
            painter.drawRoundedRect(rect, 7, 7)
            return
        if node_type == "sheet":
            painter.drawPolygon(QPolygonF([
                QPointF(sx, sy - radius),
                QPointF(sx + radius, sy),
                QPointF(sx, sy + radius),
                QPointF(sx - radius, sy),
            ]))
            return
        if node_type == "index":
            points = []
            for step in range(6):
                angle = math.pi / 6 + step * math.pi / 3
                points.append(QPointF(sx + math.cos(angle) * radius, sy + math.sin(angle) * radius))
            painter.drawPolygon(QPolygonF(points))
            return
        if node_type == "region":
            painter.drawRect(QRectF(sx - radius * 0.86, sy - radius * 0.86, radius * 1.72, radius * 1.72))
            return
        painter.drawEllipse(QPointF(sx, sy), radius, radius)

    def _label(self, painter: QPainter, text: str, sx: float, sy: float, lit: bool) -> None:
        label = text if len(text) <= 18 else text[:17] + "…"
        width = painter.fontMetrics().horizontalAdvance(label) + 8
        box = QRectF(sx - width / 2, sy, width, 15)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(theme.color("app", 220 if lit else 70))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(theme.color("text") if lit else theme.color("text", 70))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, label)

    def _arrow(self, painter: QPainter, x1: float, y1: float, x2: float, y2: float, inset: float, color: QColor) -> None:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1
        if length < inset + 8:
            return
        end_x = x2 - dx / length * inset
        end_y = y2 - dy / length * inset
        ux, uy = dx / length, dy / length
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([
            QPointF(end_x, end_y),
            QPointF(end_x - uy * 4 - ux * 8, end_y + ux * 4 - uy * 8),
            QPointF(end_x + uy * 4 - ux * 8, end_y - ux * 4 - uy * 8),
        ]))

    def _screen(self, x: float, y: float) -> tuple[float, float]:
        return x * self.zoom + self.pan_x, y * self.zoom + self.pan_y

    def _world(self, point: QPointF) -> tuple[float, float]:
        return (point.x() - self.pan_x) / self.zoom, (point.y() - self.pan_y) / self.zoom

    def _node_at(self, point: QPointF) -> dict[str, Any] | None:
        world_x, world_y = self._world(point)
        found = None
        for node in self.visible_nodes():
            limit = node["r"] + 4
            if (node["x"] - world_x) ** 2 + (node["y"] - world_y) ** 2 <= limit * limit:
                found = node
        return found

    def _edge_at(self, point: QPointF) -> Any | None:
        world_x, world_y = self._world(point)
        by_id = {node["id"]: node for node in self.visible_nodes()}
        closest = None
        closest_distance = 7
        for edge in self.visible_edges():
            src = by_id.get(edge.src)
            dst = by_id.get(edge.dst)
            if src is None or dst is None:
                continue
            distance = _segment_distance(world_x, world_y, src["x"], src["y"], dst["x"], dst["y"])
            if distance < closest_distance:
                closest = edge
                closest_distance = distance
        return closest

    @staticmethod
    def _page_sort(label: str) -> int:
        digits = "".join(character for character in label if character.isdigit())
        return int(digits or 0)


def _segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    length = dx * dx + dy * dy
    if length == 0:
        return math.hypot(px - x1, py - y1)
    ratio = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length))
    return math.hypot(px - (x1 + ratio * dx), py - (y1 + ratio * dy))


class RelationshipPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.json_paths: list[Path] = []
        self.pdf_paths: list[Path] = []
        self.output_dir: Path | None = None
        self.graph = RelationshipGraph()
        self.focus_id: str | None = None
        self._chips: list[QPushButton] = []
        self._build()

    def set_inputs(self, json_paths: list[Path], pdf_paths: list[Path], output_dir: Path | None) -> None:
        self.json_paths = [path for path in json_paths if path.is_file()]
        self.pdf_paths = [path for path in pdf_paths if path.is_file()]
        self.output_dir = output_dir
        self.rebuild()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        intro = QLabel("Feed annotated JSON and PDFs. The map links marks, schedules, callouts, and the page each one was found on.")
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        files = QHBoxLayout()
        self.file_label = QLabel("No files loaded")
        self.file_label.setObjectName("fileSummary")
        self.file_label.setWordWrap(True)
        files.addWidget(self.file_label, 1)
        add_json = QPushButton("Add JSON")
        add_json.clicked.connect(self.add_json)
        add_pdf = QPushButton("Add PDF")
        add_pdf.clicked.connect(self.add_pdf)
        fit = QPushButton("Fit")
        fit.clicked.connect(self.canvas_fit)
        export = QPushButton("Export JSON")
        export.clicked.connect(self.export_graph)
        for button in (add_json, add_pdf, fit, export):
            files.addWidget(button)
        layout.addLayout(files)

        self.node_filters = QHBoxLayout()
        self.node_filters.setSpacing(6)
        layout.addLayout(self.node_filters)

        self.edge_host = QWidget()
        self.edge_filters = QHBoxLayout(self.edge_host)
        self.edge_filters.setContentsMargins(0, 0, 0, 0)
        self.edge_filters.setSpacing(6)
        edge_scroll = QScrollArea()
        edge_scroll.setWidgetResizable(True)
        edge_scroll.setWidget(self.edge_host)
        edge_scroll.setFixedHeight(42)
        edge_scroll.setFrameShape(QFrame.Shape.NoFrame)
        edge_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(edge_scroll)

        tools = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a mark, page, or sheet")
        self.search.textChanged.connect(self.apply_filters)
        tools.addWidget(self.search, 1)
        tools.addWidget(QLabel("Neighborhood"))
        self.depth = QSpinBox()
        self.depth.setRange(1, 3)
        self.depth.setValue(1)
        self.depth.valueChanged.connect(self._depth_changed)
        tools.addWidget(self.depth)
        whole = QPushButton("Whole map")
        whole.clicked.connect(self.clear_focus)
        tools.addWidget(whole)
        layout.addLayout(tools)

        self.status = QLabel("Waiting for JSON")
        self.status.setObjectName("muted")
        layout.addWidget(self.status)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.canvas = GraphCanvas()
        self.canvas.nodeActivated.connect(self.show_node)
        self.canvas.edgeActivated.connect(self.show_edge)
        self.canvas.focusRequested.connect(self.focus_node)
        splitter.addWidget(self.canvas)

        detail = QFrame()
        detail.setObjectName("panel")
        detail.setMinimumWidth(280)
        detail.setMaximumWidth(360)
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(14, 14, 14, 14)
        self.detail_kind = QLabel("MAP")
        self.detail_kind.setObjectName("caption")
        self.detail_title = QLabel("Select a node")
        self.detail_title.setObjectName("rowTitle")
        self.detail_title.setWordWrap(True)
        self.detail_note = QLabel("Click a node for its links. Double-click to keep only that neighborhood. Drag empty space to pan, and use the wheel to zoom.")
        self.detail_note.setObjectName("muted")
        self.detail_note.setWordWrap(True)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(150)
        self.preview.setObjectName("preview")
        self.preview.setText("The annotated region shows up here")
        self.link_list = QListWidget()
        self.link_list.currentItemChanged.connect(self._activate_link_row)
        self.evidence_list = QListWidget()
        self.evidence_list.currentItemChanged.connect(self._show_evidence_row)
        detail_layout.addWidget(self.detail_kind)
        detail_layout.addWidget(self.detail_title)
        detail_layout.addWidget(self.detail_note)
        detail_layout.addWidget(self.preview)
        links_label = QLabel("Links")
        links_label.setObjectName("caption")
        evidence_label = QLabel("Where it was read")
        evidence_label.setObjectName("caption")
        detail_layout.addWidget(links_label)
        detail_layout.addWidget(self.link_list, 1)
        detail_layout.addWidget(evidence_label)
        detail_layout.addWidget(self.evidence_list, 1)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 1)
        layout.addWidget(splitter, 1)

    def canvas_fit(self) -> None:
        self.canvas.fit()

    def add_json(self) -> None:
        start = str(self.json_paths[-1].parent if self.json_paths else Path.cwd())
        paths, _ = QFileDialog.getOpenFileNames(self, "Add annotated JSON", start, "JSON (*.json)")
        if not paths:
            return
        existing = {str(path) for path in self.json_paths}
        for raw in paths:
            if raw not in existing:
                self.json_paths.append(Path(raw))
        self.rebuild()

    def add_pdf(self) -> None:
        start = str(self.pdf_paths[-1].parent if self.pdf_paths else Path.cwd())
        paths, _ = QFileDialog.getOpenFileNames(self, "Add annotated PDF", start, "PDF (*.pdf)")
        if not paths:
            return
        existing = {str(path) for path in self.pdf_paths}
        for raw in paths:
            if raw not in existing:
                self.pdf_paths.append(Path(raw))
        self.rebuild()

    def export_graph(self) -> None:
        if not self.graph.nodes:
            self.status.setText("Build a map before exporting.")
            return
        default = str((self.output_dir or Path.cwd()) / "relationship_graph.json")
        path, _ = QFileDialog.getSaveFileName(self, "Export relationship graph", default, "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(json.dumps(self.graph.to_dict(), indent=2) + "\n", encoding="utf-8")
        self.status.setText(f"Exported {Path(path).name}")

    def rebuild(self) -> None:
        names = [path.name for path in self.json_paths + self.pdf_paths]
        self.file_label.setText("  ·  ".join(names) if names else "No files loaded")
        try:
            self.graph = build_relationship_graph(self.json_paths, self.pdf_paths)
        except Exception as exc:
            self.status.setText(f"Could not read the files: {exc}")
            return
        self.focus_id = None
        self.canvas.set_graph(self.graph)
        self._rebuild_filters()
        self.apply_filters()
        self.canvas.fit()
        self._reset_detail()
        if self.graph.warnings:
            self.status.setText(self.status.text() + "  ·  " + " ".join(self.graph.warnings))

    def _rebuild_filters(self) -> None:
        self._chips.clear()
        _clear(self.node_filters)
        _clear(self.edge_filters)
        present_nodes = {node.type for node in self.graph.nodes.values()}
        present_edges = {edge.type for edge in self.graph.edges.values()}
        for name in NODE_TYPE_ORDER:
            if name not in present_nodes:
                continue
            button = chip(TYPE_LABELS.get(name, name), TYPE_COLORS.get(name, "text_muted"), name not in HIDDEN_BY_DEFAULT)
            button.setProperty("filterKind", "node")
            button.setProperty("filterName", name)
            button.toggled.connect(self.apply_filters)
            self.node_filters.addWidget(button)
            self._chips.append(button)
        self.node_filters.addStretch(1)
        for name in list(EDGE_TYPE_ORDER) + sorted(present_edges - set(EDGE_TYPE_ORDER)):
            if name not in present_edges:
                continue
            button = chip(EDGE_LABELS.get(name, name), EDGE_COLORS.get(name, "text_muted"), name not in HIDDEN_BY_DEFAULT)
            button.setProperty("filterKind", "edge")
            button.setProperty("filterName", name)
            button.toggled.connect(self.apply_filters)
            self.edge_filters.addWidget(button)
            self._chips.append(button)
        self.edge_filters.addStretch(1)

    def apply_filters(self) -> None:
        nodes = {button.property("filterName") for button in self._chips if button.property("filterKind") == "node" and button.isChecked()}
        edges = {button.property("filterName") for button in self._chips if button.property("filterKind") == "edge" and button.isChecked()}
        self.canvas.set_view_state(nodes, edges, self.search.text(), self.focus_id, self.depth.value())
        visible_nodes = self.canvas.visible_nodes()
        visible_edges = self.canvas.visible_edges()
        focus = ""
        if self.focus_id and self.focus_id in self.graph.nodes:
            focus = f"  ·  around {self.graph.nodes[self.focus_id].label}"
        self.status.setText(f"{len(visible_nodes)} nodes  ·  {len(visible_edges)} links{focus}")

    def clear_focus(self) -> None:
        self.focus_id = None
        self.search.clear()
        self.apply_filters()
        self.canvas.fit()

    def focus_node(self, node_id: str) -> None:
        self.focus_id = node_id
        self.apply_filters()
        self.canvas.fit()
        self.show_node(node_id)

    def _depth_changed(self) -> None:
        if self.canvas.selected_id and self.focus_id is None:
            self.focus_id = self.canvas.selected_id
        self.apply_filters()
        if self.focus_id:
            self.canvas.fit()

    def show_node(self, node_id: str) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None:
            return
        self.detail_kind.setText(TYPE_LABELS.get(node.type, node.type))
        self.detail_title.setText(node.label)
        links = [edge for edge in self.graph.edges.values() if node_id in (edge.src, edge.dst)]
        self.detail_note.setText(f"{node.mentions} mentions  ·  {len(links)} links")
        self._fill_links(links, node_id)
        self._fill_evidence(node.evidences)
        self._preview_evidence(node.evidences[0] if node.evidences else None)

    def show_edge(self, edge_id: str) -> None:
        edge = self.graph.edges.get(edge_id)
        if edge is None:
            return
        src = self.graph.nodes[edge.src]
        dst = self.graph.nodes[edge.dst]
        self.detail_kind.setText(EDGE_LABELS.get(edge.type, edge.type))
        self.detail_title.setText(f"{src.label}  →  {dst.label}")
        self.detail_note.setText(f"{edge.count} times in the JSON")
        self._fill_links([edge], edge.src)
        self._fill_evidence(edge.evidences)
        self._preview_evidence(edge.evidences[0] if edge.evidences else None)

    def _fill_links(self, edges: list[Any], anchor: str) -> None:
        self.link_list.blockSignals(True)
        self.link_list.clear()
        ordered = sorted(edges, key=lambda edge: (-edge.count, edge.type))
        for edge in ordered[:200]:
            other_id = edge.dst if edge.src == anchor else edge.src
            other = self.graph.nodes.get(other_id)
            other_label = other.label if other else other_id
            direction = "→" if edge.src == anchor else "←"
            item = QListWidgetItem(f"{EDGE_LABELS.get(edge.type, edge.type)}  {direction}  {other_label}")
            item.setData(Qt.ItemDataRole.UserRole, other_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, edge.id)
            self.link_list.addItem(item)
        self.link_list.blockSignals(False)

    def _fill_evidence(self, evidences: list[Evidence]) -> None:
        self.evidence_list.blockSignals(True)
        self.evidence_list.clear()
        for evidence in evidences:
            page = "" if evidence.display_page is None else f"Page {evidence.display_page}  ·  "
            text = evidence.text or evidence.kind
            item = QListWidgetItem(f"{page}{evidence.kind}  ·  {text}")
            item.setData(Qt.ItemDataRole.UserRole, evidence)
            self.evidence_list.addItem(item)
        self.evidence_list.blockSignals(False)

    def _activate_link_row(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        other_id = current.data(Qt.ItemDataRole.UserRole)
        edge_id = current.data(Qt.ItemDataRole.UserRole + 1)
        if isinstance(other_id, str) and other_id in self.graph.nodes:
            self.canvas.selected_id = other_id
            self.canvas.selected_edge = edge_id if isinstance(edge_id, str) else None
            self.canvas.update()
            self.show_node(other_id)

    def _show_evidence_row(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        evidence = current.data(Qt.ItemDataRole.UserRole)
        if isinstance(evidence, Evidence):
            self._preview_evidence(evidence)

    def _preview_evidence(self, evidence: Evidence | None) -> None:
        if evidence is None or evidence.pdf_page_index is None:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("No page region for this item")
            return
        pdf_path = self._pdf_for(evidence)
        if pdf_path is None:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Add the annotated PDF to see this region")
            return
        try:
            image = render_crop(pdf_path, evidence.pdf_page_index, evidence.bbox)
        except Exception as exc:
            self.preview.setPixmap(QPixmap())
            self.preview.setText(str(exc))
            return
        pixmap = QPixmap.fromImage(image)
        self.preview.setPixmap(pixmap.scaled(
            max(240, self.preview.width() - 8),
            max(140, self.preview.height() - 8),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def _pdf_for(self, evidence: Evidence) -> Path | None:
        if evidence.pdf and Path(evidence.pdf).is_file():
            return Path(evidence.pdf)
        for raw in self.graph.pdf_paths:
            path = Path(raw)
            if path.is_file():
                return path
        return None

    def _reset_detail(self) -> None:
        self.detail_kind.setText("MAP")
        self.detail_title.setText("Select a node")
        self.detail_note.setText("Click a node for its links. Double-click to keep only that neighborhood. Drag empty space to pan, and use the wheel to zoom.")
        self.link_list.clear()
        self.evidence_list.clear()
        self.preview.setPixmap(QPixmap())
        self.preview.setText("The annotated region shows up here")
