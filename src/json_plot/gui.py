import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pdfplumber
from PySide6.QtCore import QByteArray, QEvent, QPointF, QProcess, QProcessEnvironment, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QPlainTextEdit,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl

from json_plot.config import (
    ANNOTATION_PDF_FILE,
    CLASS_NAMES,
    DEFAULT_WEIGHTS,
    PROJECT_ROOT,
    TRAINING_PDF_FILE,
)
from json_plot.graph_view import RelationshipPage
from json_plot.hitl_coach import detection_brief, page_brief
from json_plot.hitl_session import LABEL_LEGEND, collect_page_items, pdf_page_count, pdf_page_size
from json_plot import icons, pdf_render, theme
from json_plot.json_watch import JsonWatcher, json_file_state, scan_json_folder
from json_plot.project_detect import detect_sources
from json_plot.projects import ProjectLibrary, slugify
from json_plot.qaqc import (
    CATEGORY_LABELS,
    ReviewTask,
    add_item,
    build_tasks,
    destination_label,
    item_in_category,
    load_page_map,
    load_sheet_titles,
    packet_files,
    parse_review_csv,
    related_keeps,
    remove_item,
)


def read_json(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def panel() -> QFrame:
    widget = QFrame()
    widget.setObjectName("panel")
    return widget


def metric(title: str, value: str, note: str = "") -> QFrame:
    item = panel()
    layout = QVBoxLayout(item)
    layout.setContentsMargins(18, 16, 18, 15)
    eyebrow = QLabel(title)
    eyebrow.setObjectName("caption")
    amount = QLabel(value)
    amount.setObjectName("metricValue")
    detail = QLabel(note)
    detail.setObjectName("muted")
    layout.addWidget(eyebrow)
    layout.addWidget(amount)
    layout.addWidget(detail)
    item.value_label = amount
    item.note_label = detail
    return item


class ElidedLabel(QLabel):
    """Single-line label that elides its text instead of growing taller."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._full_text = text
        self.setWordWrap(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._full_text = text
        self._elide()

    def text(self) -> str:  # noqa: N802 - Qt API
        return self._full_text

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        metrics = QFontMetrics(self.font())
        shown = metrics.elidedText(self._full_text, Qt.TextElideMode.ElideRight, self.contentsRect().width())
        super().setText(shown)


class ReviewCanvas(QWidget):
    box_created = Signal(float, float, float, float)
    detection_hovered = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(420, 400)
        self.setMouseTracking(True)
        self.image = QPixmap()
        self.detections: list[dict[str, Any]] = []
        self.review: dict[str, Any] = {}
        self.page_width = 1.0
        self.page_height = 1.0
        self.drawing = False
        self.marking = False
        self.hovered_detection_id = None
        self.zoom = 1.0
        self.start_point = None
        self.end_point = None
        self._panning = False
        self._pan_button = None
        self._pan_last = None
        self.source: tuple[Path, int] | None = None
        self.max_render_width = 4096

    def set_max_render_width(self, width: int) -> None:
        """Cap the pixel width used to rasterise PDF tiles (higher = sharper, more memory)."""
        width = max(256, int(width))
        if width == self.max_render_width:
            return
        self.max_render_width = width
        pdf_render.clear_tiles()
        self.update()

    def set_source(self, path: Path | None, page_index: int = 0) -> None:
        """Where sharp tiles come from. page_index is 0-based; None keeps the preview only."""
        self.source = (path, page_index) if path is not None else None
        self.update()

    def set_page(
        self,
        image: QPixmap,
        detections: list[dict[str, Any]],
        review: dict[str, Any],
        page_width: float,
        page_height: float,
    ) -> None:
        self.image = image
        self.detections = detections
        self.review = review
        self.page_width = page_width
        self.page_height = page_height
        self.hovered_detection_id = None
        self.resize(round(image.width() * self.zoom), round(image.height() * self.zoom))
        self.update()

    def set_zoom(self, zoom: float) -> None:
        self.zoom = min(8.0, max(0.08, zoom))
        if not self.image.isNull():
            self.resize(round(self.image.width() * self.zoom), round(self.image.height() * self.zoom))
        self.update()

    def set_marking(self, enabled: bool) -> None:
        self.marking = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def set_hovered_detection(self, detection_id: str | None) -> None:
        if self.hovered_detection_id != detection_id:
            self.hovered_detection_id = detection_id
            self.update()

    def _image_geometry(self):
        if self.image.isNull():
            return 0.0, 0.0, 1.0, 1.0
        return 0.0, 0.0, float(self.image.width() * self.zoom), float(self.image.height() * self.zoom)

    def _to_image_point(self, point):
        x, y, width, height = self._image_geometry()
        return ((point.x() - x) * self.image.width() / width,
                (point.y() - y) * self.image.height() / height)

    def _detection_rect(self, detection: dict[str, Any]) -> QRectF:
        bbox = self.review.get("annotations", {}).get(detection.get("id"), {}).get("corrected_bbox_pdf")
        bbox = bbox or detection.get("bbox_pdf", [])
        if len(bbox) != 4:
            return QRectF()
        x, y, width, height = self._image_geometry()
        scale_x = width / self.image.width()
        scale_y = height / self.image.height()
        left = x + bbox[0] * self.image.width() / self.page_width * scale_x
        top = y + bbox[1] * self.image.height() / self.page_height * scale_y
        right = x + bbox[2] * self.image.width() / self.page_width * scale_x
        bottom = y + bbox[3] * self.image.height() / self.page_height * scale_y
        return QRectF(left, top, right - left, bottom - top)

    def _paint_source(self, painter: QPainter, target: QRect, dirty: QRect) -> bool:
        """Draw only the visible tiles from the PDF. False means fall back to the preview."""
        if self.source is None or self.source[0] is None:
            return False
        visible = dirty.intersected(target)
        if visible.isEmpty():
            return True
        try:
            pdf_render.paint_page(painter, target, visible, self.source[0], self.source[1], self.max_render_width)
        except Exception:
            # The PDF engine failed for this page; keep the preview and say why.
            logging.getLogger(__name__).exception("Tile rendering failed; showing preview")
            return False
        return True

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.color("viewport"))
        if self.image.isNull():
            painter.setPen(theme.color("text_muted"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Add a PDF and JSON to see boxes")
            return

        x, y, width, height = self._image_geometry()
        target = QRectF(x, y, width, height).toRect()
        if not self._paint_source(painter, target, event.rect()):
            painter.drawPixmap(target, self.image)
        scale_x = width / self.image.width()
        scale_y = height / self.image.height()

        ordered_detections = sorted(
            self.detections,
            key=lambda item: item.get("id") == self.hovered_detection_id,
        )
        for detection in ordered_detections:
            detection_id = detection.get("id")
            rect = self._detection_rect(detection)
            if rect.isNull():
                continue
            hue = "green" if detection.get("kept") else theme.hue_for_category(detection.get("family"))
            color = theme.color(hue)
            is_hovered = detection_id == self.hovered_detection_id
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 70 if is_hovered else 36))
            painter.drawRect(rect)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(theme.color("text") if is_hovered else color, 3 if is_hovered else 2))
            painter.drawRect(rect)
            caption = str(detection.get("ocr_text") or detection.get("family_label") or "").strip()
            if not caption:
                continue
            painter.setFont(theme.ui_font(9 if not is_hovered else 10, bold=is_hovered))
            shown = caption if len(caption) <= 42 else caption[:41] + "…"
            text_width = painter.fontMetrics().horizontalAdvance(shown) + 10
            label_rect = QRectF(rect.left(), max(y, rect.top() - 18), max(text_width, 14), 18)
            if is_hovered:
                painter.fillRect(label_rect, color)
                painter.setPen(theme.color("on_hue"))
            else:
                painter.fillRect(label_rect, theme.color("app", 225))
                painter.setPen(color)
            painter.drawText(label_rect.adjusted(3, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter, shown)

        for missed in self.review.get("missed_annotations", {}).values():
            bbox = missed.get("bbox_pdf", []) if isinstance(missed, dict) else []
            if len(bbox) != 4:
                continue
            left = x + bbox[0] * self.image.width() / self.page_width * scale_x
            top = y + bbox[1] * self.image.height() / self.page_height * scale_y
            right = x + bbox[2] * self.image.width() / self.page_width * scale_x
            bottom = y + bbox[3] * self.image.height() / self.page_height * scale_y
            color = theme.color(theme.hue_for_status("recovered" if missed.get("status") == "recovered" else "missed"))
            painter.setPen(QPen(color, 3, Qt.PenStyle.DashLine))
            painter.drawRect(round(left), round(top), round(right - left), round(bottom - top))

        if self.drawing and self.start_point and self.end_point:
            painter.setPen(QPen(theme.color(theme.hue_for_status("missed")), 2, Qt.PenStyle.DashLine))
            painter.drawRect(self.start_point, self.end_point)

    def mousePressEvent(self, event) -> None:
        # Middle button always pans. Left button pans when it starts on empty page, not on a box and not while drawing.
        left_background = (
            event.button() == Qt.MouseButton.LeftButton
            and not self.marking
            and not any(self._detection_rect(d).contains(event.position()) for d in self.detections)
        )
        if event.button() == Qt.MouseButton.MiddleButton or left_background:
            self._panning = True
            self._pan_button = event.button()
            self._pan_last = event.globalPosition().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if self.marking and event.button() == Qt.MouseButton.LeftButton and not self.image.isNull():
            self.start_point = event.position().toPoint()
            self.end_point = self.start_point
            self.drawing = True
            self.update()
            return
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:
        area = self._scroll_area()
        if area is None:
            super().wheelEvent(event)
            return
        area.zoom_from_canvas(event.position(), event.angleDelta().y())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._pan_last is not None:
            current = event.globalPosition().toPoint()
            delta = current - self._pan_last
            self._pan_last = current
            area = self._scroll_area()
            if area is not None:
                area.horizontalScrollBar().setValue(area.horizontalScrollBar().value() - delta.x())
                area.verticalScrollBar().setValue(area.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self.drawing:
            self.end_point = event.position().toPoint()
            self.update()
            return
        if not self.marking:
            hovered = None
            for detection in reversed(self.detections):
                if self._detection_rect(detection).contains(event.position()):
                    hovered = detection.get("id")
                    break
            if hovered != self.hovered_detection_id:
                self.hovered_detection_id = hovered
                self.detection_hovered.emit(hovered)
                self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self.detection_hovered.emit(None)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._panning and event.button() == self._pan_button:
            self._panning = False
            self._pan_last = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if self.drawing and self.start_point and self.end_point:
            start_x, start_y = self._to_image_point(self.start_point)
            end_x, end_y = self._to_image_point(self.end_point)
            left, right = sorted((start_x, end_x))
            top, bottom = sorted((start_y, end_y))
            self.drawing = False
            self.start_point = None
            self.end_point = None
            if right - left >= 3 and bottom - top >= 3:
                self.box_created.emit(left, top, right, bottom)
            self.update()
            return
        super().mouseReleaseEvent(event)

    def _scroll_area(self) -> "ImageScrollArea | None":
        parent = self.parent()
        while parent is not None and not isinstance(parent, ImageScrollArea):
            parent = parent.parent()
        return parent


class ImageScrollArea(QScrollArea):
    zoom_changed = Signal(float)

    def __init__(self, canvas: ReviewCanvas) -> None:
        super().__init__()
        self.canvas = canvas
        self.setWidget(canvas)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundRole(self.backgroundRole())
        self.setObjectName("viewer")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def zoom_from_canvas(self, canvas_point: QPointF, delta_y: int) -> None:
        if delta_y == 0 or self.canvas.image.isNull():
            return
        factor = 1.15 if delta_y > 0 else 1 / 1.15
        viewport_point = self.canvas.mapTo(self.viewport(), canvas_point.toPoint())
        self.set_zoom(self.canvas.zoom * factor, QPointF(viewport_point))

    def set_zoom(self, zoom: float, anchor: QPointF | None = None) -> None:
        old_zoom = self.canvas.zoom
        if self.canvas.image.isNull():
            return
        point = anchor or QPointF(self.viewport().width() / 2, self.viewport().height() / 2)
        image_x = (self.horizontalScrollBar().value() + point.x()) / old_zoom
        image_y = (self.verticalScrollBar().value() + point.y()) / old_zoom
        self.canvas.set_zoom(zoom)
        self.zoom_changed.emit(self.canvas.zoom)
        self.horizontalScrollBar().setValue(round(image_x * self.canvas.zoom - point.x()))
        self.verticalScrollBar().setValue(round(image_y * self.canvas.zoom - point.y()))

    def fit_to_view(self) -> None:
        if self.canvas.image.isNull():
            return
        viewport = self.viewport().size()
        zoom = min(
            viewport.width() / self.canvas.image.width(),
            viewport.height() / self.canvas.image.height(),
        )
        self.canvas.set_zoom(zoom)
        self.zoom_changed.emit(self.canvas.zoom)
        self.horizontalScrollBar().setValue(0)
        self.verticalScrollBar().setValue(0)

    def wheelEvent(self, event) -> None:
        pointer = self.viewport().mapFrom(self, event.position().toPoint())
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.set_zoom(self.canvas.zoom * factor, QPointF(pointer))
        event.accept()


class ProjectCreationDialog(QDialog):
    LABELS = {
        "training_pdf": ("Labeled training PDF", "PDF files (*.pdf)"),
        "annotation_pdf": ("PDF to annotate", "PDF files (*.pdf)"),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create visual annotation project")
        self.setMinimumWidth(820)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(12)
        self.name_field = QLineEdit("New annotation project")
        self.root_field = QLineEdit(str(Path.home() / "json-plot-projects" / "new-annotation-project"))
        self.auto_root = True
        self.name_field.textChanged.connect(self.update_project_folder)
        self.root_field.textEdited.connect(lambda: setattr(self, "auto_root", False))
        root_row = QHBoxLayout()
        root_row.addWidget(self.root_field, 1)
        root_browse = QPushButton("Choose")
        root_browse.clicked.connect(self.choose_root)
        root_row.addWidget(root_browse)
        form.addRow("Project name", self.name_field)
        form.addRow("Project folder", root_row)
        self.data_folder_field = QLineEdit(str(Path(PROJECT_ROOT) / "raw-data"))
        self.data_folder_field.setPlaceholderText("Folder that holds the PDF and JSON files")
        data_browse = QPushButton("Choose folder")
        data_browse.clicked.connect(self.choose_data_folder)
        data_row = QHBoxLayout()
        data_row.addWidget(self.data_folder_field, 1)
        data_row.addWidget(data_browse)
        form.addRow("Data folder", data_row)
        self.source_fields: dict[str, QLineEdit] = {}
        self.pdf_choosers: dict[str, QComboBox] = {}
        defaults = {
            "training_pdf": TRAINING_PDF_FILE,
            "annotation_pdf": ANNOTATION_PDF_FILE,
        }
        for key, (label, file_filter) in self.LABELS.items():
            row = QHBoxLayout()
            field = QLineEdit(str(defaults[key]))
            field.setPlaceholderText(label)
            browse = QPushButton("Browse")
            browse.clicked.connect(lambda checked=False, k=key, f=file_filter: self.choose_file(k, f))
            row.addWidget(field, 1)
            if key.endswith("pdf"):
                chooser = QComboBox()
                chooser.setMinimumWidth(200)
                chooser.setToolTip("PDFs found in the data folder")
                chooser.activated.connect(lambda index, k=key: self.source_fields[k].setText(self.pdf_choosers[k].itemData(index)))
                self.pdf_choosers[key] = chooser
                row.addWidget(chooser)
                field.textChanged.connect(lambda text, k=key: self._sync_pdf_chooser(k, text))
            row.addWidget(browse)
            self.source_fields[key] = field
            form.addRow(label, row)
        layout.addLayout(form)
        self.detect_note = QLabel("Every JSON file in the data folder is loaded. Choose the folder and the PDFs are filled in for you.")
        self.detect_note.setWordWrap(True)
        self.detect_note.setObjectName("muted")
        layout.addWidget(self.detect_note)
        note = QLabel("The selected source files stay where they are. Project outputs and reviews are stored in the project folder.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create project")
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.validate_and_accept)
        layout.addWidget(buttons)
        if Path(self.data_folder_field.text()).is_dir():
            self.apply_data_folder(Path(self.data_folder_field.text()))

    def choose_root(self) -> None:
        current = Path(self.root_field.text()).expanduser()
        directory = QFileDialog.getExistingDirectory(self, "Choose parent folder", str(current.parent))
        if directory:
            self.root_field.setText(str(Path(directory) / slugify(self.name_field.text())))
            self.auto_root = False

    def update_project_folder(self, name: str) -> None:
        if self.auto_root:
            self.root_field.setText(str(Path.home() / "json-plot-projects" / slugify(name)))

    def choose_file(self, key: str, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose source file", self.source_fields[key].text(), file_filter)
        if path:
            self.source_fields[key].setText(path)

    def choose_data_folder(self) -> None:
        start = self.data_folder_field.text() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Choose the folder with the project files", start)
        if directory:
            self.apply_data_folder(Path(directory))

    def apply_data_folder(self, folder: Path) -> None:
        self.data_folder_field.setText(str(folder))
        found = detect_sources(folder)
        if found.is_empty:
            self.detect_note.setText("No PDF or JSON files were found in that folder. Pick the files by hand below.")
            return
        for key, chooser in self.pdf_choosers.items():
            chooser.blockSignals(True)
            chooser.clear()
            for path in found.pdf_candidates.get(key, []):
                chooser.addItem(path.name, str(path))
            chooser.blockSignals(False)
        for key, path in found.guesses.items():
            if key in self.source_fields:
                self.source_fields[key].setText(str(path))
        missing = [self.LABELS[key][0] for key in self.source_fields if key not in found.guesses]
        lines = [f"Found {len(found.pdfs)} PDF and {len(found.jsons)} JSON files. All JSON files will be loaded."]
        if found.pdfs and not found.annotation_is_pre:
            lines.append("No PDF is named '...pre', so the smallest one was picked for annotation. Check it.")
        elif len(found.pdfs) > 1:
            lines.append("Several PDFs were found. Check the picks, or choose another from the list.")
        if missing:
            lines.append("Not found, pick by hand: " + ", ".join(missing) + ".")
        self.detect_note.setText(" ".join(lines))

    def _sync_pdf_chooser(self, key: str, text: str) -> None:
        chooser = self.pdf_choosers.get(key)
        if chooser is None or not text:
            return
        index = chooser.findData(text)
        if index < 0:
            chooser.addItem(Path(text).name, text)
            index = chooser.count() - 1
        chooser.blockSignals(True)
        chooser.setCurrentIndex(index)
        chooser.blockSignals(False)

    def validate_and_accept(self) -> None:
        name = self.name_field.text().strip()
        root = Path(self.root_field.text()).expanduser()
        missing = [
            f"{self.LABELS[key][0]}: {field.text()}"
            for key, field in self.source_fields.items()
            if not Path(field.text()).expanduser().is_file()
        ]
        if not name or not str(root).strip():
            QMessageBox.warning(self, "Project details required", "Enter a project name and folder.")
            return
        if missing:
            QMessageBox.warning(self, "Source files required", "Choose an existing file for each input:\n\n" + "\n".join(missing))
            return
        invalid_types = [
            self.LABELS[key][0]
            for key, field in self.source_fields.items()
            if Path(field.text()).suffix.lower() != (".pdf" if key.endswith("pdf") else ".json")
        ]
        if invalid_types:
            QMessageBox.warning(self, "Check source file types", "Expected PDF/JSON file extensions for:\n\n" + "\n".join(invalid_types))
            return
        if not scan_json_folder(Path(self.data_folder_field.text()).expanduser()):
            QMessageBox.warning(self, "No JSON files", "The data folder has no JSON files to load.")
            return
        self.accept()

    def project_values(self) -> tuple[str, Path, dict[str, Path]]:
        sources = {key: Path(field.text()).expanduser() for key, field in self.source_fields.items()}
        sources["json_folder"] = Path(self.data_folder_field.text()).expanduser()
        return (
            self.name_field.text().strip(),
            Path(self.root_field.text()).expanduser(),
            sources,
        )


class WorkflowWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        theme.install(QApplication.instance())
        self.setWindowTitle(theme.APP_NAME)
        self.resize(1440, 940)
        self.process: QProcess | None = None
        self.library = ProjectLibrary()
        self.project: dict[str, Any] | None = None
        self.project_settings: dict[str, Any] = {}
        self.hitl_jsons: list[Path] = []
        self.hitl_pdf: Path | None = None
        self.current_review_dir: Path | None = None
        self.current_detection_doc: dict[str, Any] = {}
        self.current_review: dict[str, Any] = {}
        self.current_detection: dict[str, Any] | None = None
        self.preview_cache: dict[tuple[str, int, int], QPixmap] = {}
        self.review_tasks: list[ReviewTask] = []
        self.active_task: ReviewTask | None = None
        self.qaqc_folder: Path | None = None
        self.progress_path: Path | None = None
        self.qaqc_progress: dict[str, dict[str, int]] = {}
        self.undo_stack: list[tuple[Path, str, str, str]] = []
        self.kept_marks: set[tuple] = set()
        self.json_folder: Path | None = None  # Folder the JSON dropdown lists
        self.json_files: list[Path] = []  # Every JSON found in that folder
        self.viewed_json: Path | None = None  # None means show all loaded JSON
        self.packet_mode = False  # True when the folder is a QAQC packet, which uses its own file filter
        self._seen_json: dict[Path, tuple[int, int] | None] = {}  # File state the page last read
        self.json_watcher = JsonWatcher(parent=self)
        self.json_watcher.file_settled.connect(self._on_json_file_settled)
        self.json_watcher.folder_changed.connect(self._on_json_folder_changed)
        self._build_ui()
        theme.decorate(self)
        self.setWindowIcon(QIcon(icons.logo_pixmap(theme.token("text"), 256)))
        self.setAcceptDrops(True)
        QApplication.instance().installEventFilter(self)
        theme.manager().changed.connect(self._on_theme_changed)
        self.image_view.zoom_changed.connect(self.sync_zoom_controls)
        self.canvas.box_created.connect(self.add_drawn_box)
        if self.library.projects:
            self.set_project(self.library.projects[0], navigate=False)
            self.refresh_dashboard()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = QWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(176)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(8, 12, 8, 10)
        side_layout.setSpacing(4)
        self.brand = QLabel()
        self.brand.setObjectName("brand")
        self.brand.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.brand.setToolTip(theme.APP_NAME)
        side_layout.addWidget(self.brand)
        side_layout.addSpacing(12)
        self.nav_buttons = []
        self.nav_titles = []
        self.page_targets = (0, 4, 5)
        self.stack = QStackedWidget()
        for title, target in (("Projects", 0), ("Labels", 4), ("Map", 5)):
            button = QPushButton(title)
            button.setObjectName("navButton")
            button.setCheckable(True)
            tooltip = "Manage projects" if target == 0 else f"Open {title.lower()} workspace"
            button.setToolTip(tooltip)
            button.clicked.connect(lambda checked=False, i=target: self.show_page(i))
            side_layout.addWidget(button)
            self.nav_buttons.append(button)
            self.nav_titles.append(title)
        side_layout.addStretch(1)
        self.sidebar_toggle = QPushButton()
        self.sidebar_toggle.setObjectName("sidebarToggle")
        self.sidebar_toggle.setToolTip("Collapse sidebar")
        self.sidebar_toggle.clicked.connect(self.toggle_sidebar)
        side_layout.addWidget(self.sidebar_toggle)
        side_layout.addSpacing(4)
        self.appearance_combo = QComboBox()
        self.appearance_combo.setObjectName("appearance")
        self.appearance_combo.setToolTip("Appearance")
        self.appearance_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.appearance_combo.setMinimumContentsLength(1)
        self.appearance_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for mode in theme.MODES:
            self.appearance_combo.addItem(icons.icon(theme.MODE_ICONS[mode], theme.token("text_muted")), theme.MODE_LABELS[mode], mode)
        self.appearance_combo.setCurrentIndex(theme.MODES.index(theme.manager().mode))
        self.appearance_combo.currentIndexChanged.connect(self._change_appearance)
        side_layout.addWidget(self.appearance_combo)
        # Collapsed sidebar: one icon button that cycles System, Light, Dark.
        self.appearance_cycle = QToolButton()
        self.appearance_cycle.setObjectName("sidebarToggle")
        self.appearance_cycle.setFixedSize(40, 30)
        self.appearance_cycle.clicked.connect(self._cycle_appearance)
        self.appearance_cycle.hide()
        side_layout.addWidget(self.appearance_cycle, 0, Qt.AlignmentFlag.AlignHCenter)
        side_layout.addSpacing(6)
        self._pipeline_status_full = "●  Ready"
        self.pipeline_status = QLabel(self._pipeline_status_full)
        self.pipeline_status.setObjectName("sideStatus")
        self.pipeline_status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        side_layout.addWidget(self.pipeline_status)
        self.sidebar_collapsed = False
        root_layout.addWidget(self.sidebar)

        content = QWidget()
        content.setObjectName("content")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(6, 4, 6, 4)
        content_layout.setSpacing(6)
        header = QHBoxLayout()
        header.setSpacing(8)
        self.heading = QLabel("Overview")
        self.heading.setObjectName("compactTitle")
        header.addWidget(self.heading)
        header.addStretch(1)
        self.header_status = ElidedLabel("Project library")
        self.header_status.setObjectName("caption")
        self.header_status.setMaximumWidth(180)
        header.addWidget(self.header_status)
        self.help_button = QPushButton()
        self.help_button.setObjectName("ghostButton")
        self.help_button.setToolTip("Open workflow help")
        self.help_button.clicked.connect(self.show_help)
        theme.set_icon(self.help_button, "circle-help", size=14)
        header.addWidget(self.help_button)
        content_layout.addLayout(header)

        self.projects_page = self._build_projects()
        self.overview_page = self._build_overview()
        self.dataset_page = self._build_dataset()
        self.train_page = self._build_training()
        self.review_page = self._build_review()
        self.relationship_page = RelationshipPage()
        for page in (self.projects_page, self.overview_page, self.dataset_page, self.train_page, self.review_page, self.relationship_page):
            self.stack.addWidget(page)
        content_layout.addWidget(self.stack, 1)
        root_layout.addWidget(content, 1)
        self.setCentralWidget(root)
        self._apply_sidebar_state()
        self.show_page(0)

    def _build_projects(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        intro = QHBoxLayout()
        copy = QVBoxLayout()
        title = QLabel("Your annotation projects")
        title.setObjectName("sectionTitle")
        subtitle = QLabel("Each project keeps its source data, detector runs, and human reviews connected.")
        subtitle.setObjectName("muted")
        copy.addWidget(title)
        copy.addWidget(subtitle)
        intro.addLayout(copy, 1)
        self.new_project_button = QPushButton("＋  New project")
        self.new_project_button.setObjectName("primaryButton")
        self.new_project_button.clicked.connect(self.create_project)
        intro.addWidget(self.new_project_button, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(intro)

        library_panel = panel()
        library_layout = QVBoxLayout(library_panel)
        library_layout.setContentsMargins(16, 16, 16, 16)
        library_header = QHBoxLayout()
        library_title = QLabel("Project library")
        library_title.setObjectName("caption")
        self.project_count = QLabel("0 projects")
        self.project_count.setObjectName("muted")
        library_header.addWidget(library_title)
        library_header.addStretch(1)
        library_header.addWidget(self.project_count)
        library_layout.addLayout(library_header)
        self.project_list = QListWidget()
        self.project_list.setObjectName("projectList")
        self.project_list.itemDoubleClicked.connect(lambda item: self.open_project())
        library_layout.addWidget(self.project_list, 1)
        actions = QHBoxLayout()
        self.open_project_button = QPushButton("Open project")
        self.open_project_button.setObjectName("primaryButton")
        self.open_project_button.clicked.connect(self.open_project)
        self.rename_project_button = QPushButton("Rename")
        self.rename_project_button.clicked.connect(self.rename_project)
        self.folder_project_button = QPushButton("Show folder")
        self.folder_project_button.clicked.connect(self.show_project_folder)
        self.remove_project_button = QPushButton("Remove from library")
        self.remove_project_button.clicked.connect(self.remove_project)
        for button in (self.open_project_button, self.rename_project_button, self.folder_project_button, self.remove_project_button):
            actions.addWidget(button)
        actions.addStretch(1)
        library_layout.addLayout(actions)
        layout.addWidget(library_panel, 1)
        self.refresh_project_list()
        return page

    @property
    def project_root(self) -> Path:
        return Path(self.project["root"]) if self.project else Path(PROJECT_ROOT)

    @property
    def output_dir(self) -> Path:
        return Path(self.project_settings.get("output_dir", self.project_root / "outputs"))

    @property
    def review_dir(self) -> Path:
        return self.output_dir / "visual_annotations_pre"

    @property
    def training_pdf(self) -> Path:
        return Path(self.project_settings.get("sources", {}).get("training_pdf", TRAINING_PDF_FILE))

    @property
    def annotation_pdf(self) -> Path:
        return Path(self.project_settings.get("sources", {}).get("annotation_pdf", ANNOTATION_PDF_FILE))

    def refresh_project_list(self) -> None:
        if not hasattr(self, "project_list"):
            return
        self.project_list.clear()
        for project in self.library.projects:
            item = QListWidgetItem(f"{project['name']}\n{project['root']}")
            item.setData(Qt.ItemDataRole.UserRole, project["id"])
            self.project_list.addItem(item)
        self.project_count.setText(f"{len(self.library.projects)} project" + ("s" if len(self.library.projects) != 1 else ""))
        if self.project_list.count() and self.project_list.currentRow() < 0:
            self.project_list.setCurrentRow(0)

    def selected_project(self) -> dict[str, Any] | None:
        item = self.project_list.currentItem()
        if item is None:
            return None
        project_id = item.data(Qt.ItemDataRole.UserRole)
        return next((project for project in self.library.projects if project["id"] == project_id), None)

    def create_project(self, *_args, folder: Path | None = None) -> None:
        dialog = ProjectCreationDialog(self)
        theme.decorate(dialog)
        if folder is not None:
            dialog.apply_data_folder(folder)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, root, sources = dialog.project_values()
        try:
            project = self.library.add(name, root, sources)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Project was not created", str(error))
            return
        self.refresh_project_list()
        for row in range(self.project_list.count()):
            if self.project_list.item(row).data(Qt.ItemDataRole.UserRole) == project["id"]:
                self.project_list.setCurrentRow(row)
                break
        self.set_project(project)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.KeyPress and self.isActiveWindow() and self.stack.currentIndex() == 4:
            if self._handle_page_key(event):
                return True
        return super().eventFilter(obj, event)

    def _handle_page_key(self, event) -> bool:
        modifiers = event.modifiers()
        if event.key() == Qt.Key.Key_G and modifiers == Qt.KeyboardModifier.ControlModifier:
            self.page_number.setFocus()
            self.page_number.lineEdit().selectAll()
            return True
        if modifiers not in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.KeypadModifier):
            return False
        step = {
            Qt.Key.Key_Left.value: -1,
            Qt.Key.Key_PageUp.value: -1,
            Qt.Key.Key_Right.value: 1,
            Qt.Key.Key_PageDown.value: 1,
        }.get(event.key())
        if step is None or self._focus_takes_arrows():
            return False
        self._step_page(step)
        return True

    def _focus_takes_arrows(self) -> bool:
        # Text boxes, spin boxes, lists and combos use the arrow keys for themselves.
        return isinstance(
            QApplication.focusWidget(),
            (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit, QComboBox, QAbstractItemView, QAbstractSlider),
        )

    def _step_page(self, delta: int) -> None:
        self.page_number.setValue(self.page_number.value() + delta)  # The spin box clamps to the page range.

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile() and Path(urls[0].toLocalFile()).is_dir():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.acceptProposedAction()
        self.create_project(folder=Path(event.mimeData().urls()[0].toLocalFile()))

    def rename_project(self) -> None:
        project = self.selected_project()
        if project is None:
            return
        name, accepted = QInputDialog.getText(self, "Rename project", "Project name", text=project["name"])
        if not accepted or not name.strip():
            return
        self.library.rename(project, name)
        self.refresh_project_list()
        self._select_project_row(project["id"])
        if self.project and self.project["id"] == project["id"]:
            self.project["name"] = name.strip()
            self.header_status.setText(name.strip())

    def remove_project(self) -> None:
        project = self.selected_project()
        if project is None:
            return
        answer = QMessageBox.question(
            self,
            "Remove project",
            f"Remove {project['name']} from this library?\n\nProject files are kept on disk.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        was_active = self.project and self.project["id"] == project["id"]
        self.library.remove(project)
        if was_active:
            self.project = None
            self.project_settings = {}
            self.hitl_jsons = []
            self.hitl_pdf = None
            self.refresh_dashboard()
            self.refresh_pages()
            self.show_page(0)
        self.refresh_project_list()

    def _select_project_row(self, project_id: str) -> None:
        for row in range(self.project_list.count()):
            if self.project_list.item(row).data(Qt.ItemDataRole.UserRole) == project_id:
                self.project_list.setCurrentRow(row)
                return

    def open_project(self) -> None:
        project = self.selected_project()
        if project is not None:
            self.set_project(project)

    def main_page(self) -> None:
        """Open the project manager/library view."""
        self.show_page(0)

    def set_project(self, project: dict[str, Any], navigate: bool = True) -> None:
        self.project = project
        self.project_settings = read_json(Path(project["config"]), {})
        sources = self.project_settings.get("sources", {})
        self.current_detection = None
        self.preview_cache.clear()
        self.hitl_jsons = []
        json_folder = Path(sources["json_folder"]) if sources.get("json_folder") else None
        if json_folder is not None and json_folder.is_dir():
            self.hitl_jsons = scan_json_folder(json_folder)
            self._set_json_folder(json_folder)
        else:
            # Projects made before the data folder existed list their three JSON files.
            for key in ("schedule_data", "schedule_outputs", "reference_data"):
                raw = sources.get(key)
                if raw and Path(raw).is_file():
                    self.hitl_jsons.append(Path(raw))
            self._set_json_folder(self.hitl_jsons[0].parent if self.hitl_jsons else None)
        raw_pdf = sources.get("annotation_pdf")
        self.hitl_pdf = Path(raw_pdf) if raw_pdf else None
        if hasattr(self, "pdf_path") and self.hitl_pdf:
            self.pdf_path.setText(str(self.hitl_pdf))
            self.weights.setText(DEFAULT_WEIGHTS)
        self.header_status.setText(project["name"])
        self._load_relationships()
        self.load_hitl_page()
        if navigate:
            self.show_page(4)

    def show_project_folder(self) -> None:
        project = self.selected_project()
        if project is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(project["root"]))

    def show_help(self) -> None:
        QMessageBox.information(
            self,
            "JSON Plot help",
            "Open the review CSV. The list jumps to each sheet the reviewer marked.\n\n"
            "Add means the object is on the drawing and missing from the JSON: click Draw missing and drag a box. "
            "Fix means a box is wrong: select it and Remove. That deletes one box, not the file. "
            "Leave means a real drawing problem: Keep it.\n\n"
            "The clean pre PDF is shown under the boxes. A .bak copy of a JSON is written the first time you edit it. Undo puts the last edit back.",
        )

    def sync_zoom_controls(self, zoom: float) -> None:
        if not hasattr(self, "zoom_slider"):
            return
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(round(zoom * 100))
        self.zoom_slider.blockSignals(False)
        self.zoom_label.setText(f"{round(zoom * 100)}%")

    def _build_overview(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        intro = QHBoxLayout()
        copy = QVBoxLayout()
        title = QLabel("A practical loop for better labels.")
        title.setObjectName("sectionTitle")
        description = QLabel("Prepare the source set, train the detector, then review page-level proposals.")
        description.setObjectName("muted")
        copy.addWidget(title)
        copy.addWidget(description)
        intro.addLayout(copy, 1)
        self.quick_annotate_button = QPushButton("Annotate a page  →")
        self.quick_annotate_button.setObjectName("primaryButton")
        self.quick_annotate_button.clicked.connect(lambda: self.show_page(4))
        intro.addWidget(self.quick_annotate_button, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(intro)

        metrics = QGridLayout()
        metrics.setSpacing(12)
        self.metric_pages = metric("Reviewed pages", "0", "pages with saved detections")
        self.metric_tiles = metric("Dataset tiles", "0", "train / validation")
        self.metric_feedback = metric("Human decisions", "0", "accepted and rejected")
        self.metric_checkpoint = metric("Checkpoint", "Not trained", "best detector weights")
        for index, widget in enumerate((self.metric_pages, self.metric_tiles, self.metric_feedback, self.metric_checkpoint)):
            metrics.addWidget(widget, 0, index)
        layout.addLayout(metrics)

        lower = QHBoxLayout()
        lower.setSpacing(14)
        flow = panel()
        flow_layout = QVBoxLayout(flow)
        flow_layout.setContentsMargins(20, 18, 20, 18)
        flow_title = QLabel("Pipeline")
        flow_title.setObjectName("caption")
        flow_layout.addWidget(flow_title)
        for number, name, detail in (
            ("01", "Prepare dataset", "Tile source pages and apply saved feedback"),
            ("02", "Train detector", "Fine-tune YOLO on CPU"),
            ("03", "Annotate + review", "Run inference, OCR, and human decisions"),
        ):
            row = QHBoxLayout()
            marker = QLabel(number)
            marker.setObjectName("stepNumber")
            text = QVBoxLayout()
            label = QLabel(name)
            label.setObjectName("rowTitle")
            note = QLabel(detail)
            note.setObjectName("muted")
            text.addWidget(label)
            text.addWidget(note)
            row.addWidget(marker)
            row.addLayout(text, 1)
            flow_layout.addLayout(row)
        flow_layout.addStretch(1)
        lower.addWidget(flow, 1)

        files = panel()
        files_layout = QVBoxLayout(files)
        files_layout.setContentsMargins(20, 18, 20, 18)
        files_title = QLabel("Workspace inputs")
        files_title.setObjectName("caption")
        files_layout.addWidget(files_title)
        self.source_summary = QLabel()
        self.source_summary.setObjectName("fileSummary")
        self.source_summary.setWordWrap(True)
        files_layout.addWidget(self.source_summary)
        files_layout.addStretch(1)
        lower.addWidget(files, 1)
        layout.addLayout(lower, 1)

        self.activity_log = QTextEdit()
        self.activity_log.setObjectName("log")
        self.activity_log.setReadOnly(True)
        self.activity_log.setPlaceholderText("Pipeline output will appear here.")
        self.activity_log.setMaximumHeight(150)
        layout.addWidget(self.activity_log)
        return page

    def _build_dataset(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel("Prepare the training set")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Export source annotations and human feedback into tiled YOLO data."), 0)
        status_panel = panel()
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(20, 18, 20, 18)
        status_layout.addWidget(QLabel("Current export"))
        self.dataset_detail = QLabel("No dataset manifest found yet.")
        self.dataset_detail.setObjectName("fileSummary")
        self.dataset_detail.setWordWrap(True)
        status_layout.addWidget(self.dataset_detail)
        self.class_summary = QLabel("")
        self.class_summary.setObjectName("muted")
        status_layout.addWidget(self.class_summary)
        layout.addWidget(status_panel)
        button_row = QHBoxLayout()
        self.prepare_button = QPushButton("Prepare dataset")
        self.prepare_button.setObjectName("primaryButton")
        self.prepare_button.clicked.connect(self.run_prepare)
        button_row.addWidget(self.prepare_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        layout.addStretch(1)
        return page

    def _build_training(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QLabel("Train the visual detector")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Training re-exports the dataset first, so current review feedback is included."))
        settings = panel()
        form = QFormLayout(settings)
        form.setContentsMargins(20, 18, 20, 18)
        form.setHorizontalSpacing(28)
        form.setVerticalSpacing(14)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(5)
        self.image_size = QSpinBox()
        self.image_size.setRange(32, 2048)
        self.image_size.setSingleStep(32)
        self.image_size.setValue(640)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(1)
        self.weights = QLineEdit(DEFAULT_WEIGHTS)
        weights_row = QHBoxLayout()
        weights_row.addWidget(self.weights, 1)
        weights_button = QPushButton("Browse")
        weights_button.clicked.connect(self.choose_weights)
        weights_row.addWidget(weights_button)
        form.addRow("Epochs", self.epochs)
        form.addRow("Image size", self.image_size)
        form.addRow("Batch size", self.batch_size)
        form.addRow("Initial weights", weights_row)
        layout.addWidget(settings)
        self.train_status = QLabel("Training runs on CPU and can take a while for a full dataset.")
        self.train_status.setObjectName("muted")
        layout.addWidget(self.train_status)
        row = QHBoxLayout()
        self.train_button = QPushButton("Start training")
        self.train_button.setObjectName("primaryButton")
        self.train_button.clicked.connect(self.run_training)
        row.addWidget(self.train_button)
        self.cancel_button = QPushButton("Stop")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.stop_process)
        row.addWidget(self.cancel_button)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def _build_review(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        # One compact control strip. The canvas gets everything else.
        controls = QHBoxLayout()
        controls.setSpacing(5)
        self.pdf_path = QLineEdit()
        self.pdf_path.setMinimumWidth(160)
        self.pdf_path.setPlaceholderText("PDF to review")
        controls.addWidget(self.pdf_path, 1)
        self.pages = QComboBox()  # kept for page selection; invisible in the compact layout
        self.pages.hide()
        for caption, slot, tip in (
            ("PDF", self.choose_pdf, "Choose PDF to review"),
            ("JSON folder", self.choose_json_folder, "Load every JSON file in a folder"),
            ("Remove JSON", self.remove_hitl_json, "Remove the JSON picked in the dropdown from the loaded set"),
        ):
            button = QPushButton(caption)
            button.setObjectName("compactButton")
            button.setToolTip(tip)
            button.clicked.connect(slot)
            controls.addWidget(button)
        self.page_number = QSpinBox()  # Lives in the bottom bar, between the page arrows
        self.page_number.setRange(1, 9999)
        self.page_number.setValue(1)
        self.page_number.valueChanged.connect(self.load_hitl_page)
        self.refresh_button = QPushButton("↻ Reload")
        self.refresh_button.setObjectName("compactButton")
        self.refresh_button.setToolTip("Reload every loaded JSON from disk and redraw this page")
        self.refresh_button.clicked.connect(self.reload_json_now)
        controls.addWidget(self.refresh_button)
        self.annotate_button = QPushButton("Run page")
        self.annotate_button.setObjectName("compactButton")
        self.annotate_button.hide()
        controls.addWidget(self.annotate_button)
        review_csv = QPushButton("Review CSV")
        review_csv.setObjectName("compactButton")
        review_csv.setToolTip("Open the QAQC sheet and jump to the pages it marks")
        review_csv.clicked.connect(self.load_qaqc_csv)
        controls.addWidget(review_csv)
        self.task_combo = QComboBox()
        self.task_combo.setMinimumWidth(180)
        self.task_combo.addItem("Open a review CSV to get a sheet list", None)
        self.task_combo.currentIndexChanged.connect(self._select_task)
        controls.addWidget(self.task_combo, 2)
        self.only_task_boxes = QCheckBox("Only this type")
        self.only_task_boxes.setChecked(True)
        self.only_task_boxes.setToolTip("Draw only the callouts, doors, or schedule rows this review row is about")
        self.only_task_boxes.toggled.connect(self.load_hitl_page)
        controls.addWidget(self.only_task_boxes)
        layout.addLayout(controls)

        info_row = QHBoxLayout()
        info_row.setSpacing(10)
        self.task_banner = ElidedLabel(
            "Open the review CSV. Each row is a sheet to check on the clean PDF. "
            "Keep leaves a box. Remove deletes that one box from its JSON. Draw missing adds a box."
        )
        self.task_banner.setObjectName("banner")
        info_row.addWidget(self.task_banner, 1)

        files_panel = QWidget()
        files_layout = QVBoxLayout(files_panel)
        files_layout.setContentsMargins(0, 0, 0, 0)
        files_layout.setSpacing(2)
        files_header = QHBoxLayout()
        files_header.setSpacing(4)
        files_title = QLabel("JSON file")
        files_title.setObjectName("caption")
        files_header.addWidget(files_title)
        self.json_select = QComboBox()
        self.json_select.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.json_select.setMinimumContentsLength(16)
        self.json_select.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.json_select.setToolTip("Choose which JSON file the page shows. All loaded shows every JSON that is loaded.")
        self.json_select.currentIndexChanged.connect(self._on_json_view_changed)
        files_header.addWidget(self.json_select)
        files_header.addStretch(1)
        files_layout.addLayout(files_header)
        self.hitl_pdf_label = ElidedLabel("No PDF loaded")
        self.hitl_pdf_label.setObjectName("muted")
        files_layout.addWidget(self.hitl_pdf_label)
        info_row.addWidget(files_panel, 1)
        layout.addLayout(info_row)

        review_layout = QHBoxLayout()
        review_layout.setSpacing(4)

        left = panel()
        left.setObjectName("compactPanel")
        left.setMinimumWidth(160)
        left.setMaximumWidth(210)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(3)
        detection_header = QHBoxLayout()
        detection_title = QLabel("Labels")
        detection_title.setObjectName("caption")
        self.detection_summary = QLabel("0")
        self.detection_summary.setObjectName("countBadge")
        detection_header.addWidget(detection_title)
        detection_header.addStretch(1)
        detection_header.addWidget(self.detection_summary)
        left_layout.addLayout(detection_header)
        self.detection_filter = QComboBox()
        self.detection_filter.setObjectName("detectionFilter")
        self.detection_filter.addItem("All labels", "")
        self.detection_filter.currentIndexChanged.connect(self.populate_detections)
        left_layout.addWidget(self.detection_filter)
        self.detection_list = QListWidget()
        self.detection_list.setMouseTracking(True)
        self.detection_list.viewport().setMouseTracking(True)
        self.detection_list.currentItemChanged.connect(self.select_detection)
        self.detection_list.itemEntered.connect(self.hover_detection)
        left_layout.addWidget(self.detection_list, 1)
        action_row = QHBoxLayout()
        action_row.setSpacing(4)
        self.keep_button = QPushButton("Keep")
        self.keep_button.setObjectName("compactButton")
        self.keep_button.setToolTip("Leave this box in the JSON")
        self.keep_button.clicked.connect(self.keep_selected)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setObjectName("compactButton")
        self.remove_button.setToolTip("Delete this one box from its JSON file")
        self.remove_button.clicked.connect(self.remove_selected)
        action_row.addWidget(self.keep_button)
        action_row.addWidget(self.remove_button)
        left_layout.addLayout(action_row)
        draw_row = QHBoxLayout()
        draw_row.setSpacing(4)
        self.draw_button = QPushButton("Draw missing")
        self.draw_button.setObjectName("compactButton")
        self.draw_button.setCheckable(True)
        self.draw_button.setToolTip("Drag a box on the page. It is appended to the JSON for this review row.")
        self.draw_button.toggled.connect(self.toggle_draw)
        self.undo_button = QPushButton("Undo")
        self.undo_button.setObjectName("compactButton")
        self.undo_button.setToolTip("Put back the last box you removed or added")
        self.undo_button.clicked.connect(self.undo_json_edit)
        draw_row.addWidget(self.draw_button)
        draw_row.addWidget(self.undo_button)
        left_layout.addLayout(draw_row)
        review_layout.addWidget(left)

        preview_panel = panel()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(2, 2, 2, 2)
        preview_layout.setSpacing(2)
        preview_header = QHBoxLayout()
        preview_header.setSpacing(8)
        self.review_page_badge = QLabel("No page loaded")
        self.review_page_badge.setObjectName("caption")
        preview_header.addWidget(self.review_page_badge)
        preview_header.addStretch(1)
        for title, family in LABEL_LEGEND:
            legend = QLabel(f"● {title}")
            theme.set_property(legend, "category", family)
            preview_header.addWidget(legend)
        preview_layout.addLayout(preview_header)
        self.canvas = ReviewCanvas()
        self.canvas.detection_hovered.connect(self.select_canvas_detection)
        self.image_view = ImageScrollArea(self.canvas)
        preview_layout.addWidget(self.image_view, 1)
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(4)
        self.zoom_out_button = QPushButton("−")
        self.zoom_out_button.setObjectName("compactButton")
        self.zoom_out_button.setToolTip("Zoom out")
        self.zoom_out_button.clicked.connect(lambda: self.image_view.set_zoom(self.canvas.zoom / 1.2))
        self.zoom_in_button = QPushButton("＋")
        self.zoom_in_button.setObjectName("compactButton")
        self.zoom_in_button.setToolTip("Zoom in")
        self.zoom_in_button.clicked.connect(lambda: self.image_view.set_zoom(self.canvas.zoom * 1.2))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(8, 800)
        self.zoom_slider.setFixedWidth(110)
        self.zoom_slider.valueChanged.connect(lambda value: self.image_view.set_zoom(value / 100))
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(38)
        self.fit_button = QPushButton()
        self.fit_button.setObjectName("compactButton")
        self.fit_button.setToolTip("Fit to view")
        theme.set_icon(self.fit_button, "maximize", size=16)
        self.fit_button.clicked.connect(self.image_view.fit_to_view)
        self.actual_size_button = QPushButton("100%")
        self.actual_size_button.setObjectName("compactButton")
        self.actual_size_button.clicked.connect(lambda: self.image_view.set_zoom(1.0))
        max_render_label = QLabel("Max render")
        max_render_label.setObjectName("caption")
        max_render_label.setToolTip("Maximum pixel width used to rasterise PDF pages")
        self.max_render_spin = QSpinBox()
        self.max_render_spin.setObjectName("compactSpin")
        self.max_render_spin.setRange(512, 8192)
        self.max_render_spin.setSingleStep(512)
        self.max_render_spin.setValue(self.canvas.max_render_width)
        self.max_render_spin.setSuffix(" px")
        self.max_render_spin.setToolTip("Maximum pixel width used to rasterise PDF pages. Higher is sharper but uses more memory.")
        self.max_render_spin.valueChanged.connect(self.canvas.set_max_render_width)
        zoom_row.addWidget(max_render_label)
        zoom_row.addWidget(self.max_render_spin)
        zoom_row.addStretch(1)
        self.prev_page_button = QPushButton()
        self.prev_page_button.setObjectName("compactButton")
        self.prev_page_button.setToolTip("Previous page (Left arrow or Page Up)")
        theme.set_icon(self.prev_page_button, "chevron-left", size=16)
        self.prev_page_button.clicked.connect(lambda: self._step_page(-1))
        self.next_page_button = QPushButton()
        self.next_page_button.setObjectName("compactButton")
        self.next_page_button.setToolTip("Next page (Right arrow or Page Down)")
        theme.set_icon(self.next_page_button, "chevron-right", size=16)
        self.next_page_button.clicked.connect(lambda: self._step_page(1))
        self.page_number.setObjectName("pageNumber")
        self.page_number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_number.setPrefix("Page ")
        self.page_number.setSuffix(" / –")
        self.page_number.setMinimumWidth(130)
        self.page_number.setToolTip("Type a page number, then press Enter. Ctrl+G jumps to the box.")
        self.page_number.setKeyboardTracking(False)  # Load the page on Enter, not on every keystroke.
        page_nav = QHBoxLayout()
        page_nav.setSpacing(4)
        page_nav.addWidget(self.prev_page_button)
        page_nav.addWidget(self.page_number)
        page_nav.addWidget(self.next_page_button)
        zoom_row.addLayout(page_nav)
        zoom_row.addStretch(1)
        for widget in (self.zoom_out_button, self.zoom_in_button, self.zoom_slider, self.zoom_label, self.fit_button, self.actual_size_button):
            zoom_row.addWidget(widget)
        preview_layout.addLayout(zoom_row)
        review_layout.addWidget(preview_panel, 1)
        layout.addLayout(review_layout, 1)
        return page

    def toggle_sidebar(self) -> None:
        self.sidebar_collapsed = not self.sidebar_collapsed
        self._apply_sidebar_state()

    def _cycle_appearance(self) -> None:
        index = theme.MODES.index(theme.manager().mode)
        theme.manager().set_mode(theme.MODES[(index + 1) % len(theme.MODES)])

    def _refresh_appearance_cycle(self) -> None:
        mode = theme.manager().mode
        self.appearance_cycle.setIcon(icons.icon(theme.MODE_ICONS[mode], theme.token("text_muted"), 16))
        self.appearance_cycle.setToolTip(f"Appearance: {theme.MODE_LABELS[mode]}. Click to change.")

    def _apply_sidebar_state(self) -> None:
        collapsed = self.sidebar_collapsed
        self.appearance_combo.setVisible(not collapsed)
        self.appearance_cycle.setVisible(collapsed)
        self._refresh_appearance_cycle()
        self.sidebar.setFixedWidth(56 if collapsed else 176)
        theme.set_property(self.sidebar, "collapsed", "true" if collapsed else "false")
        self._refresh_brand()
        self.sidebar_toggle.setToolTip("Expand sidebar" if collapsed else "Collapse sidebar")
        theme.set_icon(self.sidebar_toggle, "panel-left-open" if collapsed else "panel-left-close", size=16)
        for button, title in zip(self.nav_buttons, self.nav_titles):
            button.setText("" if collapsed else title)
            button.setToolTip(title)
        self.appearance_combo.blockSignals(True)
        for index, mode in enumerate(theme.MODES):
            self.appearance_combo.setItemText(index, "" if collapsed else theme.MODE_LABELS[mode])
        self.appearance_combo.setMinimumContentsLength(0 if collapsed else 1)
        if collapsed:
            self.appearance_combo.setFixedWidth(40)
        else:
            self.appearance_combo.setMinimumWidth(0)
            self.appearance_combo.setMaximumWidth(16777215)
        self.appearance_combo.blockSignals(False)
        self.pipeline_status.setText("●" if collapsed else self._pipeline_status_full)
        self.pipeline_status.setToolTip(self._pipeline_status_full if collapsed else "")

    def _refresh_brand(self) -> None:
        height = 28 if self.sidebar_collapsed else 44
        self.brand.setPixmap(icons.logo_pixmap(theme.token("text"), height))

    def _on_theme_changed(self, _scheme: str) -> None:
        # Custom painters cache nothing colour-related, so a repaint is enough.
        theme.refresh_icons(self)
        self._refresh_brand()
        self._refresh_appearance_cycle()
        self.appearance_combo.blockSignals(True)
        for index, mode in enumerate(theme.MODES):
            self.appearance_combo.setItemIcon(index, icons.icon(theme.MODE_ICONS[mode], theme.token("text_muted")))
            self.appearance_combo.setItemText(index, "" if self.sidebar_collapsed else theme.MODE_LABELS[mode])
        self.appearance_combo.setCurrentIndex(theme.MODES.index(theme.manager().mode))
        self.appearance_combo.blockSignals(False)
        if hasattr(self, "sidebar_toggle"):
            theme.set_icon(self.sidebar_toggle, "panel-left-open" if self.sidebar_collapsed else "panel-left-close", size=16)
        for widget in self.findChildren(QWidget):
            widget.update()

    def _change_appearance(self, index: int) -> None:
        theme.manager().set_mode(self.appearance_combo.itemData(index))

    def show_page(self, index: int) -> None:
        titles = {0: "Projects", 1: "Overview", 2: "Dataset", 3: "Training", 4: "Labels", 5: "Map"}
        self.stack.setCurrentIndex(index)
        self.heading.setText(titles.get(index, theme.APP_NAME))
        for button, target in zip(self.nav_buttons, self.page_targets):
            button.setChecked(target == index)
        if index == 4:
            QTimer.singleShot(0, self.image_view.fit_to_view)
        if index == 5:
            QTimer.singleShot(0, self.relationship_page.canvas_fit)

    def _load_relationships(self) -> None:
        pdf_paths = [self.hitl_pdf] if self.hitl_pdf and self.hitl_pdf.is_file() else []
        output = self.project_settings.get("output_dir")
        self.relationship_page.set_inputs(
            self._relationship_inputs(),
            pdf_paths,
            Path(output) if output else self.output_dir,
        )
        self._sync_hitl_files()
        self._remember_json_state()

    def log(self, text: str) -> None:
        if text:
            self.activity_log.append(text.rstrip())

    def _set_running(self, running: bool, label: str = "Ready") -> None:
        for button in (self.prepare_button, self.train_button, self.annotate_button, self.refresh_button):
            button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self._pipeline_status_full = f"●  {label}" if running else "●  Ready"
        self.pipeline_status.setText("●" if self.sidebar_collapsed else self._pipeline_status_full)
        self.pipeline_status.setToolTip(self._pipeline_status_full if self.sidebar_collapsed else "")
        theme.set_property(self.pipeline_status, "state", "running" if running else "idle")

    def run_command(self, arguments: list[str], label: str) -> None:
        if self.process is not None:
            return
        if self.project is None:
            QMessageBox.information(self, "Choose a project", "Open or create a project before running the pipeline.")
            return
        process = QProcess(self)
        process.setWorkingDirectory(str(self.project_root))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("JSON_PLOT_PROJECT_CONFIG", self.project["config"])
        process.setProcessEnvironment(environment)
        process.setProgram(sys.executable)
        process.setArguments(["-m", "json_plot.visual_model", *arguments])
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(lambda: self._read_process_output(process))
        process.finished.connect(lambda code, status: self._process_finished(process, code, label))
        process.errorOccurred.connect(lambda error: self.log(f"Process error: {error.name}"))
        self.process = process
        self._set_running(True, label[:1].upper() + label[1:])
        self.log(f"Starting {label}: {sys.executable} -m json_plot.visual_model {' '.join(arguments)}")
        process.start()

    def _read_process_output(self, process: QProcess) -> None:
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log(text)

    def _process_finished(self, process: QProcess, code: int, label: str) -> None:
        self._read_process_output(process)
        self.log(f"{label.title()} finished with exit code {code}.")
        self.process = None
        process.deleteLater()
        self._set_running(False)
        self.refresh_dashboard()
        self.refresh_pages()
        if code:
            QMessageBox.warning(self, "Pipeline command failed", f"{label.title()} exited with code {code}. See the activity log for details.")

    def stop_process(self) -> None:
        if self.process is not None:
            self.process.terminate()
            QTimer.singleShot(4000, lambda process=self.process: process.kill() if process and process.state() != QProcess.ProcessState.NotRunning else None)

    def run_prepare(self) -> None:
        self.run_command(["prepare"], "prepare dataset")

    def run_training(self) -> None:
        self.run_command([
            "train", "--weights", self.weights.text().strip() or DEFAULT_WEIGHTS,
            "--epochs", str(self.epochs.value()), "--imgsz", str(self.image_size.value()),
            "--batch", str(self.batch_size.value()),
        ], "train detector")

    def run_annotation(self) -> None:
        pdf = Path(self.pdf_path.text()).expanduser()
        if not pdf.is_absolute():
            pdf = self.project_root / pdf
        if not pdf.exists():
            QMessageBox.warning(self, "PDF not found", f"Could not find {pdf}")
            return
        self.run_command(["annotate", "--page", str(self.page_number.value()), "--pdf", str(pdf)], "annotate page")

    def choose_pdf(self) -> None:
        start = str(self.hitl_pdf.parent) if self.hitl_pdf else str(self.project_root)
        path, _ = QFileDialog.getOpenFileName(self, "Choose PDF", start, "PDF files (*.pdf)")
        if path:
            self.hitl_pdf = Path(path)
            self.pdf_path.setText(path)
            self._load_relationships()
            self.load_hitl_page()

    def choose_json_folder(self) -> None:
        start = str(self.json_folder or (self.hitl_jsons[-1].parent if self.hitl_jsons else self.project_root))
        chosen = QFileDialog.getExistingDirectory(self, "Choose the folder with JSON files", start)
        if not chosen:
            return
        folder = Path(chosen)
        files = scan_json_folder(folder)
        if not files:
            self.show_toast(f"No JSON files in {folder.name or folder}")
            return
        self.hitl_jsons = files
        self._set_json_folder(folder)
        self._load_relationships()
        self.load_hitl_page()
        self.show_toast(f"Loaded {len(files)} JSON from {folder.name or folder}")

    def _viewed_paths(self) -> list[Path]:
        return [self.viewed_json] if self.viewed_json is not None else list(self.hitl_jsons)

    def _relationship_inputs(self) -> list[Path]:
        json_paths = self._viewed_paths()
        for path in list(json_paths):
            ocr = path.with_name("01_ocr_word_dict.json")
            if ocr.is_file() and ocr not in json_paths:
                json_paths.append(ocr)
        return json_paths

    def _remember_json_state(self) -> None:
        # Called after every read, so a watcher event for a file we just wrote is recognised as our own change.
        for path in {*self._viewed_paths(), *self._relationship_inputs()}:
            self._seen_json[path] = json_file_state(path)

    def _set_json_folder(self, folder: Path | None, packet: bool = False) -> None:
        self.json_folder = folder
        self.packet_mode = packet
        self.viewed_json = None
        self._rescan_json_folder()

    def _rescan_json_folder(self) -> None:
        folder = self.json_folder
        if folder is None or not folder.is_dir():
            self.json_files = []
        elif self.packet_mode:
            self.json_files = packet_files(folder)[1]
        else:
            self.json_files = scan_json_folder(folder)
        if self.viewed_json is not None and self.viewed_json not in self.json_files:
            self.viewed_json = None
        self._refresh_json_select()
        self.json_watcher.watch_files({*self.json_files, *self.hitl_jsons, *self._relationship_inputs()})
        self.json_watcher.watch_folder(folder)

    def _refresh_json_select(self) -> None:
        if not hasattr(self, "json_select"):
            return
        self.json_select.blockSignals(True)
        self.json_select.clear()
        self.json_select.addItem("All loaded", None)
        for path in self.json_files:
            label = path.name if path in self.hitl_jsons else f"{path.name}  (not loaded)"
            self.json_select.addItem(label, str(path))
            self.json_select.setItemData(self.json_select.count() - 1, str(path), Qt.ItemDataRole.ToolTipRole)
        index = self.json_select.findData(str(self.viewed_json)) if self.viewed_json is not None else 0
        self.json_select.setCurrentIndex(max(0, index))
        self.json_select.blockSignals(False)

    def _on_json_view_changed(self, index: int) -> None:
        data = self.json_select.itemData(index)
        self.viewed_json = Path(data) if data else None
        if self.viewed_json is not None and self.viewed_json not in self.hitl_jsons:
            self.hitl_jsons.append(self.viewed_json)  # Picking a file loads it
        self._load_relationships()
        self.load_hitl_page()

    def reload_json_now(self, *_args) -> None:
        self.json_watcher.cancel_pending()
        self.reload_json_files("Reloaded JSON")

    def reload_json_files(self, message: str) -> None:
        self._rescan_json_folder()
        self._load_relationships()
        self.load_hitl_page()
        self.show_toast(f"{message}  ·  {len(self._viewed_paths())} JSON")

    def _on_json_file_settled(self, path: Path) -> None:
        if path not in self._relationship_inputs():
            return  # Not in use. It is read fresh the next time someone selects it.
        if json_file_state(path) == self._seen_json.get(path):
            return  # The page already shows this version, for example after our own save.
        self.reload_json_files(f"Reloaded {path.name} after a change")

    def _on_json_folder_changed(self, folder: Path) -> None:
        if folder == self.json_folder:
            self._rescan_json_folder()

    def remove_hitl_json(self) -> None:
        path = self.viewed_json
        if path is None:
            self.show_toast("Pick a JSON in the dropdown to remove it")
            return
        if path not in self.hitl_jsons:
            self.show_toast(f"{path.name} is not in the loaded set")
            return
        self.hitl_jsons = [existing for existing in self.hitl_jsons if existing != path]
        self.viewed_json = None
        self._load_relationships()
        self.load_hitl_page()
        self.show_toast(f"Removed  ·  {path.name or 'JSON'}")

    def choose_weights(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose model weights", str(self.project_root), "Model weights (*.pt);;All files (*)")
        if path:
            self.weights.setText(path)

    def refresh_dashboard(self) -> None:
        manifest = read_json(self.output_dir / "visual_dataset" / "manifest.json", {})
        summary = read_json(self.output_dir / "visual_model" / "training_summary.json", {})
        page_files = sorted(self.review_dir.glob("page_*/detections.json"))
        accepted = rejected = 0
        for path in self.review_dir.glob("page_*/human_review.json"):
            review = read_json(path, {})
            for entry in review.get("annotations", {}).values():
                if isinstance(entry, dict):
                    accepted += entry.get("decision") == "accepted"
                    rejected += entry.get("decision") == "rejected"
        self.metric_pages.value_label.setText(str(len(page_files)))
        tile_counts = manifest.get("tile_image_counts", {})
        tile_total = sum(tile_counts.values()) if tile_counts else 0
        self.metric_tiles.value_label.setText(str(tile_total))
        self.metric_tiles.note_label.setText(f"{tile_counts.get('train', 0)} train  /  {tile_counts.get('val', 0)} validation")
        self.metric_feedback.value_label.setText(str(accepted + rejected))
        self.metric_feedback.note_label.setText(f"{accepted} accepted  /  {rejected} rejected")
        best_path = Path(summary.get("best_weights", self.output_dir / "visual_model" / "detector" / "weights" / "best.pt"))
        if not best_path.is_absolute():
            best_path = self.project_root / best_path
        trained = bool(summary) and best_path.exists()
        self.metric_checkpoint.value_label.setText("Ready" if trained else "Not trained")
        self.metric_checkpoint.note_label.setText(best_path.name if trained else "best detector weights")
        training_exists = self.training_pdf.exists()
        annotation_exists = self.annotation_pdf.exists()
        self.source_summary.setText(
            f"Training source\n{self.training_pdf.name}\n{'Available' if training_exists else 'Missing'}\n\n"
            f"Annotation PDF\n{self.annotation_pdf.name}\n{'Available' if annotation_exists else 'Missing'}"
        )
        if manifest:
            self.dataset_detail.setText(
                f"Exported {manifest.get('created_at', 'unknown time')}\n"
                f"Source boxes: {sum(manifest.get('source_box_counts', {}).values())}\n"
                f"Tiles: {tile_counts.get('train', 0)} train / {tile_counts.get('val', 0)} validation"
            )
            self.class_summary.setText("  ·  ".join(
                f"{name.replace('_', ' ')}: {count}"
                for name, count in manifest.get("tile_label_counts", {}).items()
            ))
        else:
            self.dataset_detail.setText("No dataset manifest found yet. Run Prepare dataset to export labels and page tiles.")
            self.class_summary.setText("")

    def refresh_pages(self) -> None:
        self.load_hitl_page()

    def _sync_hitl_files(self) -> None:
        if hasattr(self, "hitl_pdf_label"):
            pdf_name = self.hitl_pdf.name if self.hitl_pdf else "No PDF loaded"
            self.hitl_pdf_label.setText(f"PDF: {pdf_name}")
            self.hitl_pdf_label.setToolTip(str(self.hitl_pdf) if self.hitl_pdf else "")
        self._refresh_json_select()

    def load_qaqc_csv(self) -> None:
        start = str(self.hitl_pdf.parent) if self.hitl_pdf else str(self.project_root.parent)
        path, _ = QFileDialog.getOpenFileName(self, "Open the review CSV", start, "Review sheets (*.csv)")
        if not path:
            return
        csv_path = Path(path)
        folder = csv_path.parent
        page_map = load_page_map(folder)
        if not page_map:
            QMessageBox.warning(self, "No page map", "Put page_to_sheet JSON in the same folder as the CSV.")
            return
        tasks, missing = build_tasks(parse_review_csv(csv_path), page_map, load_sheet_titles(folder, page_map))
        if not tasks:
            QMessageBox.warning(self, "No review rows", "The CSV has no sheet rows this packet can open.")
            return
        self.qaqc_folder = folder
        self.review_tasks = tasks
        self.progress_path = csv_path.with_name(csv_path.stem + ".progress.json")
        self.qaqc_progress = read_json(self.progress_path, {}) if self.progress_path.is_file() else {}
        if not isinstance(self.qaqc_progress, dict):
            self.qaqc_progress = {}
        pre, jsons = packet_files(folder)
        if pre is not None:
            self.hitl_pdf = pre
            self.pdf_path.setText(str(pre))
        if jsons:
            self.hitl_jsons = jsons
            self._set_json_folder(folder, packet=True)
        self._load_relationships()
        self.task_combo.blockSignals(True)
        self.task_combo.clear()
        for task in tasks:
            self.task_combo.addItem(task.label(), task)
        self.task_combo.setCurrentIndex(0)
        self.task_combo.blockSignals(False)
        self._select_task(0)
        if missing:
            self.show_toast("No page for " + ", ".join(missing))

    def _select_task(self, index: int) -> None:
        task = self.task_combo.itemData(index) if index >= 0 else None
        self.active_task = task if isinstance(task, ReviewTask) else None
        self._update_task_banner()
        if self.active_task is None:
            return
        if self.page_number.value() != self.active_task.pdf_page:
            self.page_number.setValue(self.active_task.pdf_page)
        else:
            self.load_hitl_page()

    def _update_task_banner(self) -> None:
        if not hasattr(self, "task_banner"):
            return
        task = self.active_task
        if task is None:
            self.task_banner.setText(
                "Open the review CSV. Each row is a sheet to check on the clean PDF. "
                "Keep leaves a box. Remove deletes that one box from its JSON. Draw missing adds a box."
            )
            return
        kind = CATEGORY_LABELS[task.category]
        if task.verb == "add":
            instruction = f"Draw {task.count} missing {kind}. Boxes already drawn were detected; add only the ones that are not there."
        elif task.verb == "fix":
            instruction = f"Remove the {task.count} wrong {kind}. Keep leaves a box in the file."
        else:
            instruction = f"These {task.count} {kind} are real drawing issues. Leave them in the JSON."
        keeps = related_keeps(self.review_tasks, task)
        extra = ""
        if keeps and task.verb != "keep":
            extra = " Also leave " + ", ".join(f"{item.count} {CATEGORY_LABELS[item.category]}" for item in keeps) + "."
        where = ""
        if task.verb == "add" and self.qaqc_folder is not None:
            where = f" New boxes go into {destination_label(self.qaqc_folder, task)}."
        done = self.qaqc_progress.get(task.key, {})
        tally = ""
        if task.verb == "fix":
            tally = f" Removed {int(done.get('removed', 0))} of {task.count}."
        elif task.verb == "add":
            tally = f" Added {int(done.get('added', 0))} of {task.count}."
        self.task_banner.setText(f"{task.sheet} · page {task.pdf_page}. {instruction}{extra}{where}{tally}")

    def _kept_key(self, item: dict[str, Any]) -> tuple:
        box = tuple(round(float(value), 2) for value in item.get("source_bbox") or [])
        return (item.get("source_path"), item.get("page_key"), item.get("record"), box)

    def _bump_progress(self, field: str) -> None:
        if self.active_task is None:
            return
        entry = self.qaqc_progress.setdefault(self.active_task.key, {})
        entry[field] = int(entry.get(field, 0)) + 1
        if self.progress_path is not None:
            write_json(self.progress_path, self.qaqc_progress)
        self._update_task_banner()

    def _rewind_progress(self, task_key: str, field: str) -> None:
        entry = self.qaqc_progress.get(task_key)
        if not isinstance(entry, dict) or not field:
            return
        entry[field] = max(0, int(entry.get(field, 0)) - 1)
        if self.progress_path is not None:
            write_json(self.progress_path, self.qaqc_progress)
        self._update_task_banner()

    def keep_selected(self) -> None:
        item = self.current_detection
        if not item:
            self.show_toast("Click a box first")
            return
        self.kept_marks.add(self._kept_key(item))
        label = str(item.get("ocr_text") or item.get("record") or "box").strip()
        self.load_hitl_page()
        self.show_toast(f"Kept  ·  {label}")

    def remove_selected(self) -> None:
        item = self.current_detection
        if not item:
            self.show_toast("Click a box first")
            return
        label = str(item.get("ocr_text") or item.get("record") or "box").strip()
        if self.active_task and self.active_task.verb == "keep":
            answer = QMessageBox.question(
                self,
                "Leave this box",
                "The review marks this sheet as a real drawing issue. Remove the box anyway?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        answer = QMessageBox.question(
            self,
            "Remove this box",
            f"Remove “{label}” from {item.get('source')}?\n\nThe rest of that file stays. The first edit writes a .bak copy.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            previous = remove_item(item)
        except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as error:
            QMessageBox.warning(self, "Could not remove the box", str(error))
            return
        task_key = self.active_task.key if self.active_task else ""
        self.undo_stack.append((Path(str(item["source_path"])), previous, task_key, "removed"))
        self.undo_stack = self.undo_stack[-20:]
        self._bump_progress("removed")
        self.load_hitl_page()
        self.show_toast(f"Removed  ·  {label}")

    def toggle_draw(self, checked: bool) -> None:
        self.canvas.set_marking(checked)
        if checked:
            self.show_toast("Drag a box around the missing object")

    def add_drawn_box(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.draw_button.setChecked(False)
        if self.active_task is None or self.qaqc_folder is None:
            QMessageBox.information(self, "Open the review first", "Open the review CSV so the new box is written into the right JSON.")
            return
        if self.active_task.verb == "keep":
            QMessageBox.information(self, "Leave this sheet", "This review row is a real drawing issue. Draw a box on an Add row instead.")
            return
        from PySide6.QtWidgets import QDialog

        dialog = QDialog(self)
        dialog.setWindowTitle("Add the missing object")
        form = QFormLayout(dialog)
        form.addRow(QLabel(destination_label(self.qaqc_folder, self.active_task)))
        text_input = QLineEdit()
        text_input.setPlaceholderText("Leave blank for a door. Type the label for a callout or equipment tag.")
        form.addRow("Text", text_input)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        text = text_input.text().strip()
        page_width, page_height = self.current_detection_doc.get("pdf_page_size_points", [1, 1])
        image_width = max(1, self.canvas.image.width())
        image_height = max(1, self.canvas.image.height())
        bbox = [
            round(min(x0, x1) * page_width / image_width, 2),
            round(min(y0, y1) * page_height / image_height, 2),
            round(max(x0, x1) * page_width / image_width, 2),
            round(max(y0, y1) * page_height / image_height, 2),
        ]
        try:
            previous, path, _record = add_item(self.qaqc_folder, self.active_task, text, bbox)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            QMessageBox.warning(self, "Could not add the box", str(error))
            return
        if path not in self.hitl_jsons:
            self.hitl_jsons.append(path)
        if self.viewed_json not in (None, path):
            self.viewed_json = None  # Show all, so the new box is visible right away
            self._refresh_json_select()
        self.undo_stack.append((path, previous, self.active_task.key, "added"))
        self.undo_stack = self.undo_stack[-20:]
        self._bump_progress("added")
        self.load_hitl_page()
        self.show_toast(f"Added  ·  {text}")

    def undo_json_edit(self) -> None:
        if not self.undo_stack:
            self.show_toast("Nothing to undo")
            return
        path, previous, task_key, field = self.undo_stack.pop()
        try:
            path.write_text(previous, encoding="utf-8")
        except OSError as error:
            QMessageBox.warning(self, "Could not undo", str(error))
            return
        self._rewind_progress(task_key, field)
        self.load_hitl_page()
        self.show_toast(f"Undid the last edit in {path.name}")

    def load_hitl_page(self, *_args) -> None:
        if getattr(self, "_loading_hitl", False) or not hasattr(self, "canvas"):
            return
        self._loading_hitl = True
        try:
            pdf = self.hitl_pdf if self.hitl_pdf and self.hitl_pdf.is_file() else None
            typed = Path(self.pdf_path.text().strip()).expanduser() if self.pdf_path.text().strip() else None
            if typed and typed.is_file():
                pdf = typed
                self.hitl_pdf = typed
            if pdf is None:
                self._clear_review()
                return
            count = pdf_page_count(pdf)
            self.page_number.blockSignals(True)
            self.page_number.setMaximum(max(1, count))
            if self.page_number.value() > count:
                self.page_number.setValue(count)
            page = self.page_number.value()
            self.page_number.setSuffix(f" / {count}")
            self.page_number.blockSignals(False)
            width, height = pdf_page_size(pdf, page)
            items = collect_page_items(self._viewed_paths(), page, width, height)
            if self.active_task and self.only_task_boxes.isChecked():
                items = [item for item in items if item_in_category(item, self.active_task.category)]
            for item in items:
                if self._kept_key(item) in self.kept_marks:
                    item["kept"] = True
            pixmap = self.load_page_image(pdf, page, 80)
            self.current_review_dir = None
            self.current_detection_doc = {
                "page_number": page,
                "source_pdf": str(pdf),
                "pdf_page_size_points": [width, height],
                "detections": items,
                "render_dpi": 80,
            }
            self.current_review = {}
            self.current_detection = None
            self.canvas.set_page(pixmap, items, {}, width, height)
            self.canvas.set_source(pdf, page - 1)
            QTimer.singleShot(0, self.image_view.fit_to_view)
            sheet = ""
            if self.active_task and self.active_task.pdf_page == page:
                sheet = f"  {self.active_task.sheet}"
            self.review_page_badge.setText(f"Page {page:03d}{sheet}  /  {len(items)} labels")
            self._sync_label_filter(items)
            self.populate_detections()
            self._remember_json_state()
        except Exception as error:
            if hasattr(self, "hitl_pdf_label"):
                self.hitl_pdf_label.setText(f"Could not read this page. {error}")
            elif hasattr(self, "hitl_file_label"):
                self.hitl_file_label.setText(f"Could not read this page. {error}")
        finally:
            self._loading_hitl = False

    def _clear_review(self) -> None:
        self.current_review_dir = None
        self.page_number.setSuffix(" / –")
        self.detection_list.clear()
        self.current_detection = None
        self.current_review = {}
        self.current_detection_doc = {}
        self.detection_summary.setText("0")
        self.review_page_badge.setText("No page loaded")
        self.canvas.set_page(QPixmap(), [], {}, 1, 1)
        self.canvas.set_source(None)

    def load_review_page(self, index: int) -> None:
        if index < 0:
            return
        page_dir = Path(self.pages.itemData(index))
        detections_doc = read_json(page_dir / "detections.json", {})
        review = read_json(page_dir / "human_review.json", {})
        source = Path(detections_doc.get("source_pdf", self.pdf_path.text() or str(self.annotation_pdf)))
        if not source.is_absolute():
            source = self.project_root / source
        page_number = detections_doc.get("page_number", 1)
        dpi = detections_doc.get("render_dpi", 36)
        pixmap = self.load_page_image(source, page_number, dpi)
        page_size = detections_doc.get("pdf_page_size_points", [1, 1])
        self.current_review_dir = page_dir
        self.current_detection_doc = detections_doc
        self.current_review = review
        self.canvas.set_page(pixmap, detections_doc.get("detections", []), review, *page_size)
        self.canvas.set_source(source, page_number - 1)
        QTimer.singleShot(0, self.image_view.fit_to_view)
        self.page_number.setValue(page_number)
        self.pdf_path.setText(str(source))
        self.review_page_badge.setText(f"Page {page_number:03d}  /  {len(detections_doc.get('detections', []))} proposals")
        self.current_detection = None
        self.populate_detections()

    def _sync_label_filter(self, detections: list[dict[str, Any]]) -> None:
        current = self.detection_filter.currentData()
        titles = {
            "tag": "Tags",
            "door": "Doors",
            "callout": "Callouts",
            "area": "Areas",
            "verified": "Verified",
            "missing": "Missing",
            "bad": "Bad callouts",
            "ocr": "OCR",
        }
        seen: list[tuple[str, str]] = []
        known = set()
        for detection in detections:
            family = str(detection.get("family") or "")
            if family and family not in known:
                known.add(family)
                seen.append((family, titles.get(family, str(detection.get("family_label") or family))))
        self.detection_filter.blockSignals(True)
        self.detection_filter.clear()
        self.detection_filter.addItem("All labels", "")
        for family, title in seen:
            self.detection_filter.addItem(title, family)
        index = self.detection_filter.findData(current)
        self.detection_filter.setCurrentIndex(index if index >= 0 else 0)
        self.detection_filter.blockSignals(False)

    def populate_detections(self) -> None:
        detections = self.current_detection_doc.get("detections", [])
        selected_id = self.current_detection.get("id") if self.current_detection else None
        family_filter = self.detection_filter.currentData()
        self.detection_list.blockSignals(True)
        self.detection_list.clear()
        for detection in detections:
            if family_filter and detection.get("family") != family_filter:
                continue
            title = detection.get("record") or detection.get("family_label") or "Label"
            text = str(detection.get("ocr_text") or "").strip()
            prefix = "Kept  ·  " if detection.get("kept") else ""
            item = QListWidgetItem(f"{prefix}{title}  ·  {text[:48]}" if text else f"{prefix}{title}")
            item.setData(Qt.ItemDataRole.UserRole, detection)
            self.detection_list.addItem(item)
        self.detection_summary.setText(f"{self.detection_list.count()} / {len(detections)}")
        self.detection_list.blockSignals(False)
        if self.detection_list.count():
            selected_row = next((
                row for row in range(self.detection_list.count())
                if self.detection_list.item(row).data(Qt.ItemDataRole.UserRole).get("id") == selected_id
            ), 0)
            self.detection_list.setCurrentRow(selected_row)
        else:
            self.current_detection = None
            self.canvas.set_hovered_detection(None)

    def load_page_image(self, source: Path, page_number: int, dpi: int) -> QPixmap:
        key = (str(source), page_number, dpi)
        if key in self.preview_cache:
            return self.preview_cache[key]
        if not source.exists():
            return QPixmap()
        try:
            with pdfplumber.open(source) as pdf:
                page = pdf.pages[page_number - 1]
                image = page.to_image(resolution=dpi).original
                buffer = QByteArray()
                from PySide6.QtCore import QBuffer, QIODevice
                byte_buffer = QBuffer(buffer)
                byte_buffer.open(QIODevice.OpenModeFlag.WriteOnly)
                image.save(byte_buffer, "PNG")
                byte_buffer.close()
            pixmap = QPixmap.fromImage(QImage.fromData(buffer))
            self.preview_cache[key] = pixmap
            return pixmap
        except Exception as error:
            self.log(f"Could not render review page: {error}")
            return QPixmap()

    def select_detection(self, current: QListWidgetItem | None, previous: QListWidgetItem | None = None) -> None:
        if current is None:
            return
        self.current_detection = current.data(Qt.ItemDataRole.UserRole)
        self.canvas.set_hovered_detection(self.current_detection.get("id"))

    def update_coach(self) -> None:
        if not hasattr(self, "coach"):
            return
        page = int(self.current_detection_doc.get("page_number") or self.page_number.value())
        detections = self.current_detection_doc.get("detections", [])
        annotations = self.current_review.get("annotations", {})
        pending = sum(
            annotations.get(item.get("id"), {}).get("decision", "pending") == "pending"
            for item in detections
            if isinstance(item, dict)
        )
        graph = self.relationship_page.graph
        if self.current_detection:
            entry = annotations.get(self.current_detection.get("id"), {})
            note = detection_brief(
                graph,
                page,
                str(self.current_detection.get("ocr_text") or ""),
                str(self.current_detection.get("class_name") or "object"),
                str(entry.get("decision", "pending")),
                self.current_detection.get("ocr_confidence"),
            )
        else:
            note = page_brief(graph, page, pending, len(detections))
        self.coach_title.setText(note.title)
        self.coach_body.setText(note.body)
        issue = (self.current_detection or {}).get("issue")
        extra = {
            "missing_ref": " This one is a missing reference. Accept it only if that callout is really on the drawing.",
            "missing": " The JSON says this is missing from the schedule. Accept it if you can see it.",
            "bad_call": " The JSON marks this callout as wrong. Accept the box only if it sits on that callout.",
        }.get(issue)
        if extra:
            self.coach_body.setText(self.coach_body.text() + extra)
        theme.set_property(self.coach, "tone", note.action)
        self.next_pending_button.setEnabled(pending > 0)

    def focus_next_pending(self) -> None:
        detections = [item for item in self.current_detection_doc.get("detections", []) if isinstance(item, dict)]
        annotations = self.current_review.get("annotations", {})
        pending_ids = [
            item.get("id")
            for item in detections
            if annotations.get(item.get("id"), {}).get("decision", "pending") == "pending"
        ]
        if not pending_ids:
            return
        current_id = self.current_detection.get("id") if self.current_detection else None
        start = pending_ids.index(current_id) + 1 if current_id in pending_ids else 0
        target = pending_ids[start % len(pending_ids)]
        if self.detection_filter.currentData() not in ("", "pending"):
            self.detection_filter.setCurrentIndex(self.detection_filter.findData("pending"))
        for row in range(self.detection_list.count()):
            item = self.detection_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole).get("id") == target:
                self.detection_list.setCurrentItem(item)
                self.detection_list.scrollToItem(item)
                return

    def hover_detection(self, item: QListWidgetItem) -> None:
        detection = item.data(Qt.ItemDataRole.UserRole)
        self.canvas.set_hovered_detection(detection.get("id"))
        if self.detection_list.currentItem() is not item:
            self.detection_list.setCurrentItem(item)

    def select_canvas_detection(self, detection_id: str | None) -> None:
        if detection_id is None:
            self.canvas.set_hovered_detection(None)
            return
        matching_item = next((
            self.detection_list.item(row)
            for row in range(self.detection_list.count())
            if self.detection_list.item(row).data(Qt.ItemDataRole.UserRole).get("id") == detection_id
        ), None)
        if matching_item is None:
            self.detection_filter.setCurrentIndex(0)
            matching_item = next((
                self.detection_list.item(row)
                for row in range(self.detection_list.count())
                if self.detection_list.item(row).data(Qt.ItemDataRole.UserRole).get("id") == detection_id
            ), None)
        if matching_item is not None and self.detection_list.currentItem() is not matching_item:
            self.detection_list.setCurrentItem(matching_item)

    def set_decision(self, decision: str) -> None:
        if not self.current_detection:
            return
        label = str(self.current_detection.get("ocr_text") or self.current_detection.get("class_name") or "box").strip()
        entry = self.current_review.setdefault("annotations", {}).setdefault(self.current_detection["id"], {})
        entry["decision"] = decision
        corrected = self.corrected_label.currentData()
        entry["corrected_label"] = corrected or None
        entry["comment"] = self.comment.text()
        detection_id = self.current_detection["id"]
        self.save_current_review(refresh=False)
        self.current_detection = next((
            detection for detection in self.current_detection_doc.get("detections", [])
            if detection.get("id") == detection_id
        ), None)
        self.populate_detections()
        self.canvas.set_page(self.canvas.image, self.current_detection_doc.get("detections", []), self.current_review,
                             *self.current_detection_doc.get("pdf_page_size_points", [1, 1]))
        self.refresh_dashboard()
        if decision == "accepted":
            self.show_toast(f"Accepted  ·  {label}")

    def show_toast(self, message: str) -> None:
        if not hasattr(self, "toast"):
            self.toast = QLabel(self)
            self.toast.setObjectName("toast")
            self.toast_timer = QTimer(self)
            self.toast_timer.setSingleShot(True)
            self.toast_timer.timeout.connect(self.toast.hide)
        self.toast.setText(message)
        self.toast.adjustSize()
        self.toast.move(max(18, (self.width() - self.toast.width()) // 2), self.height() - self.toast.height() - 28)
        self.toast.show()
        self.toast.raise_()
        self.toast_timer.start(2200)

    def save_current_review(self, checked: bool = False, refresh: bool = True) -> None:
        if self.current_review_dir is None:
            return
        if self.current_detection:
            entry = self.current_review.setdefault("annotations", {}).setdefault(self.current_detection["id"], {})
            entry["corrected_label"] = self.corrected_label.currentData() or None
            entry["comment"] = self.comment.text()
        try:
            write_json(self.current_review_dir / "human_review.json", self.current_review)
            if refresh:
                self.log(f"Saved review: {self.current_review_dir / 'human_review.json'}")
                self.refresh_dashboard()
        except OSError as error:
            QMessageBox.critical(self, "Could not save review", str(error))

    def add_missed_annotation(self, x0: float, y0: float, x1: float, y1: float) -> None:
        if self.current_review_dir is None or self.canvas.image.isNull():
            self.mark_missed_button.setChecked(False)
            return
        from PySide6.QtWidgets import QDialog, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Add missed object")
        form = QFormLayout(dialog)
        class_choice = QComboBox()
        for name in CLASS_NAMES:
            class_choice.addItem(name.replace("_", " ").title(), name)
        text_input = QLineEdit()
        text_input.setPlaceholderText("Optional OCR text")
        form.addRow("Class", class_choice)
        form.addRow("Text", text_input)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        self.mark_missed_button.setChecked(False)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        page_width, page_height = self.current_detection_doc.get("pdf_page_size_points", [1, 1])
        image_width = self.canvas.image.width()
        image_height = self.canvas.image.height()
        bbox = [
            round(x0 * page_width / image_width, 2),
            round(y0 * page_height / image_height, 2),
            round(x1 * page_width / image_width, 2),
            round(y1 * page_height / image_height, 2),
        ]
        missed = self.current_review.setdefault("missed_annotations", {})
        key_number = len(missed) + 1
        missed_id = f"missed-{key_number:03d}"
        while missed_id in missed:
            key_number += 1
            missed_id = f"missed-{key_number:03d}"
        missed[missed_id] = {
            "class_name": class_choice.currentData(),
            "bbox_pdf": bbox,
            "corrected_text": text_input.text().strip() or None,
            "comment": "Added in visual review",
            "status": "open",
        }
        self.save_current_review(refresh=False)
        self.canvas.set_page(self.canvas.image, self.current_detection_doc.get("detections", []), self.current_review,
                             *self.current_detection_doc.get("pdf_page_size_points", [1, 1]))
        self.log(f"Added missed object {missed_id} on page {self.current_detection_doc.get('page_number')}.")
        self.refresh_dashboard()


def main() -> None:
    # Use the desktop's own file picker (via xdg-desktop-portal) for every Open/Save dialog.
    os.environ["QT_QPA_PLATFORMTHEME"] = "xdgdesktopportal"
    app = QApplication(sys.argv)
    app.setApplicationName(theme.APP_NAME)
    app.setApplicationDisplayName(theme.APP_NAME)
    app.setDesktopFileName("json-plot")
    theme.install(app)
    window = WorkflowWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()