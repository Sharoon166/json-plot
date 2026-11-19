"""Design tokens, theme modes, and the application stylesheet.

The look follows Cursor's interface: neutral surfaces, hairline borders, one
monochrome primary action, and colour kept for status and categories. Every
colour in the app resolves through a token here, so light, dark, and system
modes stay consistent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PySide6.QtCore import QObject, QSettings, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QPalette
from PySide6.QtWidgets import QAbstractButton, QApplication, QWidget

from json_plot import icons


MODES = ("system", "light", "dark")
MODE_LABELS = {"system": "System", "light": "Light", "dark": "Dark"}
APP_NAME = "JSON Plot"
RADIUS = 6
PANEL_RADIUS = 8
FONT_CANDIDATES = (
    "Inter",
    "SF Pro Text",
    "Segoe UI",
    "Helvetica Neue",
    "Noto Sans",
    "Cantarell",
    "DejaVu Sans",
)

# Semantic colour roles -> hue tokens. Data modules name roles; this file owns the hex.
STATUS_HUE = {
    "pending": "amber",
    "accepted": "green",
    "rejected": "red",
    "missed": "violet",
    "recovered": "green",
}
CATEGORY_HUE = {
    "tag": "blue",
    "door": "teal",
    "callout": "amber",
    "area": "slate",
    "verified": "green",
    "missing": "orange",
    "bad": "red",
    "ocr": "sage",
    "other": "violet",
}
TONE_HUE = {
    "accept": "green",
    "reject": "red",
    "check": "amber",
    "mark": "violet",
    "done": "green",
    "wait": "blue",
}
# Every name a widget may ask for: hues plus the neutral text tokens.
HUE_NAMES = (
    "blue", "teal", "green", "amber", "orange", "red", "violet", "slate", "sage",
    "text", "text_muted",
)


@dataclass(frozen=True)
class Palette:
    scheme: str
    # Surfaces
    app: str
    sidebar: str
    sidebar_hover: str
    sidebar_selected: str
    surface: str
    raised: str
    hover: str
    selected: str
    viewport: str
    # Lines
    border: str
    border_strong: str
    # Text
    text: str
    text_muted: str
    text_faint: str
    # Primary action: inverted, as in Cursor's main buttons
    primary: str
    primary_hover: str
    primary_text: str
    focus: str
    # Hues for status and categories
    on_hue: str
    blue: str
    teal: str
    green: str
    amber: str
    orange: str
    red: str
    violet: str
    slate: str
    sage: str


LIGHT = Palette(
    scheme="light",
    app="#fafafa",
    sidebar="#f3f3f3",
    sidebar_hover="#e8e8e8",
    sidebar_selected="#e2e2e2",
    surface="#ffffff",
    raised="#f4f4f4",
    hover="#ececec",
    selected="#e4e4e4",
    viewport="#ebebeb",
    border="#e5e5e5",
    border_strong="#d2d2d2",
    text="#1f1f1f",
    text_muted="#6b6b6b",
    text_faint="#9d9d9d",
    primary="#1f1f1f",
    primary_hover="#000000",
    primary_text="#ffffff",
    focus="#2f6fd6",
    on_hue="#ffffff",
    blue="#2f6fd6",
    teal="#1d8c7b",
    green="#1f8a4f",
    amber="#b7791f",
    orange="#c2631c",
    red="#c2453d",
    violet="#7e4fb8",
    slate="#5f6e7b",
    sage="#5d8a6c",
)

DARK = Palette(
    scheme="dark",
    app="#191919",
    sidebar="#141414",
    sidebar_hover="#222222",
    sidebar_selected="#2a2a2a",
    surface="#1d1d1d",
    raised="#252525",
    hover="#2b2b2b",
    selected="#333333",
    viewport="#111111",
    border="#2b2b2b",
    border_strong="#3a3a3a",
    text="#e6e6e6",
    text_muted="#a1a1a1",
    text_faint="#6f6f6f",
    primary="#e6e6e6",
    primary_hover="#ffffff",
    primary_text="#141414",
    focus="#6ea8fe",
    on_hue="#101010",
    blue="#6ea8fe",
    teal="#4fb8a6",
    green="#5fc48a",
    amber="#e5b04f",
    orange="#ef9050",
    red="#f07068",
    violet="#b58cf0",
    slate="#8b98a5",
    sage="#8fb39a",
)

PALETTES = {"light": LIGHT, "dark": DARK}


MODE_ICONS = {"system": "monitor", "light": "sun", "dark": "moon"}


def token(name: str) -> str:
    """A token's hex value, for stylesheets and icon colours."""
    return getattr(manager().palette, name)


def hue_for_status(status: str) -> str:
    return STATUS_HUE.get(status, "amber")


def hue_for_category(family: str | None) -> str:
    return CATEGORY_HUE.get(family or "", "blue")


def hue_for_tone(tone: str) -> str:
    return TONE_HUE.get(tone, "text_muted")


def _ui_family() -> str:
    installed = set(QFontDatabase.families())
    for name in FONT_CANDIDATES:
        if name in installed:
            return name
    return QApplication.font().family()


def build_stylesheet(p: Palette) -> str:
    rules: list[str] = []
    add = rules.append

    add(f"""
        QWidget {{ color: {p.text}; font-size: 13px; }}
        QMainWindow, QWidget#content, QDialog {{ background: {p.app}; }}
        QToolTip {{ background: {p.surface}; color: {p.text}; border: 1px solid {p.border_strong};
                    border-radius: 4px; padding: 4px 6px; }}
        QLabel {{ background: transparent; }}
        QLabel#pageTitle {{ font-size: 20px; font-weight: 600; }}
        QLabel#sectionTitle {{ font-size: 15px; font-weight: 600; }}
        QLabel#rowTitle {{ font-size: 13px; font-weight: 600; }}
        QLabel#caption {{ color: {p.text_muted}; font-size: 11px; font-weight: 600; }}
        QLabel#muted {{ color: {p.text_muted}; }}
        QLabel#fileSummary {{ color: {p.text_muted}; font-size: 12px; }}
        QLabel#metricValue {{ font-size: 24px; font-weight: 600; }}
        QLabel#countBadge {{ color: {p.text_muted}; background: {p.raised}; border: 1px solid {p.border};
                             border-radius: 9px; padding: 1px 8px; font-size: 11px; font-weight: 600; }}
        QLabel#stepNumber {{ color: {p.text_muted}; font-size: 13px; font-weight: 600; min-width: 28px; }}
        QLabel#banner {{ color: {p.text_muted}; background: {p.surface}; border: 1px solid {p.border};
                         border-radius: {RADIUS}px; padding: 8px 10px; }}
        QLabel#toast {{ color: {p.primary_text}; background: {p.primary}; border-radius: {RADIUS}px;
                        padding: 8px 14px; font-weight: 600; }}
        QLabel#preview {{ background: {p.viewport}; border-radius: {RADIUS}px; color: {p.text_muted}; }}
    """)

    add(f"""
        QWidget#sidebar {{ background: {p.sidebar}; }}
        QLabel#brand {{ font-size: 15px; font-weight: 600; }}
        QLabel#sideCaption {{ color: {p.text_muted}; font-size: 11px; }}
        QLabel#sideStatus {{ color: {p.text_muted}; font-size: 12px; }}
        QLabel#sideStatus[state="running"] {{ color: {p.amber}; }}
        QPushButton#navButton {{ background: transparent; color: {p.text_muted}; border: 0;
                                 border-radius: {RADIUS}px; padding: 7px 10px; text-align: left; }}
        QPushButton#navButton:hover {{ background: {p.sidebar_hover}; color: {p.text}; }}
        QPushButton#navButton:checked {{ background: {p.sidebar_selected}; color: {p.text}; font-weight: 600; }}
        QPushButton#sidebarToggle {{ background: transparent; border: 1px solid transparent; border-radius: {RADIUS}px; padding: 4px; }}
        QPushButton#sidebarToggle:hover {{ background: {p.sidebar_hover}; }}
        QComboBox#appearance {{ background: {p.sidebar_hover}; border: 1px solid {p.border}; }}
        QComboBox#appearance QAbstractItemView {{ background: {p.surface}; }}
        QWidget#sidebar[collapsed="true"] QLabel#brand {{ font-size: 11px; }}
        QWidget#sidebar[collapsed="true"] QPushButton#navButton {{ padding: 8px 2px; }}
    """)

    add(f"""
        QPushButton {{ background: {p.raised}; color: {p.text}; border: 1px solid {p.border_strong};
                       border-radius: {RADIUS}px; padding: 6px 12px; min-height: 16px; }}
        QPushButton:hover {{ background: {p.hover}; }}
        QPushButton:pressed {{ background: {p.selected}; }}
        QPushButton:focus {{ border-color: {p.focus}; }}
        QPushButton:disabled {{ color: {p.text_faint}; background: {p.raised}; border-color: {p.border}; }}
        QPushButton#primaryButton {{ background: {p.primary}; color: {p.primary_text}; border: 1px solid {p.primary};
                                     font-weight: 600; padding: 7px 14px; }}
        QPushButton#primaryButton:hover {{ background: {p.primary_hover}; border-color: {p.primary_hover}; }}
        QPushButton#primaryButton:disabled {{ background: {p.raised}; color: {p.text_faint}; border-color: {p.border}; }}
        QPushButton#ghostButton {{ background: transparent; border: 1px solid transparent; }}
        QPushButton#ghostButton:hover {{ background: {p.hover}; border-color: {p.border}; }}
        QPushButton#chip {{ background: {p.surface}; border: 1px solid {p.border_strong}; border-radius: {RADIUS}px;
                            padding: 4px 9px; font-size: 12px; }}
        QPushButton#chip:checked {{ color: {p.on_hue}; }}
    """)

    add(f"""
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: {p.surface}; color: {p.text};
            border: 1px solid {p.border_strong}; border-radius: {RADIUS}px; padding: 5px 8px;
            selection-background-color: {p.selected}; selection-color: {p.text}; min-height: 16px; }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {p.focus}; }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{ color: {p.text_faint}; }}
        QComboBox::drop-down {{ border: 0; width: 26px; }}
        QComboBox::down-arrow {{ image: url({icons.asset('chevron-down', p.text_muted)}); width: 12px; height: 12px; }}
        QComboBox QAbstractItemView {{ background: {p.surface}; border: 1px solid {p.border};
            selection-background-color: {p.selected}; selection-color: {p.text}; outline: 0; padding: 4px; }}
        QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            width: 16px; border: 0; background: transparent; }}
        QCheckBox {{ color: {p.text_muted}; spacing: 8px; }}
        QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {p.border_strong};
            border-radius: 3px; background: {p.surface}; }}
        QCheckBox::indicator:checked {{ background: {p.primary}; border-color: {p.primary};
            image: url({icons.asset('check', p.primary_text)}); }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({icons.asset('chevron-up', p.text_muted)}); width: 10px; height: 10px; }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({icons.asset('chevron-down', p.text_muted)}); width: 10px; height: 10px; }}
        QSlider::groove:horizontal {{ height: 4px; background: {p.border_strong}; border-radius: 2px; }}
        QSlider::sub-page:horizontal {{ background: {p.text_muted}; border-radius: 2px; }}
        QSlider::handle:horizontal {{ width: 12px; height: 12px; margin: -4px 0; border-radius: 6px;
            background: {p.text}; }}
        QSplitter::handle {{ background: transparent; }}
    """)

    add(f"""
        QListWidget, QListView {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: {RADIUS}px;
            outline: 0; padding: 4px; }}
        QListWidget::item, QListView::item {{ padding: 7px 8px; border-radius: 4px; }}
        QListWidget::item:hover, QListView::item:hover {{ background: {p.hover}; }}
        QListWidget::item:selected, QListView::item:selected {{ background: {p.selected}; color: {p.text}; }}
        QListWidget#projectList::item {{ padding: 12px 10px; min-height: 40px; }}
        QComboBox#detectionFilter {{ padding: 6px 8px; }}
        QTextEdit#log {{ background: {p.sidebar}; color: {p.text_muted}; border: 1px solid {p.border};
            border-radius: {RADIUS}px; padding: 8px; font-family: monospace; font-size: 12px; }}
        QMenu {{ background: {p.surface}; border: 1px solid {p.border}; padding: 4px; }}
        QMenu::item {{ padding: 6px 12px; border-radius: 4px; }}
        QMenu::item:selected {{ background: {p.hover}; }}
    """)

    add(f"""
        QFrame#panel {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: {PANEL_RADIUS}px; }}
        QFrame#coach {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: {PANEL_RADIUS}px;
            border-left: 3px solid {p.border_strong}; }}
        QScrollArea#viewer {{ background: {p.viewport}; border: 0; }}
        QScrollArea#viewer > QWidget#qt_scrollarea_viewport {{ background: {p.viewport}; }}
        QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
        QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
        QScrollBar::handle:vertical {{ background: {p.border_strong}; border-radius: 4px; min-height: 28px; margin: 2px; }}
        QScrollBar::handle:horizontal {{ background: {p.border_strong}; border-radius: 4px; min-width: 28px; margin: 2px; }}
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{ background: {p.text_faint}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
    """)

    add(f"""
        QLabel#compactTitle {{ font-size: 15px; font-weight: 600; }}
        QLabel#banner {{ padding: 3px 6px; font-size: 12px; }}
        QLabel#sideCaption {{ font-size: 10px; }}
        QLabel#brand {{ font-size: 13px; line-height: 110%; }}
        QPushButton#navButton {{ padding: 6px 1px; font-size: 11px; text-align: center; }}
        QComboBox#appearance {{ padding: 3px 2px; font-size: 11px; }}
        QComboBox#appearance::drop-down {{ width: 14px; }}
        QPushButton#compactButton {{ padding: 4px 6px; font-size: 12px; min-height: 12px; }}
        QSpinBox#compactSpin {{ padding: 3px 4px; font-size: 12px; min-height: 12px; }}
        QSpinBox#compactSpin::up-button, QSpinBox#compactSpin::down-button {{ width: 12px; }}
        QFrame#compactPanel {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: {PANEL_RADIUS}px; }}
    """)

    # Dynamic properties, so widgets pick a hue without inline stylesheets.
    for family, hue in CATEGORY_HUE.items():
        add(f'QLabel[category="{family}"] {{ color: {getattr(p, hue)}; font-size: 11px; font-weight: 600; }}')
    for tone, hue in TONE_HUE.items():
        add(f'QFrame#coach[tone="{tone}"] {{ border-left: 3px solid {getattr(p, hue)}; }}')
    for hue in HUE_NAMES:
        color = getattr(p, hue)
        add(f'QPushButton#chip[hue="{hue}"]:checked {{ background: {color}; border-color: {color}; }}')
        add(f'QLabel[hue="{hue}"] {{ color: {color}; }}')

    return "\n".join(rules)


def build_qpalette(p: Palette) -> QPalette:
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: p.app,
        QPalette.ColorRole.WindowText: p.text,
        QPalette.ColorRole.Base: p.surface,
        QPalette.ColorRole.AlternateBase: p.raised,
        QPalette.ColorRole.Text: p.text,
        QPalette.ColorRole.PlaceholderText: p.text_faint,
        QPalette.ColorRole.Button: p.raised,
        QPalette.ColorRole.ButtonText: p.text,
        QPalette.ColorRole.Highlight: p.selected,
        QPalette.ColorRole.HighlightedText: p.text,
        QPalette.ColorRole.ToolTipBase: p.surface,
        QPalette.ColorRole.ToolTipText: p.text,
        QPalette.ColorRole.Link: p.focus,
    }
    for role, value in roles.items():
        palette.setColor(role, QColor(value))
    return palette


class ThemeManager(QObject):
    """Holds the chosen mode, resolves "system", and re-applies on OS changes."""

    changed = Signal(str)  # resolved scheme: "light" or "dark"

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)
        self._app = app
        self._settings = QSettings(APP_NAME, APP_NAME)
        saved = str(self._settings.value("appearance", "system"))
        self._mode = saved if saved in MODES else "system"
        self._palette = LIGHT
        app.setStyle("Fusion")
        app.setFont(QFont(_ui_family(), 10))
        hints = QGuiApplication.styleHints()
        hints.colorSchemeChanged.connect(self._on_system_scheme_changed)
        self._apply()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def palette(self) -> Palette:
        return self._palette

    def set_mode(self, mode: str) -> None:
        if mode not in MODES or mode == self._mode:
            return
        self._mode = mode
        self._settings.setValue("appearance", mode)
        self._apply()

    def _system_scheme(self) -> str:
        scheme = QGuiApplication.styleHints().colorScheme()
        return "dark" if scheme == Qt.ColorScheme.Dark else "light"

    def _on_system_scheme_changed(self, _scheme) -> None:
        if self._mode == "system":
            self._apply()

    def _apply(self) -> None:
        scheme = self._system_scheme() if self._mode == "system" else self._mode
        palette = PALETTES[scheme]
        self._palette = palette
        self._app.setPalette(build_qpalette(palette))
        self._app.setStyleSheet(build_stylesheet(palette))
        self.changed.emit(scheme)


_manager: ThemeManager | None = None


def install(app: QApplication) -> ThemeManager:
    """Create the theme manager once, for the given application."""
    global _manager
    if _manager is None:
        _manager = ThemeManager(app)
    return _manager


def manager() -> ThemeManager:
    if _manager is None:
        raise RuntimeError("Theme is not installed. Call theme.install(app) first.")
    return _manager


def color(name: str, alpha: int = 255) -> QColor:
    """A token colour as QColor. Names are palette fields such as 'text' or 'teal'."""
    qcolor = QColor(getattr(manager().palette, name))
    qcolor.setAlpha(alpha)
    return qcolor


def ui_font(point_size: float = 10, bold: bool = False) -> QFont:
    font = QFont(QApplication.font())
    font.setPointSizeF(point_size)
    font.setWeight(QFont.Weight.DemiBold if bold else QFont.Weight.Normal)
    return font


def set_property(widget, name: str, value: str) -> None:
    """Set a dynamic QSS property and re-polish the widget so the rule applies."""
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


# Button captions that get an icon. Matched after stripping decorative glyphs.
ICON_BY_TEXT = {
    "Help": "circle-help",
    "PDF": "file-text",
    "JSON": "braces",
    "Add JSON": "upload",
    "Add PDF": "upload",
    "Export JSON": "download",
    "Review CSV": "list",
    "Browse": "folder",
    "Choose": "folder",
    "Open project": "folder",
    "Show folder": "folder",
    "Rename": "pencil",
    "Remove from library": "trash",
    "Remove JSON": "x",
    "New project": "plus",
    "Annotate a page": "arrow-right",
    "Keep": "check",
    "Remove": "x",
    "Draw missing": "square-plus",
    "Undo": "undo",
    "Fit": "maximize",
    "100%": "actual",
    "Start training": "play",
    "Run page": "play",
    "Stop": "square",
    "Prepare dataset": "database",
    "Projects": "layout-dashboard",
    "Labels": "tag",
    "Map": "network",
    "Collapse": "panel-left-close",
    "Expand": "panel-left-open",
}
# Icon-only buttons, matched on their whole caption.
ICON_BY_GLYPH = {"\u21bb": "refresh", "\u2212": "minus", "\uff0b": "plus"}


def _paint_icon(button: QAbstractButton) -> None:
    name = str(button.property("iconName"))
    role = str(button.property("iconRole") or "")
    if not role:
        role = "primary_text" if button.objectName() == "primaryButton" else "text"
    size = int(button.property("iconPx") or 16)
    button.setIcon(icons.icon(name, getattr(manager().palette, role), size))
    button.setIconSize(QSize(size, size))


def set_icon(button: QAbstractButton, name: str, role: str = "", size: int = 16) -> None:
    """Attach a themed icon. The icon is recoloured automatically when the theme changes."""
    button.setProperty("iconName", name)
    button.setProperty("iconRole", role)
    button.setProperty("iconPx", size)
    _paint_icon(button)


def refresh_icons(root: QWidget) -> None:
    for button in root.findChildren(QAbstractButton):
        if button.property("iconName"):
            _paint_icon(button)


def decorate(root: QWidget) -> None:
    """Give every known button under `root` its icon, matched by caption."""
    for button in root.findChildren(QAbstractButton):
        if button.property("iconName"):
            continue
        caption = button.text().strip()
        if not caption:
            continue
        if caption in ICON_BY_GLYPH:
            button.setText("")
            set_icon(button, ICON_BY_GLYPH[caption])
            continue
        label = re.sub(r"^[\uff0b+\s]+|[\s\u2192]+$", "", caption)
        name = ICON_BY_TEXT.get(label)
        if name:
            button.setText(label)
            set_icon(button, name)
