"""Watch JSON files on disk and report when one has settled after a change.

Editors, scripts and our own writes all touch the same files. A change is only
reported once the file has been quiet for RELOAD_DEBOUNCE_MS, so a burst of
writes causes one reload instead of many.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QObject, QTimer, Signal


RELOAD_DEBOUNCE_MS = 8000  # Inside the 5 to 10 second window asked for.
SKIPPED_SUFFIXES = (".progress.json",)


def scan_json_folder(folder: Path) -> list[Path]:
    """Every usable JSON file directly inside folder, sorted by name."""
    if not folder.is_dir():
        return []
    files = []
    for path in sorted(folder.glob("*.json")):
        if path.name.lower().endswith(SKIPPED_SUFFIXES) or not path.is_file():
            continue
        if path.stat().st_size <= 2:  # "{}" or "[]" is not worth loading.
            continue
        files.append(path)
    return files


def json_file_state(path: Path) -> tuple[int, int] | None:
    """Cheap fingerprint of a file, used to tell a real change from our own reads."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


class JsonWatcher(QObject):
    file_settled = Signal(object)  # Path, once the file has been quiet for the debounce window
    folder_changed = Signal(object)  # Path of the watched folder

    def __init__(self, debounce_ms: int = RELOAD_DEBOUNCE_MS, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.debounce_ms = debounce_ms
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_file_changed)
        self._watcher.directoryChanged.connect(self._on_directory_changed)
        self._timers: dict[str, QTimer] = {}

    def watch_files(self, paths: Iterable[Path]) -> None:
        wanted = {str(path) for path in paths if path.is_file()}
        watched = set(self._watcher.files())
        for raw in watched - wanted:
            self._watcher.removePath(raw)
        for raw in wanted - watched:
            self._watcher.addPath(raw)

    def watch_folder(self, folder: Path | None) -> None:
        for raw in self._watcher.directories():
            if folder is None or raw != str(folder):
                self._watcher.removePath(raw)
        if folder is not None and folder.is_dir() and str(folder) not in self._watcher.directories():
            self._watcher.addPath(str(folder))

    def cancel_pending(self) -> None:
        for timer in self._timers.values():
            timer.stop()

    def _on_file_changed(self, raw: str) -> None:
        # Atomic saves replace the file, which drops it from the watcher. Re-arm it.
        if raw not in self._watcher.files() and Path(raw).is_file():
            self._watcher.addPath(raw)
        timer = self._timers.get(raw)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(partial(self._emit_settled, raw))
            self._timers[raw] = timer
        timer.start(self.debounce_ms)  # Restarting on every change makes this a trailing debounce.

    def _emit_settled(self, raw: str) -> None:
        self.file_settled.emit(Path(raw))

    def _on_directory_changed(self, raw: str) -> None:
        self.folder_changed.emit(Path(raw))
