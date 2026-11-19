"""Render PDF pages at the exact on-screen size, one tile at a time.

Drawing a whole sheet at a fixed resolution and scaling it up is what makes
zoomed views look blocky. Rendering only the visible tiles at the current
zoom keeps lines and text sharp at any level, and memory stays bounded.
"""

from __future__ import annotations

import ctypes
from collections import OrderedDict
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as raw
from PySide6.QtCore import QRect
from PySide6.QtGui import QImage, QPainter


TILE = 512
TILE_CACHE_LIMIT = 96
_documents: OrderedDict[str, pdfium.PdfDocument] = OrderedDict()
_tiles: OrderedDict[tuple, QImage] = OrderedDict()


def _document(path: Path) -> pdfium.PdfDocument:
    key = str(path)
    document = _documents.get(key)
    if document is None:
        document = pdfium.PdfDocument(key)
        _documents[key] = document
        while len(_documents) > 2:
            _documents.popitem(last=False)
    else:
        _documents.move_to_end(key)
    return document


def render_tile(
    path: Path,
    page_index: int,
    full_width: int,
    full_height: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> QImage:
    """One width x height tile of a page drawn at full_width x full_height pixels."""
    key = (str(path), page_index, full_width, full_height, x0, y0, width, height)
    cached = _tiles.get(key)
    if cached is not None:
        _tiles.move_to_end(key)
        return cached

    page = _document(path)[page_index]
    bitmap = raw.FPDFBitmap_Create(width, height, 0)
    try:
        raw.FPDFBitmap_FillRect(bitmap, 0, 0, width, height, 0xFFFFFFFF)
        raw.FPDF_RenderPageBitmap(
            bitmap, page.raw, -x0, -y0, full_width, full_height, 0, raw.FPDF_ANNOT
        )
        stride = raw.FPDFBitmap_GetStride(bitmap)
        data = ctypes.string_at(raw.FPDFBitmap_GetBuffer(bitmap), stride * height)
        image = QImage(data, width, height, stride, QImage.Format.Format_RGB32).copy()
    finally:
        raw.FPDFBitmap_Destroy(bitmap)

    _tiles[key] = image
    while len(_tiles) > TILE_CACHE_LIMIT:
        _tiles.popitem(last=False)
    return image


def paint_page(painter: QPainter, target: QRect, visible: QRect, path: Path, page_index: int) -> None:
    """Draw the visible part of a page that occupies `target` on screen."""
    full_width = target.width()
    full_height = target.height()
    left = max(0, visible.left() - target.left())
    top = max(0, visible.top() - target.top())
    right = min(full_width, visible.right() + 1 - target.left())
    bottom = min(full_height, visible.bottom() + 1 - target.top())
    for row in range(top // TILE, (bottom - 1) // TILE + 1):
        for column in range(left // TILE, (right - 1) // TILE + 1):
            x0 = column * TILE
            y0 = row * TILE
            width = min(TILE, full_width - x0)
            height = min(TILE, full_height - y0)
            tile = render_tile(path, page_index, full_width, full_height, x0, y0, width, height)
            painter.drawImage(target.left() + x0, target.top() + y0, tile)
