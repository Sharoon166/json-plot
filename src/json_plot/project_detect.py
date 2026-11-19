"""Find a project's source files inside one folder.

The clean "pre" PDF is the annotation source. Packets often hold several PDFs,
so the rest are offered as training candidates, largest first. JSON files are
matched by the names the packets already use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


PRE_PATTERN = re.compile(r"pre(\s*\(\d+\))?$", re.IGNORECASE)
JSON_PATTERNS = {
    "schedule_data": ("schedule_data",),
    "schedule_outputs": ("schedule_output",),
    "reference_data": ("data_out",),
}


@dataclass
class DetectedSources:
    folder: Path
    pdfs: list[Path] = field(default_factory=list)
    jsons: list[Path] = field(default_factory=list)
    guesses: dict[str, Path] = field(default_factory=dict)
    pdf_candidates: dict[str, list[Path]] = field(default_factory=dict)
    annotation_is_pre: bool = False  # False means the annotation PDF was only a size guess

    @property
    def is_empty(self) -> bool:
        return not self.pdfs and not self.jsons


def _is_pre(path: Path) -> bool:
    return bool(PRE_PATTERN.search(path.stem.strip()))


def detect_sources(folder: Path) -> DetectedSources:
    found = DetectedSources(folder=folder)
    if not folder.is_dir():
        return found
    found.pdfs = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    found.jsons = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".json" and not path.name.lower().endswith(".progress.json")
        and path.stat().st_size > 2
    )

    pre_pdfs = [path for path in found.pdfs if _is_pre(path)]
    annotation = pre_pdfs[0] if pre_pdfs else None
    found.annotation_is_pre = annotation is not None
    if annotation is None and found.pdfs:
        annotation = min(found.pdfs, key=lambda path: path.stat().st_size)  # The smallest is the clean drawing.
    training_pool = [path for path in found.pdfs if path != annotation]
    training = max(training_pool, key=lambda path: path.stat().st_size) if training_pool else None

    if annotation is not None:
        found.guesses["annotation_pdf"] = annotation
    if training is not None:
        found.guesses["training_pdf"] = training
    found.pdf_candidates = {
        "annotation_pdf": sorted(found.pdfs, key=lambda path: (not _is_pre(path), path.name)),
        "training_pdf": sorted(found.pdfs, key=lambda path: (path != training, path.name)),
    }

    for key, needles in JSON_PATTERNS.items():
        match = next((path for path in found.jsons if any(needle in path.name.lower() for needle in needles)), None)
        if match is not None:
            found.guesses[key] = match
    return found
