import json
import re
from pathlib import Path
from typing import Any

from json_plot.config import (
    ANNOTATION_PDF_FILE,
    DEFAULT_PROJECT_ROOT,
    REFERENCE_DATA_FILE,
    SCHEDULE_DATA_FILE,
    SCHEDULE_OUTPUT_FILE,
    TRAINING_PDF_FILE,
)


REGISTRY_DIR = Path.home() / ".local" / "share" / "json-plot"
REGISTRY_FILE = REGISTRY_DIR / "projects.json"
PDF_KEYS = ("training_pdf", "annotation_pdf")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "annotation-project"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class ProjectLibrary:
    def __init__(self) -> None:
        self.projects: list[dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if REGISTRY_FILE.exists():
            data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
            self.projects = data.get("projects", [])
            self._prune_missing_configurations()
            return

        default_config = DEFAULT_PROJECT_ROOT / ".json-plot" / "project.json"
        project = {
            "id": slugify(DEFAULT_PROJECT_ROOT.name),
            "name": "Current workspace",
            "root": str(DEFAULT_PROJECT_ROOT),
            "config": str(default_config),
        }
        write_json(default_config, {
            "name": project["name"],
            "root": project["root"],
            "output_dir": str(DEFAULT_PROJECT_ROOT / "outputs"),
            "sources": {
                "training_pdf": str(TRAINING_PDF_FILE),
                "schedule_data": str(SCHEDULE_DATA_FILE),
                "schedule_outputs": str(SCHEDULE_OUTPUT_FILE),
                "reference_data": str(REFERENCE_DATA_FILE),
                "annotation_pdf": str(ANNOTATION_PDF_FILE),
            },
        })
        self.projects = [project]
        self.save()

    def _prune_missing_configurations(self) -> None:
        self.projects = [project for project in self.projects if Path(project.get("config", "")).is_file()]

    def save(self) -> None:
        write_json(REGISTRY_FILE, {"version": 1, "projects": self.projects})

    def add(
        self,
        name: str,
        root: Path,
        sources: dict[str, Path],
    ) -> dict[str, Any]:
        root = root.expanduser().resolve()
        if any(Path(item["root"]).resolve() == root for item in self.projects):
            raise ValueError("A project already exists at that folder.")
        config_path = root / ".json-plot" / "project.json"
        if config_path.exists():
            raise ValueError("That folder already contains a JSON Plot project configuration.")
        root.mkdir(parents=True, exist_ok=True)
        project = {
            "id": slugify(name) + "-" + str(len(self.projects) + 1),
            "name": name.strip(),
            "root": str(root),
            "config": str(config_path),
        }
        write_json(config_path, {
            "name": project["name"],
            "root": str(root),
            "output_dir": str(root / "outputs"),
            "sources": {
                **{key: str(sources[key].expanduser().resolve()) for key in PDF_KEYS},
                "json_folder": str(sources["json_folder"].expanduser().resolve()),
            },
        })
        self.projects.append(project)
        self.save()
        return project

    def rename(self, project: dict[str, Any], new_name: str) -> None:
        project["name"] = new_name.strip()
        settings = json.loads(Path(project["config"]).read_text(encoding="utf-8"))
        settings["name"] = new_name.strip()
        write_json(Path(project["config"]), settings)
        self.save()

    def remove(self, project: dict[str, Any]) -> None:
        self.projects = [item for item in self.projects if item["id"] != project["id"]]
        self.save()