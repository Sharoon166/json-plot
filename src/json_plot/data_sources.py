import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "raw-data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PDF_FILE = RAW_DATA_DIR / "32d890089ff749a18f9d3ea4e523758bpre (1).pdf"
OCR_FILE = RAW_DATA_DIR / "01_ocr_word_dict.json"
SCHEDULE_FILE = RAW_DATA_DIR / "02_schedule_data.json"
SCHEDULE_OUTPUT_FILE = RAW_DATA_DIR / "03_schedule_outputs.json"
DATA_OUT_FILE = RAW_DATA_DIR / "04_data_out.json"


@dataclass(frozen=True)
class DrawingData:
    ocr_data: dict[str, Any]
    schedule_data: dict[str, Any]
    schedule_outputs: dict[str, Any]
    data_out: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_datasets() -> DrawingData:
    return DrawingData(
        ocr_data=load_json(OCR_FILE),
        schedule_data=load_json(SCHEDULE_FILE),
        schedule_outputs=load_json(SCHEDULE_OUTPUT_FILE),
        data_out=load_json(DATA_OUT_FILE),
    )