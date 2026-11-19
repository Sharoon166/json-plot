import json
import os
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_DIR = DEFAULT_PROJECT_ROOT / "raw-data"
PROJECT_CONFIG_FILE = os.environ.get("JSON_PLOT_PROJECT_CONFIG")
PROJECT_SETTINGS = {}
if PROJECT_CONFIG_FILE:
	with Path(PROJECT_CONFIG_FILE).expanduser().open("r", encoding="utf-8") as file:
		PROJECT_SETTINGS = json.load(file)

PROJECT_ROOT = Path(PROJECT_SETTINGS.get("root", DEFAULT_PROJECT_ROOT)).expanduser().resolve()
RAW_DATA_DIR = PROJECT_ROOT / "raw-data"
OUTPUT_DIR = Path(PROJECT_SETTINGS.get("output_dir", PROJECT_ROOT / "outputs")).expanduser().resolve()
SOURCES = PROJECT_SETTINGS.get("sources", {})
TRAINING_PDF_FILE = Path(SOURCES.get("training_pdf", DEFAULT_RAW_DATA_DIR / "32d890089ff749a18f9d3ea4e523758b (6).pdf")).expanduser()
ANNOTATION_PDF_FILE = Path(SOURCES.get("annotation_pdf", DEFAULT_RAW_DATA_DIR / "32d890089ff749a18f9d3ea4e523758bpre (1).pdf")).expanduser()
SCHEDULE_DATA_FILE = Path(SOURCES.get("schedule_data", DEFAULT_RAW_DATA_DIR / "02_schedule_data.json")).expanduser()
SCHEDULE_OUTPUT_FILE = Path(SOURCES.get("schedule_outputs", DEFAULT_RAW_DATA_DIR / "03_schedule_outputs.json")).expanduser()
REFERENCE_DATA_FILE = Path(SOURCES.get("reference_data", DEFAULT_RAW_DATA_DIR / "04_data_out.json")).expanduser()
DATASET_DIR = OUTPUT_DIR / "visual_dataset"
TRAINING_DIR = OUTPUT_DIR / "visual_model"
REVIEW_DIR = OUTPUT_DIR / "visual_annotations_pre"
LEGACY_REVIEW_DIR = OUTPUT_DIR / "visual_annotations"
DEFAULT_WEIGHTS = "yolo11n.pt"
CLASS_NAMES = ["tag", "door_reference", "callout_area", "schedule_entry", "reference"]