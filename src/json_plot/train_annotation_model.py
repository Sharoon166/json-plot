import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "raw-data"
MODEL_FILE = PROJECT_ROOT / "outputs" / "models" / "annotation_text_model.json"

SCHEDULE_LABELS = {
    "verified_sched_entry": "drawing_identifier",
    "verified_tag": "drawing_identifier",
    "missing_from_schedule": "drawing_identifier",
}

REFERENCE_LABELS = {
    "verified": "reference",
    "verified_from_index": "reference",
    "missing_from_index": "reference",
    "verified_sheets": "reference",
}


def load_json(filename: str) -> dict[str, Any]:
    with (RAW_DATA_DIR / filename).open("r", encoding="utf-8") as file:
        return json.load(file)


def text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [text for item in value for text in text_values(item)]
    return []


def build_training_examples() -> list[dict[str, str]]:
    schedule_outputs = load_json("03_schedule_outputs.json")
    reference_data = load_json("04_data_out.json")
    examples = []

    for page_id, page in schedule_outputs.items():
        if not isinstance(page, dict):
            continue
        for source_label, target_label in SCHEDULE_LABELS.items():
            record = page.get(source_label)
            if not isinstance(record, dict):
                continue
            for text in text_values(record.get("texts", [])):
                examples.append({
                    "text": text,
                    "label": target_label,
                    "page_id": page_id,
                    "source": source_label,
                })

    for page_id, page in reference_data.items():
        if not isinstance(page, dict):
            continue
        for source_label, target_label in REFERENCE_LABELS.items():
            record = page.get(source_label)
            if not isinstance(record, dict):
                continue
            values = [
                *text_values(record.get("texts", [])),
                *text_values(record.get("text", [])),
                *text_values(record.get("callout_texts", [])),
                *text_values(record.get("value", "")),
            ]
            for text in values:
                examples.append({
                    "text": text,
                    "label": target_label,
                    "page_id": page_id,
                    "source": source_label,
                })

    return examples


def text_features(text: str) -> list[str]:
    normalized = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not normalized:
        return []

    bounded = f"^{normalized}$"
    features = [f"length:{min(len(normalized), 12)}"]
    features.append("has_digit" if any(char.isdigit() for char in normalized) else "no_digit")
    features.append("has_letter" if any(char.isalpha() for char in normalized) else "no_letter")
    for size in range(1, min(4, len(bounded)) + 1):
        features.extend(
            f"char:{bounded[index:index + size]}"
            for index in range(len(bounded) - size + 1)
        )
    return features


def train_model(examples: list[dict[str, str]]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    feature_counts: dict[str, Counter[str]] = defaultdict(Counter)
    total_features: Counter[str] = Counter()

    for example in examples:
        label = example["label"]
        features = text_features(example["text"])
        if not features:
            continue
        label_counts[label] += 1
        feature_counts[label].update(features)
        total_features[label] += len(features)

    if len(label_counts) < 2:
        raise ValueError("Training needs examples from at least two annotation categories.")

    vocabulary = sorted({
        feature
        for counts in feature_counts.values()
        for feature in counts
    })

    return {
        "format_version": 1,
        "algorithm": "character_ngram_naive_bayes",
        "alpha": 0.5,
        "class_counts": dict(label_counts),
        "feature_counts": {
            label: dict(counts)
            for label, counts in feature_counts.items()
        },
        "total_features": dict(total_features),
        "vocabulary_size": len(vocabulary),
        "training_example_count": sum(label_counts.values()),
    }


def predict_text(text: str, model: dict[str, Any]) -> tuple[str, float]:
    features = text_features(text)
    if not features:
        raise ValueError("Cannot classify text with no letters or digits.")

    class_counts = model["class_counts"]
    total_examples = sum(class_counts.values())
    alpha = model["alpha"]
    vocabulary_size = model["vocabulary_size"]
    scores = {}

    for label, count in class_counts.items():
        prior = math.log(count / total_examples)
        denominator = model["total_features"][label] + alpha * vocabulary_size
        score = prior
        for feature in features:
            observed = model["feature_counts"][label].get(feature, 0)
            score += math.log((observed + alpha) / denominator)
        scores[label] = score

    best_label = max(scores, key=scores.get)
    max_score = scores[best_label]
    exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
    confidence = exp_scores[best_label] / sum(exp_scores.values())
    return best_label, confidence


def main() -> None:
    examples = build_training_examples()
    model = train_model(examples)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Training examples: {model['training_example_count']}")
    for label, count in sorted(model["class_counts"].items()):
        print(f"  {label}: {count}")
    print(f"Model saved to: {MODEL_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()