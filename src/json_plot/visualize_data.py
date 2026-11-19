
# scripts/visualize_data.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from json_plot.data_sources import OUTPUTS_DIR, load_datasets


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OUTPUT_DIR = OUTPUTS_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = load_datasets()
ocr_data = datasets.ocr_data
schedule_data = datasets.schedule_data
schedule_outputs = datasets.schedule_outputs
data_out = datasets.data_out


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def safe_list(value):
    return value if isinstance(value, list) else []


def get_page_count(page_data):
    return len(page_data)


def count_schedule_annotations(page):
    """Count annotation bounding boxes across all annotation groups."""
    total = 0

    for group in page.values():
        if isinstance(group, dict):
            total += len(safe_list(group.get("bboxes")))

    return total


def count_schedule_output_items(page):
    """Count verified and missing schedule/tag entries."""
    result = {
        "verified_schedule": 0,
        "missing_schedule": 0,
        "verified_tags": 0,
        "missing_tags": 0,
    }

    if not isinstance(page, dict):
        return result

    for key, record in page.items():
        if not isinstance(record, dict):
            continue

        count = len(safe_list(record.get("bbox")))

        if key == "verified_sched_entry":
            result["verified_schedule"] = count

        elif key == "missing_sched_entry":
            result["missing_schedule"] = count

        elif key == "verified_tag":
            result["verified_tags"] = count

        elif key == "missing_from_schedule":
            result["missing_tags"] = count

    return result


def count_reference_statuses(page):
    """Count reference records by their status."""
    result = {
        "verified": 0,
        "verified_from_index": 0,
        "incorrect_reference_detail": 0,
        "other": 0,
    }

    if not isinstance(page, dict):
        return result

    for key, record in page.items():
        if not isinstance(record, dict):
            continue

        if key in result:
            result[key] += 1
        elif key.startswith("verified_from_index"):
            result["verified_from_index"] += 1
        elif key.startswith("verified"):
            result["verified"] += 1
        elif key.startswith("incorrect_reference_detail"):
            result["incorrect_reference_detail"] += 1
        else:
            result["other"] += 1

    return result


# --------------------------------------------------
# BUILD PAGE-LEVEL STATISTICS
# --------------------------------------------------

all_page_ids = sorted(
    set(ocr_data)
    | set(schedule_data)
    | set(schedule_outputs)
    | set(data_out),
    key=lambda x: int(x)
)

rows = []

for page_id in all_page_ids:

    words = safe_list(ocr_data.get(page_id, []))

    confidences = [
        float(word["confidence"])
        for word in words
        if isinstance(word, dict)
        and isinstance(word.get("confidence"), (int, float))
    ]

    avg_confidence = (
        sum(confidences) / len(confidences)
        if confidences else 0
    )

    schedule_count = count_schedule_annotations(
        schedule_data.get(page_id, {})
    )

    output_stats = count_schedule_output_items(
        schedule_outputs.get(page_id, {})
    )

    reference_stats = count_reference_statuses(
        data_out.get(page_id, {})
    )

    rows.append({
        "page_id": int(page_id),
        "ocr_words": len(words),
        "average_confidence": avg_confidence,
        "annotation_count": schedule_count,
        **output_stats,
        **reference_stats,
    })


df = pd.DataFrame(rows)

df = df.sort_values("page_id")

df.to_csv(
    OUTPUT_DIR / "page_statistics.csv",
    index=False
)


# --------------------------------------------------
# PRINT SUMMARY
# --------------------------------------------------

print("\n========== DOCUMENT SUMMARY ==========\n")

print(f"Total pages: {len(all_page_ids)}")
print(f"Total OCR words: {df['ocr_words'].sum()}")
print(f"Total annotations: {df['annotation_count'].sum()}")

print(
    f"Average OCR confidence: "
    f"{df.loc[df['ocr_words'] > 0, 'average_confidence'].mean():.3f}"
)

print(
    f"Verified schedule entries: "
    f"{df['verified_schedule'].sum()}"
)

print(
    f"Missing schedule entries: "
    f"{df['missing_schedule'].sum()}"
)

print(
    f"Verified tags: "
    f"{df['verified_tags'].sum()}"
)

print(
    f"Missing tags: "
    f"{df['missing_tags'].sum()}"
)

print(
    f"Verified references: "
    f"{df['verified'].sum() + df['verified_from_index'].sum()}"
)

print(
    f"Incorrect reference details: "
    f"{df['incorrect_reference_detail'].sum()}"
)


# --------------------------------------------------
# VISUALIZATION STYLE
# --------------------------------------------------

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(
    3, 2,
    figsize=(17, 16)
)

fig.suptitle(
    "Drawing Processing Analysis Dashboard",
    fontsize=20,
    fontweight="bold"
)


# 1. OCR WORDS PER PAGE

sns.barplot(
    data=df,
    x="page_id",
    y="ocr_words",
    ax=axes[0, 0],
    color="#3498db"
)

axes[0, 0].set_title("OCR Words per Page")
axes[0, 0].set_xlabel("Page")
axes[0, 0].set_ylabel("Word Count")


# 2. OCR CONFIDENCE

sns.barplot(
    data=df,
    x="page_id",
    y="average_confidence",
    ax=axes[0, 1],
    color="#2ecc71"
)

axes[0, 1].set_title("Average OCR Confidence per Page")
axes[0, 1].set_xlabel("Page")
axes[0, 1].set_ylabel("Average Confidence")

axes[0, 1].set_ylim(0, 1)


# 3. ANNOTATIONS PER PAGE

sns.barplot(
    data=df,
    x="page_id",
    y="annotation_count",
    ax=axes[1, 0],
    color="#9b59b6"
)

axes[1, 0].set_title("Drawing Annotations per Page")
axes[1, 0].set_xlabel("Page")
axes[1, 0].set_ylabel("Annotation Count")


# 4. SCHEDULE VERIFICATION

schedule_plot = df.set_index("page_id")[
    [
        "verified_schedule",
        "missing_schedule",
        "verified_tags",
        "missing_tags",
    ]
]

schedule_plot.plot(
    kind="bar",
    ax=axes[1, 1],
    color=[
        "#27ae60",
        "#e74c3c",
        "#2980b9",
        "#f39c12",
    ]
)

axes[1, 1].set_title("Schedule and Tag Verification")
axes[1, 1].set_xlabel("Page")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend(fontsize=8)


# 5. REFERENCE VERIFICATION

reference_plot = df.set_index("page_id")[
    [
        "verified",
        "verified_from_index",
        "incorrect_reference_detail",
    ]
]

reference_plot.plot(
    kind="bar",
    stacked=True,
    ax=axes[2, 0],
    color=[
        "#2ecc71",
        "#3498db",
        "#e74c3c",
    ]
)

axes[2, 0].set_title("Reference Verification Status")
axes[2, 0].set_xlabel("Page")
axes[2, 0].set_ylabel("Record Count")


# 6. OCR WORDS VS ANNOTATIONS

sns.scatterplot(
    data=df,
    x="ocr_words",
    y="annotation_count",
    hue="average_confidence",
    size="ocr_words",
    sizes=(50, 300),
    palette="viridis",
    ax=axes[2, 1],
)

axes[2, 1].set_title("OCR Words vs Drawing Annotations")
axes[2, 1].set_xlabel("OCR Word Count")
axes[2, 1].set_ylabel("Annotation Count")


plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(
    OUTPUT_DIR / "analysis_dashboard.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# ADDITIONAL VISUALIZATION: OCR CONFIDENCE
# DISTRIBUTION
# --------------------------------------------------

all_confidences = [
    float(word["confidence"])
    for page_words in ocr_data.values()
    for word in safe_list(page_words)
    if isinstance(word, dict)
    and isinstance(word.get("confidence"), (int, float))
]

if all_confidences:

    plt.figure(figsize=(10, 6))

    sns.histplot(
        all_confidences,
        bins=30,
        kde=True,
        color="#3498db"
    )

    plt.title("Overall OCR Confidence Distribution")
    plt.xlabel("OCR Confidence")
    plt.ylabel("Number of Words")

    plt.savefig(
        OUTPUT_DIR / "ocr_confidence_distribution.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


print("\nVisualization completed!")
print(f"Results saved to: {OUTPUT_DIR}")
