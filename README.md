# JSON Plot

The visual detector trains from `raw-data/32d890089ff749a18f9d3ea4e523758b (6).pdf` and the supplied JSON bounding boxes. By default, it runs annotations on `raw-data/32d890089ff749a18f9d3ea4e523758bpre (1).pdf`. It learns five classes: tags, door references, callout areas, schedule entries, and references. The current dataset has 15 pages, so treat its accuracy as an early baseline, not a production result.

## Run It

Run these commands from the project directory. `uv run` selects the project environment automatically; there is no need to activate `.venv` or type its Python path.

```bash
uv sync
uv run json-plot-gui
uv run json-plot-model prepare
uv run json-plot-model train --epochs 5 --imgsz 640 --batch 1
uv run json-plot-model annotate --page 1
```

The Qt 6 desktop app opens to a project library. Create a project by selecting its labeled training PDF, the three learning JSON files (`02_schedule_data.json`, `03_schedule_outputs.json`, and `04_data_out.json`), and the PDF to annotate. Source files remain in place; project-specific datasets, detector weights, and reviews are stored under that project's `outputs/` folder. Projects can be renamed, opened, removed from the library, or revealed in the file manager.

Inside a project, the app provides the same prepare, train, and annotate actions, plus a page review workspace for accepting or rejecting detections and marking missed objects. Hold Ctrl and scroll to zoom around the pointer; use the scrollbars to pan, or Fit to restore the full page. Human decisions are included in the next dataset export. A Help button in the top bar summarizes the workflow.

`prepare` exports page tiles and labels to `outputs/visual_dataset/`. It holds out pages 2, 8, 11, and 14 for validation until a reviewer makes an accept/reject decision on that page. Once feedback exists, that page is moved into training to avoid evaluation leakage.

`train` fine-tunes the small YOLO11 model on CPU. The initial weights are downloaded on the first run. The best checkpoint is saved at `outputs/visual_model/detector/weights/best.pt`, and metrics and reward history are saved in `outputs/visual_model/training_summary.json`.

`annotate` runs the detector and Tesseract OCR over one 1-based page of the pre PDF and creates `outputs/visual_annotations_pre/page_NNN/` containing `annotated_page.png`, `detections.json`, and `human_review.json`. The image labels show the predicted class, matched OCR text, detector confidence (`D`), and OCR confidence (`O`). `detections.json` contains all page-level OCR words with boxes in pixels and PDF points, plus text matched to each detection. OCR confidence is separate from detector confidence; low-confidence text should be checked. Use `--pdf PATH` to annotate a different PDF. The review JSON records that PDF page's effective dimensions so corrected boxes map back to the training coordinate frame correctly. Existing `outputs/visual_annotations/` reviews are preserved and still included as legacy feedback when preparing training data. Tesseract must be installed and available on `PATH`.

## Human Feedback And Reward

In `human_review.json`, keep each detection ID unchanged. Set `decision` to `accepted` or `rejected`; optionally fill `corrected_label`, `corrected_bbox_pdf`, `comment`, and `notes`. `bbox_pdf` and corrected boxes use `[x0, y0, x1, y1]` in the annotated PDF's effective page coordinates.

For a false negative (an object that is absent from the model's detections), add a record under `missed_annotations` with a unique ID, `class_name`, and `bbox_pdf`, for example:

```json
"missed_annotations": {
	"missed-001": {
		"class_name": "tag",
		"bbox_pdf": [100, 200, 140, 240],
		"corrected_text": "C1",
		"comment": "Was not proposed"
	}
}
```

An unresolved miss counts as reward `-1` and its box becomes a positive training label. On the next annotation run, a same-class detection overlapping at least 50% marks that miss as `recovered`; it then counts as `+1`. The image marks unresolved misses in magenta and recovered ones in green. This lets feedback represent false negatives as well as accepted and rejected proposals.

Accepted items receive reward `+1` and corrected accepted boxes are added to the next training export. Rejected items receive `-1`; a matching source label is removed, and unmatched false-positive areas remain background examples. Per-class acceptance rates use a smoothed reward factor to adjust reported detection confidence. Re-run `prepare` and `train` to incorporate decided feedback, then rerun `annotate` for the page.

This is a simple human-feedback reward, not policy-gradient reinforcement learning. Human decisions influence the training labels and confidence calibration; they do not directly update detector weights until training is run again.
