"""Validate pattern detection and spheroid segmentation, and export XLSX results."""
import argparse
import re
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

from src.datasets import SpheroidDataset
from src.models import load_pattern_model, load_spheroid_model
from src.segmentation import predict_spheroid_prob, remove_close_spheroids, remove_holes
from src.toolkit import format_day, pattern_region


def parse_name(name):
    def extract(pattern):
        match = re.search(pattern, name)
        return match.group(1) if match else ""

    return (
        extract(r"Plate_(\d+)"),
        extract(r"(?<![A-Za-z0-9])(D\d{2})(?![0-9])"),
        extract(r"(?<![A-Za-z0-9])([A-D][1-6])(?![0-9])"),
    )


def label_rows(labels, image_number, path, um_per_px):
    plate, day, well = parse_name(path.name)
    rows = []
    for label_number, label in enumerate(np.unique(labels[labels > 0]), start=1):
        area = (labels == label).sum() * um_per_px**2
        rows.append({
            "label": label_number,
            "ImageNumber": image_number,
            "AreaShape_Area": area,
            "AreaShape_EquivalentDiameter": np.sqrt(4 * area / np.pi),
            "ImagePath": str(path.resolve()),
            "Day": format_day(day),
            "Well": well,
            "Tile": "",
            "z": "",
            "Condition": f"Plate_{plate}" if plate else "",
        })
    return rows


def tile(img, title, width=380):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (width, width))
    cv2.putText(img, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


def validate(images_dir, pattern_path, model_path, output_dir, threshold=0.5,
             um_per_px=1.3, min_pattern_mm=1.0, show_segmented=False):
    images_dir, output_dir = Path(images_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern_model, _, _ = load_pattern_model(pattern_path)
    spheroid_model, _, _ = load_spheroid_model(model_path)
    helper = SpheroidDataset(images_dir, images_dir)
    object_rows, metric_rows = [], []

    for image_number, image_path in enumerate(sorted(images_dir.glob("*.jpg"))):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        pattern_mask, box, pattern_probability = pattern_region(pattern_model, image)
        if pattern_mask is None:
            print(f"{image_path.name}: NO PATTERN")
            continue

        x1, y1, x2, y2 = box
        crop_mask = pattern_mask[y1:y2, x1:x2]
        denoised = (image * pattern_mask)[y1:y2, x1:x2]
        json_path = image_path.with_suffix(".json")
        ground_truth = (
            helper._load_spheroid_mask(str(json_path), image.shape)[y1:y2, x1:x2] > 0
            if json_path.exists() else None
        )
        probability = predict_spheroid_prob(spheroid_model, denoised, crop_mask)
        predicted = probability > threshold
        labels = remove_close_spheroids(remove_holes(predicted), denoised)
        object_rows.extend(label_rows(labels, image_number, image_path, um_per_px))
        crop_width, crop_height = x2 - x1, y2 - y1
        pattern_width_mm = crop_width * um_per_px / 1000
        iou = dice = np.nan
        gt_count = pred_count = 0
        if ground_truth is not None:
            intersection = int((predicted & ground_truth).sum())
            union = int((predicted | ground_truth).sum())
            iou = intersection / max(union, 1)
            dice = 2 * intersection / max(int(predicted.sum() + ground_truth.sum()), 1)
            gt_count = cv2.connectedComponents(ground_truth.astype(np.uint8))[0] - 1
            pred_count = cv2.connectedComponents(predicted.astype(np.uint8))[0] - 1

        msg = (f"{image_path.name}: pattern_cover={pattern_mask.mean():.3f} "
               f"crop={crop_width}x{crop_height} pattern_width={pattern_width_mm:.2f} mm "
               f"threshold={threshold:.2f} IoU={iou:.3f} Dice={dice:.3f}")
        if ground_truth is not None:
            msg += f" spheroids GT/pred={gt_count}/{pred_count}"
        print(msg)
        if pattern_width_mm < min_pattern_mm:
            print(f"WARNING {image_path.name}: pattern is {pattern_width_mm:.2f} mm wide")

        full = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(full, (x1, y1), (x2, y2), (0, 0, 255), 40)
        overlay = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        for mask, color in ((ground_truth, (0, 255, 0)), (predicted, (0, 0, 255))):
            if mask is not None:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 3)
        ground_truth_img = (ground_truth.astype(np.uint8) * 255 if ground_truth is not None
                            else np.zeros_like(denoised))
        panel = np.hstack([
            tile(full, "1 original + crop"),
            tile((pattern_probability * 255).astype(np.uint8), "2 pattern prob"),
            tile(pattern_mask * 255, "3 largest region"),
            tile(denoised, "4 denoised crop"),
            tile(ground_truth_img, "5 GT spheroids"),
            tile((probability * 255).astype(np.uint8), "6 spheroid prob"),
            tile(overlay, "7 pred red / GT green"),
        ])
        cv2.imwrite(str(output_dir / f"{image_path.stem}.jpg"), panel)
        if show_segmented:
            cv2.imwrite(str(output_dir / f"{image_path.stem}_segmented.jpg"), denoised)

    with pd.ExcelWriter(output_dir / "results.xlsx") as writer:
        pd.DataFrame(object_rows).to_excel(writer, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/segmentation/images")
    parser.add_argument("--pattern", default="pattern_model.pt")
    parser.add_argument("--model", default="segmentation_model.pt")
    parser.add_argument("--out", default="debug_out")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--um-per-px", type=float, default=1.3,
                        help="pixel size in um (Leica Thunder, 5x, 6.5 um sensor pixel)")
    parser.add_argument("--min-pattern-mm", type=float, default=1.0)
    parser.add_argument("--show-segmented", action="store_true")
    args = parser.parse_args()
    validate(args.images, args.pattern, args.model, args.out, args.threshold,
             args.um_per_px, args.min_pattern_mm, args.show_segmented)


if __name__ == "__main__":
    main()
