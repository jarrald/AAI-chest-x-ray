"""Create a cleaned copy of the chest X-ray dataset.

This script removes bright border marker artifacts, such as orientation letters,
from each image and writes a second dataset with the same split/class folder
structure. It intentionally does not train, evaluate, or compare models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def remove_letters(
    gray: np.ndarray,
    bright_thr: int = 220,
    min_area: int = 25,
    max_area_frac: float = 0.006,
    edge_frac: float = 0.16,
    bottom_excl: float = 0.12,
    min_extent: float = 0.22,
    ring_max_mean: int = 110,
    close_px: int = 9,
    bbox_margin: float = 0.35,
    dilate_px: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and inpaint bright letter/marker artifacts in a grayscale X-ray."""
    h, w = gray.shape
    bright = (gray >= bright_thr).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, close_kernel)

    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        bright, connectivity=8
    )

    mask = np.zeros((h, w), np.uint8)
    max_area = max_area_frac * h * w

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        in_edge_band = (
            cx < edge_frac * w or cx > w - edge_frac * w or cy < edge_frac * h
        )
        if not in_edge_band or cy > (1.0 - bottom_excl) * h:
            continue

        aspect = bw / max(bh, 1)
        if aspect < 0.2 or aspect > 5.0:
            continue
        if area / (bw * bh + 1e-6) < min_extent:
            continue

        margin = max(4, int(0.5 * max(bw, bh)))
        ox0, oy0 = max(0, x - margin), max(0, y - margin)
        ox1, oy1 = min(w, x + bw + margin), min(h, y + bh + margin)
        ring = gray[oy0:oy1, ox0:ox1].astype(np.float32).copy()
        ring[y - oy0 : y - oy0 + bh, x - ox0 : x - ox0 + bw] = np.nan
        if np.nanmean(ring) > ring_max_mean:
            continue

        mx = int(bbox_margin * bw) + 3
        my = int(bbox_margin * bh) + 3
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
        mask[y0:y1, x0:x1] = 255

    if not mask.any():
        return gray, mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    mask = cv2.dilate(mask, kernel)
    cleaned = cv2.inpaint(gray, mask, 5, cv2.INPAINT_TELEA)
    return cleaned, mask


def iter_dataset_images(source: Path):
    for split in SPLITS:
        for class_name in CLASSES:
            class_dir = source / split / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing expected directory: {class_dir}")
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTS:
                    yield split, class_name, image_path


def save_cleaned_image(source_path: Path, output_path: Path) -> bool:
    gray = np.array(Image.open(source_path).convert("L"), dtype=np.uint8)
    cleaned, mask = remove_letters(gray)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cleaned, mode="L").save(output_path)
    return bool(mask.any())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a cleaned chest X-ray dataset without model evaluation."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("chest_xray"),
        help="Source dataset root containing train/val/test folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chest_xray_cleaned"),
        help="Destination root for the cleaned dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite images that already exist in the output dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if source == output:
        raise ValueError("Source and output directories must be different.")

    total = 0
    written = 0
    skipped = 0
    cleaned_count = 0

    for split, class_name, image_path in iter_dataset_images(source):
        total += 1
        relative_path = Path(split) / class_name / image_path.name
        output_path = output / relative_path

        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        had_artifact = save_cleaned_image(image_path, output_path)
        written += 1
        cleaned_count += int(had_artifact)

        if written % 250 == 0:
            print(f"Written {written} images...")

    print("Cleaned dataset creation complete.")
    print(f"Source:  {source}")
    print(f"Output:  {output}")
    print(f"Images found:      {total}")
    print(f"Images written:    {written}")
    print(f"Images skipped:    {skipped}")
    print(f"Images inpainted:  {cleaned_count}")


if __name__ == "__main__":
    main()
