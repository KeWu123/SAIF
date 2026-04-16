#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kvasir-SEG dataset preprocessing (single output directory version).

Functions:
- Read images and masks.
- Read bounding boxes from kavsir_bboxes.json.
- Resize image and mask with letterbox padding to target size (default: 1024).
- Save processed images and masks into one folder:
    out_dir/processed_1024/
        xxx.jpg
        xxx_mask.png
- Generate meta1.csv with:
    id, img, mask, bbox_1024

Defaults in this script:
    images: D:/MedSAM-main/Kvasir-SEG/images/*.jpg
    masks:  D:/MedSAM-main/Kvasir-SEG/masks/*.jpg
    bbox:   D:/MedSAM-main/Kvasir-SEG/kavsir_bboxes.json
    output: D:/MedSAM-main/KVasir
Change paths in main() if your layout differs.
"""

import os
import json
import pandas as pd
from skimage import io, transform, img_as_ubyte
import numpy as np
from skimage.color import gray2rgb
from skimage.draw import rectangle
from skimage.morphology import remove_small_objects


def ensure_dir(p: str):
    """Create directory if it does not exist."""
    os.makedirs(p, exist_ok=True)


def read_image(path):
    """Read an image and ensure it is 3-channel RGB."""
    img = io.imread(path)
    # Grayscale -> RGB
    if img.ndim == 2:
        img = gray2rgb(img)
    # RGBA -> RGB (drop alpha)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def read_mask(path):
    """Read a mask and convert to 0/1 binary."""
    m = io.imread(path)
    # If 3-channel, take the first channel
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 0).astype(np.uint8)


def compute_bbox(mask, min_area=20):
    """
    Extract a bounding box from a mask (unused in this script).
    This can be helpful if you want to compute bboxes directly from masks.
    """
    m = remove_small_objects(mask.astype(bool), min_size=min_area)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def resize_with_bbox(img, mask, bbox, target=1024):
    """
    Resize image and mask with letterbox padding to a square target size,
    and apply the same scale/shift to the bounding box.

    Args:
        img:   H x W x 3 image
        mask:  H x W binary mask
        bbox:  [xmin, ymin, xmax, ymax] or None
        target: target side length

    Returns:
        img_1024:  target x target x 3 image
        mask_1024: target x target binary mask (bool)
        bbox_1024: resized/padded bbox or None
    """
    H, W = img.shape[:2]
    scale = min(target / H, target / W)
    new_h, new_w = int(H * scale), int(W * scale)

    # Resize image and mask (nearest for mask to avoid smoothing)
    img_r = transform.resize(
        img, (new_h, new_w),
        preserve_range=True,
        anti_aliasing=True
    )
    m_r = transform.resize(
        mask.astype(float), (new_h, new_w),
        order=0,
        preserve_range=True
    )

    # Letterbox padding to square
    canvas = np.zeros((target, target, 3), dtype=img_r.dtype)
    canvas_m = np.zeros((target, target), dtype=m_r.dtype)
    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_r
    canvas_m[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = m_r

    if bbox is None:
        return canvas, canvas_m > 0.5, None

    xmin, ymin, xmax, ymax = bbox
    # Scale + shift bbox to the padded canvas
    xmin_new = int(xmin * scale + pad_x)
    ymin_new = int(ymin * scale + pad_y)
    xmax_new = int(xmax * scale + pad_x)
    ymax_new = int(ymax * scale + pad_y)

    return canvas, canvas_m > 0.5, [xmin_new, ymin_new, xmax_new, ymax_new]


def generate_pseudo_mask(image_shape, bbox):
    """
    Generate a simple rectangular pseudo-mask from a bbox (unused here).
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    rr, cc = rectangle(
        start=(bbox[1], bbox[0]),
        end=(bbox[3], bbox[2]),
        extent=None,
        shape=image_shape
    )
    mask[rr, cc] = 1
    return mask


def main():
    # ====== 1. Set paths (edit for your environment) ======
    img_dir = "D:/MedSAM-main/Kvasir-SEG/images"          # source images
    mask_dir = "D:/MedSAM-main/Kvasir-SEG/masks"          # source masks
    bbox_path = "D:/MedSAM-main/Kvasir-SEG/kavsir_bboxes.json"  # bbox json
    out_dir = "D:/MedSAM-main/KVasir"                     # output root

    # ====== 2. Single output directory ======
    out_single = os.path.join(out_dir, "processed_1024")
    ensure_dir(out_single)

    # ====== 3. Load bounding boxes ======
    with open(bbox_path, 'r') as f:
        bboxes = json.load(f)

    print(f"Loaded {len(bboxes)} entries from bounding box JSON file.")

    rows = []
    # ====== 4. Process each image with a bbox ======
    for img_name, bbox_info in bboxes.items():
        # File names in JSON do not include extension
        img_path = os.path.join(img_dir, img_name + ".jpg")
        mask_path = os.path.join(mask_dir, img_name + ".jpg")

        print(f"\nProcessing image: {img_name}")
        print(f"  Image path: {img_path}")
        print(f"  Mask path:  {mask_path}")

        # Validate file existence
        if not os.path.exists(img_path):
            print(f"  [WARN] Image file missing, skip: {img_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"  [WARN] Mask file missing, skip: {mask_path}")
            continue

        # Load image and mask
        img = read_image(img_path).astype(float) / 255.0
        mask = read_mask(mask_path)
        print(f"  Mask shape: {mask.shape}")

        # Read bbox list from JSON
        bboxes_in_image = []
        for bbox in bbox_info.get("bbox", []):
            bbox_coords = [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]
            bboxes_in_image.append(bbox_coords)

        # Skip if no bbox
        if not bboxes_in_image:
            print(f"  [WARN] No valid bounding box found for {img_name}, skipping.")
            continue

        # Use the first bbox only (modify here if you need multi-object support)
        img_1024, mask_1024, bbox_1024 = resize_with_bbox(
            img, mask, bboxes_in_image[0], target=1024
        )

        # ====== 5. Save processed image and mask to the same folder ======
        out_name = f"{img_name}.jpg"
        img_save_path = os.path.join(out_single, out_name)
        mask_name = out_name.replace(".jpg", "_mask.png")
        mask_save_path = os.path.join(out_single, mask_name)

        io.imsave(img_save_path, img_as_ubyte(np.clip(img_1024, 0, 1)))
        io.imsave(mask_save_path, (mask_1024.astype(np.uint8) * 255))

        # Record meta information
        rows.append({
            "id": img_name,
            "img": out_name,
            "mask": mask_name,
            "bbox_1024": bbox_1024
        })

    # ====== 6. Save meta1.csv ======
    meta_path = os.path.join(out_dir, "meta1.csv")
    pd.DataFrame(rows).to_csv(meta_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Processed {len(rows)} samples.")
    print(f"     Images & masks saved to: {out_single}")
    print(f"     Meta CSV saved to:       {meta_path}")


if __name__ == "__main__":
    main()
