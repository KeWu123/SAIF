#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kvasir-SEG dataset preprocessing (outputs images/masks + meta1.csv).

- Reads images and masks
- Reads bounding boxes from kavsir_bboxes.json
- Resizes with letterbox padding to target size (default 1024)
- Saves processed images and masks into one output directory
- Writes meta1.csv: id, img, mask, bbox_1024
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
    """Read image and ensure 3-channel RGB."""
    img = io.imread(path)
    if img.ndim == 2:
        img = gray2rgb(img)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def read_mask(path):
    """Read mask and convert to 0/1 binary."""
    m = io.imread(path)
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 0).astype(np.uint8)


def compute_bbox(mask, min_area=20):
    """Compute bbox from a mask (unused in this script)."""
    m = remove_small_objects(mask.astype(bool), min_size=min_area)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def resize_with_bbox(img, mask, bbox, target=1024):
    """Resize image/mask with letterbox padding and map bbox accordingly."""
    H, W = img.shape[:2]
    scale = min(target / H, target / W)
    new_h, new_w = int(H * scale), int(W * scale)

    img_r = transform.resize(img, (new_h, new_w), preserve_range=True, anti_aliasing=True)
    m_r = transform.resize(mask.astype(float), (new_h, new_w), order=0, preserve_range=True)

    canvas = np.zeros((target, target, 3))
    canvas_m = np.zeros((target, target))
    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_r
    canvas_m[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = m_r

    if bbox is None:
        return canvas, canvas_m > 0.5, None

    xmin, ymin, xmax, ymax = bbox
    return canvas, canvas_m > 0.5, [
        int(xmin * scale + pad_x), int(ymin * scale + pad_y),
        int(xmax * scale + pad_x), int(ymax * scale + pad_y)
    ]


def generate_pseudo_mask(image_shape, bbox):
    """Generate a rectangular pseudo-mask from bbox (unused)."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    rr, cc = rectangle(start=(bbox[1], bbox[0]), end=(bbox[3], bbox[2]), extent=None, shape=image_shape)
    mask[rr, cc] = 1
    return mask


def main():
    # Set paths
    img_dir = "D:/MedSAM-main/Kvasir-SEG/images"
    mask_dir = "D:/MedSAM-main/Kvasir-SEG/masks"
    bbox_path = "D:/MedSAM-main/Kvasir-SEG/kavsir_bboxes.json"
    out_dir = "D:/MedSAM-main/KVasir"

    out_img, out_msk = os.path.join(out_dir, "images_1024"), os.path.join(out_dir, "masks_1024")
    ensure_dir(out_img)
    ensure_dir(out_msk)

    with open(bbox_path, 'r') as f:
        bboxes = json.load(f)

    print(f"Loaded {len(bboxes)} entries from bounding box JSON file.")

    rows = []
    for img_name, bbox_info in bboxes.items():
        img_path = os.path.join(img_dir, img_name + ".jpg")
        mask_path = os.path.join(mask_dir, img_name + ".jpg")

        print(f"Processing image: {img_name}")
        print(f"Image path: {img_path}")
        print(f"Mask path: {mask_path}")

        if not os.path.exists(img_path):
            print(f"Image file missing: {img_path}")
            continue

        if not os.path.exists(mask_path):
            print(f"Mask file missing: {mask_path}")
            continue

        img = read_image(img_path).astype(float) / 255.0
        mask = read_mask(mask_path)

        # Pull bboxes from JSON
        bboxes_in_image = []
        for bbox in bbox_info["bbox"]:
            bbox_coords = [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]
            bboxes_in_image.append(bbox_coords)

        if not bboxes_in_image:
            print(f"No valid bounding box found for {img_name}, skipping.")
            continue

        img_1024, mask_1024, bbox_1024 = resize_with_bbox(img, mask, bboxes_in_image[0])

        out_name = f"{img_name}.jpg"
        io.imsave(os.path.join(out_img, out_name), img_as_ubyte(np.clip(img_1024, 0, 1)))
        io.imsave(os.path.join(out_msk, out_name.replace(".jpg", "_mask.png")), (mask_1024 * 255).astype(np.uint8))

        rows.append({
            "id": img_name,
            "img": out_name,
            "mask": out_name.replace(".jpg", "_mask.png"),
            "bbox_1024": bbox_1024
        })

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "meta1.csv"), index=False)
    print(f"[OK] processed {len(rows)} samples, saved to {out_dir}")


if __name__ == "__main__":
    main()
