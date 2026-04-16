# -*- coding: utf-8 -*-
"""
CVC-ClinicDB processor (Supervisely ann JSON: ann file name = <image_filename>.json e.g. 1.png.json)
- img: ds/img/*.png
- ann: ds/ann/*.png.json
- 解码 bitmap.data (base64+zlib png) + origin 贴回整图 mask
- mask -> bbox -> resize+letterbox 1024
- 输出 processed_1024/{images_1024,masks_1024,meta.csv}
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from skimage import io, transform, img_as_ubyte
from skimage.color import gray2rgb

import base64
import zlib
from io import BytesIO

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("需要 pillow：pip install pillow") from e


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_image(path):
    img = io.imread(path)
    if img.ndim == 2:
        img = gray2rgb(img)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


def decode_supervisely_bitmap(data_b64_zlib_png: str) -> np.ndarray:
    raw = base64.b64decode(data_b64_zlib_png)
    png_bytes = zlib.decompress(raw)
    im = Image.open(BytesIO(png_bytes))
    arr = np.array(im)

    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, 3]
        else:
            arr = arr[:, :, 0]

    if arr.max() > 1:
        arr = (arr > 127).astype(np.uint8)
    else:
        arr = (arr > 0).astype(np.uint8)
    return arr


def build_mask_from_ann_json(ann_json_path: str) -> np.ndarray:
    with open(ann_json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    H = int(ann["size"]["height"])
    W = int(ann["size"]["width"])
    full = np.zeros((H, W), dtype=np.uint8)

    for obj in ann.get("objects", []):
        if obj.get("geometryType") != "bitmap":
            continue
        bmp = obj.get("bitmap", None)
        if not bmp:
            continue

        data = bmp.get("data", None)
        origin = bmp.get("origin", None)
        if data is None or origin is None:
            continue

        patch = decode_supervisely_bitmap(data)  # (ph,pw) 0/1
        ox, oy = int(origin[0]), int(origin[1])  # [x,y]
        ph, pw = patch.shape[:2]

        x1, y1 = ox, oy
        x2, y2 = ox + pw, oy + ph

        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)
        if x2c <= x1c or y2c <= y1c:
            continue

        px1 = x1c - x1
        py1 = y1c - y1
        px2 = px1 + (x2c - x1c)
        py2 = py1 + (y2c - y1c)

        full[y1c:y2c, x1c:x2c] = np.maximum(full[y1c:y2c, x1c:x2c], patch[py1:py2, px1:px2])

    return full  # 0/1


def compute_bbox(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def resize_letterbox_1024(img_rgb01: np.ndarray, mask01: np.ndarray, bbox, target=1024):
    H, W = img_rgb01.shape[:2]
    scale = min(target / H, target / W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))

    img_r = transform.resize(img_rgb01, (new_h, new_w), preserve_range=True, anti_aliasing=True)
    mask_r = transform.resize(mask01.astype(float), (new_h, new_w), order=0, preserve_range=True)

    canvas = np.zeros((target, target, 3), dtype=np.float32)
    canvas_m = np.zeros((target, target), dtype=np.float32)

    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_r
    canvas_m[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = mask_r

    bbox_1024 = None
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        bbox_1024 = [
            int(round(x1 * scale + pad_x)),
            int(round(y1 * scale + pad_y)),
            int(round(x2 * scale + pad_x)),
            int(round(y2 * scale + pad_y)),
        ]
        bbox_1024[0] = max(0, min(bbox_1024[0], target - 1))
        bbox_1024[1] = max(0, min(bbox_1024[1], target - 1))
        bbox_1024[2] = max(bbox_1024[0] + 1, min(bbox_1024[2], target - 1))
        bbox_1024[3] = max(bbox_1024[1] + 1, min(bbox_1024[3], target - 1))

    return canvas, (canvas_m > 0.5).astype(np.uint8), bbox_1024


def main(args):
    out_img_dir = os.path.join(args.out_dir, "images_1024")
    out_msk_dir = os.path.join(args.out_dir, "masks_1024")
    ensure_dir(out_img_dir)
    ensure_dir(out_msk_dir)

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    img_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(exts)]
    img_files.sort()

    rows = []
    skip_missing = 0
    skip_empty = 0

    for f in img_files:
        img_path = os.path.join(args.img_dir, f)
        img_stem = os.path.splitext(f)[0]

        # ✅ 关键：ann 文件名是 <image_filename>.json，例如 1.png.json
        ann_path = os.path.join(args.ann_dir, f + ".json")
        if not os.path.exists(ann_path):
            # 兜底：1.json
            ann_path = os.path.join(args.ann_dir, img_stem + ".json")

        if not os.path.exists(ann_path):
            skip_missing += 1
            continue

        img = read_image(img_path).astype(np.float32) / 255.0
        mask01 = build_mask_from_ann_json(ann_path)

        bbox = compute_bbox(mask01)
        if bbox is None:
            skip_empty += 1
            continue

        img_1024, mask_1024, bbox_1024 = resize_letterbox_1024(img, mask01, bbox, target=args.target)

        out_img_name = f"{img_stem}.png"
        out_msk_name = f"{img_stem}_mask.png"

        io.imsave(os.path.join(out_img_dir, out_img_name), img_as_ubyte(np.clip(img_1024, 0, 1)))
        io.imsave(os.path.join(out_msk_dir, out_msk_name), (mask_1024.astype(np.uint8) * 255))

        rows.append({
            "id": img_stem,
            "img": out_img_name,
            "mask": out_msk_name,
            "bbox_1024": bbox_1024
        })

    meta_csv = os.path.join(args.out_dir, "meta.csv")
    pd.DataFrame(rows).to_csv(meta_csv, index=False)

    print(f"[OK] processed: {len(rows)} samples")
    print(f"[INFO] skipped (ann json missing): {skip_missing}")
    print(f"[INFO] skipped (empty mask): {skip_empty}")
    print(f"[OK] meta saved: {meta_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default=r"D:\MedSAM-main\cvc-clinic\ds\img")
    ap.add_argument("--ann_dir", default=r"D:\MedSAM-main\cvc-clinic\ds\ann")
    ap.add_argument("--out_dir", default=r"D:\MedSAM-main\cvc-clinic\processed_1024")
    ap.add_argument("--target", type=int, default=1024)
    args = ap.parse_args()
    main(args)
