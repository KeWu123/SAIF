#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CVC-300 数据集预处理（适配你当前目录：images / masks，文件名为数字编号如 149.png）：

输入:
  --img_dir   D:\MedSAM-main\CVC-300\images
  --mask_dir  D:\MedSAM-main\CVC-300\masks

输出(默认 target=1024):
  out_dir/
    images_1024/   (letterbox 后的图)
    masks_1024/    (同步处理后的GT)
    meta.csv       (id, img, mask, bbox_1024)

说明:
- bbox 直接由 GT mask 自动计算
- resize 采用等比例缩放 + 居中填充到 1024x1024 (letterbox)
- bbox 会同步映射到 1024 坐标系
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io, transform, img_as_ubyte
from skimage.color import gray2rgb
from skimage.morphology import remove_small_objects


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_image(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 2:
        img = gray2rgb(img)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


def read_mask(path: str) -> np.ndarray:
    m = io.imread(path)
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m > 0).astype(np.uint8)


def compute_bbox(mask: np.ndarray, min_area: int = 20):
    """bbox: [xmin, ymin, xmax, ymax]，坐标基于原图像素坐标"""
    m = remove_small_objects(mask.astype(bool), min_size=min_area)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def resize_with_bbox(img: np.ndarray, mask: np.ndarray, bbox, target: int = 1024):
    """等比例缩放 + 居中填充到 target，同时映射 bbox 到 target 坐标"""
    H, W = img.shape[:2]
    scale = min(target / H, target / W)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))

    img_r = transform.resize(img, (new_h, new_w), preserve_range=True, anti_aliasing=True)
    m_r = transform.resize(mask.astype(float), (new_h, new_w), order=0, preserve_range=True)

    canvas = np.zeros((target, target, 3), dtype=np.float32)
    canvas_m = np.zeros((target, target), dtype=np.float32)

    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_r
    canvas_m[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = m_r

    if bbox is None:
        return canvas, (canvas_m > 0.5), None

    xmin, ymin, xmax, ymax = bbox
    bbox_1024 = [
        int(round(xmin * scale + pad_x)),
        int(round(ymin * scale + pad_y)),
        int(round(xmax * scale + pad_x)),
        int(round(ymax * scale + pad_y)),
    ]

    # clamp
    bbox_1024[0] = max(0, min(bbox_1024[0], target - 1))
    bbox_1024[1] = max(0, min(bbox_1024[1], target - 1))
    bbox_1024[2] = max(1, min(bbox_1024[2], target - 1))
    bbox_1024[3] = max(1, min(bbox_1024[3], target - 1))
    if bbox_1024[2] <= bbox_1024[0]:
        bbox_1024[2] = min(target - 1, bbox_1024[0] + 1)
    if bbox_1024[3] <= bbox_1024[1]:
        bbox_1024[3] = min(target - 1, bbox_1024[1] + 1)

    return canvas, (canvas_m > 0.5), bbox_1024


def find_matching_mask(mask_dir: str, stem: str):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    for ext in exts:
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    # 兜底：mask 可能就是纯数字无后缀/其他后缀，做一次扫描匹配 stem
    for f in os.listdir(mask_dir):
        s, _ = os.path.splitext(f)
        if s == stem:
            return os.path.join(mask_dir, f)
    return None


def main():
    ap = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    cvc_root = repo_root / "CVC-300"

    ap.add_argument("--img_dir", default=str(cvc_root / "images"), help="CVC-300 images directory")
    ap.add_argument("--mask_dir", default=str(cvc_root / "masks"), help="CVC-300 masks directory")
    ap.add_argument("--out_dir", default=str(cvc_root / "processed_1024"), help="Output directory")
    ap.add_argument("--target", type=int, default=1024)
    ap.add_argument("--min_area", type=int, default=20)
    args = ap.parse_args()

    out_img = os.path.join(args.out_dir, "images_1024")
    out_msk = os.path.join(args.out_dir, "masks_1024")
    ensure_dir(out_img)
    ensure_dir(out_msk)

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    img_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(exts)]
    img_files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

    rows = []
    for f in img_files:
        img_path = os.path.join(args.img_dir, f)
        stem, _ = os.path.splitext(f)

        mask_path = find_matching_mask(args.mask_dir, stem)
        if mask_path is None:
            print(f"[SKIP] mask not found for {f}")
            continue

        img = read_image(img_path).astype(np.float32) / 255.0
        mask = read_mask(mask_path)

        bbox = compute_bbox(mask, min_area=args.min_area)
        if bbox is None:
            print(f"[SKIP] empty mask for {f}")
            continue

        img_1024, mask_1024, bbox_1024 = resize_with_bbox(img, mask, bbox, target=args.target)

        out_img_name = f"{stem}.png"
        out_mask_name = f"{stem}.png"

        io.imsave(os.path.join(out_img, out_img_name), img_as_ubyte(np.clip(img_1024, 0, 1)))
        io.imsave(os.path.join(out_msk, out_mask_name), (mask_1024.astype(np.uint8) * 255))

        rows.append({
            "id": stem,
            "img": out_img_name,
            "mask": out_mask_name,
            "bbox_1024": bbox_1024
        })

    meta_path = os.path.join(args.out_dir, "meta.csv")
    pd.DataFrame(rows).to_csv(meta_path, index=False)
    print(f"[OK] processed {len(rows)} samples")
    print(f"[OK] meta saved: {meta_path}")


if __name__ == "__main__":
    main()
