# -*- coding: utf-8 -*-
"""
Synapse Processor (multi-organ 1..8)
- 输入:
    img_dir:   .../Training/img   (detXXXX_avg.nii.gz)
    label_dir: .../Training/label (detXXXX_avg_seg.nii.gz)
- 输出 (out_dir/processed_1024):
    images_1024/   : 每 slice 一张 1024 PNG（3通道 uint8）
    masks_1024/    : 每 organ 二值 mask PNG（0/255, uint8）
    meta.csv       : 每行一个 (slice, organ) 样本，含 organ_id + bbox_1024
"""

import os
import re
import argparse
from glob import glob

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from skimage.transform import resize
from skimage import io


A_MIN = -125.0
A_MAX = 275.0

# 你说 label 里面类别是 1..8（不需要 hashmap）
VALID_ORGANS = list(range(1, 9))


def normalize_ct(img2d: np.ndarray, a_min=A_MIN, a_max=A_MAX) -> np.ndarray:
    """CT clip + normalize -> [0,1] float32"""
    img2d = img2d.astype(np.float32)
    img2d = np.clip(img2d, a_min, a_max)
    img2d = (img2d - a_min) / (a_max - a_min + 1e-8)
    img2d = np.clip(img2d, 0.0, 1.0)
    return img2d


def to_uint8_rgb(img01: np.ndarray) -> np.ndarray:
    """[0,1] -> uint8 RGB"""
    u8 = (img01 * 255.0 + 0.5).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)  # (H,W,3)


def resize_img01(img01: np.ndarray, target: int) -> np.ndarray:
    """resize float [0,1] to (target,target)"""
    return resize(
        img01,
        (target, target),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True
    ).astype(np.float32)


def resize_mask(mask: np.ndarray, target: int) -> np.ndarray:
    """resize binary mask to (target,target) with nearest"""
    return resize(
        mask.astype(np.uint8),
        (target, target),
        order=0,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True
    ).astype(np.uint8)


def mask_to_bbox_xyxy(mask01: np.ndarray):
    """mask01: uint8 {0,1} -> bbox [x1,y1,x2,y2] in image coord"""
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    # 注意: x2,y2 这里给到 max 像素位置(包含)，与你之前一致
    return [x1, y1, x2, y2]


def get_vol_id_from_imgname(img_path: str) -> str:
    # det0000101_avg.nii.gz -> det0000101_avg
    base = os.path.basename(img_path)
    if base.endswith(".nii.gz"):
        base = base[:-7]
    return base


def find_label_path(label_dir: str, vol_id: str) -> str:
    # det0000101_avg -> det0000101_avg_seg.nii.gz
    cand = os.path.join(label_dir, f"{vol_id}_seg.nii.gz")
    return cand


def main(args):
    out_root = os.path.join(args.out_dir, f"processed_{args.target}")
    out_img = os.path.join(out_root, f"images_{args.target}")
    out_msk = os.path.join(out_root, f"masks_{args.target}")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)

    meta_rows = []
    saved_rows = 0
    skipped_no_label = 0
    skipped_empty = 0

    img_files = sorted(glob(os.path.join(args.img_dir, "*.nii.gz")))
    print("=" * 60)
    print("Synapse processor start")
    print("=" * 60)
    print(f"found volumes: {len(img_files)}")

    for img_path in tqdm(img_files, desc="volumes"):
        vol_id = get_vol_id_from_imgname(img_path)
        label_path = find_label_path(args.label_dir, vol_id)
        if not os.path.exists(label_path):
            skipped_no_label += 1
            continue

        # 读 nii.gz
        img_nii = nib.load(img_path)
        lab_nii = nib.load(label_path)

        img = img_nii.get_fdata()  # float
        lab = lab_nii.get_fdata()  # float

        # 关键：label 转 int，避免 float 比较/映射出锅
        lab = np.rint(lab).astype(np.int16)

        # 常见 Synapse shape: (H,W,D)
        if img.ndim != 3 or lab.ndim != 3:
            continue
        H, W, D = img.shape
        if lab.shape != img.shape:
            continue

        for z in range(D):
            img2d = img[:, :, z]
            lab2d = lab[:, :, z]

            # 只保存一次 image（每个 slice 一张）
            img01 = normalize_ct(img2d)
            img01_1024 = resize_img01(img01, args.target)
            img_rgb = to_uint8_rgb(img01_1024)

            img_name = f"{vol_id}_z{z:04d}.png"
            io.imsave(os.path.join(out_img, img_name), img_rgb, check_contrast=False)

            # 对每个器官生成一个二值 mask
            any_saved_this_slice = False

            for organ_id in VALID_ORGANS:
                organ_mask = (lab2d == organ_id).astype(np.uint8)  # 0/1
                if organ_mask.sum() == 0:
                    if args.keep_empty:
                        # 保留空行：bbox 空，mask 全0
                        pass
                    else:
                        continue

                any_saved_this_slice = True

                m01_1024 = resize_mask(organ_mask, args.target)  # 0/1
                bbox = mask_to_bbox_xyxy(m01_1024)

                is_empty = 1 if (m01_1024.sum() == 0) else 0

                mask_name = f"{vol_id}_z{z:04d}_organ{organ_id}.png"
                # 保存 0/255 uint8
                io.imsave(
                    os.path.join(out_msk, mask_name),
                    (m01_1024 * 255).astype(np.uint8),
                    check_contrast=False
                )

                meta_rows.append({
                    "id": f"{vol_id}_z{z:04d}_organ{organ_id}",
                    "vol_id": vol_id,
                    "slice": z,
                    "organ_id": int(organ_id),  # ✅ 不会再全是1
                    "img": img_name,
                    "mask": mask_name,
                    "orig_h": int(H),
                    "orig_w": int(W),
                    "orig_d": int(D),
                    "bbox_1024": "" if bbox is None else str(bbox),
                    "is_empty": int(is_empty),
                })
                saved_rows += 1

            if (not any_saved_this_slice) and (not args.keep_empty):
                skipped_empty += 1

    # 保存 meta
    meta_csv = os.path.join(out_root, "meta.csv")
    df = pd.DataFrame(meta_rows)
    df.to_csv(meta_csv, index=False)

    print("=" * 60)
    print("Synapse processor done")
    print("=" * 60)
    print(f"[OK] saved rows: {saved_rows}")
    print(f"[INFO] skipped (no label file): {skipped_no_label}")
    print(f"[INFO] skipped (empty slices): {skipped_empty}")
    print(f"[OK] meta saved: {meta_csv}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()

    # ✅ 直接可跑：给默认值，不再 required
    ap.add_argument("--img_dir",
                    default=r"D:\MedSAM-main\Synapes\RawData\Training\img",
                    help=r"Synapse img dir (nii.gz)")

    ap.add_argument("--label_dir",
                    default=r"D:\MedSAM-main\Synapes\RawData\Training\label",
                    help=r"Synapse label dir (nii.gz)")

    ap.add_argument("--out_dir",
                    default=r"D:\MedSAM-main\Synapes",
                    help=r"output root dir")

    ap.add_argument("--target", type=int, default=1024, help="输出尺寸 (默认1024)")
    ap.add_argument("--keep_empty", action="store_true", help="是否保留空mask行（默认不保留）")

    args = ap.parse_args()
    main(args)
