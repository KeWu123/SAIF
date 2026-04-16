# -*- coding: utf-8 -*-
import os
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io, img_as_ubyte
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from segment_anything import sam_model_registry, SamPredictor


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 0.0 if union == 0 else float(inter / union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    s = pred.sum() + gt.sum()
    return 0.0 if s == 0 else float(2 * inter / s)


def parse_bbox(v):
    """
    meta.csv 里 bbox_1024 一般是字符串形式: "[x1, y1, x2, y2]"
    """
    if isinstance(v, str):
        box = eval(v)
    else:
        box = list(v)

    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"bbox_1024 format error: {v}")

    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2), float(y2)]


def read_mask_bin(path: str) -> np.ndarray:
    m = io.imread(path)
    if m.ndim == 3:
        m = m[:, :, 0]
    # 0/255 或 0/1 都兼容
    return (m > 127) if m.max() > 1 else (m > 0)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load SAM/MedSAM
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt)
    sam.to(device)
    predictor = SamPredictor(sam)

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.meta)
    results = []

    print("开始 baseline 推理（无框抖动 / 无SCC）...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="进度"):
        sid = row["id"] if "id" in row else os.path.splitext(str(row["img"]))[0]

        img_path = os.path.join(args.img_dir, str(row["img"]))
        mask_path = os.path.join(args.mask_dir, str(row["mask"]))

        if not os.path.exists(img_path):
            tqdm.write(f"❌ Image not found: {img_path}")
            continue
        if not os.path.exists(mask_path):
            tqdm.write(f"❌ Mask not found: {mask_path}")
            continue

        # read image
        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        if img.shape[-1] == 4:
            img = img[:, :, :3]

        # read gt mask
        gt = read_mask_bin(mask_path)

        # bbox from meta
        try:
            box = parse_bbox(row["bbox_1024"])
        except Exception as e:
            tqdm.write(f"⚠️ Invalid bbox for {sid}: {e}")
            continue

        # predict once
        try:
            predictor.set_image(img.astype(np.uint8))
            b_np = np.array(box, dtype=np.float32)[None, :]  # shape (1,4)

            masks_pred, _, _ = predictor.predict(
                box=b_np,
                multimask_output=False
            )
            pred = masks_pred[0] > 0.5

            iou_val = iou(pred, gt)
            dice_val = dice(pred, gt)

            results.append({
                "id": sid,
                "mIoU_single": iou_val,
                "mDice_single": dice_val,
            })

            if args.save_pred:
                out_mask = os.path.join(args.out, f"{sid}_pred.png")
                io.imsave(out_mask, img_as_ubyte(pred), check_contrast=False)

        except Exception as e:
            tqdm.write(f"❌ Failed on {sid}: {e}")
            continue

    # metrics: mean over images
    if results:
        mIoU = float(np.mean([r["mIoU_single"] for r in results]))
        mDice = float(np.mean([r["mDice_single"] for r in results]))
    else:
        mIoU, mDice = 0.0, 0.0

    out_csv = os.path.join(args.out, "results_baseline.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print("Baseline 结果（无框抖动 / 无SCC）")
    print("=" * 60)
    print(f"成功处理: {len(results)} / {len(df)}")
    print(f"mIoU:  {mIoU:.4f}")
    print(f"mDice: {mDice:.4f}")
    print(f"结果保存: {out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    repo_root = REPO_ROOT
    cvc_root = repo_root / "CVC-300"

    # 默认直接可跑（不传参数也行）
    ap.add_argument("--ckpt", default=str(repo_root / "work_dir" / "MedSAM" / "medsam_vit_b.pth"))
    ap.add_argument("--model_type", default="vit_b")

    ap.add_argument("--meta", default=str(cvc_root / "processed_1024" / "meta.csv"))
    ap.add_argument("--img_dir", default=str(cvc_root / "processed_1024" / "images_1024"))
    ap.add_argument("--mask_dir", default=str(cvc_root / "processed_1024" / "masks_1024"))
    ap.add_argument("--out", default=str(cvc_root / "processed_1024" / "result_baseline"))

    ap.add_argument("--save_pred", action="store_true", help="保存每张预测mask（默认不保存）")

    args = ap.parse_args()

    # 跑前检查，保证“直接可跑”
    missing = []
    if not os.path.exists(args.ckpt):
        missing.append(f"Checkpoint not found: {args.ckpt}")
    if not os.path.exists(args.meta):
        missing.append(f"Meta not found: {args.meta}")
    if not os.path.isdir(args.img_dir):
        missing.append(f"Image dir not found: {args.img_dir}")
    if not os.path.isdir(args.mask_dir):
        missing.append(f"Mask dir not found: {args.mask_dir}")

    if missing:
        print("\n".join(["❌ 运行失败：路径不存在/不正确"] + missing))
        raise SystemExit(1)

    os.makedirs(args.out, exist_ok=True)
    main(args)
