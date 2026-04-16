# -*- coding: utf-8 -*-
"""
Baseline inference (NO box jitter, NO SCC)
- 输入: processed_1024/meta.csv + images_1024 + masks_1024
- 推理: 每张图仅用 bbox_1024 跑一次 SAM predictor.predict(multimask_output=False)
- 指标: mIoU / mDice（逐图平均）+ micro 参考
- 可选: --save_pred 保存 *_pred.png
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io, img_as_ubyte
import torch

from segment_anything import sam_model_registry, SamPredictor


def iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def dice(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    s = m1.sum() + m2.sum()
    return 0.0 if s == 0 else float(2 * inter) / float(s)


def parse_bbox_1024(v):
    # meta.csv 里可能是 "[x1, y1, x2, y2]" 字符串
    if isinstance(v, str):
        box = eval(v)
    else:
        box = list(v)
    if len(box) != 4:
        raise ValueError("Box must have 4 elements")
    return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt)
    sam.to(device)
    predictor = SamPredictor(sam)

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.meta)
    results = []

    total_inter = 0
    total_pred = 0
    total_gt = 0

    print("开始处理图像 (Baseline)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
        sid = str(row.get("id", ""))

        img_path = os.path.join(args.img_dir, str(row["img"]))
        mask_path = os.path.join(args.mask_dir, str(row["mask"]))

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            tqdm.write(f"[SKIP] missing file id={sid}")
            continue

        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[:, :, :3]

        gt = io.imread(mask_path)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        gt = gt > 127

        try:
            box = parse_bbox_1024(row["bbox_1024"])
        except Exception as e:
            tqdm.write(f"[SKIP] bad bbox id={sid}: {e}")
            continue

        try:
            predictor.set_image(img.astype(np.uint8))
            b_np = np.array(box, dtype=np.float32)[None, :]
            masks_pred, _, _ = predictor.predict(
                box=b_np,
                multimask_output=False
            )
            pred = masks_pred[0].astype(bool)

            iou_val = iou(pred, gt)
            dice_val = dice(pred, gt)

            if args.print_each:
                print(f"[{sid}] IoU={iou_val:.4f} Dice={dice_val:.4f}")

            results.append({
                "id": sid,
                "mIoU_single": iou_val,
                "mDice_single": dice_val,
            })

            inter = np.logical_and(pred, gt).sum()
            total_inter += inter
            total_pred += pred.sum()
            total_gt += gt.sum()

            if args.save_pred:
                io.imsave(
                    os.path.join(args.out, f"{sid}_pred.png"),
                    img_as_ubyte(pred),
                    check_contrast=False
                )

        except Exception as e:
            tqdm.write(f"❌ Failed id={sid}: {e}")
            continue

    # 逐图平均：mIoU / mDice（你要的主指标）
    if results:
        mIoU = float(np.mean([r["mIoU_single"] for r in results]))
        mDice = float(np.mean([r["mDice_single"] for r in results]))
    else:
        mIoU, mDice = 0.0, 0.0

    # micro 参考
    overall_dice = float(2 * total_inter / (total_pred + total_gt + 1e-6))
    overall_iou = float(total_inter / (total_pred + total_gt - total_inter + 1e-6))

    out_csv = os.path.join(args.out, "results_baseline.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print("推理完成总结 (Baseline)")
    print("=" * 60)
    print(f"成功处理图像: {len(results)} / {len(df)}")
    print("\n【目标指标】")
    print(f"mIoU:  {mIoU:.4f}")
    print(f"mDice: {mDice:.4f}")
    print("\n【参考(整体micro)】")
    print(f"overall IoU:  {overall_iou:.4f}")
    print(f"overall Dice: {overall_dice:.4f}")
    print(f"\n结果已保存: {out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", default=r"D:\MedSAM-main\work_dir\MedSAM\medsam_vit_b.pth")
    ap.add_argument("--model_type", default="vit_b")

    ap.add_argument("--meta", default=r"D:\MedSAM-main\cvc-clinic\processed_1024\meta.csv")
    ap.add_argument("--img_dir", default=r"D:\MedSAM-main\cvc-clinic\processed_1024\images_1024")
    ap.add_argument("--mask_dir", default=r"D:\MedSAM-main\cvc-clinic\processed_1024\masks_1024")
    ap.add_argument("--out", default=r"D:\MedSAM-main\cvc-clinic\processed_1024\result_baseline")

    ap.add_argument("--save_pred", action="store_true", default=True,
                    help="是否保存预测mask（默认保存）")
    ap.add_argument("--no_save_pred", action="store_false", dest="save_pred",
                    help="不保存预测mask")

    ap.add_argument("--print_each", action="store_true", default=True,
                    help="是否逐张打印IoU/Dice（默认打印）")
    ap.add_argument("--no_print_each", action="store_false", dest="print_each",
                    help="不逐张打印")

    args = ap.parse_args()
    main(args)
