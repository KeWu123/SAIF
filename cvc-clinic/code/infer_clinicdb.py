# -*- coding: utf-8 -*-
"""
CVC-ClinicDB inference (Box jitter + SCC)
- 输入: processed_1024/meta.csv + images_1024 + masks_1024
- 输出: results.csv + 终端打印 mDice/mIoU
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


def jitter_boxes(box, img_shape, N=5, jitter=0.05):
    H, W = img_shape[:2]
    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))

    out = []
    for _ in range(N):
        dx1 = np.random.uniform(-jitter * w, jitter * w)
        dy1 = np.random.uniform(-jitter * h, jitter * h)
        dx2 = np.random.uniform(-jitter * w, jitter * w)
        dy2 = np.random.uniform(-jitter * h, jitter * h)

        nx1 = max(0, min(x1 + dx1, W - 1))
        ny1 = max(0, min(y1 + dy1, H - 1))
        nx2 = max(nx1 + 1, min(x2 + dx2, W - 1))
        ny2 = max(ny1 + 1, min(y2 + dy2, H - 1))
        out.append([nx1, ny1, nx2, ny2])
    return out


def iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def dice(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    s = m1.sum() + m2.sum()
    return 0.0 if s == 0 else float(2 * inter) / float(s)


def parse_bbox_1024(v):
    if isinstance(v, str):
        box = eval(v)
    else:
        box = list(v)
    if len(box) != 4:
        raise ValueError("Box must have 4 elements")
    return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]


def medsam_scope_scc(img, base_box, predictor, num_candidate_boxes=10, K=8):
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]

    predictor.set_image(img.astype(np.uint8))

    candidate_boxes = [base_box] + jitter_boxes(
        base_box, img.shape, N=max(0, num_candidate_boxes - 1), jitter=0.03
    )

    tau_list = [0.2, 0.4, 0.5, 0.6, 0.8]
    best_sc, best_mask, best_info = -1e9, None, {}

    for b in candidate_boxes:
        jittered = [b] + jitter_boxes(b, img.shape, N=max(0, K - 1), jitter=0.01)

        prob_maps = []
        for jb in jittered:
            b_np = np.array(jb, dtype=np.float32)[None, :]
            masks_pred, _, _ = predictor.predict(box=b_np, multimask_output=False)
            prob_maps.append(masks_pred[0].astype(np.float32))

        prob_maps = np.stack(prob_maps, axis=0)  # (K,H,W)

        for tau in tau_list:
            masks_tau = prob_maps >= float(tau)
            base_mask = masks_tau[0]

            ious = [iou(base_mask, masks_tau[k]) for k in range(1, masks_tau.shape[0])]
            mu = float(np.mean(ious)) if ious else 0.0
            sigma = float(np.std(ious)) if ious else 0.0

            mask_area = float(np.mean([float(np.mean(m)) for m in masks_tau]))
            area_penalty = 1.0 if (0.05 < mask_area < 0.95) else 0.1

            weighted_sc = (mu * 0.8 + (1.0 - sigma) * 0.2) * area_penalty

            if weighted_sc > best_sc:
                best_sc = weighted_sc
                final_prob = prob_maps.mean(axis=0)
                best_mask = final_prob >= float(tau)
                best_info = {"box": b, "tau": float(tau), "mu": mu, "sigma": sigma, "mask_area": mask_area}

    return best_mask, best_sc, best_info


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

    print("开始处理图像...")
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
            pred, sc, info = medsam_scope_scc(
                img, box, predictor,
                num_candidate_boxes=args.num_candidate_boxes,
                K=args.K
            )
            if pred is None:
                predictor.set_image(img.astype(np.uint8))
                b_np = np.array(box, dtype=np.float32)[None, :]
                masks_pred, _, _ = predictor.predict(box=b_np, multimask_output=False)
                pred = masks_pred[0] > 0.5
                sc = 0.0
                info = {"tau": 0.5}

            pred = pred.astype(bool)

            iou_val = iou(pred, gt)
            dice_val = dice(pred, gt)

            # 一行打印
            print(f"[{sid}] IoU={iou_val:.4f} Dice={dice_val:.4f} tau={info.get('tau', 0.5)} sc={float(sc):.4f}")

            results.append({
                "id": sid,
                "mIoU_single": iou_val,
                "mDice_single": dice_val,
                "stability": float(sc),
                "best_tau": float(info.get("tau", 0.5)),
            })

            inter = np.logical_and(pred, gt).sum()
            total_inter += inter
            total_pred += pred.sum()
            total_gt += gt.sum()

            if args.save_pred:
                io.imsave(os.path.join(args.out, f"{sid}_pred.png"),
                          img_as_ubyte(pred), check_contrast=False)

        except Exception as e:
            tqdm.write(f"❌ Failed id={sid}: {e}")
            continue

    if results:
        mIoU = float(np.mean([r["mIoU_single"] for r in results]))
        mDice = float(np.mean([r["mDice_single"] for r in results]))
    else:
        mIoU, mDice = 0.0, 0.0

    overall_dice = float(2 * total_inter / (total_pred + total_gt + 1e-6))
    overall_iou = float(total_inter / (total_pred + total_gt - total_inter + 1e-6))

    out_csv = os.path.join(args.out, "results.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print("推理完成总结 (CVC-ClinicDB)")
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
    ap.add_argument("--out", default=r"D:\MedSAM-main\cvc-clinic\processed_1024\result_scc")

    ap.add_argument("--num_candidate_boxes", type=int, default=20)
    ap.add_argument("--K", type=int, default=20)

    ap.add_argument("--save_pred", action="store_true", default=True,
                    help="是否保存预测mask（默认保存）")
    ap.add_argument("--no_save_pred", action="store_false", dest="save_pred",
                    help="不保存预测mask")

    args = ap.parse_args()
    main(args)
