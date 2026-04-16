#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedSAM batch inference for Kvasir-SEG (baseline).
Reports per-image IoU/Dice/Accuracy and overall micro metrics.
"""

import os, sys, numpy as np, pandas as pd, torch
from skimage import io, img_as_ubyte
from segment_anything import sam_model_registry, SamPredictor

# ---------- Utilities ----------

def iou(m1, m2):
    """Intersection over Union (IoU)."""
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / (union + 1e-6)


def dice(m1, m2, eps=1e-6):
    """Dice score."""
    inter = (m1 & m2).sum()
    return 2 * inter / (m1.sum() + m2.sum() + eps)


def accuracy(m1, m2):
    """Pixel accuracy."""
    tp = np.logical_and(m1 == 1, m2 == 1).sum()
    tn = np.logical_and(m1 == 0, m2 == 0).sum()
    total_pixels = m1.size
    return (tp + tn) / total_pixels


def medsam_scope(img, box, predictor):
    """Single-box MedSAM inference (choose the best mask by score)."""
    predictor.set_image(img)
    masks, scores, _ = predictor.predict(box=np.array(box), multimask_output=True)
    best = masks[np.argmax(scores)]
    return best, scores.max()


# ---------- Entry ----------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt)
    sam.to(device)
    predictor = SamPredictor(sam)
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.meta)
    results = []

    # Accumulators for overall metrics
    total_inter = total_pred = total_gt = total_tp = total_tn = 0
    total_processed_images = 0

    for _, row in df.iterrows():
        # Skip normal images if the filename contains "normal"
        if "normal" in row["img"].lower():
            print(f"Skipping normal image: {row['img']}")
            continue

        img_path = os.path.join(args.img_dir, row["img"])
        mask_path = os.path.join(args.mask_dir, row["mask"])
        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)

        # Parse bbox_1024 safely
        try:
            bbox_str = row["bbox_1024"]
            if isinstance(bbox_str, str):
                box = eval(bbox_str)
            else:
                raise ValueError(f"Invalid bbox_1024 format for image {row['img']}. Expected a string.")
        except Exception as e:
            print(f"Warning: Failed to parse bbox_1024 for {row['img']}. Error: {str(e)}")
            print(f"  bbox_1024 content: {row['bbox_1024']}")
            continue

        mask_gt = io.imread(mask_path) > 127

        try:
            mask_pred, score = medsam_scope(img, box, predictor)

            iou_val = iou(mask_pred, mask_gt)
            dice_val = dice(mask_pred, mask_gt)
            acc_val = accuracy(mask_pred, mask_gt)
            results.append({"id": row["id"], "IoU": iou_val, "Dice": dice_val, "Accuracy": acc_val, "stability": score})

            # Update global counters
            total_inter += np.logical_and(mask_pred, mask_gt).sum()
            total_pred += mask_pred.sum()
            total_gt += mask_gt.sum()

            # Track TP/TN for overall accuracy
            total_tp += np.logical_and(mask_pred == 1, mask_gt == 1).sum()
            total_tn += np.logical_and(mask_pred == 0, mask_gt == 0).sum()

            total_processed_images += 1

            out_mask = os.path.join(args.out, f"{row['id']}_pred.png")
            io.imsave(out_mask, img_as_ubyte(mask_pred), check_contrast=False)
            print(f"{row['id']}  IoU={iou_val:.3f}  Dice={dice_val:.3f}  Accuracy={acc_val:.3f}")

        except Exception as e:
            print(f"Warning: Failed to process image {row['id']}. Error: {str(e)}")
            continue

    # Overall metrics (micro)
    overall_dice = 2 * total_inter / (total_pred + total_gt + 1e-6)
    overall_iou  = total_inter / (total_pred + total_gt - total_inter + 1e-6)
    overall_accuracy = (total_tp + total_tn) / (total_pred.size)

    pd.DataFrame(results).to_csv(os.path.join(args.out, "results.csv"), index=False)
    print("\n[OK] Batch inference completed.")
    print("Mean IoU :", np.mean([r["IoU"] for r in results]))
    print("Mean Dice:", np.mean([r["Dice"] for r in results]))
    print("Mean Accuracy:", np.mean([r["Accuracy"] for r in results]))
    print("Overall IoU :", overall_iou)
    print("Overall Dice:", overall_dice)


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D:/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth", help="Checkpoint")
    ap.add_argument("--model_type", default="vit_b")
    ap.add_argument("--meta", default="D:/MedSAM-main/KVasir/meta1.csv", help="meta.csv")
    ap.add_argument("--img_dir", default="D:/MedSAM-main/KVasir/images_1024", help="Image directory")
    ap.add_argument("--mask_dir", default="D:/MedSAM-main/KVasir/masks_1024", help="GT directory")
    ap.add_argument("--out", default="D:/MedSAM-main/KVasir/result_basline", help="Output directory")
    args = ap.parse_args()
    main(args)
