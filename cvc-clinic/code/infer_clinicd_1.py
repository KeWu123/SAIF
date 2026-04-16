# -*- coding: utf-8 -*-
"""
CVC-ClinicDB inference (Best-of SCC upgraded to SC-only Top-k, NO image affine)

核心策略（更贴近你原始 SCC，高概率更高分）：
- 不做图像增强/warp（避免插值模糊）
- 候选框 candidates: base + multi-scale + jitter (unique)
- 对每个 candidate，再做 K 次 box-jitter -> 得到 K 个 prob
- prob 使用 low_res_masks logits 上采样 sigmoid（根治版）
- 每个 (candidate, tau) 的 SC 分数：mu - lambda_sc*sigma，并加 area penalty
- 全局 Top-k（默认3）按 score softmax(temp) 融合 mean_prob
- tau* 取 Top-1 的 tau（best-of tau）
- 输出 mIoU / mDice
"""

import os
import ast
import argparse
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io, img_as_ubyte

import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor


# -----------------------------
# Metrics
# -----------------------------
def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-8)

def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2.0 * inter) / float(denom + 1e-8)

def softmax(x: np.ndarray, temp: float):
    x = x.astype(np.float32) / max(1e-6, float(temp))
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-8)


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_bbox_1024(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        return [float(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ["none", "nan", ""]:
            return None
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)) and len(obj) == 4:
                return [float(x) for x in obj]
        except Exception:
            return None
    return None

def clamp_box(box, H, W):
    x0, y0, x1, y1 = box
    x0 = float(max(0, min(W - 1, x0)))
    x1 = float(max(0, min(W - 1, x1)))
    y0 = float(max(0, min(H - 1, y0)))
    y1 = float(max(0, min(H - 1, y1)))
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    if x1 == x0: x1 = min(W - 1, x0 + 1.0)
    if y1 == y0: y1 = min(H - 1, y0 + 1.0)
    return [x0, y0, x1, y1]


# -----------------------------
# Box sampling
# -----------------------------
def jitter_one_box(box, H, W, jitter: float = 0.01):
    x0, y0, x1, y1 = box
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)

    dx1 = np.random.uniform(-jitter * bw, jitter * bw)
    dy1 = np.random.uniform(-jitter * bh, jitter * bh)
    dx2 = np.random.uniform(-jitter * bw, jitter * bw)
    dy2 = np.random.uniform(-jitter * bh, jitter * bh)

    nx0 = x0 + dx1
    ny0 = y0 + dy1
    nx1 = x1 + dx2
    ny1 = y1 + dy2
    return clamp_box([nx0, ny0, nx1, ny1], H, W)

def build_candidate_boxes(base_box, H, W, num: int, jitter: float = 0.04, multi_scale: bool = True):
    x0, y0, x1, y1 = base_box
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    boxes = [clamp_box(base_box, H, W)]
    if multi_scale:
        for s in [0.90, 1.00, 1.10, 1.20]:
            nw = bw * s
            nh = bh * s
            boxes.append(clamp_box([cx - nw/2, cy - nh/2, cx + nw/2, cy + nh/2], H, W))

    for _ in range(max(0, num - len(boxes))):
        boxes.append(jitter_one_box(base_box, H, W, jitter=jitter))

    # unique
    uniq, seen = [], set()
    for b in boxes:
        t = (int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3])))
        if t not in seen:
            uniq.append([float(t[0]), float(t[1]), float(t[2]), float(t[3])])
            seen.add(t)
    return uniq[:num]


# -----------------------------
# MedSAM prob (low_res -> upsample -> sigmoid)
# -----------------------------
def predict_prob_one_after_set_image(predictor: SamPredictor, box: List[float], H: int, W: int) -> np.ndarray:
    b_np = np.array(box, dtype=np.float32)[None, :]
    masks, _, low_res = predictor.predict(box=b_np, multimask_output=False)

    if low_res is None:
        # fallback
        return masks[0].astype(np.float32)

    lr = torch.from_numpy(low_res).float()
    if lr.ndim == 3:
        lr = lr.unsqueeze(1)  # (1,1,h,w)
    elif lr.ndim == 4:
        if lr.shape[1] != 1:
            lr = lr[:, :1, :, :]
    else:
        return masks[0].astype(np.float32)

    up = F.interpolate(lr, size=(H, W), mode="bilinear", align_corners=False)
    prob = torch.sigmoid(up)[0, 0].cpu().numpy().astype(np.float32)
    return prob


# -----------------------------
# SC-only + Top-k fusion (NO BC, NO affine)
# -----------------------------
def scope_sc_only_boxjitter(
    predictor: SamPredictor,
    img_rgb: np.ndarray,
    base_box: List[float],
    num_candidates: int = 12,
    K: int = 8,                      # <=8
    tau_list: Optional[List[float]] = None,
    lambda_sc: float = 0.3,
    temp: float = 0.05,
    topk: int = 3,
    area_min: float = 0.05,
    area_max: float = 0.95,
    area_penalty_low: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:

    if tau_list is None:
        tau_list = [0.35, 0.45, 0.55, 0.65]

    if img_rgb.ndim == 2:
        img_rgb = np.repeat(img_rgb[:, :, None], 3, axis=-1)
    elif img_rgb.shape[-1] == 4:
        img_rgb = img_rgb[:, :, :3]
    img_rgb = img_rgb.astype(np.uint8)

    H, W = img_rgb.shape[:2]
    base_box = clamp_box(base_box, H, W)

    candidates = build_candidate_boxes(base_box, H, W, num=num_candidates, jitter=0.04, multi_scale=True)

    predictor.set_image(img_rgb)

    cand_scores, cand_probs, cand_taus, cand_meta = [], [], [], []

    for b in candidates:
        # K box-jitter around THIS candidate (no image augment)
        boxes_k = [b] + [jitter_one_box(b, H, W, jitter=0.01) for _ in range(max(0, K - 1))]
        probs_k = [predict_prob_one_after_set_image(predictor, bb, H, W) for bb in boxes_k]  # list of (H,W)
        probs_k = [p.astype(np.float32) for p in probs_k]

        ref_prob = probs_k[0]
        mean_prob = np.mean(np.stack(probs_k, axis=0), axis=0).astype(np.float32)

        for tau in tau_list:
            tau = float(tau)
            ref_mask = (ref_prob >= tau)

            ious_k = []
            for pk in probs_k:
                ious_k.append(iou(ref_mask, pk >= tau))

            mu = float(np.mean(ious_k)) if ious_k else 0.0
            sig = float(np.std(ious_k)) if ious_k else 0.0
            sc = mu - float(lambda_sc) * sig

            # area penalty（保留你原始 SCC 的核心）
            masks_tau = [(pk >= tau) for pk in probs_k]
            mask_area = float(np.mean([float(np.mean(m)) for m in masks_tau]))
            area_pen = 1.0 if (area_min < mask_area < area_max) else float(area_penalty_low)

            score = float(sc) * area_pen  # SC-only * area_penalty

            cand_scores.append(score)
            cand_probs.append(mean_prob)   # 注意：mean_prob 与 tau 无关，但 OK
            cand_taus.append(tau)
            cand_meta.append({"box": b, "tau": tau, "SC": float(sc), "mask_area": mask_area, "area_pen": area_pen})

    scores_np = np.array(cand_scores, dtype=np.float32)

    k_top = max(1, int(topk))
    k_top = min(k_top, len(scores_np))
    idx = np.argsort(scores_np)[::-1][:k_top]
    w = softmax(scores_np[idx], temp=temp)

    fused_prob = np.zeros((H, W), dtype=np.float32)
    for wi, j in zip(w, idx):
        fused_prob += float(wi) * cand_probs[j].astype(np.float32)

    # tau* 用 Top-1 tau
    best_j = int(idx[0])
    tau_star = float(cand_taus[best_j])
    final_mask = fused_prob >= tau_star

    debug = {"topk_idx": idx, "topk_w": w, "tau_star": tau_star}
    return final_mask, fused_prob, tau_star, debug


# -----------------------------
# Main
# -----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt)
    sam.to(device)
    predictor = SamPredictor(sam)

    ensure_dir(args.out)
    df = pd.read_csv(args.meta)

    tau_list = [float(x) for x in str(args.tau_list).split(",") if x.strip() != ""]

    miou_list, mdice_list = [], []
    total_inter = 0
    total_pred = 0
    total_gt = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
        sid = str(row.get("id", ""))

        img_path = os.path.join(args.img_dir, str(row.get("img", "")))
        mask_path = os.path.join(args.mask_dir, str(row.get("mask", "")))
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        img = io.imread(img_path)
        gt = io.imread(mask_path)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        gt = (gt > 127)

        box = parse_bbox_1024(row.get("bbox_1024", None))
        if box is None:
            continue

        pred, prob, tau_star, debug = scope_sc_only_boxjitter(
            predictor=predictor,
            img_rgb=img,
            base_box=box,
            num_candidates=args.num_candidate_boxes,
            K=min(8, int(args.K)),
            tau_list=tau_list,
            lambda_sc=args.lambda_sc,
            temp=args.temp,
            topk=args.topk,
            area_min=args.area_min,
            area_max=args.area_max,
            area_penalty_low=args.area_penalty_low,
        )

        pred = pred.astype(bool)
        iou_val = iou(pred, gt)
        dice_val = dice(pred, gt)
        miou_list.append(iou_val)
        mdice_list.append(dice_val)

        inter = np.logical_and(pred, gt).sum()
        total_inter += inter
        total_pred += pred.sum()
        total_gt += gt.sum()

        if args.save_pred:
            io.imsave(os.path.join(args.out, f"{sid}_pred.png"),
                      img_as_ubyte(pred), check_contrast=False)

    mIoU = float(np.mean(miou_list)) if miou_list else 0.0
    mDice = float(np.mean(mdice_list)) if mdice_list else 0.0
    overall_dice = float(2 * total_inter / (total_pred + total_gt + 1e-6))
    overall_iou = float(total_inter / (total_pred + total_gt - total_inter + 1e-6))

    print("\n" + "=" * 60)
    print("推理完成总结 (CVC-ClinicDB) - SC-only + Top-k + Box-jitter (NO affine)")
    print("=" * 60)
    print(f"mIoU:  {mIoU:.4f}")
    print(f"mDice: {mDice:.4f}")
    print(f"overall IoU:  {overall_iou:.4f}")
    print(f"overall Dice: {overall_dice:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", default=r"D:\MedSAM-main\work_dir\MedSAM\medsam_vit_b.pth")
    ap.add_argument("--model_type", default="vit_b")

    ap.add_argument("--meta", default=r"D:\MedSAM-main\cvc-clinic\processed_1024\meta.csv")
    ap.add_argument("--img_dir", default=r"D:\MedSAM-main\cvc-clinic\processed_1024\images_1024")
    ap.add_argument("--mask_dir", default=r"D:\MedSAM-main\cvc-clinic\processed_1024\masks_1024")
    ap.add_argument("--out", default=r"D:\MedSAM-main\cvc-clinic\processed_1024\result_sc_only_boxjitter")

    ap.add_argument("--num_candidate_boxes", type=int, default=20)
    ap.add_argument("--K", type=int, default=8)  # cap 8

    ap.add_argument("--tau_list", type=str, default="0.2,0.35,0.45,0.5,0.55,0.65,0.8")
    ap.add_argument("--lambda_sc", type=float, default=0.3)

    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.05)

    # area penalty（和你原始 SCC 一样的思想，参数可调）
    ap.add_argument("--area_min", type=float, default=0.05)
    ap.add_argument("--area_max", type=float, default=0.95)
    ap.add_argument("--area_penalty_low", type=float, default=0.1)

    ap.add_argument("--save_pred", action="store_true", default=True)
    ap.add_argument("--no_save_pred", action="store_false", dest="save_pred")

    args = ap.parse_args()
    main(args)
