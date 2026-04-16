#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kvasir-SEG inference (our plugin - full version).
SC-only + same-tau Top-k fusion + low_res prob + AIS + auto tau range.
"""

import os
import ast
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io, img_as_ubyte
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects, remove_small_holes
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from segment_anything import sam_model_registry, SamPredictor


# ================= Basics =================

def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure RGB image in uint8."""
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]
    return img.astype(np.uint8)


def parse_bbox_1024(v):
    """Parse bbox_1024 from list/tuple or string."""
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
    """Clamp bbox to image boundaries and fix invalid ranges."""
    x1, y1, x2, y2 = box
    x1 = float(max(0, min(W - 1, x1)))
    x2 = float(max(0, min(W - 1, x2)))
    y1 = float(max(0, min(H - 1, y1)))
    y2 = float(max(0, min(H - 1, y2)))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    if x2 == x1: x2 = min(W - 1, x1 + 1.0)
    if y2 == y1: y2 = min(H - 1, y1 + 1.0)
    return [x1, y1, x2, y2]


def iou(mask1, mask2):
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    inter = np.logical_and(m1, m2).sum()
    uni = np.logical_or(m1, m2).sum()
    return 0.0 if uni == 0 else float(inter) / float(uni)


def dice(mask1, mask2):
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    inter = np.logical_and(m1, m2).sum()
    s = m1.sum() + m2.sum()
    return 0.0 if s == 0 else float(2 * inter / s)


def accuracy(m1, m2):
    tp = np.logical_and(m1 == 1, m2 == 1).sum()
    tn = np.logical_and(m1 == 0, m2 == 0).sum()
    total_pixels = m1.size
    return (tp + tn) / total_pixels


def softmax(x: np.ndarray, temp: float):
    x = x.astype(np.float32) / max(1e-6, float(temp))
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-8)


# ================= AIS (adaptive budget) =================

def estimate_complexity(img_rgb: np.ndarray) -> float:
    """Edge-density based complexity estimate."""
    gray = rgb2gray(img_rgb)
    edges = sobel(gray)
    thr = float(edges.mean() + edges.std())
    edge_density = float((edges > thr).mean())
    return float(np.clip(edge_density * 6.0, 0.0, 1.0))


def estimate_box_scale(base_box, H, W) -> float:
    """Approximate object scale from bbox area ratio."""
    x1, y1, x2, y2 = base_box
    area = max(1.0, float((x2 - x1) * (y2 - y1)))
    return float(np.clip(area / float(H * W), 0.0, 1.0))


def ais_pick_budget(img_rgb, base_box, H, W,
                    K_max=8, cand_min=12, cand_mid=16, cand_max=20):
    c = estimate_complexity(img_rgb)
    s = estimate_box_scale(base_box, H, W)

    if c < 0.25 and s > 0.06:
        K = 4
    elif c < 0.55 and s > 0.03:
        K = 6
    else:
        K = 8
    K = int(min(K_max, K))

    if c < 0.25 and s > 0.06:
        n = cand_min
    elif c < 0.55 and s > 0.03:
        n = cand_mid
    else:
        n = cand_max
    return int(n), int(K), float(c), float(s)


# ================= Box sampling =================

def jitter_one_box(box, H, W, jitter=0.01):
    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))

    dx1 = np.random.uniform(-jitter * w, jitter * w)
    dy1 = np.random.uniform(-jitter * h, jitter * h)
    dx2 = np.random.uniform(-jitter * w, jitter * w)
    dy2 = np.random.uniform(-jitter * h, jitter * h)

    return clamp_box([x1 + dx1, y1 + dy1, x2 + dx2, y2 + dy2], H, W)


def build_candidate_boxes(base_box, img_shape, num_candidate_boxes=16, jitter=0.04, multi_scale=True):
    H, W = img_shape[:2]
    base_box = clamp_box(base_box, H, W)

    x1, y1, x2, y2 = base_box
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    boxes = [base_box]

    if multi_scale:
        for s in [0.90, 1.00, 1.10, 1.20]:
            nw = bw * s
            nh = bh * s
            b = [cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0]
            boxes.append(clamp_box(b, H, W))

    for _ in range(max(0, num_candidate_boxes - len(boxes))):
        boxes.append(jitter_one_box(base_box, H, W, jitter=jitter))

    uniq, seen = [], set()
    for b in boxes:
        t = (int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3])))
        if t not in seen:
            uniq.append([float(t[0]), float(t[1]), float(t[2]), float(t[3])])
            seen.add(t)
    return uniq[:num_candidate_boxes]


# ================= prob: low_res logits -> upsample -> sigmoid =================

def predict_prob_one_after_set_image(predictor: SamPredictor, box, H, W) -> np.ndarray:
    b_np = np.array(box, dtype=np.float32)[None, :]
    masks, _, low_res = predictor.predict(box=b_np, multimask_output=False)

    if low_res is None:
        return masks[0].astype(np.float32)

    lr = torch.from_numpy(low_res).float()
    if lr.ndim == 3:
        lr = lr.unsqueeze(1)
    elif lr.ndim == 4:
        if lr.shape[1] != 1:
            lr = lr[:, :1, :, :]
    else:
        return masks[0].astype(np.float32)

    up = F.interpolate(lr, size=(H, W), mode="bilinear", align_corners=False)
    prob = torch.sigmoid(up)[0, 0].cpu().numpy().astype(np.float32)
    return prob


# ================= Postprocess (adaptive) =================

def postprocess_mask_adaptive(mask: np.ndarray, obj_scale: float,
                              min_obj_big=200, min_hole_big=200,
                              min_obj_small=60, min_hole_small=60) -> np.ndarray:
    if obj_scale < 0.03:
        min_obj, min_hole = min_obj_small, min_hole_small
    else:
        min_obj, min_hole = min_obj_big, min_hole_big

    m = mask.astype(bool)
    m = remove_small_objects(m, min_size=int(min_obj))
    m = remove_small_holes(m, area_threshold=int(min_hole))
    return m.astype(bool)


# ================= Core: SC-only + same tau Top-k + auto tau =================

def score_candidate_for_tau(probs_k, tau: float, lambda_sc: float,
                           area_min: float, area_max: float, area_penalty_low: float):
    ref_prob = probs_k[0]
    ref_mask = (ref_prob >= tau)

    ious_k = [iou(ref_mask, (pk >= tau)) for pk in probs_k]
    mu = float(np.mean(ious_k)) if ious_k else 0.0
    sigma = float(np.std(ious_k)) if ious_k else 0.0
    sc = mu - float(lambda_sc) * sigma

    masks_tau = [(pk >= tau) for pk in probs_k]
    mask_area = float(np.mean([float(np.mean(m)) for m in masks_tau]))
    area_penalty = 1.0 if (area_min < mask_area < area_max) else float(area_penalty_low)

    score = float(sc) * area_penalty
    meta = {"SC": sc, "mu": mu, "sigma": sigma, "mask_area": mask_area, "area_penalty": area_penalty}
    return score, meta


def auto_tau_range_from_prob(prob: np.ndarray, base_lo=0.25, base_hi=0.80):
    q10 = float(np.quantile(prob, 0.10))
    q90 = float(np.quantile(prob, 0.90))
    lo = max(base_lo, min(0.60, q10 + 0.10))
    hi = min(base_hi, max(0.40, q90 - 0.10))
    if hi - lo < 0.15:
        mid = 0.5 * (lo + hi)
        lo = max(base_lo, mid - 0.10)
        hi = min(base_hi, mid + 0.10)
    return float(lo), float(hi)


def medsam_scope_sc_only_topk_v3(
    img,
    base_box,
    predictor,
    use_ais=True,
    num_candidate_boxes=16,
    K=8,
    tau_list=None,
    lambda_sc=0.3,
    topk=3,
    temp=0.05,
    area_min=0.05,
    area_max=0.95,
    area_penalty_low=0.1,
    refine_step=0.02,
    refine_radius=0.06,
):
    img = ensure_rgb(img)
    H, W = img.shape[:2]
    base_box = clamp_box(base_box, H, W)

    if use_ais:
        n_cand, K_use, cplx, obj_scale = ais_pick_budget(img, base_box, H, W)
    else:
        n_cand, K_use, cplx, obj_scale = int(num_candidate_boxes), int(min(8, K)), 0.0, estimate_box_scale(base_box, H, W)

    predictor.set_image(img.astype(np.uint8))
    candidates = build_candidate_boxes(base_box, img.shape, num_candidate_boxes=n_cand, jitter=0.04, multi_scale=True)
    K_use = int(min(8, max(1, K_use)))

    base_prob = predict_prob_one_after_set_image(predictor, base_box, H, W)
    lo, hi = auto_tau_range_from_prob(base_prob, base_lo=0.25, base_hi=0.80)

    if tau_list is None:
        tau_list = np.linspace(lo, hi, 9).tolist()

    cand_pack = []
    for b in candidates:
        boxes_k = [b] + [jitter_one_box(b, H, W, jitter=0.01) for _ in range(K_use - 1)]
        probs_k = [predict_prob_one_after_set_image(predictor, bb, H, W) for bb in boxes_k]
        mean_prob = np.mean(np.stack(probs_k, axis=0), axis=0).astype(np.float32)
        cand_pack.append((b, probs_k, mean_prob))

    if len(cand_pack) == 0:
        return None, None, 0.5, {"reason": "no candidates"}, obj_scale

    best_tau = None
    best_tau_score = -1e9
    best_tau_top1 = None

    for tau in tau_list:
        tau = float(tau)
        best_s = -1e9
        best_m = None
        for (b, probs_k, _) in cand_pack:
            s, meta = score_candidate_for_tau(probs_k, tau, lambda_sc, area_min, area_max, area_penalty_low)
            if s > best_s:
                best_s = s
                best_m = {"box": b, "tau": tau, **meta}
        if best_s > best_tau_score:
            best_tau_score = best_s
            best_tau = tau
            best_tau_top1 = best_m

    if best_tau is None:
        return None, None, 0.5, {"reason": "no tau"}, obj_scale

    lo2 = max(0.05, best_tau - refine_radius)
    hi2 = min(0.95, best_tau + refine_radius)
    fine_taus = []
    t = lo2
    while t <= hi2 + 1e-9:
        fine_taus.append(float(np.round(t, 4)))
        t += float(refine_step)

    best_tau_f = best_tau
    best_tau_score_f = best_tau_score
    best_tau_top1_f = best_tau_top1

    for tau in fine_taus:
        best_s = -1e9
        best_m = None
        for (b, probs_k, _) in cand_pack:
            s, meta = score_candidate_for_tau(probs_k, tau, lambda_sc, area_min, area_max, area_penalty_low)
            if s > best_s:
                best_s = s
                best_m = {"box": b, "tau": tau, **meta}
        if best_s > best_tau_score_f:
            best_tau_score_f = best_s
            best_tau_f = tau
            best_tau_top1_f = best_m

    tau_star = float(best_tau_f)

    scores = []
    probs = []
    sigmas = []
    for (b, probs_k, mean_prob) in cand_pack:
        s, meta = score_candidate_for_tau(probs_k, tau_star, lambda_sc, area_min, area_max, area_penalty_low)
        scores.append(float(s))
        probs.append(mean_prob)
        sigmas.append(float(meta["sigma"]))

    scores_np = np.array(scores, dtype=np.float32)
    sig_np = np.array(sigmas, dtype=np.float32)
    fused_logits = scores_np / (sig_np + 1e-3)

    k_top = int(max(1, topk))
    k_top = min(k_top, len(fused_logits))
    idx = np.argsort(fused_logits)[::-1][:k_top]
    w = softmax(fused_logits[idx], temp=temp)

    fused_prob = np.zeros((H, W), dtype=np.float32)
    for wi, j in zip(w, idx):
        fused_prob += float(wi) * probs[int(j)].astype(np.float32)

    final_mask = fused_prob >= tau_star

    debug = {
        "ais": {"use_ais": bool(use_ais), "num_candidates": int(n_cand), "K": int(K_use), "complexity": float(cplx), "obj_scale": float(obj_scale)},
        "tau_range": {"auto_lo": float(lo), "auto_hi": float(hi)},
        "tau_star": float(tau_star),
        "coarse_top1": best_tau_top1,
        "fine_top1": best_tau_top1_f,
        "topk_idx": idx.tolist(),
        "topk_weights": w.tolist(),
    }
    return final_mask, fused_prob, tau_star, debug, obj_scale


# ================= Entry =================

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
    total_tp = 0
    total_tn = 0
    total_pixels = 0

    tau_list = None
    if args.tau_list.strip():
        tau_list = [float(x) for x in str(args.tau_list).split(",") if x.strip() != ""]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        if args.skip_normal and "normal" in str(row.get("img", "")).lower():
            continue

        img_path = os.path.join(args.img_dir, str(row["img"]))
        mask_path = os.path.join(args.mask_dir, str(row["mask"]))

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue

        img = io.imread(img_path)
        box = parse_bbox_1024(row.get("bbox_1024", None))
        if box is None:
            continue

        mask_gt = io.imread(mask_path)
        if mask_gt.ndim == 3:
            mask_gt = mask_gt[:, :, 0]
        mask_gt = mask_gt > 127

        try:
            mask_pred, prob, tau_star, debug, obj_scale = medsam_scope_sc_only_topk_v3(
                img=img,
                base_box=box,
                predictor=predictor,
                use_ais=args.use_ais,
                num_candidate_boxes=args.num_candidate_boxes,
                K=min(8, int(args.K)),
                tau_list=tau_list,
                lambda_sc=args.lambda_sc,
                topk=args.topk,
                temp=args.temp,
                area_min=args.area_min,
                area_max=args.area_max,
                area_penalty_low=args.area_penalty_low,
                refine_step=args.refine_step,
                refine_radius=args.refine_radius,
            )

            if mask_pred is None:
                img_rgb = ensure_rgb(img)
                H, W = img_rgb.shape[:2]
                predictor.set_image(img_rgb.astype(np.uint8))
                prob = predict_prob_one_after_set_image(predictor, clamp_box(box, H, W), H, W)
                tau_star = 0.5
                mask_pred = prob >= tau_star
                obj_scale = estimate_box_scale(clamp_box(box, H, W), H, W)

            mask_pred = postprocess_mask_adaptive(mask_pred, obj_scale=obj_scale)

            iou_val = iou(mask_pred, mask_gt)
            dice_val = dice(mask_pred, mask_gt)
            acc_val = accuracy(mask_pred, mask_gt)

            results.append({
                "id": row.get("id", ""),
                "IoU": float(iou_val),
                "Dice": float(dice_val),
                "Accuracy": float(acc_val),
                "best_tau": float(tau_star),
            })

            inter = np.logical_and(mask_pred, mask_gt).sum()
            total_inter += inter
            total_pred += mask_pred.sum()
            total_gt += mask_gt.sum()
            total_tp += np.logical_and(mask_pred == 1, mask_gt == 1).sum()
            total_tn += np.logical_and(mask_pred == 0, mask_gt == 0).sum()
            total_pixels += mask_gt.size

            if args.save_pred:
                out_mask = os.path.join(args.out, f"{row.get('id','')}_pred.png")
                io.imsave(out_mask, img_as_ubyte(mask_pred), check_contrast=False)

        except Exception as e:
            tqdm.write(f"Warning: Failed id={row.get('id','')}: {str(e)}")
            continue

    if results:
        mIoU = float(np.mean([r["IoU"] for r in results]))
        mDice = float(np.mean([r["Dice"] for r in results]))
        mAcc = float(np.mean([r["Accuracy"] for r in results]))
    else:
        mIoU = mDice = mAcc = 0.0

    overall_dice = float(2 * total_inter / (total_pred + total_gt + 1e-6))
    overall_iou = float(total_inter / (total_pred + total_gt - total_inter + 1e-6))
    overall_accuracy = float((total_tp + total_tn) / max(1, total_pixels))

    pd.DataFrame(results).to_csv(os.path.join(args.out, "results.csv"), index=False)

    print("\n" + "=" * 60)
    print("Inference summary (Kvasir-SEG) - Our method")
    print("=" * 60)
    print(f"Processed images: {len(results)} / {len(df)}")
    print("\n[Mean metrics]")
    print(f"mIoU:  {mIoU:.4f}")
    print(f"mDice: {mDice:.4f}")
    print(f"mAcc:  {mAcc:.4f}")
    print("\n[Overall micro]")
    print(f"overall IoU:  {overall_iou:.4f}")
    print(f"overall Dice: {overall_dice:.4f}")
    print(f"overall Acc:  {overall_accuracy:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    repo_root = REPO_ROOT
    kvasir_root = repo_root / "KVasir"

    ap.add_argument("--ckpt", default=str(repo_root / "work_dir" / "MedSAM" / "medsam_vit_b.pth"))
    ap.add_argument("--model_type", default="vit_b")

    ap.add_argument("--meta", default=str(kvasir_root / "meta1.csv"))
    ap.add_argument("--img_dir", default=str(kvasir_root / "images_1024"))
    ap.add_argument("--mask_dir", default=str(kvasir_root / "masks_1024"))
    ap.add_argument("--out", default=str(kvasir_root / "result_ourwork"))

    ap.add_argument("--skip_normal", action="store_true", default=True)

    g_save = ap.add_mutually_exclusive_group()
    g_save.add_argument("--save_pred", dest="save_pred", action="store_true", help="Save prediction masks to --out.")
    g_save.add_argument("--no_save_pred", dest="save_pred", action="store_false", help="Do not save prediction masks.")
    ap.set_defaults(save_pred=True)

    g_ais = ap.add_mutually_exclusive_group()
    g_ais.add_argument("--use_ais", dest="use_ais", action="store_true", help="Enable AIS (adaptive budget).")
    g_ais.add_argument("--no_use_ais", dest="use_ais", action="store_false", help="Disable AIS.")
    ap.set_defaults(use_ais=True)

    ap.add_argument("--num_candidate_boxes", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)

    ap.add_argument("--tau_list", type=str, default="")

    ap.add_argument("--lambda_sc", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.05)

    ap.add_argument("--area_min", type=float, default=0.05)
    ap.add_argument("--area_max", type=float, default=0.95)
    ap.add_argument("--area_penalty_low", type=float, default=0.1)

    ap.add_argument("--refine_step", type=float, default=0.02)
    ap.add_argument("--refine_radius", type=float, default=0.06)

    args = ap.parse_args()

    print("use_ais =", args.use_ais, "| save_pred =", args.save_pred)
    main(args)
