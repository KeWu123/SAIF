import os
import glob
import argparse
import warnings

import numpy as np
import h5py
from tqdm import tqdm

import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor

from scipy.ndimage import binary_erosion, distance_transform_edt

# skimage: AIS + adaptive postprocess
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects, remove_small_holes

warnings.filterwarnings("ignore", message="The NumPy module was reloaded*")


# ===================== Metrics =====================

def dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    denom = mask1.sum() + mask2.sum()
    if denom == 0:
        return 0.0
    return float(2 * intersection / denom)


def iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def bbox_from_mask(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return [float(x1), float(y1), float(x2), float(y2)]


def to_uint8_rgb(image_2d: np.ndarray) -> np.ndarray:
    """Synapse slices are typically normalized to 0~1; convert to uint8 RGB."""
    img = np.clip(image_2d, 0.0, 1.0)
    img_u8 = (img * 255.0).astype(np.uint8)
    return np.repeat(img_u8[:, :, None], 3, axis=-1)


# ===================== HD95 (2D, pixel units) =====================

def hd95_2d(pred: np.ndarray, gt: np.ndarray) -> float:
    """2D HD95. If pred or gt is empty, returns NaN (skipped in mean)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")

    pred_er = binary_erosion(pred)
    gt_er = binary_erosion(gt)
    pred_surf = np.logical_xor(pred, pred_er)
    gt_surf = np.logical_xor(gt, gt_er)

    dt_gt = distance_transform_edt(~gt_surf)
    dt_pred = distance_transform_edt(~pred_surf)

    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]

    if d_pred_to_gt.size == 0 or d_gt_to_pred.size == 0:
        return float("nan")

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))


# ===================== Utilities =====================

def clamp_box(box, H, W):
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


def softmax_np(x: np.ndarray, temp: float):
    x = x.astype(np.float32) / max(1e-6, float(temp))
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-8)


# ===================== 1) AIS: adaptive K / num_candidates =====================

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


# ===================== Box sampling: multi-scale + jitter =====================

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


# ===================== prob: low_res logits -> upsample -> sigmoid =====================

def predict_prob_one_after_set_image(predictor: SamPredictor, box, H, W) -> np.ndarray:
    """Call after predictor.set_image(); returns HxW float prob."""
    b_np = np.array(box, dtype=np.float32)[None, :]
    masks, _, low_res = predictor.predict(box=b_np, multimask_output=False)

    if low_res is None:
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


# ===================== 2) auto tau range =====================

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


# ===================== 4) adaptive postprocess =====================

def postprocess_mask_adaptive(mask: np.ndarray, obj_scale: float,
                              min_obj_big=200, min_hole_big=200,
                              min_obj_small=60, min_hole_small=60) -> np.ndarray:
    """Use milder thresholds for small objects to avoid wiping true positives."""
    if obj_scale < 0.03:
        min_obj, min_hole = min_obj_small, min_hole_small
    else:
        min_obj, min_hole = min_obj_big, min_hole_big

    m = mask.astype(bool)
    m = remove_small_objects(m, min_size=int(min_obj))
    m = remove_small_holes(m, area_threshold=int(min_hole))
    return m.astype(bool)


# ===================== 3) stable fusion: score/(stdIoU+eps) + Top-k =====================

def score_candidate_for_tau(probs_k, tau: float, lambda_sc: float,
                           area_min: float, area_max: float, area_penalty_low: float):
    """SC-only: mu - lambda*sigma, with area penalty."""
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


def medsam_scope_sc_only_topk_synapse(
    img_rgb_uint8: np.ndarray,
    base_box,
    predictor: SamPredictor,

    # AIS
    use_ais=True,
    num_candidate_boxes=16,
    K=8,

    # tau search
    tau_list=None,
    refine_step=0.02,
    refine_radius=0.06,

    # scoring & fusion
    lambda_sc=0.3,
    topk=3,
    temp=0.05,
    area_min=0.05,
    area_max=0.95,
    area_penalty_low=0.1,
):
    """Returns: final_mask, fused_prob, tau_star, debug, obj_scale."""
    img = img_rgb_uint8.astype(np.uint8)
    H, W = img.shape[:2]
    base_box = clamp_box(base_box, H, W)

    # 1) AIS budget
    if use_ais:
        n_cand, K_use, cplx, obj_scale = ais_pick_budget(img, base_box, H, W)
    else:
        n_cand, K_use = int(num_candidate_boxes), int(min(8, K))
        cplx, obj_scale = 0.0, estimate_box_scale(base_box, H, W)

    n_cand = int(max(1, n_cand))
    K_use = int(min(8, max(1, K_use)))

    predictor.set_image(img)

    # Candidate boxes
    candidates = build_candidate_boxes(base_box, img.shape, num_candidate_boxes=n_cand, jitter=0.04, multi_scale=True)
    if len(candidates) == 0:
        return None, None, 0.5, {"reason": "no candidates"}, obj_scale

    # 2) auto tau range from base prob
    base_prob = predict_prob_one_after_set_image(predictor, base_box, H, W)
    lo, hi = auto_tau_range_from_prob(base_prob, base_lo=0.25, base_hi=0.80)

    if tau_list is None:
        tau_list = np.linspace(lo, hi, 9).tolist()
    else:
        tau_list = [float(t) for t in tau_list]

    # Pack candidates: K jittered probs + mean prob
    cand_pack = []
    for b in candidates:
        boxes_k = [b] + [jitter_one_box(b, H, W, jitter=0.01) for _ in range(K_use - 1)]
        probs_k = [predict_prob_one_after_set_image(predictor, bb, H, W) for bb in boxes_k]
        mean_prob = np.mean(np.stack(probs_k, axis=0), axis=0).astype(np.float32)
        cand_pack.append((b, probs_k, mean_prob))

    # ---- coarse best tau (best candidate per tau)
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

    # ---- refine tau locally
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

    # 3) fix tau_star, fuse Top-k candidates with stability weighting
    scores = []
    sigmas = []
    probs = []

    for (b, probs_k, mean_prob) in cand_pack:
        s, meta = score_candidate_for_tau(probs_k, tau_star, lambda_sc, area_min, area_max, area_penalty_low)
        scores.append(float(s))
        sigmas.append(float(meta["sigma"]))
        probs.append(mean_prob)

    scores_np = np.array(scores, dtype=np.float32)
    sig_np = np.array(sigmas, dtype=np.float32)

    fused_logits = scores_np / (sig_np + 1e-3)

    k_top = int(max(1, topk))
    k_top = min(k_top, len(fused_logits))
    idx = np.argsort(fused_logits)[::-1][:k_top]
    w = softmax_np(fused_logits[idx], temp=temp)

    fused_prob = np.zeros((H, W), dtype=np.float32)
    for wi, j in zip(w, idx):
        fused_prob += float(wi) * probs[int(j)].astype(np.float32)

    final_mask = fused_prob >= tau_star

    debug = {
        "ais": {"use_ais": bool(use_ais), "num_candidates": int(n_cand), "K": int(K_use),
                "complexity": float(cplx), "obj_scale": float(obj_scale)},
        "tau_range": {"auto_lo": float(lo), "auto_hi": float(hi)},
        "tau_star": float(tau_star),
        "coarse_top1": best_tau_top1,
        "fine_top1": best_tau_top1_f,
        "topk_idx": idx.tolist(),
        "topk_weights": w.tolist(),
    }

    return final_mask.astype(bool), fused_prob, tau_star, debug, float(obj_scale)


# ===================== Main (Synapse test_vol_h5) =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="D:/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth", help="Checkpoint path")
    ap.add_argument("--model_type", default="vit_b", help="Model type")
    ap.add_argument("--data_dir", default="D:/MedSAM-main/Synapes/test_vol_h5", help="test_vol_h5 path")

    ap.add_argument("--organ_ids", type=str, default="1,2,3,4,5,6,7,8",
                    help="Organ IDs (1~8), comma-separated")

    # Strategy switches/params
    ap.add_argument("--use_ais", action="store_true", default=True, help="Enable AIS adaptive budget")
    ap.add_argument("--num_candidate_boxes", type=int, default=16, help="Num candidates when AIS is off")
    ap.add_argument("--K", type=int, default=8, help="Jitter count per candidate when AIS is off (<=8)")

    ap.add_argument("--tau_list", type=str, default="", help="Empty=auto tau range; or comma list")
    ap.add_argument("--refine_step", type=float, default=0.02)
    ap.add_argument("--refine_radius", type=float, default=0.06)

    ap.add_argument("--lambda_sc", type=float, default=0.3, help="SC: mu - lambda*sigma")
    ap.add_argument("--topk", type=int, default=3, help="Top-k fusion under same tau")
    ap.add_argument("--temp", type=float, default=0.05, help="Softmax temperature")

    ap.add_argument("--area_min", type=float, default=0.05)
    ap.add_argument("--area_max", type=float, default=0.95)
    ap.add_argument("--area_penalty_low", type=float, default=0.1)

    # Postprocess thresholds (adaptive)
    ap.add_argument("--min_obj_big", type=int, default=200)
    ap.add_argument("--min_hole_big", type=int, default=200)
    ap.add_argument("--min_obj_small", type=int, default=60)
    ap.add_argument("--min_hole_small", type=int, default=60)

    args = ap.parse_args()

    organ_list = [int(x) for x in args.organ_ids.split(",") if x.strip() != ""]
    organ_list = [x for x in organ_list if 1 <= x <= 8]
    organ_list = sorted(list(set(organ_list)))
    if len(organ_list) == 0:
        raise ValueError("organ_ids parsed empty; expected 1~8")

    tau_list = None
    if args.tau_list.strip():
        tau_list = [float(x) for x in args.tau_list.split(",") if x.strip() != ""]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt)
    sam.to(device)
    predictor = SamPredictor(sam)

    h5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No .h5 files found under {args.data_dir}")

    dice_by_organ = {oid: [] for oid in organ_list}
    hd95_by_organ = {oid: [] for oid in organ_list}
    all_dice = []

    for h5p in tqdm(h5_files, desc="Infer test_vol_h5 (AIS+autoTau+stableTopK+adaptivePost)"):
        with h5py.File(h5p, "r") as f:
            vol_img = f["image"][:]  # [D, W, H]
            vol_lab = f["label"][:]

        D = vol_img.shape[0]
        for d in range(D):
            image2d = vol_img[d].astype(np.float32)
            label2d = vol_lab[d].astype(np.int32)

            img_rgb = to_uint8_rgb(image2d)
            H, W = img_rgb.shape[:2]

            for organ_id in organ_list:
                mask_gt = (label2d == organ_id)
                base_box = bbox_from_mask(mask_gt)
                if base_box is None:
                    continue

                # v3-style inference
                mask_pred, prob, tau_star, debug, obj_scale = medsam_scope_sc_only_topk_synapse(
                    img_rgb_uint8=img_rgb,
                    base_box=base_box,
                    predictor=predictor,

                    use_ais=args.use_ais,
                    num_candidate_boxes=args.num_candidate_boxes,
                    K=min(8, int(args.K)),

                    tau_list=tau_list,
                    refine_step=args.refine_step,
                    refine_radius=args.refine_radius,

                    lambda_sc=args.lambda_sc,
                    topk=args.topk,
                    temp=args.temp,
                    area_min=args.area_min,
                    area_max=args.area_max,
                    area_penalty_low=args.area_penalty_low,
                )

                # Fallback: direct base-box prediction
                if mask_pred is None:
                    predictor.set_image(img_rgb.astype(np.uint8))
                    base_box_c = clamp_box(base_box, H, W)
                    prob = predict_prob_one_after_set_image(predictor, base_box_c, H, W)
                    tau_star = 0.5
                    mask_pred = (prob >= tau_star)
                    obj_scale = estimate_box_scale(base_box_c, H, W)

                # Adaptive postprocess
                mask_pred = postprocess_mask_adaptive(
                    mask_pred,
                    obj_scale=float(obj_scale),
                    min_obj_big=args.min_obj_big,
                    min_hole_big=args.min_hole_big,
                    min_obj_small=args.min_obj_small,
                    min_hole_small=args.min_hole_small,
                )

                dsc = dice(mask_pred, mask_gt)
                hd = hd95_2d(mask_pred, mask_gt)

                dice_by_organ[organ_id].append(dsc)
                all_dice.append(dsc)
                if not np.isnan(hd):
                    hd95_by_organ[organ_id].append(hd)

    print("\n" + "=" * 60)
    print("Synapse test_vol_h5 summary (AIS + auto tau + stable Top-k + adaptive postprocess)")
    print("=" * 60)
    for organ_id in organ_list:
        md = float(np.mean(dice_by_organ[organ_id])) if len(dice_by_organ[organ_id]) > 0 else float("nan")
        mh = float(np.mean(hd95_by_organ[organ_id])) if len(hd95_by_organ[organ_id]) > 0 else float("nan")
        print(f"organ {organ_id}  mean_dice: {md:.6f}   mean_hd95: {mh:.6f}")

    overall_mean_dice = float(np.mean(all_dice)) if len(all_dice) > 0 else float("nan")
    print("-" * 60)
    print(f"overall mean_dice: {overall_mean_dice:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
