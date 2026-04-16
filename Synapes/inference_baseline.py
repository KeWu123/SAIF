import os
import glob
import argparse
import warnings

import numpy as np
import h5py
from tqdm import tqdm

import torch
from segment_anything import sam_model_registry, SamPredictor

warnings.filterwarnings("ignore", message="The NumPy module was reloaded*")

# HD95 需要 scipy
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    s = mask1.sum() + mask2.sum()
    return 0.0 if s == 0 else float(2 * inter / s)


def bbox_from_mask(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return [float(x1), float(y1), float(x2), float(y2)]


def to_uint8_rgb(image_2d: np.ndarray) -> np.ndarray:
    # 你的预处理一般是 0~1；这里转 uint8，并复制成3通道
    img = np.clip(image_2d, 0.0, 1.0)
    img_u8 = (img * 255.0).astype(np.uint8)
    return np.repeat(img_u8[:, :, None], 3, axis=-1)


def hd95_2d(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    2D HD95（像素单位）。
    pred / gt: bool 2D
    返回：95th percentile Hausdorff distance
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # 如果任一为空，HD95 没意义：返回 NaN（后面会被排除）
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")

    # 取边界（surface）
    pred_er = binary_erosion(pred)
    gt_er = binary_erosion(gt)
    pred_surf = np.logical_xor(pred, pred_er)
    gt_surf = np.logical_xor(gt, gt_er)

    # distance transform：到对方 surface 的最近距离
    # dt(x)=距离到最近的“非零点”，所以要对 surface 取反
    dt_gt = distance_transform_edt(~gt_surf)
    dt_pred = distance_transform_edt(~pred_surf)

    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]

    if d_pred_to_gt.size == 0 or d_gt_to_pred.size == 0:
        return float("nan")

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_d, 95))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", default="D:/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth", help="权重文件路径")
    ap.add_argument("--model_type", default="vit_b", help="模型类型 (vit_b/vit_l/vit_h 等)")
    ap.add_argument("--data_dir", default="D:/MedSAM-main/Synapes/test_vol_h5", help="test_vol_h5 文件夹路径")

    # 默认跑全部器官 1~8；也可指定子集：--organ_ids 1,3,7
    ap.add_argument("--organ_ids", type=str, default="1,2,3,4,5,6,7,8",
                    help="要跑的器官ID列表，用逗号分隔，如 '1,3,7'。默认跑 1~8")

    args = ap.parse_args()

    organ_list = [int(x) for x in args.organ_ids.split(",") if x.strip() != ""]
    organ_list = [x for x in organ_list if 1 <= x <= 8]
    organ_list = sorted(list(set(organ_list)))
    if len(organ_list) == 0:
        raise ValueError("organ_ids 解析为空，请传入如 --organ_ids 1,3,7（范围 1~8）")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt)
    sam.to(device)
    predictor = SamPredictor(sam)

    h5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))
    if len(h5_files) == 0:
        raise FileNotFoundError(f"在 {args.data_dir} 下没找到 .h5 文件，请确认路径是否正确。")

    dice_list = []
    hd95_list = []

    for h5p in tqdm(h5_files, desc="Infer test_vol_h5"):
        with h5py.File(h5p, "r") as f:
            vol_img = f["image"][:]  # [D, W, H]
            vol_lab = f["label"][:]

        D = vol_img.shape[0]
        for d in range(D):
            image2d = vol_img[d].astype(np.float32)
            label2d = vol_lab[d].astype(np.int32)

            # ✅ 同一个 slice 只 set_image 一次
            img_rgb = to_uint8_rgb(image2d)
            predictor.set_image(img_rgb)

            for organ_id in organ_list:
                mask_gt = (label2d == organ_id)
                box = bbox_from_mask(mask_gt)
                if box is None:
                    continue  # GT 没这个器官就跳过（不计入均值）

                b_np = np.array(box, dtype=np.float32)[None, :]
                masks_pred, _, _ = predictor.predict(box=b_np, multimask_output=False)
                mask_pred = (masks_pred[0] > 0.5)

                dsc = dice(mask_pred, mask_gt)
                hd = hd95_2d(mask_pred, mask_gt)

                dice_list.append(dsc)
                if not np.isnan(hd):
                    hd95_list.append(hd)

    mean_dice = float(np.mean(dice_list)) if len(dice_list) > 0 else float("nan")
    mean_hd95 = float(np.mean(hd95_list)) if len(hd95_list) > 0 else float("nan")

    # ✅ 按你要求：最后只输出这两个
    print(f"mean_dice: {mean_dice:.6f}")
    print(f"mean_hd95: {mean_hd95:.6f}")


if __name__ == "__main__":
    main()
