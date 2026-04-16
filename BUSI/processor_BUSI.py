#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BUSI 数据集预处理:
- 支持 Dataset_BUSI_with_GT/ 下 benign/malignant/normal 文件夹结构
- 自动匹配 *_mask.png 文件
- resize + letterbox 到 1024×1024
- 保存图像、mask、bbox 到输出目录
- 输出 meta.csv (包含文件名和 bbox 信息)
"""

import os, re, argparse
import numpy as np
import pandas as pd
from glob import glob
from skimage import io, transform, img_as_ubyte
from skimage.color import gray2rgb
from skimage.morphology import remove_small_objects

def natural_key(s: str):
    """用于排序: busi_1.png, busi_2.png, busi_10.png"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def read_image(path):
    img = io.imread(path)
    if img.ndim == 2: img = gray2rgb(img)
    elif img.shape[2] == 4: img = img[:,:,:3]
    return img

def read_mask(path):
    m = io.imread(path)
    if m.ndim == 3: m = m[:,:,0]
    return (m>0).astype(np.uint8)

def compute_bbox(mask, min_area=20):
    """从 mask 提取 bbox"""
    m = remove_small_objects(mask.astype(bool), min_size=min_area)
    ys, xs = np.where(m)
    if len(xs)==0: return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def resize_with_bbox(img, mask, bbox, target=1024):
    """等比例缩放 + 填充到 target"""
    H,W = img.shape[:2]
    scale = min(target/H, target/W)
    new_h,new_w = int(H*scale), int(W*scale)
    img_r = transform.resize(img,(new_h,new_w),preserve_range=True,anti_aliasing=True)
    m_r = transform.resize(mask.astype(float),(new_h,new_w),order=0,preserve_range=True)
    canvas = np.zeros((target,target,3))
    canvas_m = np.zeros((target,target))
    pad_y = (target-new_h)//2; pad_x=(target-new_w)//2
    canvas[pad_y:pad_y+new_h,pad_x:pad_x+new_w]=img_r
    canvas_m[pad_y:pad_y+new_h,pad_x:pad_x+new_w]=m_r
    if bbox is None: return canvas, canvas_m>0.5, None
    xmin,ymin,xmax,ymax=bbox
    return canvas, canvas_m>0.5, [
        int(xmin*scale+pad_x), int(ymin*scale+pad_y),
        int(xmax*scale+pad_x), int(ymax*scale+pad_y)
    ]

def main(args):
    out_img, out_msk = os.path.join(args.out,"images_1024"), os.path.join(args.out,"masks_1024")
    ensure_dir(out_img); ensure_dir(out_msk)

    rows=[]
    # 遍历 benign, malignant, normal 子文件夹
    for cls in ["benign","malignant","normal"]:
        cls_dir=os.path.join(args.root,cls)
        if not os.path.exists(cls_dir): continue
        for ip in sorted(glob(os.path.join(cls_dir,"*.png")), key=natural_key):
            if "_mask" in ip: continue  # 跳过 mask 文件
            base=os.path.splitext(os.path.basename(ip))[0]
            mp=os.path.join(cls_dir,f"{base}_mask.png")
            if not os.path.exists(mp): continue
            img=read_image(ip).astype(float)/255.0
            mask=read_mask(mp)
            bbox=compute_bbox(mask)
            img_1024,mask_1024,bbox_1024=resize_with_bbox(img,mask,bbox)
            # 保存
            out_name=f"{cls}_{base}.png"
            io.imsave(os.path.join(out_img,out_name),img_as_ubyte(np.clip(img_1024,0,1)))
            io.imsave(os.path.join(out_msk,out_name.replace(".png","_mask.png")),(mask_1024*255).astype(np.uint8))
            rows.append({
                "id":f"{cls}_{base}",
                "class":cls,
                "img":out_name,
                "mask":out_name.replace(".png","_mask.png"),
                "bbox_1024":bbox_1024
            })

    pd.DataFrame(rows).to_csv(os.path.join(args.out,"meta.csv"),index=False)
    print(f"[OK] processed {len(rows)} samples, saved to {args.out}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True,help="D:/MedSAM-main/Dataset_BUSI_with_GT")
    ap.add_argument("--out",required=True,help="D:/MedSAM-main/BUSI/after_processor")
    args=ap.parse_args()
    main(args)