# SAIF: A Stability-Aware Inference Framework for Medical Image Segmentation with Segment Anything Model

This repository contains the implementation of **SAIF**, a **training-free** and **plug-and-play** inference framework for improving the robustness of **box-prompted medical image segmentation** with frozen SAM-based backbones.

SAIF explicitly models **inference-time uncertainty** from two sources:

- **Bounding-box perturbation**
- **Threshold variation during binarization**

It then performs:

- **stability-consistency scoring**
- **candidate filtering**
- **stability-weighted fusion**

to produce more reliable segmentation outputs **without retraining or architectural modification**.

---

## Highlights

- Training-free inference framework
- Plug-and-play for SAM-based medical segmentation models
- Explicit uncertainty modeling in the **box-threshold joint space**
- Stability-aware candidate ranking and fusion
- No additional learnable parameters
- Validated on **Synapse**, **CVC-ClinicDB**, **Kvasir-SEG**, and **CVC-300**

---

## Method Overview

Given an input image and a box prompt, SAIF works in three stages:

1. **Uncertainty Modeling**
   - Generate multiple perturbed box candidates
   - Build an image-adaptive threshold set

2. **Stability-Consistency Scoring**
   - Evaluate candidate masks under different thresholds
   - Rank candidates by stability and boundary consistency

3. **Stability-Aware Fusion**
   - Select the top-ranked candidates
   - Fuse them in probability space using stability-based weights
   - Apply a shared threshold to obtain the final binary mask

---

## Repository Structure

```text
.
├── BUSI/
│   ├── inference_baseline.py
│   ├── inference_ourwork_busi.py
│   └── processor_BUSI.py
├── CVC-300/
│   └── code/
│       ├── infer_cvc300_baseline.py
│       ├── infer_cvc300_ourwork.py
│       └── processor_cvc300.py
├── cvc-clinic/
│   └── code/
│       ├── infer_clinicdb_baseline.py
│       ├── infer_clinicdb.py
│       ├── infer_clinicd_1.py
│       └── processor_clinicdb.py
├── KVasir/
│   ├── infer_basline.py
│   ├── infer_ourwork.py
│   ├── processor.py
│   └── dataset_kvasir.py
├── Synapes/
│   ├── inference_baseline.py
│   ├── inference_ourwork.py
│   └── processor.py
├── segment_anything/
├── utils/
└── work_dir/
