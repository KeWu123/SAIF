# MedSAM Research Workspace

This repository collects baseline and plugin-augmented inference code for multiple medical segmentation datasets. The goal is to provide clean, reproducible entry points for:

- Baseline inference
- Our plugin inference (AIS + auto tau + stable top-k + low_res prob + adaptive postprocess)

Large datasets, checkpoints, and generated outputs are not included.

## Project Structure

```
MedSAM-main/
  CVC-300/
    code/
      infer_cvc300_baseline.py
      infer_cvc300_ourwork.py
      infer_cvc300.py
      processor_cvc300.py
  BUSI/
    inference_baseline.py
    inference_ourwork_busi.py
  KVasir/
    infer_basline.py
    infer_ourwork.py
    processor.py
    dataset_kvasir.py
  Synapes/
    inference_baseline.py
    inference_ourwork.py
    processor.py
  segment_anything/
  utils/
  work_dir/
```

Notes:
- `work_dir/` is expected to contain MedSAM checkpoints (not tracked).
- Each dataset folder has a baseline script and a plugin script.
- Preprocessing scripts exist for datasets that require them.

## Requirements

- Python 3.8+
- PyTorch
- segment_anything
- numpy, pandas, tqdm, scikit-image, h5py, scipy (for Synapse HD95)

Install dependencies according to your environment.

## Datasets and Scripts

### CVC-300

Preprocess:
```bash
python CVC-300/code/processor_cvc300.py ^
  --img_dir CVC-300/images ^
  --mask_dir CVC-300/masks ^
  --out_dir CVC-300/processed_1024
```

Baseline:
```bash
python CVC-300/code/infer_cvc300_baseline.py ^
  --ckpt work_dir/MedSAM/medsam_vit_b.pth
```

Our plugin:
```bash
python CVC-300/code/infer_cvc300_ourwork.py ^
  --ckpt work_dir/MedSAM/medsam_vit_b.pth
```

### BUSI

Baseline:
```bash
python BUSI/inference_baseline.py ^
  --ckpt work_dir/MedSAM/medsam_vit_b.pth
```

Our plugin:
```bash
python BUSI/inference_ourwork_busi.py ^
  --ckpt work_dir/MedSAM/medsam_vit_b.pth
```

### Kvasir-SEG

Preprocess (two variants exist, use one):
```bash
python KVasir/processor.py
```
or
```bash
python KVasir/dataset_kvasir.py
```

Baseline:
```bash
python KVasir/infer_basline.py ^
  --ckpt work_dir/MedSAM/medsam_vit_b.pth
```

Our plugin:
```bash
python KVasir/infer_ourwork.py ^
  --ckpt work_dir/MedSAM/medsam_vit_b.pth
```

### Synapse (test_vol_h5)

Baseline:
```bash
python Synapes/inference_baseline.py ^
  --ckpt work_dir/MedSAM/medsam_vit_b.pth
```

Our plugin:
```bash
python Synapes/inference_ourwork.py ^
  --ckpt work_dir/MedSAM/medsam_vit_b.pth
```

## Outputs

Most scripts save:
- `results.csv` with per-sample metrics
- Predicted masks (optional, controlled by flags in each script)

## Checkpoints

Place MedSAM checkpoints under:
```
work_dir/MedSAM/medsam_vit_b.pth
```
Update paths with CLI flags if your layout differs.

## Notes for GitHub

- Datasets and generated outputs are excluded via `.gitignore`.
- Keep only baseline and plugin scripts if you want a minimal public release.
