# Segmentation of Sparse Coronary Artery Calcifications in Computed Tomography via 2.5D Multi-Slice Feature Fusion

## Overview

This repository contains the code for the paper *"Segmentation of Sparse Coronary Artery Calcifications in Computed Tomography via 2.5D Multi-Slice Feature Fusion"*. The project introduces the **Multi-Slice Feature Fusion Module (MSFFM)**, a plug-and-play attention module that restores inter-slice continuity in standard 2D encoder–decoder segmentation networks without resorting to 3D convolution.

## Introduction

Coronary artery calcification (CAC) on cardiac-gated CT appears as small, sparse lesions scattered across slices. A purely 2D segmentation network sees each slice in isolation and loses the vessel continuity that a radiologist relies on, while a fully 3D network is expensive to train and tends to over-smooth lesions this small.

MSFFM takes the middle path. Three adjacent slices — previous, reference, and next — are encoded by a single weight-shared 2D backbone, and their intermediate feature maps are related to one another through self- and cross-attention. Only the reference slice is segmented; the neighbours act as context. The module is inserted into an existing encoder without changing the decoder or the segmentation head, so it drops into U-Net, SegFormer, and EMCAD alike.

## Dataset

The dataset is **COCA (Coronary Calcium and Chest CT)** from Stanford AIMI, publicly available [here](https://aimi.stanford.edu/datasets/coca-coronary-calcium-chest-ct). Of the 451 studies, 433 are used; the 18 exclusions and their reasons (missing tags, zero z-spacing, absent DICOM source, unnamed ROI) are recorded in `data/dataprep/coca_data_error.txt`.

- **Classes (5):** 0 = background, 1 = LCA, 2 = LAD, 3 = LCX, 4 = RCA. In-plane resolution is 512 × 512.
- **Cross-validation:** the 433 cases are split into five folds at the **case level** using `MultilabelStratifiedKFold` (stratification key = per-case `[LCA, LAD, LCX, RCA]` multi-hot vector, `random_state=42`). Slice-level splitting is forbidden: adjacent slices of the same case would land in both train and validation, leaking raw pixels through the previous/next channels and invalidating the 2.5D premise.
- **Normalization:** HU values are clipped and z-scored with constants computed once over all 433 cases (`hu_stats_433.json`; `lower`/`upper` are the 0.5 / 99.5 percentiles, `mean`/`std` are measured after clipping). Every fold shares the same four constants.
- **Augmentation:** rot90-plus-flip and rotation are each applied with independent 50% probability. The rotation angle is an integer in `[-20, 19]`, and both image and label are resampled with `order=0` (nearest).

Expected layout on disk (the dataset root is git-ignored):

```
data/datasets/COCA/COCA_3frames_5fold/
├── images/case{0000..0450}.npy      # (D, H, W) float32 HU volume, read as memmap
├── labels/case{0000..0450}.npy      # (D, H, W) uint8
├── lists_COCA_5fold/
│   ├── fold0.txt ~ fold4.txt        # one line per sample: case{gidx}_slice{n}, n = triplet start index
│   └── fold_assignment.csv
├── hu_stats_433.json
└── case_index.csv                   # case_id ↔ original nnUNet filename
```

`COCAVolumeDataset` lazily memory-maps each case volume and assembles `vol[n:n+3]` → `(H, W, 3)` with `vol[n+1]` as the center label. These assets are produced once by `data/dataprep/build_5fold_dataset.py` from the nnUNet-format source `Dataset001_COCA`; see [`data/dataprep/README.md`](data/dataprep/README.md) for the full preprocessing chain.

## Model architecture

MSFFM receives three feature maps from an intermediate encoder stage and computes:

```
Z_ref   = SA(X_ref)                                  # self-attention
Z_prev  = CA(X_ref, X_prev)                          # Q = X_ref, K/V = X_prev
Z_next  = CA(X_ref, X_next)
Z_fused = W_c · Concat(Z_prev, Z_ref, Z_next)        # 1x1 conv restores the channel count
Z_final = Z_fused ⊕ X_ref                            # residual, BN zero-init warm start
```

- Q, K, and V are produced by 1×1 convolutions; attention is full `(H·W)×(H·W)` multi-head scaled dot-product with `num_heads=8`.
- The module is inserted at the **32×32 and 16×16** encoder resolutions, which is the best-performing placement in the ablation. The stage index differs per backbone: stages 3 and 4 for `resnet_sa`, `mix_transformer_sa`, and `emcad_sa`; stages 4 and 5 for `densenet_sa` and `efficientnet_sa`.
- The three slices pass through **the same encoder with shared weights**, three times.

### Configurations

A single entry point covers both the baselines and the MSFFM variants through `--decoder` and `--encoder`:

| Configuration | Arguments | Input channels |
|---|---|---|
| MSFFM + U-Net | `--decoder unet --encoder resnet50_sa` | 3 |
| Single-slice baseline | `--decoder unet --encoder resnet50` | 1 |
| EMCAD + MSFFM | `--decoder emcad_sa --encoder pvt_v2_b2` | 3 |
| EMCAD baseline | `--decoder emcad --encoder pvt_v2_b2` | 1 |

Supported encoders per decoder (`utils.py`):

| Decoder | Encoders |
|---|---|
| `unet`, `segformer` | `resnet50`, `densenet201`, `efficientnet-b4`, `mit_b2` (and their `_sa` variants) |
| `emcad` | `pvt_v2_b0`~`b5`, `resnet18`~`resnet152` |
| `emcad_sa` | `pvt_v2_b1`~`b5` |

The number of input slices is not a flag; `derive_num_slices(decoder, encoder)` returns 3 for any `_sa` encoder or the `emcad_sa` decoder, and 1 otherwise.

## Training strategy

- AdamW (lr = 1e-5, weight decay = 1e-4) with a polynomial LR schedule (exponent 0.9), batch size 16, up to 300 epochs, early stopping with patience 50, and AMP enabled throughout.
- The loss is derived from the decoder family: `unet` and `segformer` use last-layer supervision with `0.5·Dice + 0.5·CE`, while `emcad` and `emcad_sa` use `mutation` deep supervision with `0.7·Dice + 0.3·CE`. Override with `--supervision`, `--dice_weight`, and `--ce_weight`.
- Dice loss excludes the background class from its average.
- Each run is trained five times, once per validation fold.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --use_5fold_cv --fold_idx 0 \
  --decoder unet --encoder resnet50_sa \
  --max_epochs 300 --early_stopping_patience 50 \
  --exp_setting msffm_resnet50_unet_fold0_seed42
```

- Run every command **from the repository root**. Dataset and checkpoint paths default to cwd-relative locations, and the in-tree fork of `segmentation_models_pytorch` only shadows the pip-installed copy when the working directory is the repository root.
- `--exp_setting` must contain the string `fold{fold_idx}`; training aborts otherwise, because the fold identity is not encoded in the path and a mismatch would overwrite another fold's checkpoints.

### Experiment naming

`exp_setting` becomes the checkpoint directory name, so this convention is the only key that leads back to an existing run. The `_sa` suffix is dropped from the encoder label.

| Configuration | Pattern | Example |
|---|---|---|
| MSFFM (SMP backbones) | `msffm_{encoder}_{decoder}_fold{k}_seed42` | `msffm_resnet50_unet_fold0_seed42` |
| Single-slice baseline | `baseline_{encoder}_{decoder}_fold{k}_seed42` | `baseline_resnet50_unet_fold0_seed42` |
| EMCAD / EMCAD + MSFFM | `emcad{,_sa}_fold{k}_seed42` | `emcad_sa_fold0_seed42` |

### Fine-tuning

Passing `--init_from` turns a run into a fine-tuning run. `--exp_setting` keeps its usual meaning — the name this run saves under — so the source checkpoint is left untouched.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --fold_idx 0 \
  --decoder unet --encoder resnet50_sa \
  --init_from   msffm_resnet50_unet_fold0_seed42 \
  --exp_setting kmu_chest_fold0
```

- Weights come from the `*best_model.pth` inside the `--init_from` directory, which is located using the same `--max_epochs` / `--batch_size` / `--base_lr` as the current run.
- Only the head keys (`segmentation_head.` for `unet`/`segformer`, `out_head` for the EMCAD family) are stripped before a `strict=False` load, so the encoder and decoder transfer even when the class count changes.
- If `--init_from` carries a `fold{k}` token it must match `--fold_idx`. Weights from another fold have already been trained on the current validation fold, which would contaminate validation.
- Training restarts from scratch, so the LR schedule resets; shorten `--max_epochs` for a brief fine-tune.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --use_5fold_cv --fold_idx 0 \
  --decoder unet --encoder resnet50_sa \
  --exp_setting msffm_resnet50_unet_fold0_seed42 \
  [--is_savenii] [--save_attention]
```

- The evaluation set is `fold{fold_idx}.txt`, the validation fold used during training. There is no separate hold-out test set.
- `--exp_setting`, `--max_epochs`, `--batch_size`, `--base_lr`, and `--img_size` must match the training run for the checkpoint path to resolve.
- **Metrics are computed in 3D.** Per-slice predictions are collected per case, assembled into a `(D, H, W)` volume, and scored with MONAI's Dice, MeanIoU, and SurfaceDistance over the foreground classes. `--is_savenii` additionally writes NIfTI volumes with spacing `(0.375, 0.375, --z_spacing)`.

Results across folds are aggregated into a Markdown table of mean ± standard deviation:

```bash
python aggregate_5fold_results.py --decoder unet --encoder resnet50_sa \
  --exp_template msffm_resnet50_unet_fold{fold}_seed42
```

## Visualization

`--save_attention` enables attention visualization with a single flag. Every `NonLocalBlock` in the model tree is discovered automatically, switched to the explicit attention path (`return_attention=True`), and hooked; the query pixel is the centroid of the ground-truth lesion on the reference slice, and slices without a lesion are skipped.

Heatmaps are drawn as a **multiple of the uniform distribution** (1.0 means no preference) on a shared color scale, so panels can be compared directly rather than each being min-max normalized into a false hotspot.

## Directory convention

Training and evaluation compose directories with the same rule; changing it requires editing `train.py` and `test.py` together.

```
model/{NetClass}_{encoder}/{dataset}_{img_size}/{exp_setting}/epo{E}_bs{B}_lr{LR}/
test_log/{NetClass}_{encoder}/{dataset}_{img_size}/{exp_setting}/epo{E}_bs{B}_lr{LR}/
```

- `NetClass` is `net.__class__.__name__`, i.e. `Unet`, `Segformer`, `EMCADNet`, or `EMCAD_SA_Net`.
- `lr` is Python's string form of the float (`1e-05`).
- Checkpoints are named `epoch_{N}_{val_loss:.4f}_best_model.pth` and written whenever validation loss improves.

## Repository structure

```
├── train.py / trainer.py            # training entry point and loop
├── test.py  / tester.py             # evaluation entry point and 3D metrics
├── dataset.py                       # COCAVolumeDataset, CT normalization, augmentation
├── utils.py                         # encoder allow-lists, PolyLR, DiceLoss, supervision combinations
├── aggregate_5fold_results.py       # per-fold results into a mean±std table
├── segmentation_models_pytorch/     # in-tree fork of SMP
│   └── encoders/{resnet,densenet,efficientnet,mix_transformer}_sa.py   # MSFFM-integrated backbones
├── networks/emcad/                  # EMCAD decoder and pvtv2 backbone (use_msffm flag)
└── data/dataprep/                   # dataset construction scripts
```

## Implementation notes

- **`in_channels=1` even though the input tensor has three channels.** `smp.Unet(..., in_channels=1)` patches the first convolution to `Conv2d(1→64)`, and the `_sa` encoder slices the three channels apart inside `forward` (`[:,0:1]`, `[:,1:2]`, `[:,2:3]`) to push them through that same convolution three times. Setting `in_channels=3` breaks both the weight sharing and the ImageNet pretrained load.
- **One training run per GPU.** `trainer.py` wraps every visible GPU in `nn.DataParallel` when `torch.cuda.device_count() > 1`, so always pin `CUDA_VISIBLE_DEVICES`. With two free GPUs, run two different experiments concurrently. `DistributedDataParallel` is not supported.
- **The DataLoader uses `shuffle=False` with `collate_fn=shuffle_within_batch`**, shuffling only within a batch. Enabling outer shuffling destroys the adjacent-slice ordering the 2.5D premise depends on.
- `--use_5fold_cv` is a backward-compatibility flag and branches on nothing; the 5-fold path is always taken.
- `emcad_sa` rejects `pvt_v2_b0` because the MSFFM `NonLocalBlock` channel widths are fixed at 320/512 while b0 uses 160/256.
- `model/pvt/` only contains `pvt_v2_b2.pth` and `pvt_v2_b3.pth`. **Training** with any other pvt encoder dies with `FileNotFoundError`; evaluation is unaffected since it never reads pretrained weights.
- `--no_pretrain` has no effect in `test.py`, which always builds the model without pretrained weights and then loads the checkpoint strictly.
- Do not merge `EMCADNet` and `EMCAD_SA_Net` into one class. Paths are composed from `net.__class__.__name__`, so merging would make the two experiments overwrite each other's checkpoints.
- `NonLocalBlock` uses fused SDPA (`F.scaled_dot_product_attention`) and must pass q/k/v through `.contiguous()`, since torch 2.0 requires the last dimension to be contiguous. The fused kernel relies on atomics, so MSFFM evaluation metrics vary run to run in the fourth decimal place.
- `segmentation_models_pytorch/encoders/multi_slice_feature_fusion.py` is a reference implementation registered nowhere; editing it does not affect training or evaluation.
- `dataset.py` lives at the repository root to avoid a name collision with the HuggingFace `datasets` package. Do not reintroduce a `datasets/` directory.
- The residual diagnostic prints in `resnet_sa.py` and `multi_slice_feature_fusion.py` are deliberate instrumentation gated behind the module constant `DEBUG_RESIDUAL` (default `False`).

## Branches

| Branch | Contents |
|---|---|
| `main` | **Unified trunk.** All four configurations are selected through `--decoder` and `--encoder`. The baseline for new work. |
| `single_slice`, `EMCAD`, `EMCAD-SA` | Pre-unification hold-out baselines. Frozen, kept only to reproduce earlier results. |
| `2.5d-baselines` | Comparison harness for CAT-Net, CSAM, and SegMate (`--net25d`). |
| `3d-baselines` | Comparison harness for fully 3D models such as SegFormer3D, LHU-Net, WaveFormer, and SuPreM (`--network`). |

All branches share the same 5-fold assets, labels, and fold assignment. `3d-baselines` differs in that it uses case-level 3D crops and sliding-window inference, bypassing the per-slice accumulation and volume assembly used by the 2.5D branches. Hold-out results from the frozen branches were produced with different normalization constants and are not directly comparable to the 5-fold numbers.

## Acknowledgements

- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) — U-Net, SegFormer, and the encoder registry (forked in-tree)
- [EMCAD](https://github.com/SLDGroup/EMCAD) — efficient multi-scale convolutional attention decoder
- [PVTv2](https://github.com/whai362/PVT) — pyramid vision transformer backbone

## License

Segmentation of Sparse Coronary Artery Calcifications in Computed Tomography via 2.5D Multi-Slice Feature Fusion is released under the [MIT License](LICENSE).

## Citation

```
Stanford AIMI, "COCA - Coronary Calcium and Chest CTs",
https://aimi.stanford.edu/datasets/coca-coronary-calcium-chest-ct
```
