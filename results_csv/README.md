# Minimal dataset: case-level results

Every mean ± SD, paired difference (Δ), 95% confidence interval, and p-value reported in Tables 2–7 and S1 Table can be recomputed from these files.

## Files

| File | Content | Supports |
|---|---|---|
| `case_overlap.csv` | DSC and IoU per case, model, and artery | Tables 2, 3, 5, 6 and S1 Table |
| `case_runtime.csv` | Inference time per case and model | Table 5(a) (inference time) and Table 7 |
| `case_lesions.csv` | Lesion counts per case, model, and artery | Table 4 |

All 433 included cases are listed once per model and assigned to exactly one test fold (0–4), as in the patient-level five-fold cross-validation. Fold sizes are 85, 93, 85, 85, and 85 cases. Rows are ordered by model (in the order the models appear in the paper), then by fold, case, and artery (LCA, LAD, LCX, RCA).

### `case_overlap.csv`

| Column | Description |
|---|---|
| `case_id` | Case identifier (433 included cases) |
| `fold` | Test fold in which the case was evaluated (0–4) |
| `model_id` | Model (see the table of model IDs below) |
| `artery` | Artery category: `LCA`, `LAD`, `LCX`, or `RCA` |
| `dsc` | Dice similarity coefficient (0–1) |
| `iou` | Intersection over Union (0–1) |
| `gt_positive` | 1 if the case has ground-truth calcification in this artery, otherwise 0 |

**How the scores are computed.** For each case and artery, the predicted reference-slice masks are stacked into a scan-level volume. TP, FP, and FN are summed over all reference slices of the scan, including slices without ground-truth calcification, and DSC and IoU are then computed once from these totals. Slice-level scores are not averaged. Because both scores come from the same TP, FP, and FN, `iou = dsc / (2 − dsc)` holds for every row.

**Empty values.** If `gt_positive = 0`, then `dsc` and `iou` are empty. These case–artery pairs are excluded from the overlap averages and tests. If a case has ground-truth calcification but the prediction is empty, both scores are 0.

### `case_runtime.csv`

| Column | Description |
|---|---|
| `case_id`, `fold`, `model_id` | As above |
| `inference_ms` | Inference time for the scan, in milliseconds |

### `case_lesions.csv`

| Column | Description |
|---|---|
| `case_id`, `fold`, `model_id`, `artery` | As above |
| `artery_negative` | 1 if the case has no ground-truth calcification in this artery, otherwise 0 |
| `gt_lesions` | Number of ground-truth lesions |
| `pred_lesions` | Number of predicted lesions |
| `matched_lesions` | Number of one-to-one matched lesion pairs |

**Lesions.** Lesions are 3D connected components (26-connectivity) of each artery class, taken from masks stacked from all reference slices. Within each scan, overlapping ground-truth and predicted lesions of the same artery are matched one-to-one to maximize the number of matches. False-positive (FP) lesions are `pred_lesions − matched_lesions`.

## Model IDs

| `model_id` | Name in the paper | Tables |
|---|---|---|
| `resnet50_unet` / `resnet50_unet_msffm` | ResNet-50-UNet / + MSFFM | 2, 3, 4, 6, 7 (+ MSFFM also in 5) |
| `densenet201_unet` / `_msffm` | DenseNet-201-UNet / + MSFFM | 2, 7 |
| `efficientnet_b4_unet` / `_msffm` | EfficientNet-b4-UNet / + MSFFM | 2, 7 |
| `mit_b2_unet` / `_msffm` | MiT-b2-UNet / + MSFFM | 2, 3, 4, 7 |
| `resnet50_segformer` / `_msffm` | ResNet-50-SegFormer / + MSFFM | 2, 7 |
| `densenet201_segformer` / `_msffm` | DenseNet-201-SegFormer / + MSFFM | 2, 7 |
| `efficientnet_b4_segformer` / `_msffm` | EfficientNet-b4-SegFormer / + MSFFM | 2, 7 |
| `mit_b2_segformer` / `_msffm` | MiT-b2-SegFormer / + MSFFM | 2, 7 |
| `pvt_v2_b2_emcad` / `_msffm` | PVTv2-b2-EMCAD / + MSFFM | 2, 7 (+ MSFFM also in 5) |
| `cat_net` | CAT-Net | 5, S1 |
| `csam` | CSAM | 5, S1 |
| `segmate` | SegMate | 5, S1 |
| `nnunet_25d` | 2.5D-nnU-Net | 5, S1 |
| `segformer3d` | SegFormer3D | 5, S1 |
| `lhu_net` | LHU-Net | 5, S1 |
| `nnunet_3d` | 3D-nnU-Net | 5, S1 |
| `resnet50_unet_stage3` | MSFFM at Stage 3 | 6(a) |
| `resnet50_unet_stage4` | MSFFM at Stage 4 | 6(a) |
| `resnet50_unet_radius2` | MSFFM ± 2 slices | 6(b) |
| `resnet50_unet_radius3` | MSFFM ± 3 slices | 6(b) |
| `resnet50_unet_sa` | Self-attention only | 6(c) |
| `resnet50_unet_ca` | Cross-attention only | 6(c) |

## Reproducing the tables

### Mean ± SD (Tables 2–7, S1)

Results are first summarized within each test fold, and the tables then report the mean ± SD over the five fold-level values. The SD is the population SD (divisor 5).

- **Artery-specific DSC and IoU:** mean over the GT-positive cases of the fold.
- **Overall DSC and mIoU:** unweighted mean of the four artery-specific means of the fold.
- **Lesion sensitivity and precision (Table 4):** `Σ matched / Σ gt_lesions` and `Σ matched / Σ pred_lesions` per artery and fold. Precision is undefined (N/A) when no lesions are predicted.
- **FP lesions per scan (Table 4):** mean of `pred_lesions − matched_lesions` per artery and fold. For Table 4(a) the mean is taken over artery-negative cases only; for Table 4(b–c) it is taken over all cases.
- **Inference time (Tables 5 and 7):** mean of `inference_ms` per fold.

Example for overall DSC (mean ± SD):

```python
import pandas as pd
ov = pd.read_csv("case_overlap.csv")
d = ov[ov.model_id == "resnet50_unet_msffm"]
artery_means = d.groupby(["fold", "artery"]).dsc.mean().unstack()
overall = artery_means.mean(axis=1)
print(f"{overall.mean():.3f} ± {overall.std(ddof=0):.3f}")  # 0.721 ± 0.049
```

### Paired statistics (Tables 2, 3, 5(b), 6)

- **Patient-level scores:**
  - Overall tests use each patient's mean over the arteries in which the patient has ground-truth calcification.
  - Artery-specific tests use patient–artery scores, restricted to GT-positive pairs.
- **Δ:** mean paired difference, computed as MSFFM − baseline, configuration − baseline, or proposed − competing method, depending on the table.
- **95% CI:** percentile interval from 10,000 paired bootstrap resamples of patients, drawn with replacement from all folds pooled. The intervals are not adjusted for multiplicity.
- **p-values:** two-sided Wilcoxon signed-rank test (SciPy defaults), Holm-adjusted separately for each metric within each family:
  - the 9 overall baseline–MSFFM contrasts (Table 2);
  - the 8 artery-specific contrasts (Table 3);
  - the 9 ablation contrasts (Table 6).

## License and citation

The files in this folder are released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license, and the code in the repository is released under the MIT License. If you use these results, please cite the associated paper. Please also cite the COCA dataset (doi: 10.71718/ge5g-ds80).
