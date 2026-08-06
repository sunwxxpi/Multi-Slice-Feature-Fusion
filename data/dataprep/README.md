# dataprep — COCA dataset construction

These scripts build `COCA_1frame`, `COCA_3frames`, `COCA_3frames_5fold`, and `Dataset001_COCA` under `data/datasets/COCA/` from the original DICOM/XML export. They are never called by the training or evaluation code.

**Every script except `build_5fold_dataset.py` uses relative paths** to `./COCA/COCA_final`, `./COCA_3frames`, and `./Dataset001_COCA`. Run them with `data/datasets/COCA/` as the working directory, not from the directory the script lives in.

```bash
cd data/datasets/COCA && python ../../dataprep/preprocess_train_test_data_3frames.py
```

`build_5fold_dataset.py` resolves `SRC` and `OUT` relative to its own location, so it can be run from anywhere.

**The original DICOM/XML trees (`COCA/COCA_final`, `COCA/Gated_release_final`) are not part of this repository.** The six scripts that read them — `xml_to_nii_label.py`, `xml_to_png_label.py`, `organize_dataset.py`, `organize_nnUNet_format.py`, `add_tag_value_to_dcm.py`, `analysis_dcm_metadata.py` — cannot be run as-is. The scripts that start from `Dataset001_COCA` (`preprocess_train_test_data_{1,3}frame(s).py` and `build_5fold_dataset.py`) still work.

Rough order of execution:

```
xml_to_nii_label.py → organize_dataset.py → organize_nnUNet_format.py
  → preprocess_train_test_data_{1,3}frame(s).py → build_5fold_dataset.py
```

`coca_data_error.txt` records why 18 of the 451 studies were excluded (missing tags, zero z-spacing, absent DICOM source, unnamed ROI), leaving the 433 cases used in the paper.
