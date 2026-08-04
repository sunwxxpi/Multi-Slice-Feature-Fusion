# dataprep — COCA 데이터셋 구축 스크립트

`data/datasets/COCA/` 아래의 `COCA_1frame`·`COCA_3frames`·`COCA_3frames_5fold`·`Dataset001_COCA` 를 원본 DICOM/XML 에서 만들어낸 스크립트다. 학습·평가 경로에서는 호출되지 않는다.

**`build_5fold_dataset.py` 를 제외한 전부가 상대 경로로 `./COCA/COCA_final`, `./COCA_3frames`, `./Dataset001_COCA` 를 참조한다.** 스크립트 위치가 아니라 `data/datasets/COCA/` 를 cwd 로 잡고 실행해야 한다.

**원본 DICOM/XML 트리(`COCA/COCA_final`, `COCA/Gated_release_final`)는 이 저장소에 없다.** 이걸 읽는 6개(`xml_to_nii_label.py`, `xml_to_png_label.py`, `organize_dataset.py`, `organize_nnUNet_format.py`, `add_tag_value_to_dcm.py`, `analysis_dcm_metadata.py`)는 지금 상태로 실행할 수 없다. `Dataset001_COCA` 부터 읽는 `preprocess_train_test_data_{1,3}frame(s).py` 와 `build_5fold_dataset.py` 는 동작한다.

```bash
cd data/datasets/COCA && python ../../dataprep/preprocess_train_test_data_3frames.py
```

`build_5fold_dataset.py` 만 절대 경로(`SRC`/`OUT`)라 어디서 실행해도 된다.

대략적인 순서: `xml_to_nii_label.py` → `organize_dataset.py` → `organize_nnUNet_format.py` → `preprocess_train_test_data_{1,3}frame(s).py` → `build_5fold_dataset.py`.

`coca_data_error.txt` 는 451건 중 제외한 18건의 사유 기록이다 (Tag 누락, Z spacing 0.0, DCM 원본 부재, Unnamed ROI) — 최종 433건.
