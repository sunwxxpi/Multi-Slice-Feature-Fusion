# Data — COCA 데이터셋 사용 규약

## 1. 원본 데이터셋

- **출처:** Stanford AIMI — COCA (Coronary Calcium and Chest CT) https://aimi.stanford.edu/datasets/coca-coronary-calcium-chest-ct
- **본 연구 사용 케이스:** 451 명 중 433 명 (train 300 / test 133, 원고 Table 1).
- **In-plane 해상도:** 512 × 512 고정. 본 코드도 `--img_size 512` 가 기본.
- **클래스 (5):** 0=background, 1=LCA, 2=LAD, 3=LCX, 4=RCA. `--num_classes 5`.
- 단일 케이스가 여러 카테고리에 동시 포함될 수 있다 (Table 1 의 Total 합계 ≠ unique 수).

## 2. 본 저장소가 기대하는 디스크 구조

데이터셋 루트는 본 코드 저장소 **밖** 에 있다.

```
/home/psw/AVS-Diagnosis/COCA/                    ← 데이터셋 루트 (사용자 로컬)
├── COCA_3frames/                                ← single hold-out 용 (기존, 그대로 보존)
│   ├── train_npz/<sample_name>.npz              ← (H, W, 3) image + (H, W) label
│   ├── test_npz/<sample_name>.npz
│   └── lists_COCA/                              ← single hold-out 분할
│       ├── train.txt                            ← 14507 lines (slice 단위)
│       └── test.txt                             ←  6283 lines
├── COCA_3frames_5fold/                          ← 5-fold CV 용 (신규, rebuild 산출물)
│   ├── images/case{0000..0450}.npy             ← per-case (D, H, W) float32 볼륨 (memmap)
│   ├── labels/case{0000..0450}.npy             ← per-case (D, H, W) uint8 라벨
│   ├── lists_COCA_5fold/                        ← case-level stratified 분할
│   │   ├── fold0.txt ~ fold4.txt                ← 한 줄당 case{gidx}_slice{n}
│   │   └── fold_assignment.csv                  ← case_id, fold, vessel multi-hot, slice_count
│   ├── hu_stats_433.json                        ← 433-case 정규화 상수 (lower/upper/mean/std)
│   └── case_index.csv                           ← case_id ↔ 원본 nnUNet 파일명, depth, n_samples
├── COCA_1frame/                                 ← 단일 슬라이스 변형 (참고용, 본 코드 미사용)
│   ├── train_npz/, test_vol_h5/, lists_COCA/
├── Dataset001_COCA/                             ← nnUNet 포맷 원본 (COCA_3frames* 의 출처)
│   ├── imagesTr/, imagesVal/, labelsTr/, labelsVal/, dataset.json
└── *.py                                         ← 전처리/조직화 스크립트
    ├── preprocess_train_test_data_3frames.py    ← COCA_3frames 생성 스크립트
    ├── preprocess_train_test_data_1frame.py
    ├── organize_dataset.py / organize_nnUNet_format.py
    ├── xml_to_nii_label.py / xml_to_png_label.py
    ├── add_tag_value_to_dcm.py / analysis_dcm_metadata.py
    └── build_5fold_dataset.py                   ← COCA_3frames_5fold 전체 1회 생성 (rebuild)
```

`train.py` / `test.py` argparse 기본값:
- `root_path = /home/psw/AVS-Diagnosis/COCA/COCA_3frames/train_npz` (학습)
- `root_path = /home/psw/AVS-Diagnosis/COCA/COCA_3frames/test_npz`  (평가)
- `list_dir  = /home/psw/AVS-Diagnosis/COCA/COCA_3frames/lists_COCA`

다른 환경에서 실행할 때는 반드시 `--root_path` / `--list_dir` 를 지정. `train.txt` 는 학습/검증을 80:20 으로 분할 (`sklearn.model_selection.train_test_split`, `shuffle=False`, seed 42).

NPZ 가 손상되거나 새 코호트가 들어왔다면 `preprocess_train_test_data_3frames.py` 가 원본 DICOM/XML 로부터 `(H, W, 3)` 슬라이스 묶음과 list 파일을 재생성하는 정식 경로다 — 단, 해당 스크립트는 본 저장소가 아니라 위 데이터셋 루트에 함께 있는 사용자 로컬 자산이다.

## 3. NPZ 파일 포맷

`np.load(<sample>.npz)` 결과:
- `data['image']` — shape `(H, W, 3)`, dtype float (HU 값). 채널 0/1/2 = prev / center / next 슬라이스.
- `data['label']` — shape `(H, W)`, dtype int. center 슬라이스의 vessel-wise 마스크.
- `sample_name` 은 보통 `<case_id>_slice<NN>` 형식이며 `tester.py:parse_case_and_slice_id` 가 `case_id` 와 `slice_id`(int) 로 분리해 3D 볼륨 합성 시 사용한다.

## 4. CT 정규화 (`datasets/dataset.py:ct_normalization`)

```python
ct_normalization(image, lower=-2.0, upper=1521.0, mean=355.3804..., std=282.9181...)
```

- `np.clip` 으로 HU 범위 제한 후 z-score.
- 상수는 COCA train set 의 분포에서 도출. 다른 코호트(KMU 등) 에 적용 시 반드시 재산정.
- 이전 커밋 `ace4439` 에서 갱신된 값이며, 더 오래된 주석(`lower=1017, upper=1801, ...`) 은 옛 단위 계의 값이므로 사용 금지 — 그대로 두되 켜지 말 것.
- **5-fold CV 는 별도 상수를 쓴다 (`TODO.md` §1.5):** 위 기본 인자(train 300-case)는 **변경하지 않고**, 433 case 전체 풀로 산출한 `hu_stats_433.json` 을 `load_hu_stats` 로 읽어 `COCAVolumeDataset` 이 `ct_normalization(image, **hu)` 로 명시 전달한다. 실제 산출값: `lower=15.0, upper=1577.0, mean=773.55, std=399.24` (`lower/upper` = 0.5% / 99.5% 분위수, `mean/std` = clip 후). 모든 fold 가 같은 4개 상수 공유 (fold 별 재산정 안 함). 심장 게이트 CT 라 FOV 가 좁아 폐/공기(<-300HU) 비율 0.2% → 0.5% 분위수가 15 로 높고 median≈967. 단일 hold-out 의 mean=355 와 정규화 자체가 다르므로 두 체제의 절대 메트릭 직접 비교 금지.

## 5. Augmentation 정책

- `RandomAugmentation` 은 50% 확률로 `random_rot_flip` (rot90 0~3회 + flip), 50% 확률로 `random_rotate` (±20° 일양 분포) 를 적용. 두 변환 모두 채널 축을 보존한다.
- 원고(Methods §2.3) 에는 `±10°` 회전이라고 명시되어 있으나 코드는 `±20°`. 변경 의도가 있다면 사용자에게 확인할 것.
- 회전 시 `order=0`(label), `order=3`(image) — label 의 정수성을 보존한다.

## 6. DataLoader 설정

`trainer.py` 가 다음과 같이 구성한다:

```python
DataLoader(db_train, batch_size=16, shuffle=False, num_workers=8,
           pin_memory=True, worker_init_fn=...,
           collate_fn=shuffle_within_batch)
```

- **shuffle=False + collate_fn shuffle** 패턴은 동일 배치 내에서만 셔플하는 의도. 인접 슬라이스의 페어링(prev/ref/next) 이 npz 단계에서 이미 고정돼 있으므로, 데이터셋 차원 셔플은 무의미하며 단지 배치 구성을 다양화하기 위함이다.
- `num_workers=8` (train), `4` (val), `1` (test). 메모리 부족 시 환경 변수가 아니라 코드를 수정해야 한다.

## 7. 평가 시 3D 재구성

`tester.py` 흐름:
1. test loader 가 한 슬라이스씩 `(image, label, case_name)` 을 내보낸다.
2. `case_id, slice_id = parse_case_and_slice_id(case_name)` 로 분리.
3. case 별로 dict 에 쌓고, 모든 케이스 추론이 끝난 후 `build_3d_volume` 가 `(D, H, W)` 볼륨을 합성한다. `D = max_z - min_z + 1` 이므로 슬라이스 누락이 있으면 0 으로 채워진다는 점에 유의.
4. NIfTI 저장 시 `SetSpacing((0.375, 0.375, z_spacing))`. `--z_spacing` (기본 3) 가 실제 데이터의 슬라이스 두께와 다르면 거리 메트릭(`SurfaceDistance`, `HausdorffDistance`)이 왜곡된다.

## 8. 새 데이터셋 추가 시 체크리스트

- [ ] `(H, W, 3)` 형태로 prev/center/next 가 미리 묶인 NPZ 를 생성.
- [ ] `list_dir/train.txt`, `test.txt` 갱신. 한 줄당 sample_name (확장자 없음).
- [ ] `ct_normalization` 의 lower/upper/mean/std 를 새 데이터로 재계산해 적용.
- [ ] `--num_classes`, `DiceLoss` 의 배경 제외 가정, `class_dice_means` 의 인덱싱을 함께 점검.
- [ ] `tester.py` 의 spacing 상수와 `parse_case_and_slice_id` 의 파싱 규칙 호환성 확인.

## 9. 5-Fold Cross-Validation 자산

본 절은 `--use_5fold_cv` 모드에서 추가로 필요한 데이터 자산의 포맷을 정리한다. 결정 배경과 생성 파이프라인 전체는 `TODO.md` §1, §2 참고.

모든 자산은 `Dataset001_COCA` 원본에서 `build_5fold_dataset.py` 가 1회 생성한다 (rebuild). case_id = 원본 nnUNet **전역 인덱스** (`COCA_Tr_<gidx>_...`→`case{gidx:04d}`, train 0~313; `COCA_Val_<gidx>_...`, test 314~450). 결손 번호는 제외된 18명. → 433 unique case.

### 9.1 `images/`, `labels/` — per-case 볼륨 `.npy`
- `images/case{gidx}.npy` = `(D, H, W) float32` HU 볼륨, `labels/case{gidx}.npy` = `(D, H, W) uint8`.
- `(D,H,W)` C-contiguous 라 `vol[n:n+3]` 가 연속 평면 → `mmap_mode='r'` 부분읽기에 최적. per-slice 3채널 NPZ(67GB, 3× 중복) 대비 중복 제거로 ~27GB.
- 데이터 내용은 기존 npz 와 바이트 동일 (`nii[:,:,n:n+3] == npz['image']` 검증).

### 9.2 `lists_COCA_5fold/` — Fold 분할 리스트
- `fold0.txt ~ fold4.txt`, 한 줄당 `case{gidx}_slice{n}` (n = triplet 시작 인덱스 0..D-3, case 당 D-2 개). 합 20,790 슬라이스 (fold별 F0 4105/F1 4580/F2 4098/F3 3947/F4 4060).
- **분할 단위는 case 단위** (1 파일 = 1 case 라 구조적으로 보장). 슬라이스 단위 분할은 prev/ref/next 픽셀 누수를 일으키므로 금지.
- **층화 키:** case 별 `[LCA, LAD, LCX, RCA]` 4비트 multi-hot. **API:** `MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)` (`iterative-stratification`). fold 별 vessel 분포는 ±1 case 로 균형 (`TODO.md` §2).
- `fold_assignment.csv` 컬럼: `case_id, fold, label_LCA, label_LAD, label_LCX, label_RCA, slice_count`.

### 9.3 `hu_stats_433.json` — 정규화 상수
- 433 case 전체 voxel 분포에서 히스토그램 1-pass 로 산출. 키: `lower`(0.5%), `upper`(99.5%), `mean`, `std`(clip 후) + 분위수/메타. 실제값 `15.0 / 1577.0 / 773.55 / 399.24`.
- `ct_normalization` 의 **기본 인자를 바꾸지 않고**, `load_hu_stats(path)` 로 읽어 `COCAVolumeDataset` 이 명시 인자로 전달 (단일 hold-out 경로와 분리). 모든 fold 공유.

### 9.4 `case_index.csv` — 추적 매핑
- 컬럼: `case_id, origin(train/test), src_basename(원본 nnUNet), depth, n_samples`. 디버깅/추적용.

### 9.5 학습 시 dataset 인스턴스화 (`COCAVolumeDataset`)
- `db_train` = `fold_idx` 제외 4개 fold sample 합집합 (≈346 case, ~16,685 슬라이스), `db_val` = `fold{fold_idx}.txt` (≈87 case, ~4,105 슬라이스). train:val ≈ 4:1.
- `COCAVolumeDataset` 가 case 볼륨을 memmap 으로 lazy 로드(case 별 캐시)해 `vol[n:n+3]`→`(H,W,3)`
  + `vol[n+1]` center label 조립, `ct_normalization(**hu)` 적용. `db_val` 은 augmentation 비활성. DataLoader 의 `shuffle=False + collate_fn=shuffle_within_batch` 패턴 유지 (CLAUDE.md §6).
