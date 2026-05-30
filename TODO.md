# TODO — 5-Fold Stratified CV 전환 계획

## 0. 목표

기존 single hold-out (train 300 / test 133) 체제를 **433 case 통합 풀 위의 Stratified 5-fold Cross-Validation** 으로 전환한다. 각 fold 는 case 단위로 분할되며, multi-hot vessel presence (LCA/LAD/LCX/RCA) 를 층화 키로 사용한다. fold 별 validation 성능의 mean ± std 를 최종 보고 메트릭으로 사용한다.

---

## 1. 결정 사항

결정의 배경 트레이드오프는 §4 "함정 / 주의 사항" 에 반영되어 있다.

### 1.1 데이터 풀 구성 — 원본 nnUNet 볼륨에서 재생성 (rebuild)

- **기존 `COCA_3frames/{train_npz,test_npz}` 와 `COCA_1frame/` 은 그대로 보존.** 단일 hold-out 파이프라인은 손대지 않는다.
- **재생성 출처:** `Dataset001_COCA/{imagesTr,labelsTr,imagesVal,labelsVal}` 의 원본 `.nii.gz` per-case 볼륨. 5-fold 전용 새 디렉터리 `COCA_3frames_5fold/` 를 만든다.
- **왜 심볼릭 링크가 아니라 rebuild 였나 (Phase 1 에서 발견):**
  - `train_npz`(case0001~0300) 와 `test_npz`(case0001~0133) 가 **각각 case0001 부터 독립 번호** → 파일명 5,853개 충돌, case_id 합집합이 433 이 아니라 300 으로 붕괴.
  - 따라서 단순 심볼릭 링크/union 은 서로 다른 환자를 같은 case 로 병합한다 (불가).
  - 원본 nnUNet 파일명에 **전역 환자 인덱스**가 있음을 발견: `COCA_Tr_<gidx>_...` (train gidx 0~313), `COCA_Val_<gidx>_...` (test gidx 314~450). 교집합 0, 합집합 정확히 **433 unique**. 이것을 case_id 로 채택.
- **저장 포맷: per-case 볼륨 `.npy` (memmap).** case 당 `images/case{gidx:04d}.npy` `(D,H,W) float32` + `labels/case{gidx:04d}.npy` `(D,H,W) uint8`. 데이터로더가 `vol[n:n+3]` 로 prev/center/next triplet 을 on-the-fly 조립.
  - 기존 per-slice 3채널 NPZ 는 슬라이스를 prev/center/next 로 3번 중복 저장(67GB, 비압축). per-case 볼륨은 중복 제거로 **~27GB** (이미지 ~22GB + 라벨 ~5GB).
  - `.npy` 비압축 + `mmap_mode='r'` → 압축 해제 0, 부분 슬라이스 읽기, OS page cache 가 전체(27GB < 가용 RAM)를 캐싱 → epoch 1 이후 사실상 RAM 속도. (HDF5/압축 npz 대비 랜덤 접근 throughput 우위. 디스크 606GB 여유라 압축 불필요.)
  - 생성 데이터가 기존 npz 와 **바이트 동일**함을 검증 (`nii[:,:,n:n+3] == npz['image']`).

### 1.2 분할 단위 — case 단위 (절대 슬라이스 단위 아님)

- **2.5D 가정 보호를 위한 필수 규칙:** 한 case 에 속하는 모든 슬라이스는 반드시 같은 fold 에 통째로 들어가야 한다. 슬라이스 단위로 fold 를 나누면 인접 슬라이스가 train/val 양쪽에 등장해 prev/ref/next 픽셀이 누수된다.
- per-case 볼륨 `.npy` 1 파일 = 1 case 이므로 case 단위 분할이 구조적으로 보장된다.
- `case_id` = 전역 환자 인덱스 `case{gidx:04d}` (train 0~313, test 314~450, 결손 번호 = 제외된 18명). `case{gidx:04d}_slice{n:03d}` (n = triplet 시작 인덱스 0..D-3) 형식의 sample 이름은 `tester.py:parse_case_and_slice_id` 의 `_slice` 분리 규칙과 호환. 합집합 **433 unique case** 확인 완료.

### 1.3 층화 키 — case 별 4비트 Multi-hot Vessel Presence

- 각 case 가 어떤 vessel 에 석회화를 가지는지 `[LCA?, LAD?, LCX?, RCA?]` 의 0/1 벡터로 표현한다 (한 case 가 여러 vessel 에 동시 양성 가능 → multi-label).
- 이 벡터를 stratification 키로 사용해 5개 fold 간에 vessel 분포가 균형 잡히도록 분할.
- sklearn 기본 `StratifiedKFold` 는 multi-label 을 지원하지 않으므로, **`iterative-stratification` 패키지의 `MultilabelStratifiedKFold`** 사용.
- 호출: `MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

### 1.4 CV 구조 — Held-out Test 없는 Standard 5-Fold

- **목적:** 매 fold 가 한 번씩 평가 셋이 되고, 나머지 4개 fold 가 학습 셋이 된다. 별도 held-out test 셋을 두지 않는다 (nested 분할 없음, 433 case 를 모두 학습에 활용).
- **fold k 학습 시:**
  - Train = (fold 0..4) \ fold k 의 sample_name 합집합 (≈346 case, 약 16,600 슬라이스)
  - Validation = fold k (≈87 case, 약 4,150 슬라이스). 비율 train:val ≈ 4:1.
- **모델 선택 정책 (= 최종 보고 정책):**
  - 학습 중 매 epoch 끝에 validation loss 측정.
  - `val_loss` 가 최저로 갱신될 때마다 `best_model.pth` 저장 (= 현재 `trainer.py` 기존 로직).
  - 학습 종료 후 그 `best_model.pth` 를 같은 validation fold 에 평가한 메트릭을 **fold k 의 최종 성능**으로 사용.
  - 5개 fold 의 최종 성능 값을 모아 **mean ± std** 로 보고.
- **Early Stopping:**
  - `val_loss` 가 **patience=50 epoch** 동안 갱신되지 않으면 학습 중단.
  - `--max_epochs 300` 은 상한선으로만 작동. fold 마다 실제 종료 epoch 이 다를 수 있음.

### 1.5 CT 정규화 상수 — 전체 433 case 로 1회 산출, 모든 fold 공통

- 433 case voxel 분포에서 1회 산출한 4개 상수(`lower=15.0, upper=1577.0, mean=773.55, std=399.24`; lower/upper = 0.5%/99.5% 분위수, mean/std = clip 후)를 `hu_stats_433.json` 에 저장, `COCAVolumeDataset` 이 런타임에 읽어 `ct_normalization(image, **hu)` 로 전달.
- 기존 `ct_normalization` 하드코딩 기본값(train 300-case)은 **그대로 두어 단일 hold-out 과 분리**. 모든 fold 가 같은 상수 공유 (fold 별 재산정 안 함). 산출 근거·단일 hold-out 과의 차이(mean 355 vs 773, 심장 게이트 FOV) 상세는 `docs/DATA.md §4·§9`, `CLAUDE.md §6`.

### 1.6 재현성

- Seed = **42** (기존 그대로). fold 분할, augmentation, 가중치 초기화 모두 동일 seed 사용.
- `--deterministic 1` (기본값) 유지: `cudnn.benchmark=False, deterministic=True`.
- AMP / GradScaler 는 그대로 활성 (fp16 비결정성은 잔존, 완전 결정성 필요 시 별도 결정).

---

## 2. 작업 단계 (Phase 별 TODO)

### Phase 1 — 사전 검증 ✅ 완료 (→ rebuild 전환의 계기)

첫 점검에서 **case_id 충돌**(`train_npz`/`test_npz` 모두 case0001 부터 독립 번호 → 파일명 5,853 개 충돌, 합집합이 433 이 아닌 300)이 드러나 심볼릭 링크/union 이 불가함을 확인 → 전역 인덱스 **rebuild** 로 전환 (= "어쩔 수 없는 상황", 결정 배경 §1.1). rebuild 결과 **433 unique case**, sample 명명 `case{gidx}_slice{n}` 이 `tester.py:parse_case_and_slice_id` 와 호환, vessel 분포가 원고 Table 1 (LCA 149/LAD 382/LCX 246/RCA 256/Total 433) 과 일치함을 확인 (fold 별 분포 표는 `docs/DATA.md §9.2`).

### Phase 2 — 데이터 자산 생성 (단일 빌더, 저장소 외부 = `/home/psw/AVS-Diagnosis/COCA/`) ✅ 완료

**단일 빌더** `build_5fold_dataset.py` 가 `Dataset001_COCA` 에서 모든 자산을 1회 생성 (per-case `.npy` 볼륨 + `lists_COCA_5fold/` + `hu_stats_433.json` + `case_index.csv`). 산출물 레이아웃·포맷 상세는 `docs/DATA.md §9`. 빌더 핵심: nii → `(D,H,W)` transpose 후 `.npy`, HU 히스토그램 누적, case 별 vessel multi-hot → `MultilabelStratifiedKFold(5, shuffle=True, random_state=42)`.

- [x] `iterative-stratification` 설치, 빌더 작성·실행 (env `SAU-Net`).
- [x] **검증:** case 433, 슬라이스 합 20,790 (F0 4105/F1 4580/F2 4098/F3 3947/F4 4060), HU stats `15.0/1577.0/773.55/399.24` (§1.5), fold 별 vessel 분포 ±1 case 균형. 분포 표 상세는 `docs/DATA.md §9.2`.

### Phase 3 — 코드 수정 (저장소 내부) ✅ 완료

> 최소 변경 원칙. 신규 인자는 default 로 단일 hold-out 경로와 하위 호환. 변경 파일 목록은 §3, 동작 상세는 코드 + `docs/DATA.md §9` 가 권위본.

- [x] `datasets/__init__.py` 신규 — 빈 파일로 로컬 패키지가 HF `datasets`(4.5.0) shadowing 을 이김 (`import datasets.dataset` 복구, 단일 hold-out 경로도 함께 수리).
- [x] `datasets/dataset.py` — `load_hu_stats` + `COCAVolumeDataset`(per-case `.npy` memmap, `vol[n:n+3]`/`vol[n+1]`, `ct_normalization(**hu)`) 추가. `ct_normalization` 기본값·`COCA_dataset` 불변.
- [x] `train.py` — 5-fold argparse(`--use_5fold_cv/--fold_idx/--root_path_5fold/--list_dir_5fold/--hu_stats_path/--early_stopping_patience(=50)/--early_stopping_min_delta`) + fold 명명 경고.
- [x] `trainer.py` — fold 분기(train=fold_idx 제외 4개, val=fold_idx) + early stopping(patience). `DataLoader shuffle=False + collate_fn=shuffle_within_batch` 유지 (`CLAUDE.md §6`).
- [x] `test.py`/`tester.py` — 동일 5-fold 인자, `inference()` 가 val fold(`fold{idx}.txt`)를 `COCAVolumeDataset` 으로 로딩. 체크포인트·3D 평가 경로 불변.
- [x] `aggregate_5fold_results.py` 신규 — 5 fold `results.txt` 의 `[3D]` 라인 + best ckpt/log 를 파싱해 Run Summary + Dice/mIoU/HD (mean±std) Markdown 출력. CLI 는 `--exp_template ...{fold}...`.

### Phase 4 — 5-fold 본 실험 (페어별 진행)

> 실행 명령은 `docs/EXPERIMENTS.md §8`(5-fold 워크플로) 가 권위본. 여기엔 페어별 상태와 게이트 결과만 남긴다.
>
> 전체 그리드 = **4 encoder × 2 decoder × 5 fold = 40 trainings (MSFFM `_sa`)** + 동수 baseline (single_slice 브랜치 표준 encoder). PVTv2-b2 비교는 별도 브랜치 페어(`EMCAD-SA` / `EMCAD`). 페어 순서·게이트 위치는 §5.

#### Phase 4.1 — resnet 계열 (resnet50_sa main / resnet50 single_slice) ✅ 완료

- [x] **single_slice baseline (A) 5-fold 완료.** `results/baseline_resnet50_unet_seed42.md`. Mean Dice 0.5089±0.1003 (LCA 0 / LAD 0.659±0.330 / LCX 0.748±0.043 / RCA 0.629±0.316). LCA 5/5 fold = 0.0000, fold3 RCA 도 0.0000.
- [x] **MSFFM main 5-fold 완료.** `results/msffm_resnet50_unet_seed42.md`. Mean Dice 0.5045±0.0790 (LCA 0 / LAD **0.837±0.032** / LCX 0.412±0.339 / RCA **0.769±0.015**). **LAD/RCA 평균↑·std↓ 명확** (MSFFM 이 vessel continuity 안정화). 단 LCX fold3/4 가 0.0000 (baseline 엔 없던 패턴). 평균 mean Dice 는 baseline 과 동등 (LCX 손실이 LAD/RCA 이득 상쇄). baseline 산출물(`model/Unet_resnet50/`, `test_log/Unet_resnet50/`) main 디렉터리 복사 + single_slice worktree `git worktree remove --force` 완료. 명명·병렬 배치 상세는 `docs/EXPERIMENTS.md §3·§8.2`, branch map 은 `CLAUDE.md §10`.

#### Phase 4.2 — MSFFM 작동성 검증 게이트 ✅ 완료

- [x] resnet 페어 5-fold 종료 후 EMCAD 진입 전, MSFFM 이 2.5D 역할을 실제 수행하는지 그리드와 분리해 검증. `resnet50_sa`+Unet fold0 best(`epoch_104_0.0505`) 1개로만 진단 (`results.txt`/`model` 무접근, `normal` 모드 3D Dice 가 `results/msffm_resnet50_unet_seed42.md` fold0 과 일치 → 파이프라인 무결성 확인). 4종 증거:
  - **(1) Residual (DEBUG_RESIDUAL)** — ‖xt‖/‖x_main‖ stage3 0.2366·stage4 0.0538 → dead branch 아님. 진단 후 `resnet_sa.py` `DEBUG_RESIDUAL=False` 복구 (`CLAUDE.md §6`).
  - **(2) 채널 ablation (결정적)** — `zero_neighbors[0,ref,0]` 시 ΔLAD −0.0669(ΔMean −0.0081), `ref_only`≈0. baseline(single_slice)은 center 만 써 구조상 ΔDice≡0 → MSFFM ΔDice ≫ baseline.
  - **(3) per-class** — 5-fold 평균 LAD(+0.178)·RCA(+0.140) 압승(연속 혈관), LCA(짧고 흩어짐)는 우위 작음 — 예측 패턴 부합.
  - **(4) attention viz** — `test.py --save_attention` hook 키 버그(90f33a4 회귀: 모듈명 `cross_attention_*_N` 이 `visualize_attention` 기대 키 `stageN_*` 와 불일치) 수정 후 lesion query→prev/self/next attend 정상 확인. `test.py`/`tester.py` 패치 상세(키 정규화·uniform-대비 colorbar·표시 회전)는 `docs/ARCHITECTURE.md §5`.
- **판정:** 하드기준 (1)+(2) 충족 → PASS (inter-slice 의존은 modest·LAD 집중형).

#### Phase 4.3 — EMCAD 계열 (EMCAD-SA / EMCAD 브랜치 페어)

- [ ] pending — Phase 4.2 게이트 통과 후 진행. 별도 브랜치 페어 (PVTv2-b2 backbone). 진행 시 본 항목에 상태/메모 추가.

#### Phase 4.4 — densenet 계열 (densenet201_sa main / densenet201 single_slice)

- [ ] pending — Phase 4.3 후 진행.

#### Phase 4.5 — efficientnet 계열 (efficientnet-b4_sa main / efficientnet-b4 single_slice)

- [ ] pending — Phase 4.4 후 진행.

#### Phase 4.6 — mit 계열 (mit_b2_sa main / mit_b2 single_slice)

- [ ] pending — Phase 4.5 후 진행.

---

## 3. 변경 파일 체크리스트

| 파일 | 변경 | 위치 |
|---|---|---|
| `datasets/__init__.py` | 신규 (HF datasets shadowing 버그 수정) | 저장소 내부 |
| `datasets/dataset.py` | `load_hu_stats`, `COCAVolumeDataset` 추가 (`ct_normalization` 기본값 불변) | 저장소 내부 |
| `train.py` | argparse 5-fold 인자, fold 명명 경고 | 저장소 내부 |
| `test.py` / `tester.py` | argparse 5-fold 인자, `inference()` fold 분기 | 저장소 내부 |
| `trainer.py` | fold 분기 + early stopping | 저장소 내부 |
| `aggregate_5fold_results.py` | 신규 (`--results_dir` 기본 `./results`, gitignored) | 저장소 내부 |
| `results/*.md` | 산출물 (aggregate MD, gitignored) | 저장소 내부 |
| `.gitignore` | `results` 추가 | 저장소 내부 |
| `build_5fold_dataset.py` | 신규, 1회 실행 (rebuild) | `/home/psw/AVS-Diagnosis/COCA/` |
| `COCA_3frames_5fold/{images,labels}/*.npy` | 산출물, 신규 | `/home/psw/AVS-Diagnosis/COCA/` |
| `COCA_3frames_5fold/lists_COCA_5fold/`, `hu_stats_433.json`, `case_index.csv` | 산출물, 신규 | 〃 |
| `CLAUDE.md` / `docs/DATA.md` / `docs/EXPERIMENTS.md` | 5-fold 사용법 기재 | 저장소 내부 |

---

## 4. 함정 / 주의 사항

5-fold 전환 고유의 주의점만 남긴다. 일반 repo 함정(augmentation 분기, DataLoader `shuffle=False`+`collate_fn`, DataParallel, AMP/GradScaler, debug print 잔재)은 `CLAUDE.md §6`·`docs/DATA.md §5~§6`·`docs/EXPERIMENTS.md §5·§7` 가 권위본.

- **case_id 단위 분할 강제:** 슬라이스 단위 stratify 는 인접 슬라이스 누수로 2.5D 가정이 깨진다. fold 리스트 생성기는 반드시 case → fold → slice 순서로 동작해야 함.

---

## 5. 진행 순서 (의존성 그래프)

```
Phase 1 (검증)  ✅ case_id 충돌 발견 → 전역 인덱스 rebuild 로 전환
   ▼
Phase 2  build_5fold_dataset.py  ✅
   │   └─ COCA_3frames_5fold/{images,labels}/*.npy + lists_COCA_5fold/ + hu_stats_433.json
   ▼
Phase 3  코드 수정  ✅
   │   └─ datasets/__init__.py(버그수정) · COCAVolumeDataset · train/test/trainer fold 분기
   │      · early stopping · aggregate_5fold_results.py
   ▼
Phase 4.1  resnet 페어 (resnet50_sa main / resnet50 single_slice)  ✅
   ▼
Phase 4.2  MSFFM 작동성 검증 게이트  ✅
   │     └─ (DEBUG_RESIDUAL · attention vis · 채널 ablation · per-class)
   ▼
Phase 4.3  EMCAD 페어 (EMCAD-SA / EMCAD 브랜치)
   ▼
Phase 4.4  densenet 페어 (densenet201_sa / densenet201)
   ▼
Phase 4.5  efficientnet 페어 (efficientnet-b4_sa / efficientnet-b4)
   ▼
Phase 4.6  mit 페어 (mit_b2_sa / mit_b2)
```

