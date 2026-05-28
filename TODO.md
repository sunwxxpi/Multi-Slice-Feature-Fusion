# TODO — 5-Fold Stratified CV 전환 계획

## 0. 목표

기존 single hold-out (train 300 / test 133) 체제를 **433 case 통합 풀 위의 Stratified 5-fold Cross-Validation** 으로 전환한다. 각 fold 는 case 단위로 분할되며, multi-hot vessel presence (LCA/LAD/LCX/RCA) 를 층화 키로 사용한다. fold 별 validation 성능의 mean ± std 를 최종 보고 메트릭으로 사용한다.

---

## 1. 결정 사항

결정의 배경 트레이드오프는 §4 "함정 / 주의 사항"과 §5 "원고 보고 문구" 에 반영되어 있다.

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
  - 같은 셋이 best epoch 선택과 최종 보고에 모두 쓰이므로 미세한 optimism 이 있음. → Methods 에 솔직히 명시 (§5 참고).
- **Early Stopping:**
  - `val_loss` 가 **patience=50 epoch** 동안 갱신되지 않으면 학습 중단.
  - `--max_epochs 300` 은 상한선으로만 작동. fold 마다 실제 종료 epoch 이 다를 수 있음.

### 1.5 CT 정규화 상수 — 전체 433 case 로 1회 산출, 모든 fold 공통

- 433 case voxel 분포에서 1회 산출한 4개 상수(`lower=15.0, upper=1577.0, mean=773.55, std=399.24`; lower/upper = 0.5%/99.5% 분위수, mean/std = clip 후)를 `hu_stats_433.json` 에 저장, `COCAVolumeDataset` 이 런타임에 읽어 `ct_normalization(image, **hu)` 로 전달.
- 기존 `ct_normalization` 하드코딩 기본값(train 300-case)은 **그대로 두어 단일 hold-out 과 분리**. 모든 fold 가 같은 상수 공유 (fold 별 재산정 안 함). 산출 근거·단일 hold-out 과의 차이(mean 355 vs 773, 심장 게이트 FOV) 상세는 `docs/DATA.md §4·§9`, `CLAUDE.md §6`.
- validation fold 의 HU 분포가 상수에 미세하게 녹아드는 약한 leakage 는 Methods 에 명시 (§5).

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

### Phase 4 — 파이프라인 검증 (Smoke ✅ / 본 실험 ⏳)

> 실행 명령은 `docs/EXPERIMENTS.md §8`(5-fold 워크플로) 가 권위본. 여기엔 상태와 확인된 사실만 남긴다.

- [x] **Smoke test (단일 fold) — 종단간 정상.** `--fold_idx 0 --max_epochs 2` 로 경로만 검증. 확인된 것: train 16,685 / val 4,105 fold 분할, best_model 저장·교체, test 4,105 iter → `results.txt` 의 3D 메트릭 생성(2 epoch 라 Dice≈0 정상), `aggregate_5fold_results.py` 가 결과를 Markdown 으로 파싱 (exit 0). smoke 산출물은 점검 후 정리, 덮어쓰인 `*_model_summary.txt` 는 원복.
- [x] **Early stopping 코드 경로 — smoke 에서 무에러 실행 확인.** 매 epoch best 비교 + 카운터 분기(`patience>0 && counter>=patience` → epoch 루프 break) 정상 실행 (2 epoch 라 실제 break 미발생). 본 실험은 `patience=50` 운용. 필요 시 `--early_stopping_patience 5` 로 강제 break 별도 확인 가능 (미수행).
- [ ] **본 실험 — main 5-fold 그리드** (⏳ 진행 중, resnet 페어 5-fold 우선). 이 브랜치 실행 가능 = **4 encoder(`resnet50_sa/densenet201_sa/efficientnet-b4_sa/mit_b2_sa`) × 2 decoder(`unet/segformer`) = 8 config**, 각 fold0~4 → **40 trainings**. 이어서 평가 8 config × 5 + config별 `aggregate_5fold_results.py`. 명명·병렬 배치는 `docs/EXPERIMENTS.md §3·§8.2`. baseline(비-MSFFM) 은 별도 `single_slice` 브랜치에서 **페어로 병렬 운영** (현 워크플로: main `_sa` 그리드 ↔ single_slice 표준 encoder 그리드 — encoder 별로 동시에 학습해 MSFFM 기여를 직접 대조). PVTv2-b2 비교는 별도 브랜치 페어(`EMCAD-SA` / `EMCAD`) 에서 5-fold 로 운영한다. 페어 진행 순서: **resnet → EMCAD → densenet → efficientnet → mit** (§6 의존성 그래프). 현재 GPU0 main `resnet50_sa` × unet × fold0~4 + GPU1 single_slice 브랜치 worktree(`/home/psw/SAU-Net-single_slice`) `resnet50` × unet × fold0~4 가 병렬로 진행 중. baseline 산출물(`model/Unet_resnet50/`, `test_log/Unet_resnet50/`)은 평가·집계 후 main 디렉터리로 복사, single_slice worktree 는 `git worktree remove --force` 로 정리 (post 스크립트 자동화).
- [ ] **Phase 4.4 — MSFFM 작동성 검증 게이트 (resnet 페어 완료 직후, densenet 진행 전 필수).** 현 학습 그리드의 resnet 짝(`resnet50_sa` main / `resnet50` single_slice) 5-fold 학습·평가·집계가 끝난 시점에, densenet 페어 진입 전에 MSFFM 이 의도대로 2.5D 역할을 수행 중인지 확인한다. 학습 그리드 자체와 분리된 진단이므로 별도 exp_setting / fold0 best ckpt 1개로 수행 (현 그리드 산출물 무오염).
  - **DEBUG_RESIDUAL 진단** — `segmentation_models_pytorch/encoders/resnet_sa.py` 의 모듈 상수 `DEBUG_RESIDUAL = True` 로 켜고 fold0 best ckpt 로 짧은 inference 1회 (소수 batch). 출력되는 `Residual branch mean abs value` 로 MSFFM 가 ref feature 를 실질 변형 중인지(≈0 이면 dead branch) 정량 확인. 끝난 뒤 `False` 로 복구 (`CLAUDE.md §6` foot-gun).
  - **Attention 시각화** — `docs/ARCHITECTURE.md §5` 절차: `test.py --save_attention --is_savenii` 로 inference 1회 (fold0 best ckpt). 플래그가 모든 `NonLocalBlock` 에서 `return_attention=True` 자동 토글 + hook 등록 + 시각화 저장을 한 번에 수행. lesion query 가 prev/next 의 어디를 보는지 히트맵을 `test_save_path/attention_vis/` 에서 확인.
  - **채널 ablation (가장 결정적 증거)** — inference 시 input `[prev, ref, next]` 을 `[ref, ref, ref]` / `[zeros, ref, zeros]` / 정상 의 3 setting 으로 비교 (dataset.py `__getitem__` 또는 임시 wrapper 로 1줄 패치). MSFFM 만 ΔDice 큼 → 2.5D 신호 실제 사용 확정. baseline(`single_slice`)은 차이 없어야 정상 (center 만 사용).
  - **Per-class 우위 패턴 확인** — fold0~4 평균에서 vessel continuity 높은 LAD/RCA 에서 MSFFM 우위가 큰지, LCA(짧고 흩어진 클래스)는 우위 작은지 점검. interim test 에서 이미 LAD 에서 MSFFM 강세 패턴 관측됨 — 5-fold 평균으로 결정적 확인.
  - **통과 기준:** 최소 (DEBUG_RESIDUAL > 0 의미 있는 값) + (채널 ablation 에서 MSFFM ΔDice ≫ baseline ΔDice) 두 가지 충족 → **EMCAD 페어 (`EMCAD-SA` / `EMCAD` 브랜치 5-fold)** 진행 → 이후 densenet 페어 (`densenet201_sa` main / `densenet201` single_slice). 실패 시 그리드 중단하고 MSFFM 구현/통합부터 점검.

---

## 3. 변경 파일 체크리스트

| 파일 | 변경 | 위치 |
|---|---|---|
| `datasets/__init__.py` | 신규 (HF datasets shadowing 버그 수정) | 저장소 내부 |
| `datasets/dataset.py` | `load_hu_stats`, `COCAVolumeDataset` 추가 (`ct_normalization` 기본값 불변) | 저장소 내부 |
| `train.py` | argparse 5-fold 인자, fold 명명 경고 | 저장소 내부 |
| `test.py` / `tester.py` | argparse 5-fold 인자, `inference()` fold 분기 | 저장소 내부 |
| `trainer.py` | fold 분기 + early stopping | 저장소 내부 |
| `aggregate_5fold_results.py` | 신규 | 저장소 내부 |
| `build_5fold_dataset.py` | 신규, 1회 실행 (rebuild) | `/home/psw/AVS-Diagnosis/COCA/` |
| `COCA_3frames_5fold/{images,labels}/*.npy` | 산출물, 신규 | `/home/psw/AVS-Diagnosis/COCA/` |
| `COCA_3frames_5fold/lists_COCA_5fold/`, `hu_stats_433.json`, `case_index.csv` | 산출물, 신규 | 〃 |
| `CLAUDE.md` / `docs/DATA.md` / `docs/EXPERIMENTS.md` | 5-fold 사용법 기재 | 저장소 내부 |

---

## 4. 함정 / 주의 사항

5-fold 전환 고유의 주의점만 남긴다. 일반 repo 함정(augmentation 분기, DataLoader `shuffle=False`+`collate_fn`, DataParallel, AMP/GradScaler, debug print 잔재)은 `CLAUDE.md §6`·`docs/DATA.md §5~§6`·`docs/EXPERIMENTS.md §5·§7` 가 권위본.

- **case_id 단위 분할 강제:** 슬라이스 단위 stratify 는 인접 슬라이스 누수로 2.5D 가정이 깨진다. fold 리스트 생성기는 반드시 case → fold → slice 순서로 동작해야 함.
- **모델 선택 optimism 명시:** val 셋이 best epoch 선택과 최종 보고를 동시에 수행함을 Methods 에 한 줄로 적기.
- **HU 정규화 누수 명시:** 정규화 상수가 전체 433 case 분포에서 산출됨을 솔직히 기술.

---

## 5. 원고 보고 문구 (Methods 에 삽입할 초안)

> "We pooled the original training (n=300) and test (n=133) sets into a single cohort of 433 cases and applied a 5-fold stratified cross-validation at the case level. Stratification used the multi-hot vector of vessel presence (LCA/LAD/LCX/RCA) as the label, with `MultilabelStratifiedKFold` (random_state=42). For each fold k, models were trained on the remaining four folds (4:1 train-to-validation ratio) with model selection (lowest validation loss) and early stopping (patience=50 epochs). HU intensity normalization statistics (lower/upper clipping bounds, mean, standard deviation) were computed once on the entire 433-case pool prior to the CV split. The validation fold served as both the model selection criterion and the final performance evaluation set; final metrics are reported as mean ± standard deviation across the five folds."

---

## 6. 진행 순서 (의존성 그래프)

```
Phase 1 (검증)  ✅ case_id 충돌 발견 → 전역 인덱스 rebuild 로 전환
   │
   ▼
Phase 2  build_5fold_dataset.py  ✅
   │   └─ COCA_3frames_5fold/{images,labels}/*.npy + lists_COCA_5fold/ + hu_stats_433.json
   ▼
Phase 3  코드 수정  ✅
   │   └─ datasets/__init__.py(버그수정) · COCAVolumeDataset · train/test/trainer fold 분기
   │      · early stopping · aggregate_5fold_results.py
   ▼
Phase 4.1  Smoke test (fold0, 2 epoch)  ✅ 종단간 검증 완료 (산출물 정리됨)
   │
   └──► Phase 4.3  본 실험 main 8-config × 5 (40 trainings)  →  aggregate_5fold_results.py
        ⏳ 진행 중: resnet 페어 (resnet_sa main / resnet single_slice) 5-fold 학습+평가+집계
           │
           └──► Phase 4.4  MSFFM 작동성 검증 게이트
                  (DEBUG_RESIDUAL · attention vis · 채널 ablation · per-class)
                  │
                  ├─ 통과 → EMCAD 페어 (EMCAD-SA/EMCAD 브랜치) → densenet 페어 → efficientnet 페어 → mit 페어
                  └─ 실패 → 그리드 중단, MSFFM 구현/통합 점검 후 재개
```

