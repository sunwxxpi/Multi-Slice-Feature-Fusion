# Experiments — 학습·평가·파인튜닝 운영 가이드

## 1. 표준 학습 명령

### 1.1 Single hold-out (기존 기본 경로)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset COCA \
  --root_path /path/to/COCA_3frames/train_npz \
  --list_dir  /path/to/COCA_3frames/lists_COCA \
  --encoder   resnet50_sa \
  --decoder   unet \
  --num_classes 5 \
  --img_size  512 \
  --max_epochs 300 \
  --batch_size 16 \
  --base_lr   1e-5 \
  --exp_setting review_msffm_resnet50_unet_fold0_seed42
```

- 모든 명령은 `CUDA_VISIBLE_DEVICES=0` 로 GPU 1개에 핀한다 — 미지정 시 다중 GPU 환경에서 DataParallel 이 보이는 GPU 를 전부 잡는다 (§5).
- 인자 기본값은 원고 §2.3 의 학습 설정과 일치한다 (AdamW, lr=1e-5, batch=16, 300 epochs, PolyLR, MHA heads=8).
- 학습이 시작되면 `./{NetClass}_{encoder}_model_summary.txt` 로 `torchinfo.summary` 결과가 덮어쓰기 저장된다 (저장소 루트). 모델 구조 비교용.

### 1.2 5-Fold Stratified CV (`TODO.md` §1 의 권장 워크플로)

§1.1 과 동일하되 hold-out 인자(`--root_path`/`--list_dir`) 대신 5-fold 인자를 쓴다. 공통 인자(`--dataset/--num_classes/--img_size/--batch_size/--base_lr`)는 기본값이라 생략 가능.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --use_5fold_cv --fold_idx 0 \
  --encoder resnet50_sa --decoder unet \
  --max_epochs 300 --early_stopping_patience 50 --early_stopping_min_delta 0.0 \
  --exp_setting review_5fold_msffm_resnet50_unet_fold0_seed42
```

- `--fold_idx` 를 0~4 로 바꿔가며 총 5회 학습. exp_setting 의 `fold{K}` 부분도 함께 바꿔야 함 (불일치 시 경고만 출력되고 학습은 진행).
- `--use_5fold_cv` 일 때 `--root_path` / `--list_dir` 는 무시되고 `--root_path_5fold` (기본 `COCA_3frames_5fold`), `--list_dir_5fold`, `--hu_stats_path` 가 쓰인다 (모두 기본값이 박혀 있어 보통 생략 가능).
- `--max_epochs 300` 은 상한선. early stopping (patience=50) 이 fold 별로 실제 종료 epoch 을 결정.
- 자세한 결정 배경, 분할 키, 정규화 상수 정책은 `TODO.md` §1.

## 2. 표준 평가 명령

### 2.1 Single hold-out

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset COCA \
  --root_path /path/to/COCA_3frames/test_npz \
  --list_dir  /path/to/COCA_3frames/lists_COCA \
  --encoder   resnet50_sa \
  --decoder   unet \
  --exp_setting review_msffm_resnet50_unet_fold0_seed42 \
  --is_savenii      # NIfTI 출력이 필요할 때만
```

- `--exp_setting`, `--max_epochs`, `--batch_size`, `--base_lr`, `--img_size` 가 학습 때와 동일해야 체크포인트 경로가 매칭된다. 불일치 시 `FileNotFoundError` 발생.
- 결과는 `./test_log/{NetClass}_{encoder}/COCA_512/{exp_setting}/epo300_bs16_lr1e-05/results.txt` 에 누적된다. `is_savenii` 가 켜져 있으면 같은 디렉터리에 `results_nii/` 도 생성.

### 2.2 5-Fold CV

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset COCA \
  --use_5fold_cv --fold_idx 0 \
  --encoder   resnet50_sa \
  --decoder   unet \
  --exp_setting review_5fold_msffm_resnet50_unet_fold0_seed42 \
  [--is_savenii]
```

- 평가 셋 = `fold{fold_idx}.txt` (= 학습 때의 validation fold). hold-out test 셋은 없음.
- `--root_path_5fold` / `--list_dir_5fold` / `--hu_stats_path` 기본값 사용 (보통 생략).
- 5 fold 결과를 모두 모아 `aggregate_5fold_results.py --exp_template ...` 로 mean ± std 보고 (§8 참고).

## 3. exp_setting 명명 규약 (관찰된 패턴)

`model/Unet_resnet50_sa/COCA_512/` 아래에 다음과 같은 디렉터리가 이미 존재한다:

| 이름 | 의미 |
|---|---|
| `default` | 초기 베이스라인 실행 |
| `kmu_chest` | KMU Chest 코호트로 파인튜닝/평가 한 실험 |
| `review_msffm_resnet50_unet_fold0_seed42` | 리뷰어 응답용 main run (seed 42, fold 0) |

새 실험을 만들 때는 이 패턴(`review_<설정>_fold{N}_seed{S}`) 을 따른다.

**5-fold CV 패턴** (`TODO.md` §2 Phase 4).

**Main 그리드 — 이 브랜치 실행 가능 8 config (4 encoder × 2 decoder, 전부 +MSFFM `_sa`):** 명명 규약 `review_5fold_msffm_{encoder}_{decoder}_fold{k}_seed42` (encoder 라벨은 `_sa` 생략).

| encoder (argparse 키) | decoder=unet | decoder=segformer |
|---|---|---|
| `resnet50_sa` | `review_5fold_msffm_resnet50_unet_fold{0..4}_seed42` | `review_5fold_msffm_resnet50_segformer_fold{0..4}_seed42` |
| `densenet201_sa` | `review_5fold_msffm_densenet201_unet_fold{0..4}_seed42` | `review_5fold_msffm_densenet201_segformer_fold{0..4}_seed42` |
| `efficientnet-b4_sa` | `review_5fold_msffm_efficientnet-b4_unet_fold{0..4}_seed42` | `review_5fold_msffm_efficientnet-b4_segformer_fold{0..4}_seed42` |
| `mit_b2_sa` | `review_5fold_msffm_mit_b2_unet_fold{0..4}_seed42` | `review_5fold_msffm_mit_b2_segformer_fold{0..4}_seed42` |

→ main = **8 config × 5 fold = 40 trainings**. baseline(비-MSFFM)·PVTv2-b2·ablation 은 본 브랜치 범위에서 제외 (다른 브랜치/추후 추가).

## 4. 파인튜닝 워크플로 (`--enable_finetuning`)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --encoder resnet50_sa --decoder unet \
  --exp_setting review_msffm_resnet50_unet_fold0_seed42 \
  --enable_finetuning \
  --finetune_exp_setting kmu_chest_finetune
```

- `--exp_setting` 의 디렉터리에서 `*best_model.pth` 를 찾아 로드한다 (없으면 `FileNotFoundError`).
- `segmentation_head.*` 키만 체크포인트에서 제거된 뒤 `load_state_dict(..., strict=False)` 호출 → 클래스 수가 달라져도 인코더·디코더는 그대로 활용된다.
- 새 학습 결과는 `--finetune_exp_setting` 디렉터리에 저장. 원본 best 는 보존된다.
- 학습 자체는 동일한 trainer 가 처음부터 돌리는 형태이므로 LR 스케줄이 리셋된다. 짧은 fine-tune 이 필요하면 `--max_epochs` 도 함께 줄일 것.

## 5. GPU 핀 — 한 학습 = 한 GPU (중요)

**한 학습(run)은 반드시 단일 GPU 로만 돌린다.** `trainer.py:77-78` 이 `torch.cuda.device_count() > 1` 이면 자동으로 `nn.DataParallel` 로 보이는 GPU 를 전부 잡으므로, 매 실행에 **`CUDA_VISIBLE_DEVICES=0` 또는 `=1` 을 명시**해 GPU 를 1개로 핀할 것 (`device_count()==1` 이 되어 DataParallel 미적용). 두 GPU 가 모두 비어 있으면 `=0`/`=1` 로 **서로 다른 실험을 GPU 별로 동시에** 돌려 처리량을 2배로 올릴 수 있다 (예: fold0→GPU0, fold1→GPU1). main 만 40 trainings 규모라 이 병렬화가 전체 소요 시간을 크게 줄인다.

저장 시에는 (DataParallel 인 경우) `model.module.state_dict()` 로 복원되므로 평가 단계의 단일 GPU 로딩이 호환된다. **`DistributedDataParallel` 은 지원하지 않는다** — 도입하려면 trainer 전체를 다시 써야 한다.

## 6. 결과 해석

`test.py` 가 남기는 `results.txt` 의 한 케이스 블록 예:

```
[HH:MM:SS] Metrics for Case: <case_id>
[HH:MM:SS]   Class 1: Dice: 0.7234, mIoU: 0.6011, HD: 12.34 (GT>0 & Pred>0)
[HH:MM:SS]   Class 2: Dice: 0.0000, mIoU: 0.0000, HD: nan   (GT==0 & Pred==0)
...
[HH:MM:SS]   [Case <case_id>] - Mean Dice: 0.6543, Mean mIoU: 0.5410, Mean HD: 18.20
```

마지막에 전체 케이스 평균이 다음 형식으로 찍힌다:

```
Overall 3D Metrics Across All Cases:
  [3D] Class 1 - Dice: ..., mIoU: ..., HD: ...
  ...
  [3D] Testing Performance - Mean Dice: ..., Mean mIoU: ..., Mean HD: ...
```

- `(GT==0 & Pred>0)` 의 HD 는 NaN 으로 기록되고 평균에서 `np.nanmean` 으로 자연 제외된다.
- `(GT>0 & Pred==0)` 는 3D 공간 대각선 길이를 max HD 로 부과한다 (큰 패널티). 새 메트릭 도입 시 같은 처리를 따를 것.

## 7. 재현성

- `--seed 42` 가 기본. `random / numpy / torch / torch.cuda` 를 모두 동일 시드로 고정한다.
- `--deterministic 1` (기본) → `cudnn.benchmark=False, deterministic=True`. 속도 < 재현성 우선.
- AMP (autocast + GradScaler) 가 항상 활성화돼 있으므로 fp16 비결정성이 약간 남는다. 완전 결정 학습이 필요하면 `trainer.py` 의 `autocast()` / `GradScaler` 를 제거해야 한다.
- **5-fold CV:** `MultilabelStratifiedKFold(random_state=42)` 로 fold 분할도 동일 seed 에 고정. fold 리스트 파일이 한 번 생성되면 그 자체가 재현성의 닻 — 같은 `lists_COCA_5fold/` 를 쓰면 fold 배정이 보존된다. fold 리스트 재생성 시 seed 를 바꾸지 말 것.

## 8. 5-Fold CV 워크플로

`TODO.md` §2 Phase 4 의 운영 측면을 압축한다.

### 8.1 실행 순서 정책

1. **Phase 1~3 완료 확인** — `COCA_3frames_5fold/`(images/labels/lists/hu_stats) 산출 + 코드 수정(`--use_5fold_cv` 분기, `datasets/__init__.py`) 모두 끝나 있어야 함 (`TODO.md` §2~§3).
2. **Smoke test** — `--fold_idx 0 --max_epochs 2 --early_stopping_patience 0` 로 짧게 1회 학습+평가가 종단간 도는지 확인 (체크포인트 저장 + `results.txt` 생성).
3. **Main 5-fold 그리드** — 8 config(4 enc × 2 dec) × fold0~4 = 40 trainings → 평가 40 → config별 `aggregate_5fold_results.py`. 두 GPU 에 4 config 씩 나눠 병렬(§8.2).

### 8.2 일괄 실행 스크립트 (예시)

> **GPU 규칙(필수, §5):** 한 학습은 GPU 1개로만. `CUDA_VISIBLE_DEVICES` 를 **항상 명시** — 미지정 시 두 GPU 를 DataParallel 로 잡는다.

main 8 config 를 두 GPU 에 4개씩 나눠, 각 GPU 가 자기 몫 config 들의 fold0~4 를 **연속으로**(wave 장벽 없이) 돌린다 — GPU idle 이 거의 없다.

```bash
# config = "encoder decoder". encoder 는 argparse 의 _sa 키.
CONFIGS=("resnet50_sa unet" "resnet50_sa segformer"
         "densenet201_sa unet" "densenet201_sa segformer"
         "efficientnet-b4_sa unet" "efficientnet-b4_sa segformer"
         "mit_b2_sa unet" "mit_b2_sa segformer")

train_cfg() {  # $1=GPU  $2="encoder decoder"
  set -- $1 $2; gpu=$1; enc=$2; dec=$3; lab=${enc%_sa}
  for k in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$gpu python train.py --use_5fold_cv --fold_idx $k \
      --encoder $enc --decoder $dec --max_epochs 300 --early_stopping_patience 50 \
      --exp_setting review_5fold_msffm_${lab}_${dec}_fold${k}_seed42
  done
}

# GPU0 ← 앞 4 config, GPU1 ← 뒤 4 config (동시 진행)
( for c in "${CONFIGS[@]:0:4}"; do train_cfg 0 "$c"; done ) &
( for c in "${CONFIGS[@]:4:4}"; do train_cfg 1 "$c"; done ) &
wait
```

평가·집계는 학습 완료 후 (평가는 가벼워 단일 GPU 로 충분), config 마다 1회씩:

```bash
for c in "${CONFIGS[@]}"; do
  set -- $c; enc=$1; dec=$2; lab=${enc%_sa}
  for k in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 python test.py --use_5fold_cv --fold_idx $k \
      --encoder $enc --decoder $dec \
      --exp_setting review_5fold_msffm_${lab}_${dec}_fold${k}_seed42
  done
  python aggregate_5fold_results.py \
    --exp_template review_5fold_msffm_${lab}_${dec}_fold{fold}_seed42 \
    --encoder $enc --decoder $dec
done
```

### 8.3 결과 집계 — `aggregate_5fold_results.py`

- 입력: fold 별 `test_log/{NetClass}_{encoder}/{dataset}_{img_size}/{exp}/{param}/results.txt` (`--exp_template` 의 `{fold}` 를 0..4 로 치환해 자동 합성) + `model/.../*_best_model.pth`.
- 출력: **Markdown** (stdout + `aggregate_5fold_{title}.md` 동시 저장).
- 포함 항목:
  - Run Summary: fold 별 best epoch / best val_loss / stop epoch.
  - Dice / mIoU / HD 표 3종 (행 = fold0..4 + mean ± std, 열 = LCA/LAD/LCX/RCA/Mean).
- CLI 예: `python aggregate_5fold_results.py --exp_template review_5fold_msffm_resnet50_unet_fold{fold}_seed42 --encoder resnet50_sa --decoder unet`.

### 8.4 모델 선택과 보고의 이중 사용

같은 validation fold 가 (1) best epoch 선택 기준과 (2) 최종 성능 보고 대상으로 동시 사용된다. 미세한 optimism 이 있음 → 원고 Methods 에 명시 (`TODO.md` §5 의 영문 초안 참고).
