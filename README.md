# MSFFM — Multi-Slice Feature Fusion Module

심장 게이트 CT 상의 **관상동맥 석회화(CAC, Coronary Artery Calcification)** 분할을 위한 2.5D 세그멘테이션 프레임워크.
표준 2D encoder–decoder 백본(U-Net / SegFormer / EMCAD)에 MSFFM 을 plug-and-play 로 끼워 넣어, 인접 3개 슬라이스(prev / reference / next)의 inter-slice continuity 를 self/cross-attention 으로 회복한다. 3D convolution 은 쓰지 않는다.

## MSFFM

encoder 중간 stage 의 feature map 세 장을 받아 다음을 계산한다.

```
Z_ref   = SA(X_ref)                                  # self-attention
Z_prev  = CA(X_ref, X_prev)                          # Q = X_ref, K/V = X_prev
Z_next  = CA(X_ref, X_next)
Z_fused = W_c · Concat(Z_prev, Z_ref, Z_next)        # 1x1 conv 로 채널 복원
Z_final = Z_fused ⊕ X_ref                            # residual (BN zero-init warm-start)
```

- Q/K/V 는 1×1 conv, attention 은 전체 `(H·W)×(H·W)` multi-head scaled dot-product (`num_heads=8`).
- 삽입 위치는 encoder 의 **32×32 / 16×16** 해상도 두 곳 (동시 적용이 최적). stage 번호는 백본마다 다르다 — `resnet_sa`/`mix_transformer_sa`/`emcad_sa` 는 stage 3·4, `densenet_sa`/`efficientnet_sa` 는 stage 4·5.
- 세 슬라이스는 **가중치를 공유하는 같은 encoder** 를 3번 통과한다.

## 구성 (4 configuration)

`--decoder` / `--encoder` 조합 하나로 baseline 과 MSFFM 을 모두 제어한다.

| 구성 | 명령 인자 | 입력 채널 |
|---|---|---|
| MSFFM + U-Net | `--decoder unet --encoder resnet50_sa` | 3 |
| single-slice baseline | `--decoder unet --encoder resnet50` | 1 |
| EMCAD + MSFFM | `--decoder emcad_sa --encoder pvt_v2_b2` | 3 |
| EMCAD baseline | `--decoder emcad --encoder pvt_v2_b2` | 1 |

**encoder 허용 목록** (`utils.py`)

| decoder | encoder |
|---|---|
| `unet`, `segformer` | `resnet50`, `densenet201`, `efficientnet-b4`, `mit_b2` (+ 각각의 `_sa`) |
| `emcad` | `pvt_v2_b0`~`b5`, `resnet18`~`resnet152` |
| `emcad_sa` | `pvt_v2_b1`~`b5` |

입력 채널 수는 별도 플래그 없이 `derive_num_slices(decoder, encoder)` 가 결정한다 — `_sa` encoder 또는 `emcad_sa` 면 3, 나머지는 1.

## 데이터

- **출처:** Stanford AIMI — [COCA (Coronary Calcium and Chest CT)](https://aimi.stanford.edu/datasets/coca-coronary-calcium-chest-ct). 451명 중 433명 사용 (제외 18건의 사유는 `data/dataprep/coca_data_error.txt`).
- **클래스 5개:** 0=background, 1=LCA, 2=LAD, 3=LCX, 4=RCA. in-plane 512×512.
- **분할:** 433 case 를 **case 단위**로 stratified 5-fold (`MultilabelStratifiedKFold`, 층화 키 = case 별 `[LCA, LAD, LCX, RCA]` multi-hot, `random_state=42`). 슬라이스 단위 분할은 같은 case 의 prev/ref/next 가 train/val 양쪽에 등장해 raw 픽셀이 누수되므로 금지.
- **정규화:** 433 case 전체 voxel 분포에서 산출한 `hu_stats_433.json` (`lower`/`upper` = 0.5/99.5 분위수, `mean`/`std` = clip 후) 로 clip + z-score.
- **Augmentation:** rot90+flip 과 rotate 를 각각 독립 50% 확률로 적용. 회전각은 정수 `[-20, 19]`, image·label 모두 `order=0` (nearest).

디스크 구조 (데이터셋 루트는 `.gitignore` 로 git 밖):

```
data/datasets/COCA/COCA_3frames_5fold/
├── images/case{0000..0450}.npy      # (D, H, W) float32 HU 볼륨 (memmap)
├── labels/case{0000..0450}.npy      # (D, H, W) uint8
├── lists_COCA_5fold/
│   ├── fold0.txt ~ fold4.txt        # 한 줄당 case{gidx}_slice{n} (n = triplet 시작 인덱스)
│   └── fold_assignment.csv
├── hu_stats_433.json
└── case_index.csv                   # case_id ↔ 원본 nnUNet 파일명
```

`COCAVolumeDataset` 이 볼륨을 memmap 으로 lazy 로드해 `vol[n:n+3]`→`(H,W,3)` 와 `vol[n+1]` center label 을 조립한다 (`num_slices=1` 이면 center 한 장만). 자산 생성은 `data/dataprep/build_5fold_dataset.py` 가 nnUNet 포맷 원본 `Dataset001_COCA` 에서 1회 수행하며, 나머지 전처리 스크립트의 실행 규약은 `data/dataprep/README.md` 참고.

## 학습

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --use_5fold_cv --fold_idx 0 \
  --decoder unet --encoder resnet50_sa \
  --max_epochs 300 --early_stopping_patience 50 \
  --exp_setting msffm_resnet50_unet_fold0_seed42
```

- AdamW (lr=1e-5, wd=1e-4) + PolyLR (exponent=0.9), batch 16, 최대 300 epoch, early stopping patience 50, AMP 상시 활성.
- 손실은 decoder 계열에서 유도된다 — `unet`/`segformer` = `last_layer` supervision + `0.5·Dice + 0.5·CE`, `emcad`/`emcad_sa` = `mutation` deep supervision + `0.7·Dice + 0.3·CE`. `--supervision`/`--dice_weight`/`--ce_weight` 로 덮어쓸 수 있다.
- `--exp_setting` 에 `fold{fold_idx}` 문자열이 없으면 즉시 종료된다. fold 정체성이 경로에 자동으로 들어가지 않아, 불일치 시 다른 fold 의 체크포인트를 덮어쓰기 때문.
- `--fold_idx` 를 0~4 로 바꿔 5회 반복.
- 명령은 **저장소 루트에서** 실행한다. 데이터·체크포인트 경로 기본값이 모두 cwd 기준 상대경로이고, `segmentation_models_pytorch` 도 루트에서 실행할 때만 in-tree fork 가 pip 설치본(`site-packages`)을 가린다.

### `exp_setting` 명명 규약

`exp_setting` 문자열이 체크포인트 디렉터리 이름이 되므로, 이 규약이 곧 기존 실행 결과를 찾아가는 유일한 키다. encoder 라벨에서는 `_sa` 접미사를 뺀다.

| 구성 | 패턴 | 예 |
|---|---|---|
| MSFFM (SMP 계열) | `msffm_{encoder}_{decoder}_fold{k}_seed42` | `msffm_resnet50_unet_fold0_seed42` |
| single-slice baseline | `baseline_{encoder}_{decoder}_fold{k}_seed42` | `baseline_resnet50_unet_fold0_seed42` |
| EMCAD / EMCAD+MSFFM | `emcad{,_sa}_fold{k}_seed42` | `emcad_sa_fold0_seed42` |

### 파인튜닝

`--init_from` 에 출발점이 될 `exp_setting` 을 주면 파인튜닝이 된다. `--exp_setting` 은 평소와 똑같이 **이번 실행의 저장 이름**이므로 원본 체크포인트는 그대로 남는다.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --fold_idx 0 \
  --decoder unet --encoder resnet50_sa \
  --init_from   msffm_resnet50_unet_fold0_seed42 \
  --exp_setting kmu_chest_fold0
```

- 체크포인트는 `--init_from` 디렉터리의 `*best_model.pth` 를 쓴다. `--max_epochs`/`--batch_size`/`--base_lr` 이 원본과 같아야 그 디렉터리를 찾는다.
- head 키(`unet`/`segformer` → `segmentation_head.`, `emcad*` → `out_head`)만 제거한 뒤 `strict=False` 로 올리므로, 클래스 수가 달라도 encoder·decoder 를 재사용한다.
- `--init_from` 이 `fold{k}` 를 달고 있으면 `--fold_idx` 와 일치해야 한다. 다른 fold 의 가중치는 지금의 validation fold 를 이미 학습한 상태라 검증이 오염된다.
- 학습은 처음부터 다시 도는 형태라 LR 스케줄이 리셋된다. 짧은 파인튜닝이면 `--max_epochs` 도 줄일 것.

## 평가

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --use_5fold_cv --fold_idx 0 \
  --decoder unet --encoder resnet50_sa \
  --exp_setting msffm_resnet50_unet_fold0_seed42 \
  [--is_savenii] [--save_attention]
```

- 평가 셋 = `fold{fold_idx}.txt` (학습 때의 validation fold). 별도 hold-out test 셋은 없다.
- `--exp_setting`/`--max_epochs`/`--batch_size`/`--base_lr`/`--img_size` 가 학습 때와 같아야 체크포인트 경로가 매칭된다.
- **메트릭은 3D.** 슬라이스 예측을 case 별로 모아 `(D,H,W)` 볼륨으로 합성한 뒤 MONAI 의 Dice / MeanIoU / SurfaceDistance 를 배경 제외 클래스에 적용한다. `--is_savenii` 시 NIfTI 도 저장 (spacing `(0.375, 0.375, --z_spacing)`).
- `--save_attention` 하나로 attention 시각화가 켜진다 — 모델 트리의 모든 `NonLocalBlock` 을 찾아 `return_attention=True` 토글 + hook 등록 + 히트맵 저장까지 자동. 히트맵 스케일은 uniform 대비 배수(1.0 = 무선호)다.

**집계:**

```bash
python aggregate_5fold_results.py --decoder unet --encoder resnet50_sa \
  --exp_template msffm_resnet50_unet_fold{fold}_seed42
```

fold 별 `results.txt` 를 읽어 Dice / mIoU / HD 를 mean ± std 표(Markdown)로 stdout + `results/` 에 저장한다.

## 경로 규약

학습과 평가가 **같은 규칙**으로 디렉터리를 합성한다. 바꾸려면 `train.py` 와 `test.py` 를 함께 고쳐야 한다.

```
model/{NetClass}_{encoder}/{dataset}_{img_size}/{exp_setting}/epo{E}_bs{B}_lr{LR}/
test_log/{NetClass}_{encoder}/{dataset}_{img_size}/{exp_setting}/epo{E}_bs{B}_lr{LR}/
```

- `NetClass` = `net.__class__.__name__` → `Unet` / `Segformer` / `EMCADNet` / `EMCAD_SA_Net`.
- `lr` 은 파이썬 float 의 문자열 표현 (`1e-05`).
- 체크포인트는 `epoch_{N}_{val_loss:.4f}_best_model.pth` (val_loss 최저 시 저장).

## 저장소 구조

```
├── train.py / trainer.py            # 학습 진입점 및 루프
├── test.py  / tester.py             # 평가 진입점 및 3D 메트릭
├── dataset.py                       # COCAVolumeDataset, CT 정규화, augmentation
├── utils.py                         # encoder 허용 목록, PolyLR, DiceLoss, supervision 조합
├── aggregate_5fold_results.py       # fold 별 결과를 mean±std 표로 집계
├── segmentation_models_pytorch/     # SMP 를 in-tree 로 fork·수정
│   └── encoders/{resnet,densenet,efficientnet,mix_transformer}_sa.py   # 백본별 MSFFM 통합
├── networks/emcad/                  # EMCAD 디코더 + pvtv2 백본 (use_msffm 플래그)
└── data/dataprep/                   # 데이터셋 구축 스크립트
```

## 구현 노트

- **`in_channels=1` 인데 실제 입력은 3채널이다.** `smp.Unet(..., in_channels=1)` 이 첫 conv 를 `Conv2d(1→64)` 로 패치하고, `_sa` encoder 가 forward 안에서 3채널을 `[:,0:1]/[:,1:2]/[:,2:3]` 로 잘라 같은 conv 를 3번 통과시킨다. `in_channels=3` 으로 바꾸면 가중치 공유와 ImageNet pretrained 로딩이 함께 깨진다.
- **한 학습 = GPU 1개.** `trainer.py` 는 `torch.cuda.device_count() > 1` 이면 보이는 GPU 를 전부 `nn.DataParallel` 로 잡는다. 매 실행에 `CUDA_VISIBLE_DEVICES` 를 명시할 것. 두 GPU 가 비어 있으면 `=0`/`=1` 로 서로 다른 실험을 동시에 돌린다. `DistributedDataParallel` 은 지원하지 않는다.
- **DataLoader 는 `shuffle=False` + `collate_fn=shuffle_within_batch`.** 셔플을 batch 내부에서만 수행한다. 외부 shuffle 을 켜면 인접 슬라이스 정렬이 깨져 2.5D 가정이 무의미해진다.
- `--use_5fold_cv` 는 하위 호환용 플래그로 아무것도 분기하지 않는다 — 켜든 안 켜든 항상 5-fold 경로를 읽는다.
- `--decoder emcad_sa` 가 `pvt_v2_b0` 을 거부하는 이유는 MSFFM `NonLocalBlock` 채널이 320/512 로 고정인데 b0 만 160/256 이기 때문이다.
- `model/pvt/` 에는 `pvt_v2_b2.pth`/`b3.pth` 만 있다. 나머지로 **학습**하면 모델 생성 중 `FileNotFoundError` 로 죽는다 (평가는 pretrained 를 안 읽으므로 무관).
- `--no_pretrain` 은 `test.py` 에서 무의미하다. 평가는 항상 pretrained 없이 모델을 만든 뒤 체크포인트를 strict 로드한다.
- `EMCADNet` 과 `EMCAD_SA_Net` 을 한 클래스로 합치지 말 것. 경로가 `net.__class__.__name__` 으로 합성되므로 합치면 두 실험의 체크포인트가 서로 덮어쓴다.
- `NonLocalBlock` 은 fused SDPA(`F.scaled_dot_product_attention`)를 쓰며 q/k/v 를 `.contiguous()` 로 넘긴다 — torch 2.0 SDPA 가 마지막 축 연속을 요구한다. fused 경로는 atomics 를 쓰므로 평가 메트릭이 소수점 4자리 수준에서 run-to-run 으로 흔들린다.
- `segmentation_models_pytorch/encoders/multi_slice_feature_fusion.py` 는 어떤 registry 에도 등록되지 않은 참고 구현이다. 수정해도 학습/평가에 영향이 없다.
- `dataset.py` 가 저장소 루트에 있는 이유는 HF `datasets` 패키지와의 이름 충돌 때문이다. `datasets/` 디렉터리를 되살리지 말 것.
- `resnet_sa.py` / `multi_slice_feature_fusion.py` 의 residual 진단 print 는 모듈 상수 `DEBUG_RESIDUAL` (기본 `False`) 로 게이팅된 의도적 instrumentation 이다.

## Acknowledgements

- [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) — U-Net / SegFormer 및 encoder registry (본 저장소에 fork 포함)
- [EMCAD](https://github.com/SLDGroup/EMCAD) — efficient multi-scale convolutional attention decoder
- [PVTv2](https://github.com/whai362/PVT) — pyramid vision transformer 백본
