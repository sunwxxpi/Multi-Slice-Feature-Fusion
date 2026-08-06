# CLAUDE.md — SAU-Net / MSFFM Repository

이 파일은 Claude(코딩 에이전트)가 본 저장소에서 작업할 때 반드시 먼저 읽어야 할 컨텍스트 문서입니다. 사용자의 전역 `~/.claude/CLAUDE.md` 규칙(한국어 우선, 최소 변경, 기존 패턴 우선)에 종속됩니다.

## 1. 프로젝트 한 줄 요약

심장 게이트 CT 상의 **관상동맥 석회화(CAC, Coronary Artery Calcification)** 분할을 위한 2.5D 세그멘테이션 프레임워크 연구 코드. 표준 2D Encoder–Decoder(U-Net / SegFormer / EMCAD) 백본에 **MSFFM(Multi-Slice Feature Fusion Module)** 을 plug-and-play 로 끼워 넣어, 인접 3개 슬라이스 (이전 / 기준 / 이후)의 inter-slice continuity 를 self/cross-attention 으로 회복한다.

자세한 동기·설계는 `docs/ARCHITECTURE.md` 참고. 원고 `MSFFM_full_20251223.pdf` 는 저장소에 없다 (git 이력에도 없음, 소유자 로컬 파일) — figure/table 의 1차 출처지만 에이전트는 열 수 없다.

## 2. 디렉터리 맵 (핵심만)

```
SAU-Net/
├── train.py / trainer.py         # 학습 진입점 및 학습 루프
├── test.py  / tester.py          # 평가 진입점 및 3D 메트릭 계산
├── utils.py                      # PolyLRScheduler, DiceLoss, FocalLoss, SMP_ENCODERS, encoder 검증, supervision 조합 빌더
├── dataset.py                    # COCAVolumeDataset, CT normalization, augmentation
├── segmentation_models_pytorch/  # SMP 라이브러리를 in-tree 로 fork·수정한 코드
│   └── encoders/
│       ├── multi_slice_feature_fusion.py   # MSFFM 본체 (cross-attn / cosine fusion)
│       ├── resnet_sa.py                    # ResNet-50 + MSFFM
│       ├── densenet_sa.py                  # DenseNet-201 + MSFFM
│       ├── efficientnet_sa.py              # EfficientNet-b4 + MSFFM
│       ├── mix_transformer_sa.py           # MiT-b2 + MSFFM
│       └── resnet.py / densenet.py / ...   # 각 백본의 registry 에 *_sa 등록됨
├── networks/emcad/               # EMCAD 디코더 백본 (pvtv2 + MSFFM 옵션)
├── tests/                        # 검증 스크립트 (pytest 아님, `PYTHONPATH=. python tests/check_*.py`)
├── data/
│   ├── datasets/COCA/            # COCA 데이터셋 실체 (gitignored)
│   └── dataprep/                 # 데이터셋 구축 스크립트 (git 추적)
├── model/                        # 학습 체크포인트 (gitignored)
├── test_log/                     # 평가 결과·NIfTI·attention vis (gitignored)
└── docs/                         # 본 문서를 포함한 추가 가이드
```

## 3. 환경 / 실행

- Conda env: `SAU-Net` (`/home/psw/anaconda3/envs/SAU-Net/bin/python3`)
- 주요 의존성: PyTorch + `segmentation_models_pytorch`(in-tree fork) + `monai` + `SimpleITK`
- 학습/평가 명령 예시는 `docs/EXPERIMENTS.md §1~§2`. train/test 의 `--exp_setting` 이 동일해야 체크포인트 경로가 매칭된다 (§5). GPU 는 실행마다 `CUDA_VISIBLE_DEVICES` 로 1개 핀 (§6).
- **데이터셋 루트:** `data/datasets/COCA/COCA_3frames_5fold/` 가 유일한 활성 경로다 (저장소 안이지만 `.gitignore` 로 git 밖). 인자는 `--root_path_5fold`/`--list_dir_5fold`/`--hu_stats_path` (모두 기본값이 `/home/psw/...` 절대경로로 박혀 있어 다른 PC 로 옮기면 깨지므로 덮어쓸 것). `--root_path`/`--list_dir` (hold-out 인자)는 통합 과정에서 제거돼 더 이상 존재하지 않는다. `--use_5fold_cv` 는 하위 호환용으로 남은 플래그일 뿐 실제로는 아무것도 분기하지 않는다 — 켜든 안 켜든 학습/평가는 항상 위 5-fold 경로를 읽는다. 경로 구조 상세는 `docs/DATA.md §2`.
- **5-fold CV 자산 생성:** `build_5fold_dataset.py` 가 `Dataset001_COCA` 원본에서 1회 생성한다 (rebuild 이유·포맷·배경은 `docs/DATA.md §9`).

## 4. 핵심 컨벤션

- **입력 텐서 모양:** `(B, C, 512, 512)`. `C=3` 이면 채널 0/1/2 = `prev / reference / next`, `C=1` 이면 reference 한 장이다. `C` 는 `--decoder`/`--encoder` 조합이 결정한다 (`_sa` encoder 또는 `--decoder emcad_sa` → 3, 나머지 → 1, `utils.py:derive_num_slices`). augmentation·resize 는 `(H,W,C)` 규약을 유지해 두 경우를 공유한다 (`dataset.py`).
- **클래스 수:** 5 (배경 + LCA / LAD / LCX / RCA). `DiceLoss` 는 배경(index 0) 을 제외하고 평균.
- **손실:** `--decoder` 계열에서 기본값이 유도된다 (`--supervision`/`--dice_weight`/`--ce_weight` 로 덮어쓸 수 있음). `unet`/`segformer` = `last_layer` supervision, `0.5·Dice + 0.5·CE`. `emcad`/`emcad_sa` = `mutation` deep supervision, `0.7·Dice + 0.3·CE`. AMP(`GradScaler`) 는 계열 무관 공통 사용.
- **옵티마이저·스케줄:** AdamW(lr=1e-5, wd=1e-4) + 커스텀 `PolyLRScheduler` (exponent=0.9).
- **MHA heads:** 논문 명시값 8. 모든 `_sa` 인코더(`resnet_sa.py` / `densenet_sa.py` / `efficientnet_sa.py` / `mix_transformer_sa.py`)와 참고 구현 `multi_slice_feature_fusion.py` 가 `num_heads=8` 로 통일됨. attention 은 전체 `(H·W)×(H·W)`.
- **MSFFM 삽입 위치:** encoder 의 32×32 / 16×16 해상도 두 곳. 동시 적용이 최적 (Ablation, 논문 Table 4). 모듈 이름의 stage 번호는 백본마다 다르다 — `resnet_sa`/`mix_transformer_sa` 는 `cross_attention_*_3`/`_4`, `densenet_sa`/`efficientnet_sa` 는 `_4`/`_5`. 해상도는 네 백본 모두 같다.
- **체크포인트 네이밍:** `epoch_{N}_{val_loss:.4f}_best_model.pth` (val_loss 최저 시 저장) 와 학습 종료 시점의 `epoch_{max}_{val_loss}.pth`.

## 5. 경로 규약 (snapshot / log)

학습/평가 모두 아래 규칙으로 디렉터리를 합성한다. 변경 시 `train.py`와 `test.py` 가 **함께** 바뀌어야 한다.

```
snapshot_path = ./model/{NetClass}_{encoder}/{dataset}_{img_size}/{exp_setting}/epo{E}_bs{B}_lr{LR}/
log_path      = ./test_log/{NetClass}_{encoder}/{dataset}_{img_size}/{exp_setting}/epo{E}_bs{B}_lr{LR}/
```

- `NetClass` 는 `net.__class__.__name__` 이므로 `--decoder` 에 따라 `Unet`, `Segformer`, `EMCADNet`, `EMCAD_SA_Net` 네 가지가 들어간다.
- `lr` 은 파이썬 float 의 문자열 표현 (`1e-05` → 그대로 문자열). 비교 시 `1.0e-05` 등으로 바꾸지 말 것.
- 파인튜닝 시 (`--enable_finetuning`) 은 동일 `exp_setting` 의 best 체크포인트를 로드한 뒤 `--finetune_exp_setting` 의 새 디렉터리에 저장한다. 무시(strict=False)되는 키는 decoder 계열마다 다르다 — `unet`/`segformer` 는 `segmentation_head.` 로 시작하는 키를, `emcad`/`emcad_sa` 는 `out_head` 로 시작하는 키(`out_head1~4`, deep supervision 출력 헤드)를 제거한다. 어느 쪽이든 클래스 수가 달라도 인코더·디코더는 재사용된다.

## 6. "건드리기 전 알아둘 것" (Foot-guns)

| 항목 | 내용 |
|---|---|
| `in_channels=1` 인데 실제 입력은 3채널 | `smp.Unet(..., in_channels=1)` 호출이 `set_in_channels(1)` 을 트리거해 첫 conv 를 `Conv2d(1→64)` 로 패치한다. SA 인코더는 `forward(x)` 안에서 3채널을 직접 `[:,0:1]/[:,1:2]/[:,2:3]` 로 잘라 같은 conv1 을 3번 통과시키므로 정상 동작한다. **임의로 `in_channels=3` 으로 바꾸지 말 것** — 가중치 공유와 ImageNet pretrained 로딩이 깨진다. |
| `test.py:add_encoder_prefix` | 현재 체크포인트는 `model.state_dict()` 가 이미 `encoder.`/`decoder.`/`segmentation_head.` prefix 를 갖고 있어(`smp.Unet`/`Segformer` 의 서브모듈 이름 자체가 그 prefix) 이 함수는 no-op 이다 (실제 체크포인트 여러 건을 로드해 전수 확인). prefix 없는 구형 체크포인트를 위한 방어 코드로 남아 있다. |
| `print("Residual branch mean abs value:", ...)` | `resnet_sa.py` / `multi_slice_feature_fusion.py` 의 forward 안 residual 진단 print. 모듈 상수 `DEBUG_RESIDUAL`(기본 `False`) 로 게이팅되어 평상시 실행 안 됨 (`.item()` CUDA 동기화·로그 오염 없음). 진단 필요 시 해당 파일의 `DEBUG_RESIDUAL=True` 로 켤 것. **의도된 instrumentation 이므로 삭제하지 말 것.** |
| `ResNetSAEncoder` 가 두 파일에 중복 정의 | `resnet_sa.py` 와 `multi_slice_feature_fusion.py` 양쪽에 동명 클래스가 있다. 둘 다 `num_heads=8` / 전체 attention 으로 동일하지만, `resnet50_sa` 등록 키는 `resnet.py` 가 `from .resnet_sa import ResNetSAEncoder` 로 가져오는 쪽만 활성이다. `multi_slice_feature_fusion.py` 의 클래스(및 `CosineDynamicFusion`/`DoubleConv`)는 어디서도 import 되지 않는 참고 구현 — 수정해도 학습/평가에 영향 없음. |
| `networks/fcbformer/` 의 소스 부재 | 저장소 어디에도 없다 (`__pycache__` 조차 남아 있지 않음). FCBFormer 경로는 현재 활성 코드 경로에 없다. |
| `EMCADNet` / `EMCAD_SA_Net` 클래스 분리 | 두 클래스를 하나로 합치지 말 것. 경로가 `net.__class__.__name__` 으로 합성되므로 합치면 두 실험의 체크포인트가 서로 덮어쓴다. `Unet`/`Segformer` 는 encoder 이름(`resnet50` vs `resnet50_sa`)이 경로를 갈라준다. |
| `--decoder emcad_sa` 의 encoder 제한 | `pvt_v2_b1`~`b5` 만 지원한다. MSFFM 의 `NonLocalBlock` 채널이 320/512 로 고정인데 `pvt_v2_b0` 만 160/256 이기 때문. `train.py`/`test.py` 가 parse 직후 거부한다. |
| hold-out 경로 부재 | 5-fold 경로(`COCA_3frames_5fold`)가 유일하게 남은 경로다. `--use_5fold_cv` 여부와 무관하게 항상 이 경로를 쓴다 (§3). 5-fold 이전 hold-out(`COCA_dataset`, `COCA_1frame`) 은 통합 시 제거됐다. 옛 hold-out 결과 재현은 동결 브랜치(`single_slice`/`EMCAD`/`EMCAD-SA`)에서 한다. |
| 학습 시 `DataLoader(shuffle=False, collate_fn=shuffle_within_batch)` | shuffle 을 batch 내부에서 수행. 외부 shuffle 을 켜지 말 것 — 인접 슬라이스 정렬이 깨지면 MSFFM 가정이 무의미해진다. |
| **학습 1개 = GPU 1개 (DataParallel 자동 함정)** | `trainer.py` 가 `torch.cuda.device_count()>1` 이면 **보이는 GPU 를 전부 `nn.DataParallel` 로 잡는다**. 한 학습(run)은 반드시 단일 GPU 로 돌려야 하므로 **매 실행에 `CUDA_VISIBLE_DEVICES=0` 또는 `=1` 을 명시**할 것 (그러면 `device_count()==1` → DataParallel 미적용). 두 GPU 가 모두 비면 `=0`/`=1` 로 서로 다른 실험을 동시에 돌려도 된다. 명령·병렬 워크플로 상세는 `docs/EXPERIMENTS.md §5·§8`. |
| `ct_normalization` 의 상수 | 시그니처의 하드코딩 기본값 `lower=-2.0, upper=1521.0, mean=355.38, std=282.92` (train 300-case) 은 **hold-out 제거 후 어떤 활성 경로도 호출하지 않는다** — `COCAVolumeDataset` 이 항상 `hu_stats_433.json` (`15.0/1577.0/773.55/399.24`, 433-case 0.5/99.5 분위수) 을 `load_hu_stats` 로 읽어 명시 인자로 전달한다. 동결 브랜치의 hold-out 결과는 이 죽은 기본값으로 산출된 것이라 5-fold 수치와 절대 비교 금지. 다른 코호트(KMU 등) 적용 시 재산정. |
| `dataset.py` 가 저장소 루트에 있는 이유 | env 에 HF `datasets`(4.5.0)가 설치돼 있어, 예전 `datasets/` 패키지는 빈 `__init__.py` 로만 우선권을 잡고 있었다. 루트 `dataset.py` 로 옮겨 이름 충돌 자체를 없앴다 — `datasets/` 를 되살리지 말 것. |
| 평가는 3D | `tester.py` 는 슬라이스 예측을 케이스별로 모아 3D 볼륨으로 합성한 뒤 MONAI 메트릭 (Dice/MeanIoU/SurfaceDistance) 을 적용한다. 2D 슬라이스 단위 메트릭이 필요하면 `compute_metrics_3d` 를 우회해야 한다. |
| 5-fold CV 시 분할 단위 | 반드시 **case 단위**로 fold 를 나눠야 한다. 슬라이스 단위 stratify 는 같은 case 의 인접 슬라이스가 train/val 양쪽에 동시 등장해 NPZ 안의 prev/ref/next 채널을 통해 raw 픽셀이 누수된다 (2.5D 가정 파괴). 층화 키는 vessel multi-hot 벡터, API 는 `MultilabelStratifiedKFold`. |
| 체크포인트 선택 방식이 3곳마다 다름 | `test.py` 는 `sorted(glob(...))` 결과가 정확히 1개인지 assert 한 뒤 그 파일을 쓴다. `train.py` 의 파인튜닝 경로는 정렬하지 않는 `os.listdir()` 에서 이름이 `best_model` 로 끝나는 첫 항목을 쓴다. `aggregate_5fold_results.py` 는 `glob(...)[0]` (정렬·개수 검증 없음) 을 쓴다. 디렉터리마다 체크포인트가 정확히 1개뿐인 지금은 무해하지만 세 곳의 보장 수준이 서로 다르다 — 의도적으로 통일하지 않고 남겨둠. |
| `--finetune_exp_setting` 에는 fold 토큰 가드가 없음 | `--exp_setting` 은 `fold{fold_idx}` 문자열 포함을 parse 직후 강제하지만 `--finetune_exp_setting` 은 같은 검증이 없다. 같은 `--finetune_exp_setting` 으로 fold 별 파인튜닝을 여러 번 돌리면 저장 디렉터리가 서로 덮어쓴다. 파인튜닝 대상은 보통 fold 개념이 없는 다른 코호트(KMU 등)라서 의도적으로 가드를 확장하지 않았다. |

## 7. 자주 헷갈리는 용어

- **MSFFM**: Multi-Slice Feature Fusion Module. 본 연구의 핵심 모듈명.
- **SA (Self-Attention)**: 인코더 이름 접미사 `_sa` 는 MSFFM 통합 버전을 의미. 단순 self-attention 만이 아니다 — self-attn + cross-attn (prev/next) + fusion 을 모두 포함한다.
- **2.5D**: 3D volumetric conv 없이 3장의 2D 슬라이스를 입력으로 받아 inter-slice 관계를 attention 으로만 모델링하는 설계.
- **DSC vs mIoU**: 동일한 표기지만 본 코드의 `mIoU` 는 클래스별 IoU 의 산술 평균(MONAI MeanIoU, 배경 제외 평균은 `tester.py` 에서 처리).

## 8. 변경할 때 따르는 절차

1. 실험적 의도와 어긋나는 수정인지 먼저 점검 — 원고가 저장소에 없으므로 `docs/ARCHITECTURE.md` 의 설계 기록으로 대조하고, 판단이 서지 않으면 원고를 가진 소유자에게 확인한다.
2. 코드 수정 시 한국어 주석 유지(전역 규칙). 영어 식별자/타입은 그대로 둘 것.
3. 학습 → 평가의 경로 규약(§5) 을 깨지 않는 한 가장 작은 변경을 적용.
4. 디버그 `print` 가 시끄럽다고 일괄 제거하지 말 것 — `residual ratio` 출력은 의도된 instrumentation 가능성이 있다 (§6).
5. `model/`, `test_log/`, `data/datasets/` 는 `.gitignore` 대상이므로 결과물을 커밋하지 않는다. `data/dataprep/` 은 추적 대상이니 주의 — `data/` 통째로 무시된다고 가정하면 안 된다.

## 9. 추가 문서

- `docs/ARCHITECTURE.md` — MSFFM 내부 동작, encoder integration 흐름, attention 시각화 hook.
- `docs/DATA.md` — COCA 데이터셋 포맷, npz 구조, list 파일, CT normalization.
- `docs/EXPERIMENTS.md` — 학습/평가 명령 예시, exp_setting 명명 규약, 파인튜닝 워크플로.
- `tests/check_*.py` — 통합 시 도입한 검증 스크립트. pytest 미사용, 저장소 루트에서 `PYTHONPATH=. python tests/check_<name>.py` 로 개별 실행하며 exit code 로 판정한다. 데이터 경로·MSFFM 배선·손실 조합·CLI 조합 검증을 담당한다.
- `MSFFM_full_20251223.pdf` — 원고 (figure / table 의 1차 출처). **저장소에 없다** — 소유자 로컬 파일이라 에이전트는 열 수 없다.

## 10. 브랜치 맵 (Branch Map)

본 저장소는 MSFFM 의 ablation·비교를 위해 백본과 MSFFM 적용 여부별로 브랜치를 나눈다. 모든 브랜치는 동일한 5-fold CV 자산(`COCA_3frames_5fold`)·center 슬라이스 집합·라벨·3D 합성 방식을 공유해 공정 비교를 보장한다 (입력 채널 수만 다름).

| 브랜치 | 진입점 / 모델 | MSFFM | 설명 |
|---|---|---|---|
| `main` | `--decoder {unet,segformer,emcad,emcad_sa}` | 선택 | **통합 트렁크.** 4개 구성을 `--decoder`/`--encoder` 로 모두 제어한다. 신규 작업의 기준 브랜치. |
| `single_slice` | (동결) | ❌ | 통합 전 단일 슬라이스 baseline. hold-out 결과 재현 전용, 신규 작업 금지. |
| `EMCAD` | (동결) | ❌ | 통합 전 EMCAD baseline. 재현 전용. |
| `EMCAD-SA` | (동결) | ✅ | 통합 전 EMCAD+MSFFM. 재현 전용. |
| `2.5d-baselines` | `--net25d {catnet,csam,segmate}` + `--network` | — | 2.5D/3D 비교 하네스. 통합 대상 아님. |
| `3d-baselines` | `--network {segformer3d,lhunet,waveformer}` | — | true-3D 비교 하네스. 통합 대상 아님. |

**통합 후 4구성 실행:**

| 구성 | 명령 |
|---|---|
| MSFFM + U-Net | `--decoder unet --encoder resnet50_sa` |
| single-slice baseline | `--decoder unet --encoder resnet50` |
| EMCAD + MSFFM | `--decoder emcad_sa --encoder pvt_v2_b2` |
| EMCAD baseline | `--decoder emcad --encoder pvt_v2_b2` |

- **공통:** 학습 `train.py --use_5fold_cv`, 평가 `test.py --use_5fold_cv` (§3·§5). HU 정규화는 `hu_stats_433.json` (§6).
- **MSFFM 브랜치(`main`/`EMCAD-SA`):** `NonLocalBlock` 은 fused SDPA(`F.scaled_dot_product_attention`)를 쓰며 q/k/v 를 반드시 `.contiguous()` 로 넘긴다 — torch 2.0 SDPA 가 마지막 축 연속을 요구하기 때문(누락 시 forward crash). attention 시각화 hook 은 기본 비활성, `test.py --save_attention` 플래그 1개로 opt-in (자동으로 모든 `NonLocalBlock` 의 `return_attention=True` 토글 + hook 등록 + `attention_vis/` 저장).
- 각 브랜치는 자체 코드/문서를 갖는다. 본 표는 `main` 기준 정리이며 다른 브랜치 세부는 해당 브랜치를 따른다.
