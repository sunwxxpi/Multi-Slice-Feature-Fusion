# CLAUDE.md — SAU-Net / MSFFM Repository

이 파일은 Claude(코딩 에이전트)가 본 저장소에서 작업할 때 반드시 먼저 읽어야 할 컨텍스트 문서입니다. 사용자의 전역 `~/.claude/CLAUDE.md` 규칙(한국어 우선, 최소 변경, 기존 패턴 우선)에 종속됩니다.

## 1. 프로젝트 한 줄 요약

심장 게이트 CT 상의 **관상동맥 석회화(CAC, Coronary Artery Calcification)** 분할을 위한 2.5D 세그멘테이션 프레임워크 연구 코드. 표준 2D Encoder–Decoder(U-Net / SegFormer / EMCAD) 백본에 **MSFFM(Multi-Slice Feature Fusion Module)** 을 plug-and-play 로 끼워 넣어, 인접 3개 슬라이스 (이전 / 기준 / 이후)의 inter-slice continuity 를 self/cross-attention 으로 회복한다.

자세한 동기·설계·실험 결과는 `MSFFM_full_20251223.pdf`(원고)와 `docs/ARCHITECTURE.md` 참고.

## 2. 디렉터리 맵 (핵심만)

```
SAU-Net/
├── train.py / trainer.py         # 학습 진입점 및 학습 루프
├── test.py  / tester.py          # 평가 진입점 및 3D 메트릭 계산
├── utils.py                      # PolyLRScheduler, DiceLoss, FocalLoss
├── datasets/dataset.py           # COCA_dataset, CT normalization, augmentation
├── segmentation_models_pytorch/  # SMP 라이브러리를 in-tree 로 fork·수정한 코드
│   └── encoders/
│       ├── multi_slice_feature_fusion.py   # MSFFM 본체 (cross-attn / cosine fusion)
│       ├── resnet_sa.py                    # ResNet-50 + MSFFM
│       ├── densenet_sa.py                  # DenseNet-201 + MSFFM
│       ├── efficientnet_sa.py              # EfficientNet-b4 + MSFFM
│       ├── mix_transformer_sa.py           # MiT-b2 + MSFFM
│       └── resnet.py / densenet.py / ...   # 각 백본의 registry 에 *_sa 등록됨
├── networks/{emcad,fcbformer}/   # __pycache__ 만 남음 (소스 미커밋 상태)
├── model/                        # 학습 체크포인트 (gitignored)
├── test_log/                     # 평가 결과·NIfTI·attention vis (gitignored)
└── docs/                         # 본 문서를 포함한 추가 가이드
```

## 3. 환경 / 실행

- Conda env: `SAU-Net` (`/home/psw/anaconda3/envs/SAU-Net/bin/python3`)
- 주요 의존성: PyTorch + `segmentation_models_pytorch`(in-tree fork) + `monai` + `SimpleITK`
- 학습 (single hold-out, 기존 기본 경로):
  ```bash
  python train.py --encoder resnet50_sa --decoder unet \
                  --exp_setting <experiment-name>
  ```
- 학습 (5-fold CV 모드 — `TODO.md` §1 의 권장 워크플로):
  ```bash
  python train.py --encoder resnet50_sa --decoder unet \
                  --use_5fold_cv --fold_idx 0 \
                  --early_stopping_patience 50 \
                  --exp_setting review_5fold_<name>_fold0_seed42
  ```
- 평가:
  ```bash
  python test.py --encoder resnet50_sa --decoder unet \
                 --exp_setting <same-as-train>  [--is_savenii]
  # 5-fold 모드는 train.py 와 동일하게 --use_5fold_cv --fold_idx K 를 함께 지정
  ```
- 두 스크립트의 `--exp_setting` 이 동일해야 체크포인트 경로가 매칭된다 (§5 참고).
- **데이터셋 루트:** `/home/psw/AVS-Diagnosis/COCA/` (본 저장소 외부). 본 코드의 학습/평가는 그 하위 `COCA_3frames/{train_npz, test_npz, lists_COCA}` 만 사용한다. 같은 부모 디렉터리에 `COCA_1frame/` (단일 슬라이스 변형), `Dataset001_COCA/` (nnUNet 포맷), 그리고 `preprocess_train_test_data_3frames.py` 등 전처리 스크립트가 함께 있다 — DICOM/XML 에서 본 코드가 기대하는 NPZ 를 만드는 출처이므로 데이터 재생성이 필요하면 그곳을 참고. 자세한 경로 구조는 `docs/DATA.md` §2.
- **5-fold CV 모드 추가 자산:** `--use_5fold_cv` 사용 시 별도 디렉터리 `/home/psw/AVS-Diagnosis/COCA/COCA_3frames_5fold/` 를 쓴다 (기존 `COCA_3frames/` 와 분리, 원본 보존). 구성: `images/case{gidx}.npy` + `labels/case{gidx}.npy` (per-case `(D,H,W)` 볼륨, memmap), `lists_COCA_5fold/fold{0..4}.txt` (case-level stratified, sample 이름 `case{gidx}_slice{n}`), `hu_stats_433.json` (433-case 정규화 상수), `case_index.csv` (case_id↔원본 nnUNet 파일명). 모두 데이터 루트의 `build_5fold_dataset.py` 가 `Dataset001_COCA` 원본에서 1회 생성. 절차·배경은 `TODO.md` §1~§2.
  - case_id = 원본 nnUNet 전역 인덱스 (train 0~313, test 314~450). train_npz/test_npz 가 각각 case0001 부터 독립 번호라 단순 통합이 불가능했던 점이 rebuild 의 이유 (TODO.md §1.1).
  - 데이터로더는 `--use_5fold_cv` 일 때 `--root_path_5fold` / `--list_dir_5fold` / `--hu_stats_path` (모두 위 경로가 기본값) 를 쓰며 `--root_path` / `--list_dir` 는 무시.
- 데이터 루트의 기본 절대경로는 위 사용자 로컬 경로다. 다른 PC 로 옮기면 깨지므로 `--root_path` / `--list_dir` 로 덮어쓸 것.

## 4. 핵심 컨벤션

- **입력 텐서 모양:** `(B, 3, 512, 512)`. 채널 0/1/2 = `prev / reference / next` 슬라이스. augmentation·resize 도 채널 축을 보존하는 방식으로 작성되어 있다 (`datasets/dataset.py`).
- **클래스 수:** 5 (배경 + LCA / LAD / LCX / RCA). `DiceLoss` 는 배경(index 0) 을 제외하고 평균.
- **손실:** `loss = 0.5 * Dice + 0.5 * CrossEntropy`, AMP(`GradScaler`) 사용.
- **옵티마이저·스케줄:** AdamW(lr=1e-5, wd=1e-4) + 커스텀 `PolyLRScheduler` (exponent=0.9).
- **MHA heads:** 논문은 8 로 명시. 다만 `resnet_sa.py` 는 `num_heads=8`, `multi_slice_feature_fusion.py`의 `ResNetSAEncoder` 는 `num_heads=1` 로 설정되어 있다 → 학습/평가에 실제로 어떤 파일이 활성화되는지는 §6 의 등록(registry) 흐름으로 추적할 것.
- **MSFFM 삽입 위치:** encoder 의 stage 3(32×32) 와 stage 4(16×16). 두 곳 동시 적용이 최적 (Ablation, 논문 Table 4).
- **체크포인트 네이밍:** `epoch_{N}_{val_loss:.4f}_best_model.pth` (val_loss 최저 시 저장) 와 학습 종료 시점의 `epoch_{max}_{val_loss}.pth`.

## 5. 경로 규약 (snapshot / log)

학습/평가 모두 아래 규칙으로 디렉터리를 합성한다. 변경 시 `train.py`와 `test.py` 가 **함께** 바뀌어야 한다.

```
snapshot_path = ./model/{NetClass}_{encoder}/{dataset}_{img_size}/{exp_setting}/epo{E}_bs{B}_lr{LR}/
log_path      = ./test_log/{NetClass}_{encoder}/{dataset}_{img_size}/{exp_setting}/epo{E}_bs{B}_lr{LR}/
```

- `NetClass` 는 `net.__class__.__name__` 이므로 `Unet`, `Segformer` 가 들어간다.
- `lr` 은 파이썬 float 의 문자열 표현 (`1e-05` → 그대로 문자열). 비교 시 `1.0e-05` 등으로 바꾸지 말 것.
- 파인튜닝 시 (`--enable_finetuning`) 은 동일 `exp_setting` 의 best 체크포인트를 로드한 뒤 `--finetune_exp_setting` 의 새 디렉터리에 저장한다. segmentation_head 만 무시(strict=False) 하므로 클래스 수가 달라도 인코더·디코더는 재사용된다.

## 6. "건드리기 전 알아둘 것" (Foot-guns)

| 항목 | 내용 |
|---|---|
| `in_channels=1` 인데 실제 입력은 3채널 | `smp.Unet(..., in_channels=1)` 호출이 `set_in_channels(1)` 을 트리거해 첫 conv 를 `Conv2d(1→64)` 로 패치한다. SA 인코더는 `forward(x)` 안에서 3채널을 직접 `[:,0:1]/[:,1:2]/[:,2:3]` 로 잘라 같은 conv1 을 3번 통과시키므로 정상 동작한다. **임의로 `in_channels=3` 으로 바꾸지 말 것** — 가중치 공유와 ImageNet pretrained 로딩이 깨진다. |
| 학습 시 인코더 prefix 미사용, 테스트 시 prefix 부여 | `test.py:add_encoder_prefix` 가 체크포인트의 키 앞에 `encoder.` 를 붙여 SMP 의 모듈 트리에 매칭시킨다. 학습 코드가 `model.state_dict()` 그대로 저장하므로 키가 평탄(flat)하기 때문. 새 체크포인트 포맷을 만들 때 둘 다 수정해야 한다. |
| `print("Residual branch mean abs value:", ...)` | `resnet_sa.py` / `multi_slice_feature_fusion.py` 의 forward 안에 디버그 print 가 남아 있다. 학습/추론 로그를 오염시키지만 **의도된 잔재이므로 함부로 지우지 말고** 변경 전 사용자 확인. |
| `num_heads` 두 파일에서 다른 값 | `resnet_sa.py` 의 `ResNetSAEncoder` 는 `num_heads=8`, `multi_slice_feature_fusion.py` 의 또 다른 `ResNetSAEncoder` 는 `num_heads=1`. `resnet50_sa` 등록 키는 `resnet.py` 가 `from .resnet_sa import ResNetSAEncoder` 로 가져온다(=heads=8). `multi_slice_feature_fusion.py` 의 클래스는 현재 어디서도 직접 import 되지 않는다 — 참고 구현. |
| `networks/{emcad,fcbformer}` 의 소스 부재 | `__pycache__` 만 남아 있다. EMCAD / FCBFormer 경로는 현재 활성 코드 경로에 없다. branch `EMCAD`, `FCBFormer` 에서 부활할 수 있으나 main 브랜치 작업 시는 dead code 로 간주. |
| 학습 시 `DataLoader(shuffle=False, collate_fn=shuffle_within_batch)` | shuffle 을 batch 내부에서 수행. 외부 shuffle 을 켜지 말 것 — 인접 슬라이스 정렬이 깨지면 MSFFM 가정이 무의미해진다. |
| `ct_normalization` 의 상수 | 단일 hold-out 기본값 `lower=-2.0, upper=1521.0, mean=355.38, std=282.92` (train 300-case). **5-fold 경로는 이 기본값을 쓰지 않는다** — `hu_stats_433.json` (`15.0/1577.0/773.55/399.24`, 433-case 0.5/99.5 분위수) 을 `load_hu_stats` 로 읽어 `COCAVolumeDataset` 이 명시 인자로 전달한다. 두 경로의 정규화가 다르므로 절대 수치 직접 비교 금지. 다른 코호트(KMU 등) 적용 시 재산정. |
| 로컬 `datasets` vs HuggingFace `datasets` | env 에 HF `datasets`(4.5.0)가 설치돼 있어, 로컬 `datasets/` 에 `__init__.py` 가 없으면 `import datasets.dataset` 이 HF 패키지에 가로채여 실패한다. **`datasets/__init__.py`(빈 파일) 를 지우지 말 것** — 로컬 패키지 우선권을 보장하는 마커다. |
| 평가는 3D | `tester.py` 는 슬라이스 예측을 케이스별로 모아 3D 볼륨으로 합성한 뒤 MONAI 메트릭 (Dice/MeanIoU/SurfaceDistance) 을 적용한다. 2D 슬라이스 단위 메트릭이 필요하면 `compute_metrics_3d` 를 우회해야 한다. |
| 5-fold CV 시 분할 단위 | 반드시 **case 단위**로 fold 를 나눠야 한다. 슬라이스 단위 stratify 는 같은 case 의 인접 슬라이스가 train/val 양쪽에 동시 등장해 NPZ 안의 prev/ref/next 채널을 통해 raw 픽셀이 누수된다 (2.5D 가정 파괴). 층화 키는 vessel multi-hot 벡터, API 는 `MultilabelStratifiedKFold`. 결정 배경은 `TODO.md` §1.2~§1.3. |

## 7. 자주 헷갈리는 용어

- **MSFFM**: Multi-Slice Feature Fusion Module. 본 연구의 핵심 모듈명.
- **SA (Self-Attention)**: 인코더 이름 접미사 `_sa` 는 MSFFM 통합 버전을 의미. 단순 self-attention 만이 아니다 — self-attn + cross-attn (prev/next) + fusion 을 모두 포함한다.
- **2.5D**: 3D volumetric conv 없이 3장의 2D 슬라이스를 입력으로 받아 inter-slice 관계를 attention 으로만 모델링하는 설계.
- **DSC vs mIoU**: 동일한 표기지만 본 코드의 `mIoU` 는 클래스별 IoU 의 산술 평균(MONAI MeanIoU, 배경 제외 평균은 `tester.py` 에서 처리).

## 8. 변경할 때 따르는 절차

1. `MSFFM_full_20251223.pdf` 의 Methods / Ablation 절을 먼저 확인 — 실험적 의도와 어긋나는 수정인지 점검.
2. 코드 수정 시 한국어 주석 유지(전역 규칙). 영어 식별자/타입은 그대로 둘 것.
3. 학습 → 평가의 경로 규약(§5) 을 깨지 않는 한 가장 작은 변경을 적용.
4. 디버그 `print` 가 시끄럽다고 일괄 제거하지 말 것 — `residual ratio` 출력은 의도된 instrumentation 가능성이 있다 (§6).
5. `model/`, `test_log/`, `data/` 는 `.gitignore` 대상이므로 결과물을 커밋하지 않는다.

## 9. 추가 문서

- `docs/ARCHITECTURE.md` — MSFFM 내부 동작, encoder integration 흐름, attention 시각화 hook.
- `docs/DATA.md` — COCA 데이터셋 포맷, npz 구조, list 파일, CT normalization.
- `docs/EXPERIMENTS.md` — 학습/평가 명령 예시, exp_setting 명명 규약, 파인튜닝 워크플로.
- `TODO.md` — 5-fold stratified CV 전환 계획. 결정 사항, Phase 별 작업 목록, 영향 받는 코드/문서 목록, 원고 Methods 삽입 문구 초안을 모두 담는다. 5-fold 관련 작업 시 1차 참고.
- `MSFFM_full_20251223.pdf` — 원고 (figure / table 의 1차 출처).
