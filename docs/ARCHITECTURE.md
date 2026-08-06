# Architecture — MSFFM 통합 2.5D 세그멘테이션 프레임워크

본 문서는 원고(`MSFFM_full_20251223.pdf`) §2 와 코드 베이스의 연결고리를 설명한다. CLAUDE.md 가 "무엇을 건드리지 말 것"의 관점이라면 여기는 "어떻게 동작하는가"의 관점이다.

## 1. 전체 데이터 흐름

```
per-case .npy (D,H,W) memmap
   └─► COCAVolumeDataset (num_slices=3 → vol[n:n+3]→(H,W,3) prev/center/next, num_slices=1 → center (H,W,1); vol[n+1] center label)
         ├─ ct_normalization  (clip + z-score)
         ├─ RandomAugmentation (rot90 / flip / rotate)
         ├─ Resize → 512x512
         └─ ToTensor → (C, H, W), int64 label
            │
            ▼
   --decoder 가 4가지 모델 경로 중 하나를 선택 (`utils.py:derive_num_slices` 가 C 를 결정)

   --decoder unet/segformer                     --decoder emcad/emcad_sa
   ────────────────────────                     ─────────────────────────
   smp.Unet / smp.Segformer                     EMCADNet / EMCAD_SA_Net
      │ in_channels=1, classes=5                    │ encoder=pvt_v2_bN
      ▼                                             ▼
   *_sa Encoder ──► features[0..5]               pvt_v2 backbone (use_msffm=False/True)
      │ stage3·4 에 MSFFM 통합(_sa 계열만)          │ stage3·4 에 MSFFM 통합(emcad_sa 만)
      ▼                                             ▼
   UnetDecoder / SegformerDecoder                EMCAD 디코더 (networks/emcad/decoders.py)
      │                                             │
      ▼                                             ▼
   SegmentationHead                              out_head1~4 (deep supervision, 최종단이 예측에 쓰임)
      │                                             │
      └───────────────────┬─────────────────────────┘
                           ▼
               logits (B, 5, H, W)
```

> `COCAVolumeDataset` 하나가 모든 `--decoder` 경로의 입력을 만든다 — encoder 이후 흐름만 `--decoder` 별로 갈린다. 5-fold 데이터 자산·정규화 상수는 `docs/DATA.md §9`.

## 2. MSFFM 핵심 식 (원고 §2.2)

세 슬라이스의 feature map 을 입력으로 받아 다음을 계산한다.

```
Z_ref  = SA(X_ref)
Z_prev = CA(X_ref, X_prev)
Z_next = CA(X_ref, X_next)
Z_fused = W_c · Concat(Z_prev, Z_ref, Z_next)        # 채널 1x1 conv 로 복원
Z_final = Z_fused ⊕ X_ref                            # residual
```

`SA`, `CA` 모두 1×1 conv 로 Q/K/V 를 만든 뒤 multi-head scaled dot-product attention 을 수행한다. cross-attention 은 `Q := W_q · X_ref`, `K, V := W_k/v · X_s` (s ∈ {prev,next}).

## 3. 코드와의 매핑

MSFFM 은 코드베이스에 두 가지 형태로 존재한다 — SMP 인코더의 `*_sa` 계열(§3.1~3.3)과 EMCAD 백본(§3.4). 둘 다 stage 3/4 에 `NonLocalBlock` 3개(prev/self/next) + `compress` conv(1×1) + residual 구조로 동일하다.

### 3.1 활성 구현 (`segmentation_models_pytorch/encoders/resnet_sa.py`)

- `NonLocalBlock` 클래스가 attention 한 쌍을 담당한다. 호출은 `cross_attention_*(x_thisBranch, x_otherBranch)` 이고 매핑은 (원고 §2.2)
  - `query := W_q · x_thisBranch`  (reference 슬라이스)
  - `key   := W_k · x_otherBranch` (이웃 슬라이스 prev/next)
  - `value := W_v · x_otherBranch` 로, `cross_attention_self_3(x_main, x_main)` 호출 시 self-attention 으로 환원된다.
- attention 은 전체 `(H·W)×(H·W)` multi-head (num_heads=8) scaled dot-product 다.
- `ResNetSAEncoder.forward` 는 입력 `(B, 3, H, W)` 를 채널 축으로 `prev/main/next` 로 분리한 뒤 stage0~stage4 를 **가중치 공유로 3번 통과**한다.
- Stage 3 종료 직후:
  ```python
  xt1, _ = cross_attention_prev_3(x_main, x_prev)
  xt2, _ = cross_attention_self_3(x_main, x_main)
  xt3, _ = cross_attention_next_3(x_main, x_next)
  xt = torch.cat([xt1, xt2, xt3], dim=1)
  xt = compress_3(xt)            # 3072 → 1024 (W_c, 1x1 conv)
  x_main = xt + x_main           # residual (W_z BN zero-init warm-start)
  ```
  `compress` 출력이 곧바로 residual 로 더해진다. Stage 4 도 동일한 구조 (채널만 2048→1024 inter, 6144→2048 compress).
- `features` 리스트는 `[input, stage0_main, stage1_main, stage2_main, stage3_fused, stage4_fused]` 순서로 채워져 SMP 디코더가 받는 형식과 호환된다.

### 3.2 참고 구현 (`multi_slice_feature_fusion.py`)

같은 디렉터리에 또 다른 `ResNetSAEncoder` 가 있다. 활성 구현(`resnet_sa.py`)과 동일한 paper-faithful fusion 을 따른다 — `MultiSliceFeatureFusion` 의 Q=ref / K,V=이웃, 전체 attention, num_heads=8, `W_z` BN zero-init residual. forward 는 `concat → compress(1x1) → +x_main` 으로 융합한다.

- `CosineDynamicFusion` (글로벌 descriptor 의 cosine similarity 로 동적 가중치 산출) 과 `DoubleConv` (3x3 conv ×2) 클래스는 파일에 **정의만 남아 있고 현재 forward 경로에서 호출되지 않는다** — 실험 변형 보관용.
- 이 인코더는 **어떤 encoder registry 에도 등록되어 있지 않다**. 새로 도입하려면 `resnet.py` 의 `resnet_encoders` 딕셔너리에 명시적으로 키를 추가해야 한다.

### 3.3 그 외 백본

- `densenet_sa.py`, `efficientnet_sa.py`, `mix_transformer_sa.py` 모두 같은 패턴을 따른다.
- 각 파일은 해당 백본의 원본(non-SA) 모듈을 `from .{backbone} import {Backbone}Encoder` 식으로 import 하지 않고, 백본별로 forward 흐름을 다시 작성해 두었다. 따라서 stage 단위 hook 위치 (어디서 MSFFM 을 끼울지) 가 백본마다 다를 수 있다 — 코드의 stage 주석을 직접 따라가야 한다.

### 3.4 EMCAD 경로 (`networks/emcad/pvtv2.py`)

- `PyramidVisionTransformerImpr.__init__` 이 `use_msffm=True` 일 때 stage3(채널 320)·stage4(채널 512) 에 각각 `cross_attention_{prev,self,next}_{3,4}` (`NonLocalBlock`, num_heads=8) 와 `compress_{3,4}` (1×1 conv) 를 만든다. `use_msffm=False` 면 아무것도 만들지 않는다 — SMP 쪽처럼 별도 클래스가 아니라 같은 백본 클래스의 생성자 플래그 분기다.
- `EMCADNet` 은 `use_msffm=False` 로 고정, `EMCAD_SA_Net(EMCADNet)` 은 `__init__` 에서 `use_msffm=True` 를 강제한다. 두 클래스가 다르므로 체크포인트 경로(`net.__class__.__name__`)가 자동으로 갈린다.
- `forward_features` 의 `_fuse(x_main, x_prev, x_next, prev_attn, self_attn, next_attn, compress)` 가 융합을 담당: `compress(cat(prev_attn(x_main,x_prev), self_attn(x_main,x_main), next_attn(x_main,x_next))) + x_main` — SMP 경로의 `compress(cat(...)) + x_main` 과 동일한 구조.
- `pvt_v2_b0` 은 stage3/4 채널이 160/256 이라 `NonLocalBlock` 의 하드코딩 채널(320/512)과 안 맞아 `use_msffm=True` 를 assert 로 거부한다 — `--decoder emcad_sa` 의 encoder 허용 목록이 `pvt_v2_b1`~`b5` 인 이유(`utils.py:EMCAD_SA_ENCODERS`).

## 4. Encoder Registry 흐름

```
train.py: smp.Unet(encoder_name="resnet50_sa", ...)
      └─► smp.encoders.get_encoder("resnet50_sa", in_channels=1, weights="imagenet")
            └─► encoders["resnet50_sa"]
                  ├─ encoder       : ResNetSAEncoder        # from resnet_sa.py
                  ├─ params        : out_channels, block, layers
                  └─ pretrained    : ImageNet-1k (resnet50 가중치 그대로)
```

- ImageNet 가중치는 SA 모듈에 대응되는 key 가 없으므로 `load_state_dict(strict=False)` 로 로드 된다 (`ResNetSAEncoder.load_state_dict` 가 `fc.bias/weight` 를 pop 한 뒤 strict=False).
- `set_in_channels(1)` 은 conv1 의 in_channels 만 패치한다. cross_attention_* 모듈은 영향 없음.

## 5. Attention 시각화 hook

- `tester.py:get_attn_hook` 가 `(z, attention_weights)` 튜플의 두 번째 원소를 `attn_dict` 에 저장한다. 평상시 forward 는 fused SDPA 경로(`attention_weights=None`) — 시각화하려면 대상 `NonLocalBlock` 의 `return_attention=True` 로 명시 계산 경로를 켜야 가중치가 나온다.
- **사용법: `test.py --save_attention` 단일 플래그.** `test.py` 가 자동으로
  1. `net.named_modules()` 로 모델 트리 전체를 순회(`net.encoder`(SMP) 든 `net.backbone`(EMCAD) 든 위치 무관하게 잡는다) → `return_attention` 속성 보유 모듈(모든 `NonLocalBlock`) 자동 검색
  2. 각 모듈의 `return_attention=True` 토글
  3. 모듈명(`cross_attention_prev_3` 등)을 `visualize_attention` 이 기대하는 키 `stage{N}_{prev|self|next}` 로 정규화한 뒤 hook 등록.
  를 수행. 백본 무관 동작 (resnet50_sa / densenet201_sa / efficientnet-b4_sa / mit_b2_sa / emcad_sa 공통).
- `tester.py:inference` 는 `args.save_attention` 가 True 일 때만 `visualize_attention(...)` 호출 + `attn_vis_dir` mkdir 수행. OFF 시 빈 디렉터리 생성도 없음.
- 저장 위치: `test_save_path/attention_vis/` (= `--is_savenii` 켜진 경우) 또는 fallback `./test_log/attention_vis_fallback/{exp_setting}/` (exp_setting 포함하여 run 간 섞임 방지).
- 메모리 안전: 매 slice 시작 시 `attn_dict.clear()` — 시각화 OFF + hook ON 같은 잘못된 조합에서도 누수 없음.
- 수치 안정: 명시 attention 경로는 autocast(fp16) 안이라도 Q/K 를 fp32 로 cast 후 softmax — overflow/underflow 방지.
- query 픽셀은 ground-truth lesion 의 평균 좌표로 자동 선정. GT 없으면 해당 슬라이스 skip.
- NonLocalBlock 의 attention head 는 8 (논문 명시값) — 의미 있는 시각화에 충분.
- **히트맵 스케일 = uniform 대비 배수** (`attn × N`, 1.0 = 무선호). 이전 per-panel min-max 정규화는 거의 균일한 분포도 풀스케일 jet 로 칠해 가짜 핫스팟을 만들었다. 6 패널이 공통 `vmin=0 / vmax=max(5, per-panel peak 의 median)` 으로 그려져 강도 직접 비교가 가능하고, 제목의 `attn@box`(query 가 자기 lesion 위치를 보는 배수)·`peak`(어디든 최대 집중 배수)로 정량 비교한다.
- 표시 방향은 `VIS_ROT90_CCW`(기본 1, CCW 90°)로 회전 — COCA axial 의 누운 cardiac FOV 를 척추 아래 표준 방향으로 맞춘다. 이미지·박스·query 좌표를 `_rot90_cell` 로 동일 회전하므로 정렬이 유지된다 (0 으로 두면 원본 array 방향).

## 6. 결과물 형식

- 추론은 슬라이스별로 수행 후 `accumulate_slice_prediction` 가 케이스별로 dict 에 쌓는다.
- `build_3d_volume` 가 slice id 순서대로 D×H×W 텐서를 만들고 `is_savenii` 시 NIfTI 로 기록한다.
  - spacing = `(0.375, 0.375, args.z_spacing)`. 실제 spacing 이 다른 데이터셋을 쓰면 수정 필요.
  - `np.flip(np.transpose(array, (0, 2, 1)), (1, 2))` 로 축을 재배열하므로 ITK-SNAP 호환을 위해 의도된 변환임 — 임의 제거 금지.
- 메트릭은 클래스별 (`1..num_classes-1`, 배경 제외) Dice / mIoU / SurfaceDistance 를 계산하고, `log_3d_metrics` 가 전체 케이스 평균을 로그한다.
