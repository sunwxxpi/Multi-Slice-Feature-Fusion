# Architecture — MSFFM 통합 2.5D 세그멘테이션 프레임워크

본 문서는 원고(`MSFFM_full_20251223.pdf`) §2 와 코드 베이스의 연결고리를 설명한다. CLAUDE.md 가 "무엇을 건드리지 말 것"의 관점이라면 여기는 "어떻게 동작하는가"의 관점이다.

## 1. 전체 데이터 흐름

```
NPZ (image: HxWx3, label: HxW)
   └─► COCA_dataset
         ├─ ct_normalization  (clip + z-score)
         ├─ RandomAugmentation (rot90 / flip / rotate)
         ├─ Resize → 512x512
         └─ ToTensor → (3, H, W), int64 label
            │
            ▼
   smp.Unet / smp.Segformer (in_channels=1, classes=5)
            │
            ▼
   *_sa Encoder  ──► features[0..5]  (stage0~stage5)
            │            stage3·stage4 에 MSFFM 통합
            ▼
   Decoder (UnetDecoder / SegformerDecoder)
            │
            ▼
   SegmentationHead  → logits (B, 5, H, W)
```

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

- `tester.py:get_attn_hook` 가 `(z, attention_weights)` 튜플의 두 번째 원소를 `attn_dict` 에 저장한다. forward 는 평상시 fused SDPA 경로로 `attention_weights=None` 을 반환하므로, 시각화하려면 대상 `NonLocalBlock` 의 `return_attention=True` 로 명시 계산 경로를 켜야 가중치가 나온다.
- `test.py` 에 hook 등록 코드가 주석 처리되어 있다 (lines 96–103). 사용 시:
  1. 대상 `NonLocalBlock` 들의 `return_attention=True` 설정
  2. `test.py` 의 hook 블록 주석 해제
  3. `tester.py:inference` 의 `visualize_attention(...)` 호출 주석 해제
  4. NonLocalBlock 의 attention head 가 1 이상이어야 의미 있는 시각화가 나온다.
- 저장 위치는 `test_save_path/attention_vis/` (없으면 `./test_log/attention_vis/`).
- query 픽셀은 ground-truth lesion 의 평균 좌표로 자동 선정된다. GT 가 없으면 해당 슬라이스는 스킵.

## 6. 결과물 형식

- 추론은 슬라이스별로 수행 후 `accumulate_slice_prediction` 가 케이스별로 dict 에 쌓는다.
- `build_3d_volume` 가 slice id 순서대로 D×H×W 텐서를 만들고 `is_savenii` 시 NIfTI 로 기록한다.
  - spacing = `(0.375, 0.375, args.z_spacing)`. 실제 spacing 이 다른 데이터셋을 쓰면 수정 필요.
  - `np.flip(np.transpose(array, (0, 2, 1)), (1, 2))` 로 축을 재배열하므로 ITK-SNAP 호환을 위해 의도된 변환임 — 임의 제거 금지.
- 메트릭은 클래스별 (`1..num_classes-1`, 배경 제외) Dice / mIoU / SurfaceDistance 를 계산하고, `log_3d_metrics` 가 전체 케이스 평균을 로그한다.
