import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet
from ._base import EncoderMixin

# 고정된 입력 크기의 경우, cudnn의 벤치마크 기능을 활성화하여 연산 최적화를 도모합니다.
torch.backends.cudnn.benchmark = True

# True 로 두면 forward 안에서 residual 경로의 절대값/norm 비율을 print (디버그용).
# 평상시 False — print 와 .item() 동기화로 인한 로그 오염·GPU stall 방지.
DEBUG_RESIDUAL = False

# =============================================================================
# MultiSliceFeatureFusion 모듈: Cross-Attention 연산 수행
# =============================================================================
class MultiSliceFeatureFusion(nn.Module):
    """
    이 모듈은 두 개의 feature map 간의 정보를 상호 보완적으로 반영하기 위해
    cross-attention 연산을 수행합니다.
    
    - 1x1 Convolution을 이용해 입력 feature로부터 query, key, value를 생성합니다.
    - Scaled Dot-Product Attention 방식을 이용해 attention 가중치를 계산합니다.
    - 최종적으로 projection layer와 residual 연결을 적용하여 출력을 생성합니다.
    """
    def __init__(self, in_channels, inter_channels=None, num_heads=8):
        super(MultiSliceFeatureFusion, self).__init__()
        self.in_channels = in_channels
        # 중간 채널 수가 명시되지 않은 경우, 입력 채널의 절반을 사용
        self.inter_channels = inter_channels or in_channels // 2
        self.num_heads = num_heads

        # multi-head attention을 위해 inter_channels가 num_heads로 나누어 떨어져야 합니다.
        assert self.inter_channels % self.num_heads == 0, "inter_channels must be divisible by num_heads"
        self.head_dim = self.inter_channels // self.num_heads

        # 1x1 Convolution 레이어로 query, key, value를 생성
        self.query_conv = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.key_conv   = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)

        # attention 결과를 원래 채널 수로 복원하는 projection layer (BatchNorm 포함)
        self.W_z = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels)
        )
        # zero-init residual: 초기엔 fusion=identity 로 시작해 pretrained feature 보호,
        # 학습하며 BatchNorm gamma 가 0 에서 자라 fusion 이 점진적으로 engage.
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)
        # 평상시 fused SDPA 사용. True 면 시각화용으로 attention 가중치를 명시 계산.
        self.return_attention = False

    def forward(self, x_thisBranch, x_otherBranch):
        # 논문 §2.2: Query=reference(x_thisBranch), Key/Value=상대 슬라이스(x_otherBranch).
        # 동일 입력이면 self-attention 으로 환원. 전체 (H*W)x(H*W) multi-head attention.
        B, C, H, W = x_thisBranch.size()

        query = self.query_conv(x_thisBranch)
        key   = self.key_conv(x_otherBranch)
        value = self.value_conv(x_otherBranch)

        N = H * W
        query = query.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        key   = key.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        value = value.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        if self.return_attention:
            # 시각화 경로: fused 커널은 가중치를 반환하지 않으므로 명시적으로 계산.
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attention_weights = F.softmax(attention_scores, dim=-1)
            out = torch.matmul(attention_weights, value)
        else:
            # fused scaled dot-product attention: 동일 연산, 메모리/속도 이득 (scale 기본 1/sqrt(head_dim)).
            out = F.scaled_dot_product_attention(query.contiguous(), key.contiguous(), value.contiguous())
            attention_weights = None

        out = out.permute(0, 1, 3, 2).contiguous().view(B, self.inter_channels, H, W)
        z = self.W_z(out)

        return z, attention_weights
    
# =============================================================================
# CosineDynamicFusion 모듈: 코사인 유사도 기반 동적 Fusion 수행
# =============================================================================
class CosineDynamicFusion(nn.Module):
    """
    이 모듈은 이전, 중앙, 이후 슬라이스의 feature map을 입력으로 받아,
    각 슬라이스의 글로벌 descriptor(평균 및 최대 풀링 결과)를 산출한 후,
    중앙 슬라이스와 인접 슬라이스 간의 코사인 유사도를 통해 동적 가중치를 계산합니다.
    이후 각 슬라이스에 해당 가중치를 적용하고, channel 차원에서 feature들을 concat하여
    최종 fusion feature map을 생성합니다.
    
    최종 출력은 (B, 3 * C, H, W) 형태의 feature map입니다.
    """
    def __init__(self, pooling_type='avgmax'):
        """
        Args:
            pooling_type (str): 글로벌 descriptor 산출 방식 선택.
                                'avgmax' (기본): 평균 풀링과 최대 풀링 결과를 결합.
                                'avg': 평균 풀링 결과만 사용.
        """
        super(CosineDynamicFusion, self).__init__()
        self.pooling_type = pooling_type

    def compute_global_descriptor(self, feat):
        """
        입력 feature map으로부터 글로벌 descriptor를 산출합니다.
        
        Args:
            feat (torch.Tensor): feature map, shape (B, C, H, W)
            
        Returns:
            descriptor (torch.Tensor): 
                - 'avgmax' 방식일 경우: (B, 2 * C) (평균 풀링과 최대 풀링 결과 결합)
                - 'avg' 방식일 경우: (B, C) (평균 풀링 결과만 사용)
        """
        B, C, H, W = feat.shape
        if self.pooling_type == 'avgmax':
            # 평균 풀링과 최대 풀링을 각각 적용 후 결과를 결합
            avg_pool = F.adaptive_avg_pool2d(feat, (1, 1)).view(B, -1)
            max_pool = F.adaptive_max_pool2d(feat, (1, 1)).view(B, -1)
            descriptor = torch.cat([avg_pool, max_pool], dim=1)
        elif self.pooling_type == 'avg':
            # 평균 풀링 결과만 사용
            descriptor = F.adaptive_avg_pool2d(feat, (1, 1)).view(B, -1)
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
            
        return descriptor
    
    def forward(self, feat_prev, feat_self, feat_next):
        """
        Args:
            feat_prev (torch.Tensor): 이전 슬라이스의 feature, shape (B, C, H, W)
            feat_self (torch.Tensor): 중앙 슬라이스의 feature, shape (B, C, H, W)
            feat_next (torch.Tensor): 이후 슬라이스의 feature, shape (B, C, H, W)
        
        Returns:
            fused_feat (torch.Tensor): 가중치가 적용되어 channel 방향으로 concat된 feature, shape (B, 3 * C, H, W)
        """
        B, C, H, W = feat_self.size()
        
        # 각 슬라이스의 글로벌 descriptor 계산 (평균, 최대 풀링 또는 둘의 결합)
        desc_prev = self.compute_global_descriptor(feat_prev)  # shape: (B, D)
        desc_self = self.compute_global_descriptor(feat_self)  # shape: (B, D)
        desc_next = self.compute_global_descriptor(feat_next)  # shape: (B, D)
        
        # 중앙 슬라이스와 이전/이후 슬라이스 간의 코사인 유사도 산출
        sim_prev = F.cosine_similarity(desc_self, desc_prev, dim=1).unsqueeze(1)  # shape: (B, 1)
        sim_self = torch.ones_like(sim_prev)  # 자기 자신과의 유사도는 1
        sim_next = F.cosine_similarity(desc_self, desc_next, dim=1).unsqueeze(1)  # shape: (B, 1)
        
        # 세 슬라이스의 유사도를 하나로 결합 후 softmax 적용하여 동적 가중치 생성
        weights = torch.cat([sim_prev, sim_self, sim_next], dim=1)  # shape: (B, 3)
        weights = F.softmax(weights, dim=1)
        
        # 계산된 가중치를 각 슬라이스에 곱하기 위해 차원 확장 (브로드캐스팅)
        w_prev = weights[:, 0].view(B, 1, 1, 1)
        w_self = weights[:, 1].view(B, 1, 1, 1)
        w_next = weights[:, 2].view(B, 1, 1, 1)
        
        # 가중치를 적용하여 feature map을 조정
        feat_prev_weighted = feat_prev * w_prev
        feat_self_weighted = feat_self * w_self
        feat_next_weighted = feat_next * w_next
        
        # 조정된 feature map들을 channel 차원에서 연결하여 최종 fusion feature map 생성
        fused_feat = torch.cat([feat_prev_weighted, feat_self_weighted, feat_next_weighted], dim=1)
        
        return fused_feat

# =============================================================================
# DoubleConv 모듈: 두 번의 3x3 Convolution + BatchNorm + ReLU 연산 수행
# =============================================================================
class DoubleConv(nn.Module):
    """
    이 모듈은 3x3 Convolution, Batch Normalization, ReLU 활성화 함수를 연속 두 번 적용하여
    입력 feature map의 정제 및 표현력 향상을 도모합니다.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels  # 중간 채널 수가 지정되지 않으면 출력 채널 수로 설정
        self.double_conv = nn.Sequential(
            # 첫 번째 Convolution: padding=1로 입력과 출력의 공간 차원 유지
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 두 번째 Convolution
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# =============================================================================
# ResNetSAEncoder 모듈: ResNet 기반 3-slice Encoder
# =============================================================================
class ResNetSAEncoder(ResNet, EncoderMixin):
    """
    이 Encoder는 3개의 슬라이스(이전, 중앙, 이후)를 입력으로 받아,
    ResNet의 여러 단계(stage)에서 각 슬라이스별로 개별 연산을 수행한 후,
    cross-attention 결과를 concat 후 1x1 conv 로 융합합니다.

    주요 처리 단계:
      - 입력 슬라이스 분리 및 초기 convolution 처리
      - 각 stage별로 동일한 네트워크 블록을 독립적으로 적용
      - Stage3와 Stage4에서 cross-attention 후 concat + 1x1 conv fusion 수행
      - residual 연결을 통해 원본 정보와 fusion 결과를 결합
      
    입력:
        x: (B, 3, H, W) — 순서대로 [이전, 중앙, 이후] 슬라이스
    출력:
        features: 각 단계에서 추출된 feature map들의 리스트
    """
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels

        # Encoder로 사용하기 위해 fc와 avgpool 레이어 제거
        del self.fc
        del self.avgpool

        self.num_heads = 8  # cross-attention 연산에 사용할 헤드 수 (필요 시 조정 가능)

        # ---------------------- Stage3: Layer3 이후 Fusion ----------------------
        # 각 슬라이스별 cross-attention 모듈 구성 (채널: 1024 -> 512)
        self.cross_attention_prev_3 = MultiSliceFeatureFusion(in_channels=1024, inter_channels=512, num_heads=self.num_heads)
        self.cross_attention_self_3 = MultiSliceFeatureFusion(in_channels=1024, inter_channels=512, num_heads=self.num_heads)
        self.cross_attention_next_3 = MultiSliceFeatureFusion(in_channels=1024, inter_channels=512, num_heads=self.num_heads)
        # 세 슬라이스 attention 출력을 채널 방향으로 concat 후 1x1 conv 로 채널 복원 (3072 -> 1024)
        self.compress_3 = nn.Conv2d(3072, 1024, kernel_size=1, bias=False)

        # ---------------------- Stage4: Layer4 이후 Fusion ----------------------
        self.cross_attention_prev_4 = MultiSliceFeatureFusion(in_channels=2048, inter_channels=1024, num_heads=self.num_heads)
        self.cross_attention_self_4 = MultiSliceFeatureFusion(in_channels=2048, inter_channels=1024, num_heads=self.num_heads)
        self.cross_attention_next_4 = MultiSliceFeatureFusion(in_channels=2048, inter_channels=1024, num_heads=self.num_heads)
        self.compress_4 = nn.Conv2d(6144, 2048, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 3-slice 입력, shape (B, 3, H, W) — 각 슬라이스 순서대로 [이전, 중앙, 이후]
        
        Returns:
            features (list of torch.Tensor): 각 단계별로 추출된 feature map들의 리스트
        """
        features = []
        # ---------- Stage0: 원본 입력 단계 ----------
        # 초기 입력 자체를 feature 리스트에 추가
        features.append(x)
        
        # 입력을 슬라이스별로 분리 (채널 차원 기준)
        x_prev = x[:, 0:1, :, :]  # 이전 슬라이스, shape: (B, 1, H, W)
        x_main = x[:, 1:2, :, :]  # 중앙 슬라이스, shape: (B, 1, H, W)
        x_next = x[:, 2:3, :, :]  # 이후 슬라이스, shape: (B, 1, H, W)

        # 각 슬라이스에 대해 초기 convolution, BatchNorm, ReLU를 독립적으로 적용 (공유 가중치)
        x_prev = self.conv1(x_prev)
        x_prev = self.bn1(x_prev)
        x_prev = self.relu(x_prev)
        x_main = self.conv1(x_main)
        x_main = self.bn1(x_main)
        x_main = self.relu(x_main)
        x_next = self.conv1(x_next)
        x_next = self.bn1(x_next)
        x_next = self.relu(x_next)

        # 중앙 슬라이스 feature를 feature 리스트에 저장
        features.append(x_main)

        # ---------- Stage1: MaxPool 및 Layer1 적용 ----------
        # 각 슬라이스별로 max pooling과 Layer1을 적용하여 공간 해상도를 축소
        x_prev = self.maxpool(x_prev)
        x_prev = self.layer1(x_prev)
        x_main = self.maxpool(x_main)
        x_main = self.layer1(x_main)
        x_next = self.maxpool(x_next)
        x_next = self.layer1(x_next)

        # Stage1의 중앙 feature 저장
        features.append(x_main)

        # ---------- Stage2: Layer2 적용 ----------
        # 각 슬라이스별로 Layer2 블록 적용
        x_prev = self.layer2(x_prev)
        x_main = self.layer2(x_main)
        x_next = self.layer2(x_next)

        # Stage2의 중앙 feature 저장
        features.append(x_main)

        # ---------- Stage3: Layer3 및 Fusion ----------
        # 각 슬라이스에 대해 Layer3의 블록들을 순차적으로 적용
        for block in self.layer3:
            x_prev = block(x_prev)
            x_main = block(x_main)
            x_next = block(x_next)
        
        # 중앙 슬라이스 feature를 기준으로 cross-attention 연산 수행
        xt1, _ = self.cross_attention_prev_3(x_main, x_prev)
        xt2, _ = self.cross_attention_self_3(x_main, x_main)
        xt3, _ = self.cross_attention_next_3(x_main, x_next)
        # cross-attention 결과들을 채널 방향으로 concat (출력 shape: (B, 3072, H, W))
        fused_xt = torch.cat([xt1, xt2, xt3], dim=1)
        # 1x1 Convolution을 통해 채널 수를 축소 (3072 -> 1024)
        fused_xt = self.compress_3(fused_xt)
        xt_downcross = fused_xt
        
        # fusion 결과의 절대값 평균과 norm 비율로 residual 경로 효과 확인 (디버그 시에만)
        if DEBUG_RESIDUAL:
            residual_mean = xt_downcross.abs().mean().item()
            print("Residual branch mean abs value:", residual_mean)
            residual_ratio = (torch.norm(xt_downcross) / torch.norm(x_main)).item()
            print("Residual branch norm ratio:", residual_ratio)
        
        # residual 연결을 통해 중앙 feature에 fusion 결과를 더함
        x_main = xt_downcross + x_main

        # Stage3의 최종 중앙 feature를 저장
        features.append(x_main)

        # ---------- Stage4: Layer4 및 Fusion ----------
        # 각 슬라이스별로 Layer4의 블록들을 적용
        for block in self.layer4:
            x_prev = block(x_prev)
            x_main = block(x_main)
            x_next = block(x_next)
        
        # Stage4에서 cross-attention 연산 수행
        xt1, _ = self.cross_attention_prev_4(x_main, x_prev)
        xt2, _ = self.cross_attention_self_4(x_main, x_main)
        xt3, _ = self.cross_attention_next_4(x_main, x_next)
        # 세 슬라이스 결과를 채널 방향으로 concat
        fused_xt = torch.cat([xt1, xt2, xt3], dim=1)
        # 1x1 Convolution으로 채널 수 축소 (6144 -> 2048)
        fused_xt = self.compress_4(fused_xt)
        xt_downcross = fused_xt
        
        # residual 연결을 통해 중앙 feature에 fusion 결과를 결합
        x_main = xt_downcross + x_main

        # Stage4의 최종 중앙 feature를 저장
        features.append(x_main)
        
        return features

    def load_state_dict(self, state_dict, **kwargs):
        # Encoder로 사용할 때 불필요한 fc 레이어 파라미터 제거
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        # 부모 클래스의 load_state_dict를 호출 (유연한 로드를 위해 strict=False)
        super().load_state_dict(state_dict, strict=False, **kwargs)