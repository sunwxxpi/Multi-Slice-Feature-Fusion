import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet
from ._base import EncoderMixin

torch.backends.cudnn.benchmark = True

# =============================================================================
# SliceFeatureFusion 모듈 (Cross-Attention)
# =============================================================================
class SliceFeatureFusion(nn.Module):
    """
    입력 feature map에 대해 cross-attention 연산을 수행하여
    다른 슬라이스의 정보를 반영한 feature를 생성합니다.
    
    - 1x1 convolution을 사용해 query, key, value를 계산합니다.
    - Attention 계산 후, residual connection을 적용합니다.
    """
    def __init__(self, in_channels, inter_channels=None, num_heads=8):
        super(SliceFeatureFusion, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2
        self.num_heads = num_heads

        assert self.inter_channels % self.num_heads == 0, "inter_channels must be divisible by num_heads"
        self.head_dim = self.inter_channels // self.num_heads

        self.query_conv = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.key_conv   = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)

        self.W_z = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels)
        )
        # BatchNorm 초기화를 0으로 하여 초기 residual 영향 최소화
        nn.init.constant_(self.W_z[1].weight, 0) # TODO: 1e-3
        nn.init.constant_(self.W_z[1].bias, 0)

    def forward(self, x_thisBranch, x_otherBranch):
        """
        Args:
            x_thisBranch (torch.Tensor): 기준 feature map, shape (B, C, H, W)
            x_otherBranch (torch.Tensor): 참조 feature map, shape (B, C, H, W)
        
        Returns:
            z (torch.Tensor): cross-attention 결과 feature, shape (B, C, H, W)
            attention_weights (torch.Tensor): attention 가중치, shape (B, num_heads, N, N)
        """
        B, C, H, W = x_thisBranch.size()
        
        # Query, Key, Value 생성
        query = self.query_conv(x_thisBranch)   # (B, inter_channels, H, W)
        key   = self.key_conv(x_otherBranch)      # (B, inter_channels, H, W)
        value = self.value_conv(x_otherBranch)    # (B, inter_channels, H, W)

        N = H * W
        # 텐서를 (B, num_heads, N, head_dim)로 변환
        query = query.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        key   = key.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        value = value.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # Attention score 계산 (Scaled Dot-Product)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, value)

        # 결과를 원래 크기로 복원 후 residual 연결
        out = out.permute(0, 1, 3, 2).contiguous().view(B, self.inter_channels, H, W)
        z = self.W_z(out)
        
        return z, attention_weights
    
# =============================================================================
# CosineDynamicFusion 모듈
# =============================================================================
class CosineDynamicFusion(nn.Module):
    """
    이전, 센터, 이후 슬라이스의 feature map을 받아서
    각 슬라이스의 글로벌 descriptor(Adaptive Avg Pooling과 Max Pooling 결합)를 계산합니다.
    센터 슬라이스와 인접 슬라이스 간의 코사인 유사도를 기반으로 동적 가중치를 산출하고,
    각 슬라이스에 가중치를 적용한 후 채널 차원에서 concat하여 fused feature를 생성합니다.
    
    출력: fused feature, shape = (B, 3 * C, H, W)
    """
    def __init__(self, pooling_type='avgmax'):
        """
        Args:
            pooling_type (str): 글로벌 descriptor 계산 방식.
                                'avgmax' (기본): 평균풀링과 최대풀링 결과를 concat.
                                'avg': 평균풀링 결과만 사용.
        """
        super(CosineDynamicFusion, self).__init__()
        self.pooling_type = pooling_type

    def forward(self, feat_prev, feat_self, feat_next):
        """
        Args:
            feat_prev (torch.Tensor): 이전 슬라이스 feature, (B, C, H, W)
            feat_self (torch.Tensor): 센터 슬라이스 feature, (B, C, H, W)
            feat_next (torch.Tensor): 이후 슬라이스 feature, (B, C, H, W)
        
        Returns:
            fused_feat (torch.Tensor): 동적 가중치가 적용되어 concat된 feature, (B, 3 * C, H, W)
        """
        B, C, H, W = feat_self.size()
        
        # 각 슬라이스의 글로벌 descriptor 계산
        desc_prev = self.compute_global_descriptor(feat_prev)  # (B, D)
        desc_self = self.compute_global_descriptor(feat_self)  # (B, D)
        desc_next = self.compute_global_descriptor(feat_next)  # (B, D)
        
        # 센터 슬라이스와 이전, 이후 슬라이스 간의 코사인 유사도 계산
        sim_prev = F.cosine_similarity(desc_self, desc_prev, dim=1).unsqueeze(1)  # (B, 1)
        sim_self = torch.ones_like(sim_prev)  # 센터 슬라이스는 자기 자신과의 유사도 1
        sim_next = F.cosine_similarity(desc_self, desc_next, dim=1).unsqueeze(1)  # (B, 1)
        
        # 3 슬라이스의 유사도를 softmax를 통해 정규화하여 가중치 산출 (B, 3)
        weights = torch.cat([sim_prev, sim_self, sim_next], dim=1)
        weights = F.softmax(weights, dim=1)
        
        # 각 슬라이스에 가중치를 적용 (브로드캐스팅)
        w_prev = weights[:, 0].view(B, 1, 1, 1)
        w_self = weights[:, 1].view(B, 1, 1, 1)
        w_next = weights[:, 2].view(B, 1, 1, 1)
        
        feat_prev_weighted = feat_prev * w_prev
        feat_self_weighted = feat_self * w_self
        feat_next_weighted = feat_next * w_next
        
        # 채널 차원에서 concat하여 fused feature 생성
        fused_feat = torch.cat([feat_prev_weighted, feat_self_weighted, feat_next_weighted], dim=1)
        
        return fused_feat

    def compute_global_descriptor(self, feat):
        """
        입력 feature map으로부터 글로벌 descriptor를 계산합니다.
        
        Args:
            feat (torch.Tensor): feature map, shape (B, C, H, W)
            
        Returns:
            descriptor (torch.Tensor): 글로벌 descriptor, shape
                - pooling_type 'avgmax': (B, 2 * C)
                - pooling_type 'avg': (B, C)
        """
        B, C, H, W = feat.shape
        if self.pooling_type == 'avgmax':
            avg_pool = F.adaptive_avg_pool2d(feat, (1, 1)).view(B, -1)
            max_pool = F.adaptive_max_pool2d(feat, (1, 1)).view(B, -1)
            descriptor = torch.cat([avg_pool, max_pool], dim=1)
        elif self.pooling_type == 'avg':
            descriptor = F.adaptive_avg_pool2d(feat, (1, 1)).view(B, -1)
        else:
            raise ValueError(f"Unknown pooling_type: {self.pooling_type}")
            
        return descriptor

# =============================================================================
# DoubleConv 모듈
# =============================================================================
class DoubleConv(nn.Module):
    """
    두 개의 3x3 convolution layer, BatchNorm, ReLU를 연속으로 적용하여
    입력 feature를 정제하는 모듈입니다.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# =============================================================================
# ResNetSAEncoder: ResNet 기반 3-slice Encoder (동적 Fusion 적용)
# =============================================================================
class ResNetSAEncoder(ResNet, EncoderMixin):
    """
    3-slice 입력(이전, 센터, 이후)을 받아서,
    ResNet의 여러 단계(stage0 ~ stage4)에서 각 슬라이스별로
    cross-attention을 적용한 후, CosineDynamicFusion 모듈로 동적 가중치 fusion을 수행합니다.
    마지막에 1x1 Conv와 DoubleConv로 채널 수를 맞추고, residual 연결을 통해 최종 feature를 생성합니다.
    
    입력 x: (B, 3, H, W) — 채널 순서: [이전, 센터, 이후]
    출력: 각 단계별 feature list
    """
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels

        # fc와 avgpool 레이어 제거 (encoder로 사용하기 위함)
        del self.fc
        del self.avgpool

        self.num_heads = 1  # 필요한 경우 변경

        # ---------------------- Stage3 ----------------------
        self.cross_attention_prev_3 = SliceFeatureFusion(in_channels=1024, inter_channels=512, num_heads=self.num_heads)
        self.cross_attention_self_3 = SliceFeatureFusion(in_channels=1024, inter_channels=512, num_heads=self.num_heads)
        self.cross_attention_next_3 = SliceFeatureFusion(in_channels=1024, inter_channels=512, num_heads=self.num_heads)
        # 동적 Fusion: 3 슬라이스의 feature를 concat (출력 채널: 3 * 1024 = 3072)
        self.dynamic_fusion_3 = CosineDynamicFusion(pooling_type='avgmax')
        # 1x1 Conv: 채널 수 축소 (3072 -> 1024)
        self.compress_3 = nn.Conv2d(3072, 1024, kernel_size=1, bias=False)
        # DoubleConv: 채널 수 1024 유지, 추가 정제 수행
        self.double_conv_3 = DoubleConv(1024, 1024, 1024)

        # ---------------------- Stage4 ----------------------
        self.cross_attention_prev_4 = SliceFeatureFusion(in_channels=2048, inter_channels=1024, num_heads=self.num_heads)
        self.cross_attention_self_4 = SliceFeatureFusion(in_channels=2048, inter_channels=1024, num_heads=self.num_heads)
        self.cross_attention_next_4 = SliceFeatureFusion(in_channels=2048, inter_channels=1024, num_heads=self.num_heads)
        self.dynamic_fusion_4 = CosineDynamicFusion(pooling_type='avgmax')
        self.compress_4 = nn.Conv2d(6144, 2048, kernel_size=1, bias=False)
        self.double_conv_4 = DoubleConv(2048, 2048, 2048)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 3-slice 입력, shape (B, 3, H, W) — [이전, 센터, 이후]
        
        Returns:
            features (list of torch.Tensor): 각 단계별 생성된 feature list
        """
        features = []
        # ------------------ Stage0: 초기 입력 ------------------
        # 원본 입력 저장
        features.append(x)
        
        # 슬라이스 분리: 이전, 센터, 이후
        x_prev = x[:, 0:1, :, :]  # (B, 1, H, W)
        x_main = x[:, 1:2, :, :]  # (B, 1, H, W)
        x_next = x[:, 2:3, :, :]  # (B, 1, H, W)

        # conv1, BatchNorm, ReLU (공유)
        x_prev = self.conv1(x_prev)
        x_prev = self.bn1(x_prev)
        x_prev = self.relu(x_prev)
        x_main = self.conv1(x_main)
        x_main = self.bn1(x_main)
        x_main = self.relu(x_main)
        x_next = self.conv1(x_next)
        x_next = self.bn1(x_next)
        x_next = self.relu(x_next)

        features.append(x_main)

        # ------------------ Stage1: MaxPool + Layer1 ------------------
        x_prev = self.maxpool(x_prev)
        x_prev = self.layer1(x_prev)
        x_main = self.maxpool(x_main)
        x_main = self.layer1(x_main)
        x_next = self.maxpool(x_next)
        x_next = self.layer1(x_next)

        features.append(x_main)

        # ------------------ Stage2: Layer2 ------------------
        x_prev = self.layer2(x_prev)
        x_main = self.layer2(x_main)
        x_next = self.layer2(x_next)

        features.append(x_main)

        # ------------------ Stage3: Layer3 + 동적 Fusion ------------------
        for block in self.layer3:
            x_prev = block(x_prev)
            x_main = block(x_main)
            x_next = block(x_next)
        
        # 각 슬라이스에 대해 cross-attention 적용
        xt1, _ = self.cross_attention_prev_3(x_main, x_prev)
        xt2, _ = self.cross_attention_self_3(x_main, x_main)
        xt3, _ = self.cross_attention_next_3(x_main, x_next)
        # CosineDynamicFusion 모듈로 동적 가중치 fusion 수행 (출력: (B, 3072, H, W))
        fused_xt = self.dynamic_fusion_3(xt1, xt2, xt3)
        # 1x1 Conv로 채널 수 축소 (3072 -> 1024)
        fused_xt = self.compress_3(fused_xt)
        # DoubleConv 적용 후 residual 연결
        xt_downcross = self.double_conv_3(fused_xt)
        
        # 잔차 경로의 평균 절대값 출력 (값이 0에 가까우면 identity 역할을 하고 있다는 의미)
        residual_mean = xt_downcross.abs().mean().item()
        print("Residual branch mean abs value:", residual_mean)
        # x_main의 norm에 대한 비율 (잔차가 x_main에 비해 매우 작으면 identity 효과가 있다는 의미)
        residual_ratio = (torch.norm(xt_downcross) / torch.norm(x_main)).item()
        print("Residual branch norm ratio:", residual_ratio)
        
        x_main = xt_downcross + x_main

        features.append(x_main)

        # ------------------ Stage4: Layer4 + 동적 Fusion ------------------
        for block in self.layer4:
            x_prev = block(x_prev)
            x_main = block(x_main)
            x_next = block(x_next)
        
        xt1, _ = self.cross_attention_prev_4(x_main, x_prev)
        xt2, _ = self.cross_attention_self_4(x_main, x_main)
        xt3, _ = self.cross_attention_next_4(x_main, x_next)
        fused_xt = self.dynamic_fusion_4(xt1, xt2, xt3)
        fused_xt = self.compress_4(fused_xt)
        xt_downcross = self.double_conv_4(fused_xt)
        
        x_main = xt_downcross + x_main

        features.append(x_main)
        
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)