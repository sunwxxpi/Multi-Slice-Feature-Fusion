import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet

from ._base import EncoderMixin

torch.backends.cudnn.benchmark = True

# True 로 두면 forward 안에서 residual 경로의 절대값/norm 비율을 print (디버그용).
# 평상시 False — print 와 .item() 동기화로 인한 로그 오염·GPU stall 방지.
DEBUG_RESIDUAL = False

def window_partition(x, window_size):
    # x: (B, C, H, W)
    B, C, H, W = x.shape
    num_win_h = H // window_size
    num_win_w = W // window_size
    x = x.view(B, C, num_win_h, window_size, num_win_w, window_size)
    # permute to (B, num_win_h, num_win_w, C, window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    # merge windows
    x = x.view(B * num_win_h * num_win_w, C, window_size, window_size)
    
    return x

def window_unpartition(x, window_size, H, W, B):
    # x: (B*num_win_h*num_win_w, C, window_size, window_size)
    C = x.size(1)
    num_win_h = H // window_size
    num_win_w = W // window_size
    x = x.view(B, num_win_h, num_win_w, C, window_size, window_size)
    # permute back
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    # reshape
    x = x.view(B, C, H, W)
    
    return x

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, num_heads=8, window_size=8, num_global_tokens=1):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens

        assert self.inter_channels % self.num_heads == 0, "inter_channels should be divisible by num_heads"
        self.head_dim = self.inter_channels // self.num_heads

        self.query_conv = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)

        
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

class DoubleConv(nn.Module):
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

###############################################
# Custom encoder with Non-Local and 3-slice input
###############################################
class ResNetSAEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels

        # fc, avgpool 제거
        del self.fc
        del self.avgpool

        # Non-local block 파라미터 (예: resnet50 기준)
        self.window_size = 16
        self.num_global_tokens = 1
        self.num_heads = 8

        # layer3용 Non-Local Block
        self.cross_attention_prev_3 = NonLocalBlock(in_channels=1024, inter_channels=512, 
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_self_3 = NonLocalBlock(in_channels=1024, inter_channels=512, 
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_next_3 = NonLocalBlock(in_channels=1024, inter_channels=512, 
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.compress_3 = nn.Conv2d(3072, 1024, kernel_size=1, bias=False)

        # layer4용 Non-Local Block
        self.cross_attention_prev_4 = NonLocalBlock(in_channels=2048, inter_channels=1024,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_self_4 = NonLocalBlock(in_channels=2048, inter_channels=1024,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_next_4 = NonLocalBlock(in_channels=2048, inter_channels=1024,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.compress_4 = nn.Conv2d(6144, 2048, kernel_size=1, bias=False)

    """ def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ] """

    def forward(self, x):
        # features 리스트 초기화
        features = []

        # 1. Identity feature (원본 입력)
        features.append(x)

        # 2. 채널 분리: prev, main, next
        x_prev = x[:, 0:1, :, :]   # (B,1,H,W)
        x_main = x[:, 1:2, :, :]   # (B,1,H,W)
        x_next = x[:, 2:3, :, :]   # (B,1,H,W)

        # 3. stage0: conv1 + bn1 + relu (가중치 공유)
        x_prev = self.conv1(x_prev)
        x_prev = self.bn1(x_prev)
        x_prev = self.relu(x_prev)

        x_main = self.conv1(x_main)
        x_main = self.bn1(x_main)
        x_main = self.relu(x_main)

        x_next = self.conv1(x_next)
        x_next = self.bn1(x_next)
        x_next = self.relu(x_next)

        # main branch의 feature append
        features.append(x_main)

        # 4. stage1: maxpool + layer1
        x_prev = self.maxpool(x_prev)
        x_prev = self.layer1(x_prev)

        x_main = self.maxpool(x_main)
        x_main = self.layer1(x_main)

        x_next = self.maxpool(x_next)
        x_next = self.layer1(x_next)

        # main branch의 feature append
        features.append(x_main)

        # 5. stage2: layer2
        x_prev = self.layer2(x_prev)
        x_main = self.layer2(x_main)
        x_next = self.layer2(x_next)

        # main branch의 feature append
        features.append(x_main)

        # 6. stage3: layer3 (Non-Local Block 포함)
        for i, b in enumerate(self.layer3):
            x_prev = b(x_prev)
            x_main = b(x_main)
            x_next = b(x_next)

        xt1, _ = self.cross_attention_prev_3(x_main, x_prev)
        xt2, _ = self.cross_attention_self_3(x_main, x_main)
        xt3, _ = self.cross_attention_next_3(x_main, x_next)
        xt = torch.cat([xt1, xt2, xt3], dim=1)
        
        xt = self.compress_3(xt)
        xt_downcross = xt
        
        # fusion 결과의 절대값 평균과 norm 비율로 residual 경로 효과 확인 (디버그 시에만)
        if DEBUG_RESIDUAL:
            residual_mean = xt_downcross.abs().mean().item()
            print("Residual branch mean abs value:", residual_mean)
            residual_ratio = (torch.norm(xt_downcross) / torch.norm(x_main)).item()
            print("Residual branch norm ratio:", residual_ratio)
        
        x_main = xt_downcross + x_main

        # main branch의 feature append
        features.append(x_main)

        # 7. stage4: layer4 (Non-Local Block 포함)
        for i, b in enumerate(self.layer4):
            x_prev = b(x_prev)
            x_main = b(x_main)
            x_next = b(x_next)

        xt1, _ = self.cross_attention_prev_4(x_main, x_prev)
        xt2, _ = self.cross_attention_self_4(x_main, x_main)
        xt3, _ = self.cross_attention_next_4(x_main, x_next)
        xt = torch.cat([xt1, xt2, xt3], dim=1)

        xt = self.compress_4(xt)
        xt_downcross = xt
        
        x_main = xt_downcross + x_main

        # main branch의 feature append
        features.append(x_main)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)