import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet

from ._base import EncoderMixin

torch.backends.cudnn.benchmark = True

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

        self.global_tokens = nn.Parameter(torch.randn(self.num_global_tokens, self.inter_channels))
        
        self.W_z = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels)
        )
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

    def forward(self, x_thisBranch, x_otherBranch):
        B, C, H, W = x_thisBranch.size()
        
        query = self.query_conv(x_otherBranch)   # (B, inter_channels, H, W)
        key   = self.key_conv(x_thisBranch)      # (B, inter_channels, H, W)
        value = self.value_conv(x_thisBranch)    # (B, inter_channels, H, W)
        
        N = H * W  # 전체 픽셀 수
        query = query.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, num_heads, N, head_dim)
        key   = key.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, num_heads, N, head_dim)
        value = value.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, num_heads, N, head_dim)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention_weights, value)

        out = out.permute(0, 1, 3, 2).contiguous()  # (B, num_heads, head_dim, N)
        out = out.view(B, self.inter_channels, H, W)

        z = self.W_z(out) # (B, C, H, W)

        """ # 윈도우 분할
        x_this_win = window_partition(x_thisBranch, self.window_size)
        x_other_win = window_partition(x_otherBranch, self.window_size)

        # 쿼리, 키, 값 생성
        query = self.query_conv(x_other_win)
        key = self.key_conv(x_this_win)
        value = self.value_conv(x_this_win)

        B_win = query.shape[0]
        N = self.window_size * self.window_size

        # 멀티헤드를 위한 차원 변환
        query = query.view(B_win, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B_win, num_heads, N, head_dim)
        key = key.view(B_win, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)      # (B_win, num_heads, N, head_dim)
        value = value.view(B_win, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B_win, num_heads, N, head_dim)

        # 글로벌 토큰 추가
        global_tokens = self.global_tokens.unsqueeze(0).expand(B_win, -1, -1)
        global_tokens = global_tokens.view(B_win, self.num_heads, self.num_global_tokens, self.head_dim)
        query = torch.cat([global_tokens, query], dim=2)  # (B_win, num_heads, num_global_tokens + N, head_dim)
        key = torch.cat([global_tokens, key], dim=2)
        value = torch.cat([global_tokens, value], dim=2)

        # 어텐션 계산
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, value)  # (B_win, num_heads, num_global_tokens + N, head_dim)

        # 글로벌 토큰 제거
        out = out[:, :, self.num_global_tokens:, :]  
        out = out.permute(0, 1, 3, 2).contiguous().view(B_win, self.inter_channels, self.window_size, self.window_size)

        # 윈도우 되돌리기
        x_un = window_unpartition(out, self.window_size, H, W, B)
        z = self.W_z(x_un) """
        
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
        self.double_conv_3 = DoubleConv(1024, 1024, 1024)

        # layer4용 Non-Local Block
        self.cross_attention_prev_4 = NonLocalBlock(in_channels=2048, inter_channels=1024,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_self_4 = NonLocalBlock(in_channels=2048, inter_channels=1024,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_next_4 = NonLocalBlock(in_channels=2048, inter_channels=1024,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.compress_4 = nn.Conv2d(6144, 2048, kernel_size=1, bias=False)
        self.double_conv_4 = DoubleConv(2048, 2048, 2048)

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
        xt_downcross = self.double_conv_3(xt)
        
        # fusion 결과의 절대값 평균과 norm 비율을 출력하여 residual 경로의 효과 확인
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
        xt_downcross = self.double_conv_4(xt)
        
        x_main = xt_downcross + x_main

        # main branch의 feature append
        features.append(x_main)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)