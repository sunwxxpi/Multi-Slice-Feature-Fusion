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
    def __init__(self, in_channels, inter_channels=None, num_heads=16, window_size=8, num_global_tokens=1):
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
            nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1),
            nn.BatchNorm2d(self.in_channels)
        )
        nn.init.constant_(self.W_z[1].weight, 0)
        nn.init.constant_(self.W_z[1].bias, 0)

    def forward(self, x_thisBranch, x_otherBranch):
        B, C, H, W = x_thisBranch.size()

        # 윈도우 분할
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
        z = self.W_z(x_un) + x_thisBranch
        return z, attention_weights

class DoubleConvDownCross(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConvDownCross, self).__init__()
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
        self.window_size = 8
        self.num_global_tokens = 1
        self.num_heads = 16

        # layer2용 Non-Local Block
        self.cross_attention_prev_2 = NonLocalBlock(in_channels=512, inter_channels=256, 
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_self_2 = NonLocalBlock(in_channels=512, inter_channels=256, 
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_next_2 = NonLocalBlock(in_channels=512, inter_channels=256, 
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.downcross_2 = DoubleConvDownCross(1536, 512, 1024)

        # layer3용 Non-Local Block
        self.cross_attention_prev_3 = NonLocalBlock(in_channels=1024, inter_channels=512,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_self_3 = NonLocalBlock(in_channels=1024, inter_channels=512,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_next_3 = NonLocalBlock(in_channels=1024, inter_channels=512,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.downcross_3 = DoubleConvDownCross(3072, 1024, 2048)

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

        # 5. stage2: layer2 (Non-Local Block 포함)
        x_prev_blocks = x_prev
        x_blocks = x_main
        x_next_blocks = x_next

        for i, b in enumerate(self.layer2):
            if i == len(self.layer2) - 1:
                skip_x_main_2 = x_blocks.clone()

                xt1, _ = self.cross_attention_prev_2(x_blocks, x_prev_blocks)
                xt2, _ = self.cross_attention_self_2(x_blocks, x_blocks)
                xt3, _ = self.cross_attention_next_2(x_blocks, x_next_blocks)

                xt = torch.cat([xt1, xt2, xt3], dim=1)
                xt_downcross = self.downcross_2(xt)

                x_blocks = xt_downcross + skip_x_main_2

            x_prev_blocks = b(x_prev_blocks)
            x_blocks = b(x_blocks)
            x_next_blocks = b(x_next_blocks)

        x_prev = x_prev_blocks
        x_main = x_blocks
        x_next = x_next_blocks

        # main branch의 feature append
        features.append(x_main)

        # 6. stage3: layer3 (Non-Local Block 포함)
        x_prev_blocks = x_prev
        x_blocks = x_main
        x_next_blocks = x_next

        for i, b in enumerate(self.layer3):
            if i == len(self.layer3) - 1:
                skip_x_main_3 = x_blocks.clone()
                
                xt1, _ = self.cross_attention_prev_3(x_blocks, x_prev_blocks)
                xt2, _ = self.cross_attention_self_3(x_blocks, x_blocks)
                xt3, _ = self.cross_attention_next_3(x_blocks, x_next_blocks)
                xt = torch.cat([xt1, xt2, xt3], dim=1)
                xt_downcross = self.downcross_3(xt)
                
                x_blocks = xt_downcross + skip_x_main_3
            
            x_prev_blocks = b(x_prev_blocks)
            x_blocks = b(x_blocks)
            x_next_blocks = b(x_next_blocks)

        x_prev = x_prev_blocks
        x_main = x_blocks
        x_next = x_next_blocks

        # main branch의 feature append
        features.append(x_main)

        # 7. stage4: layer4
        x_prev = self.layer4(x_prev)
        x_main = self.layer4(x_main)
        x_next = self.layer4(x_next)

        # main branch의 feature append
        features.append(x_main)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)