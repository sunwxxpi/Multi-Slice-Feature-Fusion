"""Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import url_map, url_map_advprop, get_model_params

from ._base import EncoderMixin


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


class EfficientNetSAEncoder(EfficientNet, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):
        blocks_args, global_params = get_model_params(model_name, override_params=None)
        super().__init__(blocks_args, global_params)

        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self._fc
        
        # Non-local block 파라미터
        self.window_size = 16
        self.num_global_tokens = 1
        self.num_heads = 8

        # stage4용 Non-Local Block
        self.cross_attention_prev_4 = NonLocalBlock(in_channels=160, inter_channels=80, 
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_self_4 = NonLocalBlock(in_channels=160, inter_channels=80, 
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_next_4 = NonLocalBlock(in_channels=160, inter_channels=80, 
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.compress_4 = nn.Conv2d(480, 160, kernel_size=1, bias=False)

        # stage5용 Non-Local Block
        self.cross_attention_prev_5 = NonLocalBlock(in_channels=448, inter_channels=224,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_self_5 = NonLocalBlock(in_channels=448, inter_channels=224,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_next_5 = NonLocalBlock(in_channels=448, inter_channels=224,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.compress_5 = nn.Conv2d(1344, 448, kernel_size=1, bias=False)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._conv_stem, self._bn0, self._swish),
            self._blocks[: self._stage_idxs[0]], # :6
            self._blocks[self._stage_idxs[0] : self._stage_idxs[1]], # 6:10
            self._blocks[self._stage_idxs[1] : self._stage_idxs[2]], # 10:22
            self._blocks[self._stage_idxs[2] :], # 22:(32)
        ]

    def forward(self, x):
        stages = self.get_stages()

        block_number = 0.0
        drop_connect_rate = self._global_params.drop_connect_rate

        features = []
        
        x_prev = x[:, 0:1, :, :]  # (B,1,H,W)
        x_main = x[:, 1:2, :, :]  # (B,1,H,W)
        x_next = x[:, 2:3, :, :]  # (B,1,H,W)
        
        for i in range(self._depth + 1):
            # Identity and Sequential stages
            if i < 2:
                x_prev = stages[i](x_prev)
                x_main = stages[i](x_main)
                x_next = stages[i](x_next)

            # Block stages need drop_connect rate
            else:
                for module in stages[i]:
                    drop_connect = drop_connect_rate * block_number / len(self._blocks)
                    block_number += 1.0
                    x_prev = module(x_prev, drop_connect)
                    x_main = module(x_main, drop_connect)
                    x_next = module(x_next, drop_connect)
                    
                if i == 4:
                    xt1, _ = self.cross_attention_prev_4(x_main, x_prev)
                    xt2, _ = self.cross_attention_self_4(x_main, x_main)
                    xt3, _ = self.cross_attention_next_4(x_main, x_next)
                    xt = torch.cat([xt1, xt2, xt3], dim=1)
                    
                    xt = self.compress_4(xt)
                    xt_downcross = xt
                    
                    x_main = xt_downcross + x_main

                elif i == 5:
                    xt1, _ = self.cross_attention_prev_5(x_main, x_prev)
                    xt2, _ = self.cross_attention_self_5(x_main, x_main)
                    xt3, _ = self.cross_attention_next_5(x_main, x_next)
                    xt = torch.cat([xt1, xt2, xt3], dim=1)
                    
                    xt = self.compress_5(xt)
                    xt_downcross = xt
                    
                    x_main = xt_downcross + x_main

            features.append(x_main)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("_fc.bias", None)
        state_dict.pop("_fc.weight", None)
        super().load_state_dict(state_dict, strict=False, **kwargs)