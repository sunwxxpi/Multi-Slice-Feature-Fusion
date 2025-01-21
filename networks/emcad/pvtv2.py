import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model

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
        
        """ # (1) 윈도우 분할 대신, 전체 (H×W)에 대한 쿼리/키/값 생성
        #     -> 기존 window_partition 제거
        query = self.query_conv(x_otherBranch)   # (B, inter_channels, H, W)
        key   = self.key_conv(x_thisBranch)      # (B, inter_channels, H, W)
        value = self.value_conv(x_thisBranch)    # (B, inter_channels, H, W)

        # (2) Multi-head을 위해 (B, inter_channels, H, W)를 (B, num_heads, head_dim, H*W)로 변환
        #     그리고 (B, num_heads, H*W, head_dim) 형태가 되도록 permute
        N = H * W  # 전체 픽셀 수
        query = query.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, num_heads, N, head_dim)
        key   = key.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, num_heads, N, head_dim)
        value = value.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, num_heads, N, head_dim)

        # (3) 어텐션 스코어 계산
        #     attention_scores: (B, num_heads, (num_global_tokens + N), (num_global_tokens + N))
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # (4) 어텐션 결과(문맥 벡터) 계산
        #     out: (B, num_heads, (num_global_tokens + N), head_dim)
        out = torch.matmul(attention_weights, value)

        # (5) 다시 (B, inter_channels, H, W) 형태로 복원
        out = out.permute(0, 1, 3, 2).contiguous()  # (B, num_heads, head_dim, N)
        out = out.view(B, self.inter_channels, H, W)

        # (6) 최종 projection
        z = self.W_z(out)  # (B, C, H, W) """

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
        z = self.W_z(x_un)
        
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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerImpr(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        
        # Non-local block 파라미터
        self.window_size = 16
        self.num_global_tokens = 1
        self.num_heads = 16

        # stage3용 Non-Local Block
        self.cross_attention_self_3 = NonLocalBlock(in_channels=320, inter_channels=160, 
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_cross_3 = NonLocalBlock(in_channels=320, inter_channels=160, 
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.compress_3 = nn.Conv2d(960, 320, kernel_size=1, bias=False)
        self.double_conv_3 = DoubleConv(320, 320, 320)

        # stage4용 Non-Local Block
        self.cross_attention_self_4 = NonLocalBlock(in_channels=512, inter_channels=256,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.cross_attention_cross_4 = NonLocalBlock(in_channels=512, inter_channels=256,
                                                    num_heads=self.num_heads, window_size=self.window_size, num_global_tokens=self.num_global_tokens)
        self.compress_4 = nn.Conv2d(1536, 512, kernel_size=1, bias=False)
        self.double_conv_4 = DoubleConv(512, 512, 512)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = 1
            #load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # def _get_pos_embed(self, pos_embed, patch_embed, H, W):
    #     if H * W == self.patch_embed1.num_patches:
    #         return pos_embed
    #     else:
    #         return F.interpolate(
    #             pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
    #             size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x_prev, x_main, x_next):
        B = x_main.shape[0]
        outs = []
        
        ####################### stage 1 #######################
        x_prev, H, W = self.patch_embed1(x_prev)
        x_main, H, W = self.patch_embed1(x_main) # (16, 3, 512, 512) -> (16, (512/4)*(512/4), 64)
        x_next, H, W = self.patch_embed1(x_next)
        
        for i, blk in enumerate(self.block1):
            x_prev = blk(x_prev, H, W)
            x_main = blk(x_main, H, W)
            x_next = blk(x_next, H, W)
            
        x_prev = self.norm1(x_prev)
        x_main = self.norm1(x_main)
        x_next = self.norm1(x_next)
        
        x_prev = x_prev.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_main = x_main.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # (16, (512/4)*(512/4), 64) -> (16, 64, (512/4), (512/4))
        x_next = x_next.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        outs.append(x_main)

        ####################### stage 2 #######################
        x_prev, H, W = self.patch_embed2(x_prev)
        x_main, H, W = self.patch_embed2(x_main) # (16, 64, (512/4), (512/4)) -> (16, (512/8)*(512/8), 128)
        x_next, H, W = self.patch_embed2(x_next)
        
        for i, blk in enumerate(self.block2):
            x_prev = blk(x_prev, H, W)
            x_main = blk(x_main, H, W)
            x_next = blk(x_next, H, W)
            
        x_prev = self.norm2(x_prev)
        x_main = self.norm2(x_main)
        x_next = self.norm2(x_next)
        
        x_prev = x_prev.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_main = x_main.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # (16, (512/8)*(512/8), 128) -> (16, 128, (512/8), (512/8))
        x_next = x_next.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        outs.append(x_main)

        ####################### stage 3 #######################
        x_prev, H, W = self.patch_embed3(x_prev)
        x_main, H, W = self.patch_embed3(x_main) # (16, 128, (512/8), (512/8)) -> (16, (512/16)*(512/16), 320)
        x_next, H, W = self.patch_embed3(x_next)
        
        for i, blk in enumerate(self.block3):
            x_prev = blk(x_prev, H, W)
            x_main = blk(x_main, H, W)
            x_next = blk(x_next, H, W)
            
        x_prev = self.norm3(x_prev)
        x_main = self.norm3(x_main)
        x_next = self.norm3(x_next)
        
        x_prev = x_prev.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_main = x_main.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # (16, (512/16)*(512/16), 320) -> (16, 320, (512/16), (512/16))
        x_next = x_next.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        xt_1, _ = self.cross_attention_cross_3(x_main, x_prev)
        xt_2, _ = self.cross_attention_self_3(x_main, x_main)
        xt_3, _ = self.cross_attention_cross_3(x_main, x_next)
        xt = torch.cat([xt_1, xt_2, xt_3], dim=1)
        
        xt = self.compress_3(xt)
        xt_downcross = self.double_conv_3(xt)
        
        x_main = xt_downcross + x_main
        
        outs.append(x_main)

        ####################### stage 4 #######################
        x_prev, H, W = self.patch_embed4(x_prev)
        x_main, H, W = self.patch_embed4(x_main) # (16, 320, (512/16), (512/16)) -> (16, (512/32)*(512/32), 512)
        x_next, H, W = self.patch_embed4(x_next)
        
        for i, blk in enumerate(self.block4):
            x_prev = blk(x_prev, H, W)
            x_main = blk(x_main, H, W)
            x_next = blk(x_next, H, W)
            
        x_prev = self.norm4(x_prev)
        x_main = self.norm4(x_main)
        x_next = self.norm4(x_next)
        
        x_prev = x_prev.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_main = x_main.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # (16, (512/32)*(512/32), 512) -> (16, 512, (512/32), (512/32))
        x_next = x_next.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        xt_1, _ = self.cross_attention_cross_4(x_main, x_prev)
        xt_2, _ = self.cross_attention_self_4(x_main, x_main)
        xt_3, _ = self.cross_attention_cross_4(x_main, x_next)
        xt = torch.cat([xt_1, xt_2, xt_3], dim=1)
        
        xt = self.compress_4(xt)
        xt_downcross = self.double_conv_4(xt)
        
        x_main = xt_downcross + x_main
        
        outs.append(x_main)

        return outs

        # return x.mean(dim=1)

    def forward(self, x_prev, x_main, x_next):
        x = self.forward_features(x_prev, x_main, x_next)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
class pvt_v2_b0(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)



@register_model
class pvt_v2_b1(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

@register_model
class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

@register_model
class pvt_v2_b3(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

@register_model
class pvt_v2_b4(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@register_model
class pvt_v2_b5(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)