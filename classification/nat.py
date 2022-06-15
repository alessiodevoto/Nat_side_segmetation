"""
Neighborhood Attention Transformer.
https://arxiv.org/abs/2204.07143

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from natten import NeighborhoodAttention


model_urls = {
    "nat_mini_1k": "http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_mini.pth",
    "nat_tiny_1k": "http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_tiny.pth",
    "nat_small_1k": "http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_small.pth",
    "nat_base_1k": "http://ix.cs.uoregon.edu/~alih/nat/checkpoints/CLS/nat_base.pth",
}


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim, kernel_size=kernel_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        print(f'      [Nat Layer] Input of size:{x.shape}')
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            print(f'      [Nat Layer] Applying NAT to tensor of size:{x.shape}')
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            print(f'      [Nat Layer] Output of size:{x.shape}')
            return x
        shortcut = x
        x = self.norm1(x)
        print(f'      [Nat Layer] Applying NAT to tensor of size:{x.shape}')
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        print(f'      [Nat Layer] Output of size:{x.shape}')
        return x


class NATBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, kernel_size, num_classes=3, downsample=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 layer_scale=None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Initialize protoypes.
        print('Initializing NAT block with prototypes.')
        self.prototypes = torch.nn.Parameter(torch.Tensor(dim, num_classes), requires_grad=True)
        self.prototypes.data.uniform_(-1, 1)

        self.blocks = nn.ModuleList([
            NATLayer(dim=dim,
                     num_heads=num_heads, kernel_size=kernel_size,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                     drop=drop, attn_drop=attn_drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer, layer_scale=layer_scale)
            for i in range(depth)])

        self.downsample = None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)

    def forward(self, x):

        """# This should go before each attention layer.
        print(f'[NAT Block] Processing tensor of size: {x.shape}')
        print(f'[NAT Block] Processing prototypes of size: {self.prototypes.shape}')
        pixel_classes = x @ self.prototypes
        print(f'[NAT Block] Computed pixel classes of shape: {pixel_classes.shape}')
        # apply softmax
        attentive_prototypes = pixel_classes @ self.prototypes.t()
        print(f'[NAT Block] Computed attentive prototypes of shape: {attentive_prototypes.shape}')"""


        for blk in self.blocks:
            print(f'    [NAT block] Input of size:{x.shape}')
            # learn prototypes
            pixel_classes = torch.nn.functional.softmax(x @ self.prototypes, dim=3)
            attentive_prototypes = pixel_classes @ self.prototypes.t()
            x = x + attentive_prototypes
            # apply NAT layer
            x = blk(x)


        if self.downsample is None:
            return x
        print(f'    [NAT block] Output of size:{x.shape}')
        return self.downsample(x)


class NAT(nn.Module):
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 depths,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 kernel_size=7,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_levels - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(dim=int(embed_dim * 2 ** i),
                             depth=depths[i],
                             num_heads=num_heads[i],
                             kernel_size=kernel_size,
                             mlp_ratio=self.mlp_ratio,
                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                             drop=drop_rate, attn_drop=attn_drop_rate,
                             drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                             norm_layer=norm_layer,
                             downsample=(i < self.num_levels - 1),
                             layer_scale=layer_scale)
            self.levels.append(level)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        print(f'  [NAT] Input of size: {x.shape}')
        x = self.patch_embed(x)
        print(f'  [NAT] Patch embed: {x.shape}')
        x = self.pos_drop(x)
        print(f'  [NAT] Pos drop: {x.shape}')

        for level in self.levels:
            x = level(x)
            print(f'  [NAT] Level out: {x.shape}')


        x = self.norm(x).flatten(1, 2)
        print(f'  [NAT] flatten 1 out: {x.shape}')
        x = self.avgpool(x.transpose(1, 2))
        print(f'  [NAT] avg pool out: {x.shape}')
        x = torch.flatten(x, 1)
        print(f'  [NAT] flatten 2 out: {x.shape}')
        return x

    def forward(self, x):
        print(f'[Transformer NAT] starting: {x.shape}')
        x = self.forward_features(x)
        print(f'[Transformer NAT] fwd: {x.shape}')
        x = self.head(x)
        print(f'[Transformer NAT] head: {x.shape}')
        return x


@register_model
def nat_mini(pretrained=False, **kwargs):
    model = NAT(depths=[3, 4, 6, 5], num_heads=[2, 4, 8, 16], embed_dim=64, mlp_ratio=3,
                 drop_path_rate=0.2, kernel_size=7, **kwargs)
    if pretrained:
        url = model_urls['nat_mini_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def nat_tiny(pretrained=False, **kwargs):
    model = NAT(depths=[3, 4, 18, 5], num_heads=[2, 4, 8, 16], embed_dim=64, mlp_ratio=3,
                drop_path_rate=0.2, kernel_size=7, **kwargs)
    if pretrained:
        url = model_urls['nat_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def nat_small(pretrained=False, **kwargs):
    model = NAT(depths=[3, 4, 18, 5], num_heads=[3, 6, 12, 24], embed_dim=96, mlp_ratio=2,
                drop_path_rate=0.3, layer_scale=1e-5, kernel_size=7, **kwargs)
    if pretrained:
        url = model_urls['nat_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def nat_base(pretrained=False, **kwargs):
    model = NAT(depths=[3, 4, 18, 5], num_heads=[4, 8, 16, 32], embed_dim=128, mlp_ratio=2,
                drop_path_rate=0.5, layer_scale=1e-5, kernel_size=7, **kwargs)
    if pretrained:
        url = model_urls['nat_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint)
    return model
