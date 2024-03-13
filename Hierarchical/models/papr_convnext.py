# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from utils import batch_index_fill, batch_index_select
from .mobileone import mobileone
import os
from torchvision.models import resnet152

class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class PaPrBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward_ffn(self, x):
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        return x

    def forward(self, x, mask=None):
        input_x = x

        if mask is None: # compatible with the original implementation 
            x = self.dwconv(x)

            x = x.permute(0, 2, 3, 1)        # (N, C, H, W) -> (N, H, W, C)
            x = self.forward_ffn(x)
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

            x = input_x + self.drop_path(x)
            return x
        else:
            idx1, idx2 = mask
            N, C, H, W = x.shape

            ############################################
            if not self.training:
                input_x = input_x.permute(0, 2, 3, 1).reshape(N, H*W, C) # (N, C, H, W) -> (N, H, W, C)

                x1 = batch_index_select(input_x, idx1)
                x2 = batch_index_select(input_x, idx2)

                x2 = torch.zeros_like(x2)

                input_x = torch.zeros_like(input_x)
                input_x = batch_index_fill(input_x, x1, x2, idx1, idx2)

                input_x = input_x.reshape(N, H, W, C).permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

            ############################################
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 1).reshape(N, H*W, C) # (N, C, H, W) -> (N, H, W, C)

            x1 = batch_index_select(x, idx1)
            x2 = batch_index_select(x, idx2)

            x2 = torch.zeros_like(x2)

            x1 = self.forward_ffn(x1)

            x = torch.zeros_like(x)
            x = batch_index_fill(x, x1, x2, idx1, idx2)

            x = x.reshape(N, H, W, C).permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

            x = input_x + self.drop_path(x)
            return x


@torch.no_grad()
def mask_predictor(x, mask, sparse_ratio):
        B, C, H, W = x.size()

        if (mask.shape[-2] != H) or (mask.shape[-1] != W):
            mask = F.interpolate(mask, size=(H, W), mode="bicubic", align_corners=True) # [batch, 1, 14, 14]

        mask = mask.view(B, -1)
        mask = F.normalize(mask, dim=-1)

        idx = torch.argsort(mask, dim=1, descending=True)

        num_keep_node = int(sparse_ratio * H * W)
        idx1 = idx[:, :num_keep_node]
        idx2 = idx[:, num_keep_node:]

        return [idx1, idx2]


class PaPrConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1., 
                 fraction=1.0, mask_block=[0,1,2,3],
                 mobileone_weights='', cnn_size = None):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[PaPrBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )

            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        # new modules
        self.fraction = fraction
        self.mask_block = mask_block
        self.mobileone_weights = mobileone_weights
        self.cnn_size = cnn_size
        # init 

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def update(self):
        self.cnn = mobileone(variant="s0", inference_mode=True)
        checkpoint = torch.load(os.path.join(self.mobileone_weights, 'mobileone_s0.pth.tar'), map_location="cpu")
        self.cnn.load_state_dict(checkpoint)
        self.cnn.fc = nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def get_costum_param_groups(self, weight_decay):
        decay = []
        no_decay = []
        new_param = []
        for name, param in self.named_parameters():
            if 'fast_path' in name or 'predictor' in name:
                new_param.append(param)
            elif not param.requires_grad:
                continue  # frozen weights
            elif 'cls_token' in name or 'pos_embed' in name: #or 'patch_embed' in name:
                continue  # frozen weights
            elif len(param.shape) == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': new_param, 'name': 'new_param', 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0., 'name': 'base_no_decay', 'small_lr_scalar': 0.01, 'fix_step': 5},
            {'params': decay, 'weight_decay': weight_decay, 'name': 'base_decay', 'small_lr_scalar': 0.01, 'fix_step': 5}
            ]
    
    def forward(self, x):
        if self.fraction != 1.0:
            with torch.no_grad():
                if (self.cnn_size is not None) and (self.cnn_size != x.shape[-1]):
                    t = F.interpolate(x, (self.cnn_size, self.cnn_size), mode='bilinear', align_corners=True)
                else:
                    t = x
                t = self.cnn.stage0(t)
                t = self.cnn.stage1(t)
                t = self.cnn.stage2(t)
                t = self.cnn.stage3(t)
                t = self.cnn.stage4(t)
                t = self.cnn.fc(t)
                z = t.mean(dim=1).unsqueeze(1)
                
                base_mask = z

        for i in range(4):
            x = self.downsample_layers[i](x)
            for _, layer in enumerate(self.stages[i]):
                if (i in self.mask_block) and (self.fraction != 1.0):
                    with torch.no_grad():
                        mask = mask_predictor(x, base_mask, self.fraction)

                    x = layer(x, mask)
                else:
                    x = layer(x)
        
        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)

        
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
