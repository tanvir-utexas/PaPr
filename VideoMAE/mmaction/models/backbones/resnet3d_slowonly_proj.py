# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmaction.registry import MODELS
from .resnet3d_slowfast import ResNet3dPathway
import torch.nn.functional as F

@MODELS.register_module()
class SlowOnly_proj(ResNet3dPathway):
    """SlowOnly backbone based on ResNet3dPathway.

    Args:
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Defaults to ``(1, 7, 7)``.
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Defaults to 1.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Defaults to 1.
        inflate (Sequence[int]): Inflate dims of each block.
            Defaults to ``(0, 0, 1, 1)``.
        with_pool2 (bool): Whether to use pool2. Defaults to False.
    """

    def __init__(self,
                 model = "res50",
                 conv1_kernel: Sequence[int] = (1, 7, 7),
                 conv1_stride_t: int = 1,
                 pool1_stride_t: int = 1,
                 inflate: Sequence[int] = (0, 0, 1, 1),
                 with_pool2: bool = False,
                 **kwargs
                 ) -> None:

        if model == "res50":
            super().__init__(
                conv1_kernel=conv1_kernel,
                conv1_stride_t=conv1_stride_t,
                pool1_stride_t=pool1_stride_t,
                inflate=inflate,
                with_pool2=with_pool2,
                depth=50,
                pretrained="./checkpoints/slowonly_r50_8xb16-8x8x1-256e_kinetics400-rgb_20220901-2132fc87.pth", 
                lateral=False, 
                norm_eval=False
                )

        elif model == "res101":
            super().__init__(
                conv1_kernel=conv1_kernel,
                conv1_stride_t=conv1_stride_t,
                pool1_stride_t=pool1_stride_t,
                inflate=inflate,
                with_pool2=with_pool2,
                depth=101, 
                pretrained="./checkpoints/slowonly_r50_8xb16-8x8x1-256e_kinetics400-rgb_20220901-2132fc87.pth", 
                lateral=False, 
                norm_eval=False
                )
        
        self.init_weights()

        assert not self.lateral


    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        if x.shape[-1] != 224 or x.shape[-3] != 8:
            x = F.interpolate(x, (8, 224, 224), align_corners=True, mode="trilinear")


        x = self.conv1(x)
        if self.with_pool1:
            x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)
            if i in self.out_indices:
                outs.append(x)
        
        x = outs[0]
        
        x = F.interpolate(x, (8, 14, 14), align_corners=True, mode="trilinear")
        # x = self.avg_pool(x)
        x = x.mean(dim=1, keepdim=True)
        x = x.flatten(2).transpose(1, 2).squeeze(-1)
        x = F.normalize(x, dim=1)

        mask = x.sort(dim=1, descending=True).indices

        return mask