from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from torchvision.models import resnet50, resnet18, resnet101, resnet152, \
                                mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small
from .mobileone import mobileone
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def make_papr_class(transformer_class):
    class PaPr(transformer_class):

        def update(self, proposal_model = 'resnet50', z = 0.5, input_size = 224, proposal_weights = ""):
            self.proposal_model = proposal_model

            dim = int(input_size * 12 / 384)

            if proposal_model == 'resnet18':     
                self.proposal = resnet18(pretrained=True)
                self.proposal.avgpool = nn.Identity()
                self.proposal.fc = nn.Sequential(nn.Unflatten(1, (512, dim, dim)))
            elif proposal_model == 'resnet50':     
                self.cnn = resnet50(pretrained=True)
                self.proposal.avgpool = nn.Identity()
                self.proposal.fc = nn.Sequential(nn.Unflatten(1, (2048, dim, dim)))
            elif proposal_model == 'resnet101':     
                self.proposal = resnet101(pretrained=True)
                self.proposal.avgpool = nn.Identity()
                self.proposal.fc = nn.Sequential(nn.Unflatten(1, (2048, dim, dim)))
            elif proposal_model == 'resnet152':     
                self.proposal = resnet152(pretrained=True)
                self.proposal.avgpool = nn.Identity()
                self.proposal.fc = nn.Sequential(nn.Unflatten(1, (2048, dim, dim)))
            elif "mobileone" in proposal_model:
                variant = self.proposal_model.split('_')[-1]
                self.proposal = mobileone(variant=variant, inference_mode=True)
                checkpoint = torch.load(proposal_weights, map_location="cpu")
                self.proposal.load_state_dict(checkpoint)
                self.proposal.fc = nn.Identity()

            self.z = z
            self.depth = len(self.blocks)


        @torch.no_grad()
        def extract_conv_features(self, x):
            if "res" in self.proposal_model:
                feat = self.proposal(x)
            elif "mobileone" in self.proposal_model:
                x = self.proposal.stage0(x)
                x = self.proposal.stage1(x)
                x = self.proposal.stage2(x)
                x = self.proposal.stage3(x)
                feat = self.proposal.stage4(x)
            else:
                raise Exception("The proposal model is not listed!")

            return feat

        def apply_papr(self, x, feat):
            batch, n, c = x.shape[0], x.shape[1], x.shape[2]
            h1 = w1 = int(np.sqrt(n))

            Fd = feat.mean(dim=1).unsqueeze(1)
            P = F.interpolate(Fd, size=(h1, w1), mode="bicubic", align_corners=True)
            P = P.view(batch, -1)

            nt = int(n * self.z)
            M = P.argsort(dim=1, descending=True)[:, :nt]
            x = x.gather(dim=1, index=M.unsqueeze(-1).expand(batch, -1, c))

            return x

        def forward_features(self, x):
            images = x
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)

            if self.z != 1.0:
                feat = self.extract_conv_features(images)                
                x = self.apply_papr(x, feat)

            for index, block in enumerate(self.blocks):
                x = block(x)

            x = self.norm(x)

            return x

    return PaPr


def create_class_token_free_ViT_model(
    model: VisionTransformer, proposal_model: str = "resnet50", z: float = 0.5, input_size: int = 224, proposal_weights: str = ""
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For training and for evaluating MAE models off the self set this to be False.
    """
    PaPr = make_papr_class(model.__class__)
    model.__class__ = PaPr
    model.update(proposal_model, z, input_size, proposal_weights)
