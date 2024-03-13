# PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference (Under Review)

This repository contains official codes for PaPr implementation. Since we apply PaPr on various architectures, we had to prepare separate environments for each setup to follow the baseline implementation. We provide brief overview of supported model architectures.


## Hierarchical Models

We apply PaPr on various version ConvNext and Swin transformers. Please follow the `Hierarchical` folder for more details.

## ViT Models

We apply PaPr on supervised Augreg (see `ViT/AugReg`), on class-token free ViTs (see `ViT/CTFree`), and on self-supervised MAEs (see `ViT/MAE`). Please follow the respective folder for more details.

## VideoMAE Models

We apply PaPr on VideoMAE models in Kinetics400 evaluation. Please follow the `VideoMAE` folder for more details.


## Acknowledgements

We borrowed codes heavily from [DynamicViT](https://github.com/raoyongming/DynamicViT), [ToMe](https://github.com/facebookresearch/ToMe), and [mmaction2](https://github.com/open-mmlab/mmaction2). We thank them for their amazing work.
