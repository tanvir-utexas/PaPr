# Prepare Environments
Run the following codes to prepare the environment.

```
conda create -n papr_hr python=3.9
conda activate papr_hr
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install timm==0.3.2
pip install tensorboardX
pip install six
pip install fvcore
```

# Prepare Data and Model Weights

Prepare the ImageNet validation set in the following structure. Also, download mobileone weights from [ml-mobileone](URL "https://github.com/apple/ml-mobileone") and put in the `mobileone_weights` directory. Download `swin` model weights from [link]() and put in `pretrained_weights/swin` directory. Download `convnext` model weights from [link]() and put in `pretrained_weights/convnext` directory.

```markdown
    Hierarchical/
    ├── data/
    │   ├── val/
    │   │   ├── n01440764
    │   │   |── ...
    │   │   └── n01443537
    ├── mobileone_weights/
    │   ├── mobileone_s0.pth.tar
    ├── pretrained_weights/
    │   ├── convnext/
    │   │   ├── convnext_base_1k_224_ema.pth
    │   │   |── convnext_base_22k_1k_224.pth
    │   │   └── ...
    │   ├── swin/
    │   │   ├── swin_base_patch4_window7_224.pth
    │   │   |── swin_base_patch4_window7_224_22kto1k.pth
    │   │   └── ...
    └── README.md
```

# Evaluate

Use the following code to run evaluation on ConvNext with ImageNet1k pre-trained weights.

```
CUDA_VISIBLE_DEVICES=0 python ./eval.py \
                --data_path ./data \
                --model papr_convnext-b \
                --model_path ./pretrained_weights/convnext \
                --batch_size 512 \
                --z 0.65 \
                --input_size 224 \
                --mask_block 4 \
                --mobileone_weights ./mobileone_weights \
                --work_dir ./work_dirs/papr_convnext-b/z_0.65
```

Use the following code to run evaluation on ConvNext with ImageNet22k pre-trained weights.

```
CUDA_VISIBLE_DEVICES=0 python ./eval.py \
                --data_path ./data \
                --model papr_convnext-b \
                --model_path ./pretrained_weights/convnext \
                --batch_size 512 \
                --z 0.65 \
                --in22k \
                --mask_block 4 \
                --input_size 224 \
                --mobileone_weights ./mobileone_weights \
                --work_dir ./work_dirs/papr_convnext-b_in22k/z_0.65
```


Use the following code to run evaluation on Swin with ImageNet1k pre-trained weights.

```
CUDA_VISIBLE_DEVICES=0 python ./eval.py \
                --data_path ./data \
                --model papr_swin-b \
                --model_path ./pretrained_weights/swin \
                --batch_size 512 \
                --z 0.65 \
                --input_size 224 \
                --mask_block 4 \
                --mobileone_weights ./mobileone_weights \
                --work_dir ./work_dirs/papr_swin-b/z_0.65
```

Use the following code to run evaluation on Swin with ImageNet22k pre-trained weights.

```
CUDA_VISIBLE_DEVICES=0 python ./eval.py \
                --data_path ./data \
                --model papr_swin-b \
                --model_path ./pretrained_weights/pretrained/swin \
                --batch_size 512 \
                --z 0.65 \
                --in22k \
                --input_size 224 \
                --mask_block 4 \
                --mobileone_weights ./mobileone_weights \
                --work_dir ./work_dirs/papr_swin-b/z_0.65
```