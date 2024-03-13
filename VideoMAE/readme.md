# Prepare Environment

We perform video experiments using [mmaction2](https://github.com/open-mmlab/mmaction2) framework. Please install the framework using following codes.

```
conda create --name papr_video python=3.8 -y
conda activate papr_video

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet

pip install -v -e .

```

# Prepare Dataset

Download the datasets from [kinetics400val](https://mycuhk-my.sharepoint.com/personal/1155136485_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155136485%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2Fkinetics%5F400%5Fval%5F320%2Etar&parent=%2Fpersonal%2F1155136485%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments&ga=1) and prepare `data` in the following format. Also, download the pretrained models from [mmaction2](https://github.com/open-mmlab/mmaction2?tab=readme-ov-file) and put in the `checkpoints` directory. 

```markdown
    VideoMAE/
    ├── data/
    │   ├── kinetics400/
    │   |   ├── videos_val/
    │   |   |    ├── __lt03EF4ao.mp4/
    │   |   |    ├── __NrybzYzUg.mp4/
    │   |   |    ├──  ....
    │   │   ├── kinetics400_val_list_videos.txt
    │   │   └── kinetics400_class2ind.txt
    ├── checkpoints/
    │   ├── vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth
    │   ├── vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth
    │   ├── slowonly_r50_8xb16-8x8x1-256e_kinetics400-rgb_20220901-2132fc87.pth
    │   ├── slowonly_r101_8xb16-8x8x1-196e_kinetics400-rgb_20220901-e6281431.pth
    │   ├── x3d_s_13x6x1_facebook-kinetics400-rgb_20201027-623825a0.pth
    │   └── x3d_m_16x5x1_facebook-kinetics400-rgb_20201027-3f42382a.pth
    |   ......
    └── readme.md
```


# Evaluation

To run `VideoMAE-Base` with `PaPr`, use the following command.

```
    bash ./tools/dist_test_mae_base_papr.sh
```


To run `VideoMAE-Large` with `PaPr`, use the following command.

```
    bash ./tools/dist_test_mae_large_papr.sh
```