# Prepare Environments
Run the following codes to prepare the environment.

```
conda create -n papr_mae python=3.9
conda activate papr_mae
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install timm==0.3.2
pip install fvcore
```

`timm==0.3.2` creates some issues with torch 1.8.1+. Please follow this approach [fix](URL "https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842") to solve it.

# Prepare Data and Model Weights

Prepare the ImageNet validation set in the following structure. Download mobileone weights from [ml-mobileone](URL "https://github.com/apple/ml-mobileone") and put in the `mobileone_weights` directory. Download the fine-tuned MAE weights from [mae](URL "https://github.com/facebookresearch/mae").

```markdown
    MAE/
    ├── data/
    │   ├── val/
    │   │   ├── n01440764
    │   │   |── ...
    │   │   └── n01443537
    ├── mobileone_weights/
    │   ├── mobileone_s0.pth.tar
    ├── mae_weights/
    │   ├── mae_finetuned_vit_base.pth
    │   ├── mae_finetuned_vit_large.pth
    │   └── mae_finetuned_vit_huge.pth
    └── README.md
```

# Evaluate

Use the following code to run evaluation.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval.py\
                    ./data \                      
                    --multiprocessing-distributed \
                    --vit_arch base \                   
                    --prop_arch mobileone_s0 \          
                    --proposal_weights "./mobileone_weights/mobileone_s0.pth.tar" \ 
                    --z 0.5  \      
                    -b 512 \
                    --input_size 256 \
                    --dist-url tcp://localhost:12341 \
                    --ngpu 8 \
                    --work_dir "./work_dirs"

```