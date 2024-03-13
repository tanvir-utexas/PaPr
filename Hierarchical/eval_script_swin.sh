# scripts for imagenet1k pretrained models
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


# scripts for imagenet22k pretrained models
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