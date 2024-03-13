# scripts for imagenet1k pretrained models
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


# scripts for imagenet22k pretrained models
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
