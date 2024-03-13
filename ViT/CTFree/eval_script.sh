CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval.py\
            ./data \
            --multiprocessing-distributed \
            --vit_arch "medium" \
            --prop_arch "mobileone_s0" \
            --proposal_weights "./mobileone_weights/mobileone_s0.pth.tar" \
            --z 0.5  \
            -b 512 \
            --dist-url tcp://localhost:12341 \
            --ngpu 8 \
            --input_size 256 \
            --work_dir "./work_dirs"