CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval.py\
            ./data \
            --multiprocessing-distributed \
            --vit_arch "base" \
            --prop_arch "mobileone_s0" \
            --proposal_weights "./mobileone_weights/mobileone_s0.pth.tar" \
            --z 0.5  \
            --r_merged 0 \
            -b 512 \
            --dist-url tcp://localhost:12341 \
            --ngpu 8 \
            --with_ct \
            --work_dir "./work_dirs"