#!/bin/bash

for (( i=0; i < 200; i++ )); do
    python infer.py --result_dir results/ucf101_wgan_64px_relu_zdim-100_svd-freq-5_no-beta_2017-03-15_22-05-47_0 \
    --iter 295000 \
    --gpu 0 \
    --n 100 \
    --video \
    --out_dir supplemental/videos_295000 \
    --seed $i
done

# python infer.py --result_dir results/ucf101_wgan_64px_relu_zdim-100_svd-freq-5_no-beta_2017-03-15_22-05-47_0 \
# --iter 295000 \
# --gpu 0 \
# --n 100 \
# --images \
# --out_dir supplemental \
# --seed 41
