
for (( i=4; i < 200; i++ )); do
    python evaluation/tile_prediction.py \
    --result_dir results/ucf101_wgan_64px_relu_zdim-100_svd-freq-5_no-beta_2017-03-15_22-05-47_0 \
    --iter 295000 --gpu 0 --out_dir suplemental/tiles --seed $i
done
