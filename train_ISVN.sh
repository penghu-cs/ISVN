#!/usr/bin/env bash
output_shape=1024
datasets="MNIST SVHN" #nus_wide xmedianet INRIA-Websearch MNIST SVHN
batch_size=16
alpha=0.5
beta=1e-2
threshold=0.7
seed=1024
K=400
epochs=200

# train View 0
python train_ISVN.py --datasets $datasets --view_id 0 --seed $seed --epochs $epochs --batch_size $batch_size --output_shape $output_shape --beta $beta --alpha $alpha --threshold $threshold --K $K --gpu_id 0 &
# train View 1
python train_ISVN.py --datasets $datasets --view_id 1 --seed $seed --epochs $epochs --batch_size $batch_size --output_shape $output_shape --beta $beta --alpha $alpha --threshold $threshold --K $K --gpu_id 1 &

# python train_ISVN.py --view_id -1 --datasets $datasets --seed $seed --epochs $epochs --batch_size $batch_size --output_shape $output_shape --beta $beta --alpha $alpha --K $K --gpu_id -1 --multiprocessing

wait
# eval
python train_ISVN.py --mode eval --datasets $datasets --view -1 --output_shape $output_shape --beta $beta --alpha $alpha --threshold $threshold --K $K --gpu_id 0 --num_workers 0
