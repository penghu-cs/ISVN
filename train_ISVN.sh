#!/usr/bin/env bash
output_shape=1024
gama=1
datasets="MNIST SVHN" #nus_wide xmedianet INRIA-Websearch MNIST SVHN
batch_size=16
beta=0.5
alpha=8e-2
threshold=0.9
seed=0
K=400
epochs=200

# train View 1
python train_ISVN.py --datasets $datasets --view_id 1 --seed $seed --epochs $epochs --batch_size $batch_size --output_shape $output_shape --beta $beta --alpha $alpha --threshold $threshold --K $K --gpu_id 1 &
# train View 0
python train_ISVN.py --datasets $datasets --view_id 0 --seed $seed --epochs $epochs --batch_size $batch_size --output_shape $output_shape --beta $beta --alpha $alpha --threshold $threshold --K $K --gpu_id 0

# python train_ISVN.py --view_id -1 --datasets $datasets --seed $seed --epochs $epochs --batch_size $batch_size --output_shape $output_shape --beta $beta --alpha $alpha --K $K --gpu_id -1 --multiprocessing

# eval
python train_ISVN.py --mode eval --datasets $datasets --view -1 --output_shape $output_shape --beta $beta --alpha $alpha --threshold $threshold --K $K --gpu_id 0 --num_workers 0
